//! Running one tool under the machine guards: an address-space cap and a wall-clock timeout, its output captured.
use crate::refusal::{Done, Kind, Refusal};
use std::collections::HashMap;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// The wall-clock limit of a version or fact probe, in seconds. It must tolerate a cold filesystem (the first start of a
/// compiler binary is pure I/O), not a slow compiler; an answer is memoised, so it is paid once.
pub const PROBE_SECONDS: u64 = 180;

/// The sh program that lowers the soft address-space limit to its first argument (KiB) and starts the command. A failed `ulimit`
/// (a cap above the inherited hard limit, or a system that cannot cap the address space) leaves the limit as it was and the
/// command still runs: the timeout and the source budget bound it then.
const CAP_PROGRAM: &str = r#"ulimit -S -v "$1" 2>/dev/null; shift; exec "$@""#;

extern "C" {
    /// POSIX `kill(2)`: a negative pid signals the whole process group.
    fn kill(pid: i32, sig: i32) -> i32;
}

/// What a finished tool left.
#[derive(Debug)]
pub struct Output {
    /// The exit code; `None` when a signal ended it.
    pub code: Option<i32>,
    /// What it wrote to standard output.
    pub stdout: String,
    /// What it wrote to standard error.
    pub stderr: String,
}

/// Reads `pipe` to its end on its own thread, so a full pipe never stalls the tool.
fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut bytes = vec![];
        let _ = pipe.read_to_end(&mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Runs `argv` with exactly the variables of `env`, under `cap_bytes` of address space (0: none) and `timeout_seconds` (0: none).
/// A program that cannot be started answers exit code 127, as a shell would. A timeout kills the tool's whole process group
/// (`nvcc` starts `cicc` and `ptxas`) and refuses by name.
///
/// # Errors
/// Refuses with a [`Refusal`](crate::refusal::Refusal) that names what is wrong with the request or the tool.
pub fn run<S: std::hash::BuildHasher>(
    argv: &[String],
    env: &HashMap<String, String, S>,
    cap_bytes: u64,
    timeout_seconds: u64,
) -> Done<Output> {
    let capped = cap_bytes > 0 && std::path::Path::new("/bin/sh").exists();
    let mut command = if capped {
        let mut c = Command::new("/bin/sh");
        c.args([
            "-c",
            CAP_PROGRAM,
            "sh",
            &cap_bytes.div_ceil(1024).to_string(),
        ])
        .args(argv);
        c
    } else {
        let mut c = Command::new(
            argv.first()
                .ok_or_else(|| Refusal::bad("an empty command line"))?,
        );
        c.args(&argv[1..]);
        c
    };
    command
        .env_clear()
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            return Ok(Output {
                code: Some(127),
                stdout: String::new(),
                stderr: format!("{}: {e}", argv[0]),
            })
        }
    };
    let (out, err) = match (child.stdout.take(), child.stderr.take()) {
        (Some(o), Some(e)) => (drain(o), drain(e)),
        _ => {
            return Err(Refusal::new(
                Kind::Io,
                "a child process has no output pipes",
            ))
        }
    };
    let deadline =
        (timeout_seconds > 0).then(|| Instant::now() + Duration::from_secs(timeout_seconds));
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if deadline.is_some_and(|d| Instant::now() >= d) => {
                stop_group(&mut child);
                return Err(Refusal::new(
                    Kind::TimedOut,
                    format!("{} did not finish within {timeout_seconds} s", argv[0]),
                ));
            }
            Ok(None) => thread::sleep(Duration::from_millis(2)),
            Err(e) => {
                return Err(Refusal::new(
                    Kind::Io,
                    format!("waiting for {}: {e}", argv[0]),
                ))
            }
        }
    };
    let join = |h: thread::JoinHandle<String>| h.join().unwrap_or_default();
    Ok(Output {
        code: status.code(),
        stdout: join(out),
        stderr: join(err),
    })
}

/// Kills the process group of `child` and reaps it.
fn stop_group(child: &mut std::process::Child) {
    if let Ok(pid) = i32::try_from(child.id()) {
        // SAFETY: `kill` takes two integers and touches no memory; the group was made by `process_group(0)` for this child.
        unsafe {
            kill(-pid, 9);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> HashMap<String, String> {
        HashMap::from([("PATH".to_string(), "/bin:/usr/bin".to_string())])
    }

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| (*p).to_string()).collect()
    }

    #[test]
    fn output_and_exit_code_are_captured() {
        let done = run(
            &argv(&["/bin/sh", "-c", "echo out; echo err >&2; exit 3"]),
            &env(),
            0,
            0,
        )
        .unwrap();
        assert_eq!(
            (done.code, done.stdout.trim(), done.stderr.trim()),
            (Some(3), "out", "err")
        );
    }

    #[test]
    fn a_program_that_does_not_exist_answers_127() {
        let done = run(&argv(&["/no/such/tool"]), &env(), 0, 0).unwrap();
        assert_eq!(done.code, Some(127));
    }

    #[test]
    fn a_tool_over_its_time_is_refused_by_name() {
        let refused = run(&argv(&["/bin/sh", "-c", "sleep 30"]), &env(), 0, 1).unwrap_err();
        assert_eq!(refused.kind, Kind::TimedOut);
    }

    #[test]
    fn a_capped_run_starts_the_command_through_the_cap_program() {
        let done = run(&argv(&["/bin/sh", "-c", "echo capped"]), &env(), 1 << 40, 0).unwrap();
        assert_eq!(done.stdout.trim(), "capped");
    }
}
