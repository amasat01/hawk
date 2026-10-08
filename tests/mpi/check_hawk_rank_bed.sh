#!/usr/bin/env bash
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
#
# tests/mpi/check_hawk_rank_bed.sh — the integrity gate for HAWK's MPI rank bed.
#
# WHAT IT DOES, IN ORDER, AND WHY EACH STEP IS HERE.
#
#   THE GATE BUILDS ITS OWN INPUTS, AND NAMES THEIR DIGEST. It emits and
#       compiles a HAWK artifact — four units, one per access class the rank bed
#       certifies — into a scratch directory, ONCE, before any rank starts (two
#       ranks compiling into the same path would race), and prints the sha256
#       over the DEPLOYED bytes. A gate that reused whatever artifact was lying
#       around would certify a body nobody in this run built; a verdict that did
#       not name the digest would not say WHICH bytes came out green.
#
#   EAGLE'S OWN RANK BED, DRIVEN BY THAT ARTIFACT. eagle's
#       python/tests/mpi/check_rank_bed.sh documents $EAGLE_RANK_BED_ARTIFACT as
# the door where the artifact is a real HAWK emission — this is that
#       door being walked through. It certifies the STRUCTURE: the contiguous
# rank cut, the replicated-input ruling for cross_sample_read, the
# refusal on an unsupported access class, the rank-ordered fold, each
# with its own pre-gather non-vacuity arm. Its rc is captured DIRECTLY
# into a variable, never read through a pipe, because a pipeline
# reports the LAST command's status and a verdict that cannot reach the
# caller is not a verdict.
#
#   HAWK'S OWN ROWS, on the same artifact: the gathered plane against
# `hawk._core`'s SERIAL oracle, bit for bit. eagle's bed compares a
# rank run against eagle's own serial arm; this compares it against
# HAWK's own oracle. Neither subsumes the other, so both run.
#
# THE JUDGEMENT, in the shape eagle's bed and the C++ bed before it both use:
#
#     pinned == collected == ran(rank 0)   AND   verdict(rank 1) == verdict(rank 0)
#
# where a verdict is (tests, failures, errors, skipped) read out of that rank's
# OWN JUnit XML — skipped included, because a skipped row is pinned and certifies
# nothing, which is the vacuity failure at the granularity of one row. `mpirun`
# returns 0 when every rank returns 0, and a rank that never reached a row
# returns 0 too, so the launcher's status alone certifies nothing.
#
# NON-VACUITY IS PROVEN, NOT ASSUMED: `-k NoSuchTest` selects nothing and
# the SAME judgement is applied to it; this gate refuses to report GREEN unless
# that probe comes out RED.
#
# The expectation is a committed NAME MANIFEST, never a count: a scalar is blind
# to a same-commit swap (delete one, add one) and an ambient $PYTEST_ADDOPTS
# silences a run and a listing identically. Re-mint is refused when $CI is set —
# a gate that regenerates its own expectation is a tautology.
#
# Usage:
#   tests/mpi/check_hawk_rank_bed.sh
#   tests/mpi/check_hawk_rank_bed.sh --remint [--allow-removals]
#
# Environment:
#   PY                       the interpreter (default: `python3` from $PATH)
#   MPIRUN                   the launcher (default: `mpirun` from $PATH)
#   HAWK_MPI_BED_RANKS       the world size (default: 2)
#   EAGLE_PYTHON_ROOT        eagle/python (default: the sibling checkout)
#   EAGLE_RANK_BED_ARTIFACT  an existing artifact root to drive (default: built here)
#   HAWK_BED_CACHE           a compile-cache directory to reuse across runs
#
set -u -o pipefail

PROG="tests/mpi/check_hawk_rank_bed.sh"
RANKS="${HAWK_MPI_BED_RANKS:-2}"
MPIRUN="${MPIRUN:-mpirun}"
PY="${PY:-python3}"

usage() {
    cat >&2 <<EOF
usage: $PROG [--remint [--allow-removals]]

  --remint            regenerate the pinned node-id manifest from THIS collection
                      (local only; refused in CI)
  --allow-removals    required when a re-mint would DELETE manifest lines

No count is ever passed on the command line — the manifest file is the expectation.
EOF
    exit 2
}

REMINT=0
ALLOW_REMOVALS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --remint)         REMINT=1 ;;
        --allow-removals) ALLOW_REMOVALS=1 ;;
        -h|--help)        usage ;;
        *)                echo "$PROG: unexpected argument '$1'" >&2; usage ;;
    esac
    shift
done

BED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # tests/mpi
TESTS_DIR="$(cd "$BED_DIR/.." && pwd)"                       # tests
REPO="$(cd "$BED_DIR/../.." && pwd)"                         # the hawk repo
MANIFEST="$BED_DIR/expected_tests_hawk_rank_bed.txt"
REMINT_CMD="$PROG --remint"
EAGLE_PYTHON_ROOT="${EAGLE_PYTHON_ROOT:-$({ cd "$REPO/../eagle/python" 2>/dev/null || cd "$REPO/../eagle-abi/python" 2>/dev/null; } && pwd)}"

for tool in "$MPIRUN" "$PY"; do
    if ! command -v "$tool" > /dev/null 2>&1 && [ ! -x "$tool" ]; then
        echo "GATE RED: '$tool' not found on \$PATH (set \$MPIRUN / \$PY)" >&2
        exit 1
    fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# The rows import `_oracle`/`_deploy` from tests/ and `_bed_artifact` from
# tests/mpi/; pytest only puts the test file's OWN directory on sys.path.
export PYTHONPATH="$TESTS_DIR:$BED_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HAWK_MPI_BED_RANKS="$RANKS"

# ---------------------------------------------------------------------------
# Build the artifact, name its digest.
# ---------------------------------------------------------------------------
if [ -n "${EAGLE_RANK_BED_ARTIFACT:-}" ]; then
    echo "== $PROG: driving the CALLER's artifact at $EAGLE_RANK_BED_ARTIFACT =="
    ARTIFACT="$EAGLE_RANK_BED_ARTIFACT"
    DIGEST="$("$PY" -c "
import sys; sys.path.insert(0, '$BED_DIR')
import _bed_artifact; print(_bed_artifact.digest('$ARTIFACT'))" 2>/dev/null)"
else
    ARTIFACT="$TMP/artifact"
    echo "== $PROG: emitting + compiling the HAWK bed artifact into $ARTIFACT =="
    if ! "$PY" "$BED_DIR/_bed_artifact.py" "$ARTIFACT" \
             "${HAWK_BED_CACHE:-$TMP/cache}" > "$TMP/build.txt" 2> "$TMP/build.err"
    then
        echo "GATE RED: building the HAWK bed artifact failed" >&2
        sed 's/^/         | /' "$TMP/build.err" >&2
        exit 1
    fi
    DIGEST="$(sed -n '2p' "$TMP/build.txt")"
    export EAGLE_RANK_BED_ARTIFACT="$ARTIFACT"
fi
if [ -z "${DIGEST:-}" ]; then
    echo "GATE RED: could not digest the bed artifact at $ARTIFACT — a verdict that" >&2
    echo "          cannot name the bytes it certified is not a verdict" >&2
    exit 1
fi
echo "   artifact digest (sha256 over the deployed manifests/sidecars/.so/.ptx):"
echo "   $DIGEST"

# ---------------------------------------------------------------------------
# The collection (a ONE-rank world: the test SET is a property of the source, and
# two ranks collecting into one merged stdout would interleave two copies of it).
# Env-scrubbed, because an ambient PYTEST_ADDOPTS filters a collection exactly as
# it filters a run — a collection that inherited it would agree with a filtered
# run about a set neither of them ran.
# ---------------------------------------------------------------------------
collect_scrubbed() {
    ( cd "$REPO" && env -u PYTEST_ADDOPTS -u PYTEST_CURRENT_TEST \
        "$MPIRUN" -np 1 --oversubscribe \
        "$PY" -m pytest tests/mpi -p no:cacheprovider -q --collect-only ) \
        2>"$TMP/collect.err" | sed -n 's#^tests/mpi/##p'
}

if [ "$REMINT" -eq 1 ]; then
    if [ -n "${CI:-}" ]; then
        echo "REFUSED: --remint is a LOCAL command; \$CI is set." >&2
        echo "         CI compares against the committed manifest and never regenerates it." >&2
        exit 2
    fi
    collect_scrubbed | LC_ALL=C sort -u > "$TMP/new.txt"
    COLLECT_RC=${PIPESTATUS[0]}
    if [ "$COLLECT_RC" -ne 0 ]; then
        echo "REFUSED: collecting the bed failed (rc $COLLECT_RC); refusing to mint from it." >&2
        sed 's/^/  | /' "$TMP/collect.err" >&2
        exit 1
    fi
    if [ ! -s "$TMP/new.txt" ]; then
        echo "REFUSED: the bed collected ZERO tests — refusing to mint an empty manifest." >&2
        exit 1
    fi
    if [ -f "$MANIFEST" ]; then
        LC_ALL=C comm -23 "$MANIFEST" "$TMP/new.txt" > "$TMP/removed.txt"
        LC_ALL=C comm -13 "$MANIFEST" "$TMP/new.txt" > "$TMP/added.txt"
        if [ -s "$TMP/removed.txt" ] && [ "$ALLOW_REMOVALS" -ne 1 ]; then
            echo "" >&2
            echo "  ####################################################################" >&2
            echo "  #  RE-MINT REFUSED: this would REMOVE $(wc -l < "$TMP/removed.txt") test(s) from the manifest." >&2
            echo "  #  Tests do not normally disappear. Read this list before you agree:" >&2
            echo "  ####################################################################" >&2
            sed 's/^/  - /' "$TMP/removed.txt" >&2
            echo "" >&2
            echo "  If every removal above is intended, re-run with:" >&2
            echo "      $REMINT_CMD --allow-removals" >&2
            echo "  and commit the manifest diff IN THE SAME COMMIT as the test change." >&2
            exit 1
        fi
        sed 's/^/  - /' "$TMP/removed.txt"
        sed 's/^/  + /' "$TMP/added.txt"
    fi
    cp "$TMP/new.txt" "$MANIFEST"
    echo "re-minted $MANIFEST ($(wc -l < "$MANIFEST") tests)"
    echo "Commit this manifest diff IN THE SAME COMMIT as the test change."
    exit 0
fi

RED=0
red() { echo "GATE RED: $*" >&2; RED=1; }

# ---------------------------------------------------------------------------
# — manifest hygiene.
# ---------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "GATE RED: no manifest at $MANIFEST — mint it with: $REMINT_CMD" >&2
    exit 1
fi
EXPECTED=$(awk 'END{print NR}' "$MANIFEST")
if [ "$EXPECTED" -lt 1 ]; then
    red "[R1] manifest $MANIFEST is EMPTY — an empty expectation certifies nothing"
fi
if [ -s "$MANIFEST" ] && [ "$(tail -c1 "$MANIFEST" | wc -l)" -ne 1 ]; then
    red "[R1] manifest $MANIFEST is not newline-terminated"
fi
if grep -nvE '^[^[:space:]#]+\.py::[^[:space:]#]+$' "$MANIFEST" > "$TMP/shape.txt"; then
    red "[R1] manifest $MANIFEST has lines that are not a bare '<file>.py::<test>' node id:"
    sed 's/^/         /' "$TMP/shape.txt" >&2
fi
if ! LC_ALL=C sort -c "$MANIFEST" 2>"$TMP/sortc.txt"; then
    red "[R1] manifest $MANIFEST is NOT sorted (LC_ALL=C): $(cat "$TMP/sortc.txt")"
fi
UNIQ=$(LC_ALL=C sort -u "$MANIFEST" | awk 'END{print NR}')
if [ "$UNIQ" -ne "$EXPECTED" ]; then
    red "[R1] manifest $MANIFEST has duplicate lines ($EXPECTED lines, $UNIQ unique):"
    LC_ALL=C sort "$MANIFEST" | uniq -d | sed 's/^/         /' >&2
fi

# ---------------------------------------------------------------------------
# — env-scrubbed collection vs the manifest, both directions.
# ---------------------------------------------------------------------------
collect_scrubbed | LC_ALL=C sort -u > "$TMP/collected.txt"
COLLECT_RC=${PIPESTATUS[0]}
COLLECTED=$(awk 'END{print NR}' "$TMP/collected.txt")
if [ "$COLLECT_RC" -ne 0 ]; then
    red "[R2] collecting the bed under '$MPIRUN -np 1' failed with rc $COLLECT_RC"
    sed 's/^/         | /' "$TMP/collect.err" >&2
fi
LC_ALL=C sort -u "$MANIFEST" > "$TMP/manifest.txt"
LC_ALL=C comm -23 "$TMP/manifest.txt" "$TMP/collected.txt" > "$TMP/missing.txt"
LC_ALL=C comm -13 "$TMP/manifest.txt" "$TMP/collected.txt" > "$TMP/extra.txt"
if [ -s "$TMP/missing.txt" ] || [ -s "$TMP/extra.txt" ]; then
    red "[R2] the bed's test SET differs from $MANIFEST (collected $COLLECTED, pinned $EXPECTED)"
    if [ -s "$TMP/missing.txt" ]; then
        echo "         MISSING (pinned but NOT collected):" >&2
        sed 's/^/         - /' "$TMP/missing.txt" >&2
    fi
    if [ -s "$TMP/extra.txt" ]; then
        echo "         UNEXPECTED (collected but NOT pinned — new rows, un-minted):" >&2
        sed 's/^/         + /' "$TMP/extra.txt" >&2
    fi
fi

# ---------------------------------------------------------------------------
# HAWK's own run + the per-rank verdicts.
#
# THE XML IS NAMED PER RANK by $OMPI_COMM_WORLD_RANK, and only a shell running
# INSIDE the rank can put the rank in the path (mpirun performs no expansion on
# the argv it is handed). Both ranks share a working directory, so one
# un-suffixed path would have them racing to write it and the gate would then
# judge whichever won.
# ---------------------------------------------------------------------------
XML_BASE="$TMP/hawk_rank_bed"
export XML_BASE PY

cat > "$TMP/rank_leg.sh" <<'RANKLEG'
#!/usr/bin/env bash
rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${OMPI_MCA_orte_ess_vpid:-0}}}"
exec "$PY" -m pytest tests/mpi -p no:cacheprovider \
     --junitxml="${XML_BASE}_rank${rank}.xml" "$@"
RANKLEG
chmod +x "$TMP/rank_leg.sh"

# One `<testsuite ...>` attribute BY NAME rather than by position: pytest and
# gtest order those attributes differently, and reading them positionally is how
# a gate ends up comparing `errors` against an expected test count.
suite_attr() {
    grep -m1 -oP '<testsuite [^>]*' "$1" \
        | grep -m1 -oP "(?<![a-zA-Z])$2=\"\K[0-9]+"
}

run_bed() {
    local log="$1"; shift
    local r
    for (( r = 0; r < RANKS; ++r )); do rm -f "${XML_BASE}_rank${r}.xml"; done
    ( cd "$REPO" && "$MPIRUN" -np "$RANKS" --oversubscribe \
        "$TMP/rank_leg.sh" "$@" ) > "$log" 2>&1
    RUN_RC=$?
    VERDICTS=()
    for (( r = 0; r < RANKS; ++r )); do
        local xml="${XML_BASE}_rank${r}.xml"
        if [ ! -s "$xml" ]; then VERDICTS+=(""); continue; fi
        VERDICTS+=("$(suite_attr "$xml" tests) $(suite_attr "$xml" failures) \
$(suite_attr "$xml" errors) $(suite_attr "$xml" skipped)")
    done
}

judge() {
    local out="$1"
    local bad=0 r
    [ "$RUN_RC" -ne 0 ] && { echo "[R3] '$MPIRUN -np $RANKS' exited with rc $RUN_RC" >> "$out"; bad=1; }
    for (( r = 0; r < RANKS; ++r )); do
        if [ -z "${VERDICTS[$r]}" ]; then
            echo "[R4] rank $r wrote no XML report — it never reported a run (crashed, aborted, or deadlocked)" >> "$out"
            bad=1
        fi
    done
    [ "$bad" -eq 1 ] && return 1
    local ran fail err skip
    read -r ran fail err skip <<< "${VERDICTS[0]}"
    if [ "$ran" -lt 1 ]; then
        echo "[R5] rank 0 ran $ran tests — a selection that matches nothing is not a pass" >> "$out"
        bad=1
    elif [ "$ran" -ne "$EXPECTED" ]; then
        echo "[R6] rank 0 ran $ran tests, manifest pins $EXPECTED (PYTEST_ADDOPTS='${PYTEST_ADDOPTS:-<unset>}')" >> "$out"
        bad=1
    fi
    if [ "$fail" -ne 0 ] || [ "$err" -ne 0 ]; then
        echo "[R7] rank 0 reported $fail failure(s) and $err error(s)" >> "$out"
        bad=1
    fi
    if [ "$skip" -ne 0 ]; then
        echo "[R7b] rank 0 SKIPPED $skip row(s); this bed must not skip — a skipped row is pinned but certifies nothing" >> "$out"
        bad=1
    fi
    for (( r = 1; r < RANKS; ++r )); do
        if [ "${VERDICTS[$r]}" != "${VERDICTS[0]}" ]; then
            echo "[R8] rank $r's verdict (tests failures errors skipped) = '${VERDICTS[$r]}' differs from rank 0's '${VERDICTS[0]}'" >> "$out"
            bad=1
        fi
    done
    return $bad
}

# The REAL run. Deliberately NOT env-scrubbed: an ambient PYTEST_ADDOPTS must
# make this gate RED, not be laundered into a clean run.
echo "== $PROG: running $MPIRUN -np $RANKS $PY -m pytest tests/mpi (pinned $EXPECTED tests) =="
run_bed "$TMP/hawk_rank_bed.log"
RUN_VERDICTS=("${VERDICTS[@]}")
RUN_LAUNCH_RC=$RUN_RC
: > "$TMP/complaints.txt"
if ! judge "$TMP/complaints.txt"; then
    while IFS= read -r line; do red "$line"; done < "$TMP/complaints.txt"
    grep -E "^(FAILED|ERROR)|^E " "$TMP/hawk_rank_bed.log" | head -40 | sed 's/^/         | /' >&2
    tail -5 "$TMP/hawk_rank_bed.log" | sed 's/^/         | /' >&2
fi

# ---------------------------------------------------------------------------
# — NON-VACUITY. The instrument must be able to fail.
# ---------------------------------------------------------------------------
echo "== $PROG: non-vacuity probe (-k NoSuchTest must come out RED) =="
run_bed "$TMP/probe.log" -k NoSuchTest
PROBE_VERDICT="${VERDICTS[0]:-<no report>}"
: > "$TMP/probe_complaints.txt"
if judge "$TMP/probe_complaints.txt"; then
    red "[R9] the NoSuchTest probe came out GREEN (rank 0 verdict '$PROBE_VERDICT') — this gate cannot fail and certifies nothing"
else
    echo "   probe RED as required: rank 0 verdict (tests failures errors skipped) = '$PROBE_VERDICT'"
    sed 's/^/     probe: /' "$TMP/probe_complaints.txt"
fi

# ---------------------------------------------------------------------------
# EAGLE's own rank bed, on the SAME artifact. Its rc is captured DIRECTLY
# into a variable; reading it through a pipe would report `tee`'s status and the
# verdict would never reach this caller.
# ---------------------------------------------------------------------------
EAGLE_RC="skipped"
if [ -x "$EAGLE_PYTHON_ROOT/tests/mpi/check_rank_bed.sh" ]; then
    echo "== $PROG: handing the SAME artifact to eagle's rank bed =="
    ( cd "$EAGLE_PYTHON_ROOT" && MPIRUN="$MPIRUN" PY="$PY" \
        EAGLE_MPI_BED_RANKS="$RANKS" \
        EAGLE_RANK_BED_ARTIFACT="$ARTIFACT" \
        tests/mpi/check_rank_bed.sh ) > "$TMP/eagle_bed.log" 2>&1
    EAGLE_RC=$?
    sed 's/^/   eagle| /' "$TMP/eagle_bed.log"
    if [ "$EAGLE_RC" -ne 0 ]; then
        red "[A1] eagle's rank bed came out RED against this HAWK artifact (rc $EAGLE_RC)"
    fi
else
    red "[A1] eagle's rank bed is not at $EAGLE_PYTHON_ROOT/tests/mpi/check_rank_bed.sh —" \
        "the structural half cannot run, and a half-run gate must not report GREEN." \
        "Set \$EAGLE_PYTHON_ROOT."
fi

echo "-----------------------------------------------------------------------"
echo "manifest=$MANIFEST  ranks=$RANKS  pinned=$EXPECTED  collected=$COLLECTED  launcher_rc=$RUN_LAUNCH_RC"
echo "artifact=$ARTIFACT"
echo "artifact_digest=$DIGEST"
echo "eagle_rank_bed_rc=$EAGLE_RC"
for (( i = 0; i < RANKS; ++i )); do
    echo "  rank $i verdict (tests failures errors skipped) = '${RUN_VERDICTS[$i]:-<no report>}'"
done
if [ "$RED" -ne 0 ]; then
    echo "" >&2
    echo "VERDICT: RED" >&2
    echo "If — and only if — the bed's row set legitimately changed, re-mint DELIBERATELY with:" >&2
    echo "    $REMINT_CMD" >&2
    echo "and commit the manifest diff in the SAME commit as the row change. Never silence." >&2
    exit 1
fi
echo "VERDICT: GREEN"
exit 0
