"""The two libraries are isolated: neither's answers change with the other present or absent.

Each library is also asked from a folder that holds nothing but itself and its own data documents.
Run through test_binding.py, which names the two libraries (HOST_LIB and CUDA_LIB).
"""

import os
import shutil
import tempfile

HOST_COMPILER = "c++ || Apple clang version 21.0.0 || 0"
HOST_STATED = (
    '{"headers": "roots", "isa": "arm64", "l1d_bytes": 131072, "l2_bytes": 16777216,'
    ' "cache_source": "stated"}'
)
CUDA_COMPILER = "/usr/local/cuda/bin/nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0"
CUDA_STATED = '{"device_arch": "sm_80"}'
# for each library: the target it serves, the compiler and stated machine of its facts question,
# and the documents it ships
ASKS = {
    "cpu": ("host", HOST_COMPILER, HOST_STATED, ("build.json", "arm.json", "x86.json")),
    "cuda": ("cuda", CUDA_COMPILER, CUDA_STATED, ("cuda_build.json", "cuda.json")),
}


def answers(o, path, facts_dir):
    """Return the identity, facts answer and shipped documents of the library at `path`."""
    c = o.Ctx()
    lib = c.backend_open(path)
    name = c.backend_name(lib)
    target, compiler, stated, cards = ASKS[name]
    record = {
        "name": name,
        "identity": c.backend_identity(lib),
        "facts": c.backend_facts(lib, facts_dir, target, compiler, stated),
        "cards": {card: c.backend_card(lib, facts_dir, card) for card in cards},
    }
    c.backend_close(lib)
    return record


def alone(o, path, facts_source):
    """Return `answers` read from a folder holding the library and a copy of its own documents."""
    folder = tempfile.mkdtemp(prefix="osprey_alone_")
    try:
        copy = os.path.join(folder, os.path.basename(path))
        shutil.copy(path, copy)
        shutil.copytree(facts_source, os.path.join(folder, "facts"))
        return answers(o, copy, None)
    finally:
        shutil.rmtree(folder)


def refused_by_name(o, raises, libs, facts_dirs, name, other):
    """Check that the library `name` refuses a facts question for the target of `other`."""
    c = o.Ctx()
    lib = c.backend_open(libs[name])
    target, compiler, stated, _ = ASKS[other]

    def ask():
        return c.backend_facts(lib, facts_dirs[name], target, compiler, stated)

    e = raises(o.BadRequest, ask, f"{name}: a facts question for the target of {other}")
    assert name in str(e) and target in str(e), str(e)
    c.backend_close(lib)


def run(done, o, raises, libs, facts_dirs, host_text):
    """Check the isolation of the libraries `libs` (name to path).

    `host_text(lib)` renders one host unit in an opened library.
    """
    cpu_before = answers(o, libs["cpu"], facts_dirs["cpu"])
    cuda_before = answers(o, libs["cuda"], facts_dirs["cuda"])
    c = o.Ctx()
    cpu = c.backend_open(libs["cpu"])
    text_before = host_text(cpu)
    cuda = c.backend_open(libs["cuda"])
    # both are open now: nothing either answers moves with the other present
    assert answers(o, libs["cpu"], facts_dirs["cpu"]) == cpu_before, "cpu moved, cuda open"
    assert answers(o, libs["cuda"], facts_dirs["cuda"]) == cuda_before, "cuda moved, cpu open"
    assert host_text(cpu) == text_before, "a host text moved, cuda open"
    c.backend_close(cuda)
    c.backend_close(cpu)
    done.append("sideways: with the other library open, every answer of each is unchanged")
    # a folder holding one library and its own documents answers as the two do side by side
    assert alone(o, libs["cpu"], facts_dirs["cpu"]) == cpu_before, (
        "cpu answers differ when it stands alone"
    )
    assert alone(o, libs["cuda"], facts_dirs["cuda"]) == cuda_before, (
        "cuda answers differ when it stands alone"
    )
    done.append("sideways: each library stands alone with its own documents, answers the same")
    refused_by_name(o, raises, libs, facts_dirs, "cpu", "cuda")
    refused_by_name(o, raises, libs, facts_dirs, "cuda", "cpu")
    done.append("sideways: each library refuses the other's target by name")
