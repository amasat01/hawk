# hawk sanitize

## Free-threading race gate (the CI `tsan-ft` job)

hawk._core, eagle._core and the host kernels hawk compiles at run time, all built with
ThreadSanitizer; the `ft` rows run under a free-threaded CPython built with
`--with-thread-sanitizer`.

```bash
tests/sanitize/build_tsan_python.sh 3.14.8 ~/.cache/tsan-python      # once (~20 min at -j2, JOBS=2)
HAWK_AETHER_INCLUDE=../aether HAWK_EAGLE_INCLUDE=../eagle RAPTOR_DIR=../raptor \
  tests/sanitize/tsan_ft.sh ~/.cache/tsan-python/bin/python3.14t
```

GREEN needs the canary race (`_unsynchronised_bump`) to be reported without
`tests/sanitize/tsan.supp`, which proves the detector reaches `_core`, AND zero reports
from the `ft` rows with it. Every suppression names its owner, its reason and its
evidence; the only hawk entry is the deliberate canary.

Clang builds the kernels here, so the host profile defaults to `native` (the
`native-vector-math` profile needs GCC).
