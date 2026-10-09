# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Builds hawk._core, the nanobind host execution path.
#
# HOST ONLY, and there is no CUDA language anywhere in this file. That is
# because a device artifact is loaded and launched by EAGLE, never by
# HAWK, so this extension has no device code, no arch list and nothing to compile
# with nvcc. It is one C++23 translation unit over eagle's ABI header plus aether's
# index typedef.
#
# WHAT THE BUILD MUST GET RIGHT, and what each block below is for:
#
# 1. THE TWO HEADER ROOTS ARE RESOLVED BY THE SAME CODE THE PYTHON DRIVER USES.
# `hawk/_roots.py` is loaded BY PATH — not imported as
#     `hawk.compile.toolchain`, because importing any hawk submodule runs
# `hawk/__init__`, whose self-check imports the very `_core` this build
# is about to produce. A hand-written second copy of the search rule here would
# be exactly the drift this shared code is meant to prevent, so there isn't one.
#
# 2. THE INCLUDE ORDER IS MANDATORY: aether first, eagle second.
#     `EAGLE_ABI_INDEX_T` binds to `AETHER_INDEX_T` only if that macro is already
#     visible when `plugin/gref_abi.h` is preprocessed; reversed, the index width
#     silently narrows to `uint32_t` and the whole parameter block shifts. The
#     source enforces it too (its two `#include`s are in that order), and the -I
#     order here keeps a same-named header from the wrong root winning.
#
# 3. THE BUILD DIGEST IS STAMPED AT CONFIGURE: a content hash over
#     the sorted `src/` sources plus this file, written into a generated header.
#     `hawk/__init__` recomputes the SAME hash from the tree at import and refuses
#     a disagreement, which is what catches the stale / wrong-arch /
#     editable-shadowed binding class at `import hawk` instead of at first launch.
#     The rule has to be stated twice — once here in CMake, once in Python — so it
#     is stated as simply as it can be: for each file, `<name>\n<sha256>\n`, in
# sorted-name order, then one sha256 over that string. A dedicated check
# proves the two agree, by making them disagree. The SAME stamp is
#     ALSO written as a small Python module (`hawk/_build_digest.py`, below)
#     and install()ed into the wheel: a wheel install carries no `src/` beside
#     the package, so `hawk/__init__` cannot recompute — it reads this stamp
# back instead.
cmake_minimum_required(VERSION 3.20)
project(hawk_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED PATHS "${nanobind_ROOT}" NO_DEFAULT_PATH)
endif()

# ---------------------------------------------------------------------------
# (1) The header roots — HAWK's own resolver, run as a script.
# ---------------------------------------------------------------------------
# `-I` keeps the script's own folder off sys.path: run by path, hawk/_roots.py
# would otherwise put hawk/ first, and hawk/types.py would shadow the
# standard-library `types` module.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I "${CMAKE_CURRENT_SOURCE_DIR}/hawk/_roots.py"
  OUTPUT_VARIABLE _hawk_roots
  ERROR_VARIABLE _hawk_roots_err
  RESULT_VARIABLE _hawk_roots_rc
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _hawk_roots_rc EQUAL 0)
  message(FATAL_ERROR
    "hawk/_roots.py could not resolve the aether/eagle include roots.\n"
    "Set $HAWK_AETHER_INCLUDE / $HAWK_EAGLE_INCLUDE and re-run.\n${_hawk_roots_err}")
endif()
string(REPLACE "\n" ";" _hawk_roots_list "${_hawk_roots}")
list(GET _hawk_roots_list 0 HAWK_AETHER_INCLUDE)
list(GET _hawk_roots_list 1 HAWK_EAGLE_INCLUDE)
message(STATUS "> hawk._core aether root: ${HAWK_AETHER_INCLUDE}")
message(STATUS "> hawk._core eagle  root: ${HAWK_EAGLE_INCLUDE}")
if(NOT EXISTS "${HAWK_EAGLE_INCLUDE}/plugin/gref_abi.h")
  message(FATAL_ERROR "resolved eagle root has no plugin/gref_abi.h: ${HAWK_EAGLE_INCLUDE}")
endif()
if(NOT EXISTS "${HAWK_AETHER_INCLUDE}/aether/typedefs.h")
  message(FATAL_ERROR "resolved aether root has no aether/typedefs.h: ${HAWK_AETHER_INCLUDE}")
endif()

# ---------------------------------------------------------------------------
# (3) The build digest.
# ---------------------------------------------------------------------------
file(GLOB _hawk_core_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
list(APPEND _hawk_core_sources "CMakeLists.txt")
list(SORT _hawk_core_sources)
set(_hawk_digest_input "")
foreach(_rel IN LISTS _hawk_core_sources)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_rel}" _one)
  string(APPEND _hawk_digest_input "${_rel}\n${_one}\n")
endforeach()
string(SHA256 HAWK_BUILD_DIGEST "${_hawk_digest_input}")
message(STATUS "> hawk._core build digest: ${HAWK_BUILD_DIGEST}")

# The compiler identity + the flags, EXPORTED by `build_info()` and never
# compared: a different compiler is not by itself a wrong binding, but
# diagnosing a build-dependent difference needs to be able to read it.
set(HAWK_COMPILER "${CMAKE_CXX_COMPILER}")
set(HAWK_COMPILER_ID "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
# libstdc++/libgcc bundled when the toolchain has their static archives (a
# distribution gcc without libstdc++-static links them dynamically instead).
include(CheckLinkerFlag)
check_linker_flag(CXX "-static-libstdc++" HAWK_STATIC_LIBSTDCXX)
if(HAWK_STATIC_LIBSTDCXX)
  set(_hawk_static_runtime -static-libstdc++ -static-libgcc)
else()
  set(_hawk_static_runtime "")
  message(WARNING "hawk: no static libstdc++ in this toolchain; linking it dynamically")
endif()
string(REPLACE ";" " " _hawk_static_runtime_text "${_hawk_static_runtime}")
set(HAWK_COMPILE_FLAGS "-std=c++23 -fvisibility=hidden ${_hawk_static_runtime_text} ${CMAKE_CXX_FLAGS}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/hawk_build_digest.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/hawk_build_digest.h" @ONLY)
# The Python twin of the same stamp: a wheel
# install carries no src/ or CMakeLists.txt beside the package, so
# hawk.__init__._self_check compares hawk._core.build_digest() against this
# instead of recomputing from the tree. install()ed below alongside _core so
# it lands in the wheel; see src/_build_digest.py.in for the non-vacuity
# argument (why the dev tree does not need this file at all).
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/_build_digest.py.in"
               "${CMAKE_CURRENT_BINARY_DIR}/hawk/_build_digest.py" @ONLY)

# ---------------------------------------------------------------------------
# The extension.
# ---------------------------------------------------------------------------
nanobind_add_module(_core MODULE FREE_THREADED src/hawk_core.cpp)
target_compile_features(_core PRIVATE cxx_std_23)
# (2) MANDATED ORDER: aether first, eagle second.
target_include_directories(_core PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${HAWK_AETHER_INCLUDE}"
  "${HAWK_EAGLE_INCLUDE}")
# The host build of aether's headers: the same mode every HAWK host TU compiles
# under (`hawk.compile.toolchain.host_flags`), so `_core` and the artifacts it
# loads agree about what aether they were built against.
target_compile_definitions(_core PRIVATE AETHER_CPP_MODE)
# Bundle libstdc++/libgcc, exactly as eagle's binding does and for the same
# reason: the runtime env's system libstdc++ can predate the build toolchain's.
target_link_options(_core PRIVATE ${_hawk_static_runtime})
# dlopen/dlsym live in libdl on older glibc and in libc since 2.34; CMake's
# ${CMAKE_DL_LIBS} resolves whichever this platform needs.
target_link_libraries(_core PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(_core PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON)

# At BUILD time as well as at test time: nothing here may find an
# OpenMP runtime, because the serial oracle must stay serial. This is not a
# find_package(OpenMP) that is then ignored — the point is that the flag never
# appears, so the target below is asserted to carry no OpenMP link library.
get_target_property(_hawk_core_libs _core LINK_LIBRARIES)
if(_hawk_core_libs MATCHES "OpenMP|gomp|iomp")
  message(FATAL_ERROR
    "hawk._core links an OpenMP runtime (${_hawk_core_libs}); the serial oracle "
    "must open no parallel region")
endif()

# scikit-build-core maps this DESTINATION into the wheel's `hawk` package.
install(TARGETS _core LIBRARY DESTINATION hawk)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hawk/_build_digest.py" DESTINATION hawk)
