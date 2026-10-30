# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# The gates of this repository, runnable from a clean checkout. CI calls these targets, so the commands live here only.
#
#   make gate            the CPU legs CI runs: gate-lint, gate-test, gate-names, gate-docs
#   make gate-gpu        the legs that need a card: the tests marked gpu, then the wide-value correctness checks
#                        (A1 zero local memory, C8 registers, A3 against NumPy and finite differences, A5 memory)
#   make gate-all        both
#   make compare-0.7.2   "did it change": the wide-value checks against the PUBLISHED 0.7.2 release (A3-twin, A3-text, SPEED),
#                        installed from PyPI by hash (ci/compare_0.7.2.txt) into a throwaway venv of its own; needs a card
#
# Run the GPU targets on a card, e.g. <pool>/gpu_run.sh --gpu 1 taskset -c 2-7 make gate-gpu
#
# The environment:
#   a venv is made under GATE_DIR (default build/gate, git-ignored) and made afresh by every run; GATE_REUSE=1 keeps it
#   (the inner loop: GATE_REUSE=1 make gate-test PYTEST_ARGS="tests/test_foo.py").
#   Siblings come from PyPI (the declared dependencies) unless a source directory is given:
#   RAPTOR_SRC, AETHER_SRC (its dsc/ is sealed and installed), EAGLE_SRC (built against the headers of AETHER_SRC and its own).
#   SITE_SRC is the checkout of amasat01.github.io (the docs skin and the freshness generator): gate-lint and gate-docs need it.
#   Toolchains come from the environment, never guessed: a C++ compiler (CXX), nvcc on PATH (or CUDACXX), cppcheck on PATH
#   (or HAWK_CPPCHECK). A missing one stops the gate before it starts. GATE_JOBS (default 2) is the build parallelism;
#   pytest runs in one process.
PYTHON ?= python3
GATE_DIR ?= build/gate
GATE_JOBS ?= 2
GATE_REUSE ?=
PYTEST_ARGS ?= tests
RAPTOR_SRC ?=
AETHER_SRC ?=
EAGLE_SRC ?=
SITE_SRC ?=
# the device class of the eagle build: the card's, else 70 (a build with no card to run on); CUDA_ARCH overrides
# the device compiler hawk tests by default is nvcc (CUDACXX, else CUDA_PATH/bin/nvcc); unset, hawk falls back to NVRTC
export HAWK_NVCC ?= $(or $(CUDACXX),$(if $(CUDA_PATH),$(CUDA_PATH)/bin/nvcc))
CUDA_ARCH ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. '),70)

GATE_ABS := $(abspath $(GATE_DIR))
VENV := $(GATE_ABS)/venv
# the notebooks a gate runs write their animations here, not into docs/_static (a gate never changes tracked files)
export DOCS_ANIM_DIR := $(GATE_ABS)/anim
VPY := $(VENV)/bin/python
LINT := $(GATE_ABS)/lint
CMP := $(GATE_ABS)/cmp072
NB_ARGS ?=
NVCC = $$(command -v "$${CUDACXX:-nvcc}")
# what the one gate venv carries: the extras the requested goals need
EXTRAS := test
ifneq ($(filter gate gate-all gate-docs,$(MAKECMDGOALS)),)
EXTRAS := test,docs
endif
ifneq ($(filter gate-all gate-gpu compare-0.7.2,$(MAKECMDGOALS)),)
EXTRAS := $(EXTRAS),test-gpu
endif
ifneq ($(filter gate-all gate-gpu,$(MAKECMDGOALS)),)
EXTRAS := $(EXTRAS),docs
endif
export CMAKE_BUILD_PARALLEL_LEVEL = $(GATE_JOBS)
export HAWK_GATE_CACHE = $(GATE_ABS)/cache
export HAWK_CACHE_DIR = $(GATE_ABS)/hawk_cache
export TMPDIR = $(GATE_ABS)/tmp
ifneq ($(AETHER_SRC),)
export HAWK_AETHER_INCLUDE = $(abspath $(AETHER_SRC))
endif
ifneq ($(EAGLE_SRC),)
export HAWK_EAGLE_INCLUDE = $(abspath $(EAGLE_SRC))
export RAPTOR_EAGLE_ROOT = $(abspath $(EAGLE_SRC))
endif

.PHONY: gate-gpu-notebooks gate gate-names gate-lint gate-test gate-docs gate-gpu gate-all compare-0.7.2 gate-tools gate-venv

gate: gate-lint gate-test gate-names gate-docs
gate-all: gate gate-gpu

gate-tools:
	@mkdir -p $(GATE_ABS)/tmp $(HAWK_GATE_CACHE) $(HAWK_CACHE_DIR)
	@test -n "$(CXX)" || { echo "gate: CXX is unset (the C++ compiler the host objects are built with)" >&2; exit 2; }
	@command -v "$${CUDACXX:-nvcc}" >/dev/null || { echo "gate: nvcc is not on PATH and CUDACXX is unset" >&2; exit 2; }
	@command -v "$${HAWK_CPPCHECK:-cppcheck}" >/dev/null || { echo "gate: cppcheck is not on PATH and HAWK_CPPCHECK is unset" >&2; exit 2; }

# -- the throwaway environment ----------------------------------------------------------------------------------------------
# (hawk._core and eagle._core build with --no-build-isolation, so their build requirements are installed first.)
gate-venv: gate-tools
	@if [ -z "$(GATE_REUSE)" ] || [ ! -x $(VPY) ]; then rm -rf $(VENV) && $(PYTHON) -m venv $(VENV); fi
	$(VPY) -m pip install --quiet --upgrade pip
	$(VPY) -m pip install --quiet "scikit-build-core>=0.10" "setuptools-scm>=8" "nanobind>=3.1,<4" cmake ninja -c ci/constraints.txt
	if [ -n "$(RAPTOR_SRC)" ]; then $(VPY) -m pip install --quiet "$(abspath $(RAPTOR_SRC))" -c ci/constraints.txt; fi
	# (the builds go to GATE_DIR, never into a sibling checkout.) aether-dsc's payload is sealed at build time (it is not tracked): seal it from the checkout, then install it
	if [ -n "$(AETHER_SRC)" ]; then \
	  $(VPY) -m pip install --quiet "cuda-bindings>=12.6,<13" "nvidia-cuda-nvrtc-cu12>=12.6,<13" "nvidia-cuda-cccl-cu12>=12.6,<13" -c ci/constraints.txt && \
	  (cd "$(abspath $(AETHER_SRC))/dsc" && $(VPY) -m aether_dsc.seal) && \
	  $(VPY) -m pip install --quiet --config-settings=build-dir=$(GATE_ABS)/build/aether-dsc "$(abspath $(AETHER_SRC))/dsc" -c ci/constraints.txt; fi
	# eagle: both header sets into a prefix of the gate's own, then its binding built against it
	if [ -n "$(EAGLE_SRC)" ]; then \
	  test -n "$(AETHER_SRC)" || { echo "gate: EAGLE_SRC needs AETHER_SRC (eagle builds against its headers)" >&2; exit 2; }; \
	  cmake -DAETHER_CPP_MODE=OFF -DCMAKE_CUDA_COMPILER=$(NVCC) -DCMAKE_INSTALL_PREFIX=$(GATE_ABS)/prefix \
	      -B $(GATE_ABS)/aether-build -S "$(abspath $(AETHER_SRC))" >/dev/null && \
	  cmake --install $(GATE_ABS)/aether-build >/dev/null && \
	  cmake -DCMAKE_CUDA_COMPILER=$(NVCC) -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) -DCMAKE_PREFIX_PATH=$(GATE_ABS)/prefix \
	      -DCMAKE_INSTALL_PREFIX=$(GATE_ABS)/prefix -B $(GATE_ABS)/eagle-hdr -S "$(abspath $(EAGLE_SRC))" >/dev/null && \
	  cmake --install $(GATE_ABS)/eagle-hdr >/dev/null && \
	  CMAKE_PREFIX_PATH=$(GATE_ABS)/prefix SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=$(NVCC);-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)" \
	    $(VPY) -m pip install --quiet --no-build-isolation --config-settings=build-dir=$(GATE_ABS)/build/eagle "$(abspath $(EAGLE_SRC))/python" -c ci/constraints.txt; fi
	CMAKE_PREFIX_PATH=$(GATE_ABS)/prefix $(VPY) -m pip install --quiet --no-build-isolation -e ".[$(EXTRAS)]" -c ci/constraints.txt

# -- the CPU legs CI runs -----------------------------------------------------------------------------------------------------
# lint: ruff (the repo's config), the manifesto copy (raptor main, or MANIFESTO_SOURCE), and the documented numbers and capability
# map re-rendered from the committed cards (the generator reads the eagle checkout EAGLE_SRC, through RAPTOR_EAGLE_ROOT).
# A venv of its own: it needs ruff and the site's tools only.
gate-lint:
	@test -n "$(SITE_SRC)" || { echo "gate-lint: SITE_SRC (a checkout of amasat01.github.io) is unset" >&2; exit 2; }
	@test -n "$(EAGLE_SRC)" || { echo "gate-lint: EAGLE_SRC (a checkout of eagle, which the generator reads) is unset" >&2; exit 2; }
	rm -rf $(LINT) && $(PYTHON) -m venv $(LINT)
	$(LINT)/bin/python -m pip install --quiet ruff -c ci/constraints.txt "$(abspath $(SITE_SRC))"
	$(LINT)/bin/python -m ruff check
	bash tools/manifesto_check.sh CONTRIBUTING.md
	$(LINT)/bin/python -m raptor_docs.fresh --check --site "$(abspath $(SITE_SRC))"

# test: everything not marked gpu (the GPU-less runner's selection); the cppcheck rows fail, never skip, without a cppcheck
gate-test: gate-venv
	$(VPY) -m pytest -q -m "not gpu" -p no:cacheprovider $(PYTEST_ARGS)

# coverage: every public name has a test that mentions it (static, seconds); the untested ones are in the allowlist, which only shrinks
gate-names: gate-venv
	$(VPY) -m raptor.conformance.public_names --src . --tests tests --allow ci/public_names_untested.txt --module hawk

# docs: every notebook filled and run on the CPU (the card is hidden: this leg is the GPU-less one; the card pool is for gate-gpu), the strict (-W) Sphinx build, every public name has a page
gate-docs: gate-venv
	@test -n "$(SITE_SRC)" || { echo "gate-docs: SITE_SRC (a checkout of amasat01.github.io) is unset" >&2; exit 2; }
	$(VPY) -m pip install --quiet --force-reinstall --no-deps "$(abspath $(SITE_SRC))"
	$(VPY) -m pip install --quiet torch -c ci/constraints.txt --index-url https://download.pytorch.org/whl/cpu
	CUDA_VISIBLE_DEVICES= PATH=$(VENV)/bin:$$PATH $(MAKE) -C docs nbcheck
	PATH=$(VENV)/bin:$$PATH $(MAKE) -C docs strict
	$(VPY) tools/docs_inventory_check.py docs/_build/html/objects.inv hawk

# -- the legs that need a card ------------------------------------------------------------------------------------------------
gate-gpu: gate-venv
	$(VPY) -m pytest -q -m gpu -p no:cacheprovider $(PYTEST_ARGS)
	$(VPY) tests/test_wide_placement.py gpu

# every notebook, run through raptor_docs.nb run_gate with the card visible: its CPU leg again, then its GPU leg (a notebook
# that declares gate.gpu_only runs only there). gate-docs hides the card, so the GPU pass has this one named place; it runs
# at the release gate and in CI's GPU job, not in every batch gate.
gate-gpu-notebooks: gate-venv
	@test -n "$(SITE_SRC)" || { echo "gate-gpu: SITE_SRC (a checkout of amasat01.github.io) is unset" >&2; exit 2; }
	$(VPY) -m pip install --quiet --force-reinstall --no-deps "$(abspath $(SITE_SRC))"
	$(VPY) -m pip install --quiet torch -c ci/constraints.txt --index-url https://download.pytorch.org/whl/cpu
	cd docs && PATH=$(VENV)/bin:$$PATH $(VPY) -m raptor_docs.nb run_gate $(NB_ARGS)

# -- did it change? ----------------------------------------------------------------------------------------------------------
# The release is fetched by hash from PyPI into a venv of its own; this tree runs beside it in the gate venv.
compare-0.7.2: gate-venv
	rm -rf $(CMP) && $(PYTHON) -m venv $(CMP)
	$(CMP)/bin/python -m pip install --quiet --require-hashes --no-deps -r ci/compare_0.7.2.txt
	HAWK_TWIN_VENV=$(CMP) $(VPY) tests/test_wide_placement.py compare
