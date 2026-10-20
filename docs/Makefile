# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for the hawk Sphinx (MyST/Markdown + notebook) documentation.
# hawk's public surface is pure Python -- there is no Doxygen/breathe step.

# `python -m sphinx`, not the bare `sphinx-build` script: a `sphinx-build`
# found on $PATH is an entry-point script pinned to WHATEVER interpreter it
# was installed under by its own shebang, which silently wins over an
# environment that layers one Python on top of another (a venv with
# `include-system-site-packages`, for instance) -- `python -m sphinx` always
# runs under the `python` this shell actually resolves, so autodoc imports
# hawk from the same tree every other command in this shell sees.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml strict linkcheck nbexec nbcheck nbcheck-hosted nbcheck-floor nbcheck-hosted-gpu nbclear

# -W --keep-going: warnings are fatal, but every warning is still reported in
# one pass rather than stopping at the first.
strict:
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Execute every tutorial/vocabulary/example/explanation notebook IN PLACE
# (outputs committed): the execution itself is the gate (any cell error
# fails the run), and CI never re-executes -- conf.py sets
# nb_execution_mode = "off".
nbexec:
	python tools/nb_execute.py content/tutorials content/vocabulary content/examples content/compile_cache.ipynb content/compile_options.ipynb content/across_samples.ipynb

# CI / pre-publish gate: refuse to build unless every notebook was executed, then
# RUN every notebook in memory: on the CPU always, and again on the GPU when one is
# present (tools/nb_run_gate.py; a notebook declares "gate" metadata for what it
# needs, and a GPU-only one skips cleanly on the CPU leg). A committed notebook can
# no longer outlive the code that produced it.
NB_PATHS = content/tutorials content/vocabulary content/examples content/compile_cache.ipynb content/compile_options.ipynb content/across_samples.ipynb

nbcheck:
	python tools/nb_colab_setup.py --check
	python tools/nb_check_filled.py $(NB_PATHS)
	python tools/nb_run_gate.py $(NB_PATHS)

# Strip all outputs + execution counts from every notebook under content/.
nbclear:
	@find ./content -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' -print0 \
	  | xargs -0 -r jupyter nbconvert --clear-output --inplace

clean:
	rm -rf $(BUILDDIR) content/generated content/api/generated

# Watch for changes and auto-recompile (requires watchdog: pip install watchdog)
livehtml: Makefile
	watchmedo shell-command -p "*.rst;*.md;*.py" -R \
	    -c "make html" \
	    -i "$(BUILDDIR)/*" \
	    --debug-force-polling .

# Catch-all target: route unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# The driver lives in eagle's docs tools; point EAGLE_ROOT at an eagle checkout.
EAGLE_ROOT ?= $(abspath ../../eagle)

# Run every docs notebook as a hosted (Colab) user does: wheels built from the checkouts, a pinned environment,
# the setup cell included (_tools/nb_hosted.py). Needs uv, a Python 3.13 and a C++23 compiler (CXX).
NBH_PYTHON ?= python3.13
NBH_ARGS   ?=
nbcheck-hosted:
	python $(EAGLE_ROOT)/docs/_tools/nb_hosted.py --python $(NBH_PYTHON) --lock colab --repos hawk --hawk $(abspath ..) $(NBH_ARGS)

# The same against the oldest numpy and torch that have Python 3.13 wheels (_tools/hosted_env/floor-cpu.txt).
nbcheck-floor:
	python $(EAGLE_ROOT)/docs/_tools/nb_hosted.py --python $(NBH_PYTHON) --lock floor --repos hawk --hawk $(abspath ..) $(NBH_ARGS)

# The GPU runtime's lock (CUDA 12 builds: see the header of _tools/hosted_env/colab-gpu.txt): eagle's CUDA plugin is built for
# NBH_GPU_ARCH, the GPU-only notebooks run too, on GPU NBH_GPU_INDEX, under the flock file NBH_GPU_LOCK if set. Needs ~5 GB free.
NBH_GPU_ARCH  ?= 61
NBH_GPU_INDEX ?= 1
NBH_GPU_LOCK  ?=
nbcheck-hosted-gpu:
	python $(EAGLE_ROOT)/docs/_tools/nb_hosted.py --python $(NBH_PYTHON) --lock colab-gpu --gpu --cuda-arch $(NBH_GPU_ARCH) --gpu-index $(NBH_GPU_INDEX) $(if $(NBH_GPU_LOCK),--gpu-lock $(NBH_GPU_LOCK)) --repos hawk --hawk $(abspath ..) $(NBH_ARGS)
