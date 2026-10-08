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

.PHONY: help Makefile clean livehtml strict linkcheck nbexec nbcheck nbclear

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
	python tools/nb_execute.py content/tutorials content/vocabulary content/examples content/compile_cache.ipynb content/compile_options.ipynb

# CI / pre-publish gate: refuse to build unless every notebook was executed.
nbcheck:
	python tools/nb_check_filled.py content/tutorials content/vocabulary content/examples content/compile_cache.ipynb content/compile_options.ipynb

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
