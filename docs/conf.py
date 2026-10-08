# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for building the hawk documentation."""

import os
import re

# -- Project information -----------------------------------------------------

project = "hawk"
copyright = "2026, Alessandro Masat"
author = "Alessandro Masat"

# Read the version from hawk/__init__.py (the single source of truth, matched
# by pyproject.toml's own scikit-build regex reader) rather than importing
# hawk: this file runs under autodoc too, and hawk's import-time self-check
# refuses a _core that is not built from the tree beside it, which a docs
# venv has no business tripping over just to read a version string.
_version = "0.0.0"
_init = os.path.join(os.path.dirname(__file__), "..", "hawk", "__init__.py")
_pattern = re.compile(r'''^__version__\s*=\s*["'](?P<value>.+?)["']''')
try:
    with open(_init) as _f:
        for _line in _f:
            _m = _pattern.match(_line)
            if _m:
                _version = _m.group("value")
                break
except FileNotFoundError:
    pass

version = _version
release = _version

# -- General configuration ---------------------------------------------------

# hawk's public surface is pure Python (the compiled hawk._core extension is
# an internal implementation detail, never documented); no breathe/doxygen.
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# -- Notebook execution (myst_nb) --------------------------------------------
# Tutorials/examples are executed LOCALLY (`make nbexec`, see docs/tools/) and
# committed WITH outputs; the published build never re-executes them, so a
# stale or missing output cannot diverge silently between a local render and
# the deployed one — it is the `make nbcheck` gate's job to refuse that.
nb_execution_mode = "off"
nb_execution_timeout = 3600
nb_merge_streams = True
nb_output_stderr = "show"

# -- Python API reference (autosummary/autodoc) ------------------------------
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
napoleon_google_docstring = True
napoleon_include_special_with_doc = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_inherit_docstrings = False
autoclass_content = "class"
# Deliberately NO 'members': True here. The autosummary/autosummary templates
# this build uses (_templates/autosummary/module.rst) already list every
# module's public functions/classes through their OWN `.. autosummary::
# :toctree: generated/` blocks, each producing one full page per member;
# adding 'members': True would ALSO render every member inline on the
# parent module's automodule page -- the same object documented twice under
# the same dotted name, which Sphinx (rightly) refuses under `-W`.
autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "exclude-members": "__weakref__",
}

# -----------------------------------------------------------------------------
# Intersphinx. The family siblings (raptor, eagle, aether) are deliberately
# NOT mapped here: their GitHub Pages sites do not exist until the family's
# Pages flip, so mapping them would make every build's intersphinx fetch fail
# until then. This page's cross-sibling references are plain hyperlinks
# instead (see content/interop.md, index.md).
# `python` is also left out: docs.python.org's inventory was observed
# returning an intermittent 503 from this build's network, which -W would
# turn into a hard failure for a fetch hawk's own pages do not depend on.
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Templates and exclusions
templates_path = ["_templates"]
exclude_patterns = ["_templates", "_build", "Thumbs.db", ".DS_Store", "README.md"]

# Two narrow, tooling-internal rough edges, not content bugs:
# - "autodoc": sphinx_autodoc_typehints' autodoc-process-signature hook
#   throws formatting the synthesised __new__ of a typing.NamedTuple
#   (hawk.compile.DeviceImage) -- a known interaction between the two, not a
#   docstring or type-annotation problem on hawk's side.
# - "sphinx_autodoc_typehints.forward_reference": the same plugin cannot
#   resolve `Mapping` while rendering hawk.ext.DEFAULT_KIND's own type (an
#   `autodata`-documented dataclass instance, not a function/class the
#   plugin's forward-reference resolver is built for).
suppress_warnings = ["autodoc", "sphinx_autodoc_typehints.forward_reference"]

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# raptor-tokens.css (kit) -> site-accent.css (this site's --accent) ->
# raptor-theme.css (shared skin, derives everything from --accent) ->
# raptor-reveal.css (kit).
html_css_files = ["raptor-tokens.css", "site-accent.css", "raptor-theme.css", "raptor-reveal.css"]
html_theme_options = {
    "repository_url": "https://github.com/amasat01/hawk",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "collapse_navigation": True,
    "navigation_with_keys": True,
    # Colab + download launch buttons on notebook pages (no Binder: not configured).
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "classic",
    },
    # One transparent logo file works on light AND dark pages (RAPTOR brand kit).
    "logo": {
        "image_light": "_static/brand/family_hawk.svg",
        "image_dark": "_static/brand/family_hawk.svg",
        "alt_text": "hawk",
    },
    # Family frame: purple "part of RAPTOR" chip in the footer, every site
    # (raptor-theme.css's .raptor-family-chip; theme's own extension point).
    "extra_footer": (
        '<div class="raptor-family-chip">part of '
        '<a href="https://amasat01.github.io/">RAPTOR</a></div>'
    ),
}
html_title = f"hawk — write one kernel, run it on CPU or GPU ({version})"

# RAPTOR brand: favicons + home-screen icon. Leave html_favicon unset: the
# hook below writes the icon links itself (SVG where supported, favicon.ico
# for Safari/older tools, 180 px icon for iOS home screens).
_RAPTOR_ICONS = [
    ("icon", "brand/favicon.ico", 'sizes="any"'),
    ("icon", "brand/favicon.svg", 'type="image/svg+xml"'),
    ("apple-touch-icon", "brand/app_icon_180.png", ""),
]


def _raptor_icons(app, pagename, templatename, context, doctree):
    pathto = context.get("pathto")
    if pathto is None:
        return
    links = "".join(
        f'<link rel="{rel}" href="{pathto("_static/" + path, 1)}" {extra}>\n'
        for rel, path, extra in _RAPTOR_ICONS
    )
    context["metatags"] = context.get("metatags", "") + links


def setup(app):
    app.connect("html-page-context", _raptor_icons)
