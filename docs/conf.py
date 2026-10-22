"""Sphinx configuration of the scote documentation (Sphinx, MyST Markdown, numpydoc docstrings, Read the Docs theme).

How to build: see ``README.md`` in this directory. Generated during the build, and not part of the repository:
``generated/``, one API page per module, function, and class, from the autosummary templates in ``_templates``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

import scote  # noqa: E402
from scote_docs import api_homes  # noqa: E402

project = "scote"
author = "Simon Mages"
copyright = "2026, The SCOTE Project Maintainers"
release = scote.__version__
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "scote_docs",
]

# API reference: one page per public module, function, and class; every object once, under the shallowest public
# module that exports it (scote_docs.api_homes)
autosummary_generate = True
autosummary_context = {"scote_homes": api_homes()[0]}
templates_path = ["_templates"]
add_module_names = False
autodoc_default_options = {"member-order": "bysource"}
autodoc_mock_imports = ["torch", "scanpy", "anndata"]
autoclass_content = "class"

# every cross-reference must resolve (with -W, an unresolved one fails the build); the words of the numpydoc type
# descriptions that name no object are exempt
nitpicky = True
nitpick_ignore = [("py:class", word) for word in ("array-like", "callable", "sequence")]

# docstrings are numpydoc style; types become links where they name documented objects
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {"scipy.sparse matrix": ":class:`scipy.sparse matrix <scipy.sparse.spmatrix>`"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# Markdown pages
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 3

# copy buttons on code blocks, without prompts and outputs
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header"]

html_theme = "sphinx_rtd_theme"
html_title = f"scote {release}"
html_static_path = ["_static"]
html_css_files = ["scote.css"]
html_show_sourcelink = False
html_theme_options = {"navigation_depth": 4, "collapse_navigation": True}
toc_object_entries_show_parents = "hide"
