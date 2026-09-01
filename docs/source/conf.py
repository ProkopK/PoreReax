import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
from porereax import __version__

project = "PoreReax"
copyright = "2026, Karl Prokop"
author = "Karl Prokop"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo_text.svg"
html_favicon = "_static/logo.svg"
