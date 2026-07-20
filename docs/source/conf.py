import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))
from porereax import __version__


project = 'PoreReax'
copyright = '2026, Karl Prokop'
author = 'Karl Prokop'
release = __version__
version = __version__

extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autoapi_type = "python"
autoapi_dirs = ["../../src/porereax"]
autoapi_ignore = ["*/templates/*"]
autoapi_output_dir = "api"
autoapi_add_toctree_entry = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ['_templates']
exclude_patterns = []


html_theme = "furo"
html_static_path = ['_static']
