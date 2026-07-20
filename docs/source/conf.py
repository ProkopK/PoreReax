import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PoreReax'
copyright = '2026, Karl Prokop'
author = 'Karl Prokop'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",       # pulls docstrings from your code
    "sphinx.ext.napoleon",      # lets you write Google/NumPy-style docstrings
    "sphinx.ext.viewcode",      # adds links to highlighted source code
    "sphinx.ext.intersphinx",   # links to other projects' docs (e.g. Python, numpy)
    "sphinx_autodoc_typehints", # shows type hints nicely instead of duplicating in docstring
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']
