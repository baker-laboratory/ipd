import os
import sys
sys.path.insert(0, os.path.abspath(".."))
import importlib.metadata

# -- Project information -----------------------------------------------------
project = "IPD"
author = "Will Sheffler"
release = importlib.metadata.version('ipd')

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",  # Supports Google/NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",  # For example galleries like Biotite
]

# -- HTML Theme Configuration ------------------------------------------------
pygments_style = 'sphinx'

html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 3,
    "show_nav_level": 3,
}
html_static_path = ["_static"]
html_css_files = ['css/custom.css']

sphinx_gallery_conf = {
    "examples_dirs": "examples/gallery",  # Source directory
    "gallery_dirs": "auto_examples",  # Output directory
}

# extensions.remove("sphinx_gallery.gen_gallery")

autosummary_generate = True
autodoc_docstring_signature = True
autodoc_inherit_docstrings = False
# suppress_warnings = ['autodoc']

intersphinx_mapping = {
    'biotite': ('https://www.biotite-python.org/latest', None),
    'numpy': ('https://numpy.org/doc/stable', None)
}
