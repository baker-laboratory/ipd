import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "IPD"
author = "Your Name"
release = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Supports Google/NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",  # For example galleries like Biotite
]

# -- HTML Theme Configuration ------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 2,
    "show_nav_level": 2,
}
html_static_path = ["_static"]
autosummary_generate = True

sphinx_gallery_conf = {
    "examples_dirs": "examples/gallery",  # Source directory
    "gallery_dirs": "auto_examples",  # Output directory
}

# extensions.remove("sphinx_gallery.gen_gallery")
