# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FastFoodSim"
copyright = "2026, Maxime Szymanski"
author = "Maxime Szymanski"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # Point to the root folder

extensions = [
    "sphinx.ext.autodoc",  # Core library for pulling docstrings
    "sphinx.ext.napoleon",  # Specifically for Google/NumPy styles
    "sphinx.ext.viewcode",  # Adds "view source" links
    "sphinx_rtd_theme",  # Modern theme
]

html_theme = "sphinx_rtd_theme"
