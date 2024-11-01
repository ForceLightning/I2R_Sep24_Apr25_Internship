import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "I2R Sep 2024-Apr 2025 Internship"
copyright = "2024, Christopher Kok, Yu Yang"
author = "Christopher Kok, Yu Yang"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "models.p3d",
    "tests",
    "thirdparty",
]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for Autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special_members": "__init__",
    "undoc-members": False,
    "private_members": False,
    "show-inheritance": True,
}


# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12/", None),
    "torch": ("https://pytorch.org/docs/2.5/", None),
    "numpy": ("https://numpy.org/doc/1.26/", None),
    "torchmetrics": ("https://lightning.ai/docs/torchmetrics/v1.5.0/", None),
    "matplotlib": ("https://matplotlib.org/3.9.0/", None),
    "segmentation_models_pytorch": ("https://smp.readthedocs.io/en/v0.3.4/", None),
}

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src/"))
