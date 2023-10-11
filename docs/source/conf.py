# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at
# https://oss.oracle.com/licenses/upl/

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

autoclass_content = "both"

sys.path.insert(0, os.path.abspath("../../"))

import guardian_ai

version = guardian_ai.__version__
release = version


# -- Project information -----------------------------------------------------
# TODO: Update project name
project = "Oracle Guardian AI Open Source Project"
copyright = (
    f"2023, {datetime.datetime.now().year} Oracle and/or its affiliates. "
    f"Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/"
)
author = "Oracle Data Science"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.todo",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_code_tabs",
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_autorun",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Get version
import guardian_ai

version = guardian_ai.__version__
release = version

# Unless we want to expose real buckets and namespaces
nbsphinx_allow_errors = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
# autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
language = "en"

# Disable the generation of the various indexes
html_use_modindex = False
html_use_index = False

html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
