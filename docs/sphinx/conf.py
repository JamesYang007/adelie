# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'adelie'
copyright = '2023, James Yang'
author = 'James Yang'
release = open("../../VERSION", "r").read()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_design",
    "numpydoc",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {
        "image_light": "../logos/adelie-penguin.svg",
        "image_dark": "../logos/adelie-penguin-dark.svg",
    },
    "github_url": "https://github.com/JamesYang007/adelie",
    "collapse_navigation": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {"default_mode": "dark"}
html_static_path = ['_static']
html_css_files = ["numpy.css"]

numpydoc_show_class_members = False