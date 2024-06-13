# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import adelie as ad

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'adelie'
copyright = '2023, James Yang'
author = 'James Yang'
release = ad.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
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

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------
import adelie
import inspect
from os.path import relpath, dirname

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    # Ignore re-exports as their source files are not within the numpy repo
    module = inspect.getmodule(obj)
    if module is not None and not module.__name__.startswith("adelie"):
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None
    fn = relpath(fn, start=dirname(adelie.__file__))

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    path = f"https://github.com/JamesYang007/adelie/blob/main/adelie/{fn}{linespec}"
    return path