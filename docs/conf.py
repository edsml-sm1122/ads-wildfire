import sys
import os

# We're working in the ./docs directory, but need the package root in the path
# This command appends the directory one level up, in a cross-platform way.
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '../'))))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ads-wildfire'
copyright = '2023, Team Creek'
author = 'Team Creek'
release = '2023'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = ['sphinx.ext.autodoc',
#               'sphinx.ext.napoleon',
#               'sphinx_automodapi.automodapi',
#               'sphinx.ext.graphviz']
extensions = ['sphinx_automodapi.automodapi', 'sphinx.ext.graphviz']
numpydoc_show_class_members = False
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'bizstyle'
html_static_path = ['_static']
