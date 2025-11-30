# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Advanced Retrieval and Re-ranking'
copyright = '2025, Abhijeet Pendyala'
author = 'Abhijeet Pendyala'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_wagtail_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_wagtail_theme'
html_static_path = ['_static']
html_theme_path = [sphinx_wagtail_theme.get_html_theme_path()]

# Wagtail theme options
html_theme_options = {
    'project_name': 'Advanced Retrieval and Re-ranking',
    'logo': '',
    'logo_alt': 'Advanced Retrieval and Re-ranking',
    'logo_height': 59,
    'logo_url': '/',
    'logo_width': 45,
    'github_url': 'https://github.com/pendu/Advanced-Retreival-and-Re-ranking',
    'footer_links': ','.join([
    'https://github.com/pendu/Advanced-Retreival-and-Re-ranking',
    ]),
}
