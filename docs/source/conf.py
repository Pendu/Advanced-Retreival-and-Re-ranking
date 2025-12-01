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

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# =============================================================================
# THEME TOGGLE: Change this to switch between themes
# Options: 'pydata' or 'wagtail'
# =============================================================================
USE_THEME = 'pydata'
#USE_THEME = 'wagtail'

# GitHub repository info (shared by both themes)
GITHUB_URL = 'https://github.com/pendu/Advanced-Retreival-and-Re-ranking'
GITHUB_USER = 'pendu'
GITHUB_REPO = 'Advanced-Retreival-and-Re-ranking'
GITHUB_BRANCH = 'main'

# -----------------------------------------------------------------------------
# Theme-specific configuration
# -----------------------------------------------------------------------------

if USE_THEME == 'pydata':
    # PyData Sphinx Theme
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/
    html_theme = 'pydata_sphinx_theme'
    
    html_theme_options = {
        # Header / Navigation
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        
        # Icon links in the navbar
        "icon_links": [
            {
                "name": "GitHub",
                "url": GITHUB_URL,
                "icon": "fa-brands fa-github",
                "type": "fontawesome",
            },
        ],
        
        # Show edit and view source buttons
        "use_edit_page_button": True,
        "show_nav_level": 2,
        "navigation_depth": 3,
        
        # Footer
        "footer_start": ["copyright"],
        "footer_end": ["theme-version"],
    }
    
    # GitHub context for "Edit on GitHub" button
    html_context = {
        "github_user": GITHUB_USER,
        "github_repo": GITHUB_REPO,
        "github_version": GITHUB_BRANCH,
        "doc_path": "docs/source",
    }

elif USE_THEME == 'wagtail':
    # Sphinx Wagtail Theme
    # https://sphinx-wagtail-theme.readthedocs.io/
    import sphinx_wagtail_theme
    
    html_theme = 'sphinx_wagtail_theme'
    html_theme_path = [sphinx_wagtail_theme.get_html_theme_path()]
    
    html_theme_options = {
        'project_name': project,
        'logo': '',
        'logo_alt': project,
        'logo_height': 59,
        'logo_url': '/',
        'logo_width': 45,
        'github_url': GITHUB_URL + '/',
        'footer_links': ','.join([
            f'GitHub|{GITHUB_URL}',
        ]),
    }
    
    # GitHub context for "Edit on GitHub" and "View source" buttons
    html_context = {
        'display_github': True,
        'github_user': GITHUB_USER,
        'github_repo': GITHUB_REPO,
        'github_version': GITHUB_BRANCH,
        'conf_py_path': '/docs/source/',
    }

else:
    raise ValueError(f"Unknown theme: {USE_THEME}. Use 'pydata' or 'wagtail'.")
