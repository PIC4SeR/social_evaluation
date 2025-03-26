# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "social_evaluation"
copyright = "2025, PIC4SeR Andrea Ostuni"
author = "Andrea Ostuni"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.todo",  # Enable TODOs in docs
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx_copybutton",  # Add copy buttons to code blocks
    "sphinx_autodoc_typehints",  # Include type hints
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/PIC4SeR/social_evaluation_graphs",
    "use_repository_button": True,
}
html_static_path = ["_static"]
html_logo = "../images/Logo-Outline.png"
html_title = "Social Evaluation"
