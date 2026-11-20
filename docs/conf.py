# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration shared by the English and Japanese user documentation.

Build one language at a time; the language is chosen by the source directory:

    sphinx-build -b html -c docs docs/en out/docs/html/en
    sphinx-build -b html -c docs -D language=ja docs/ja out/docs/html/ja
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_DOCS = Path(__file__).resolve().parent
_ROOT = _DOCS.parent

_version = re.search(r"VERSION\s+(\d+\.\d+\.\d+)", (_ROOT / "CMakeLists.txt").read_text("utf-8"))
project = "cxxgpr"
author = "cxxgpr contributors"
copyright = "cxxgpr contributors, Apache-2.0"  # noqa: A001
release = _version.group(1) if _version else "0"
version = release

language = "en"

extensions = [
    "myst_parser",
    "breathe",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]

# --- MyST ----------------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]
mermaid_version = "11.4.1"

# --- Breathe (Doxygen XML -> API reference) --------------------------------------------------
breathe_projects = {"cxxgpr": os.environ.get("CXXGPR_DOXYGEN_XML", str(_ROOT / "out" / "docs" / "doxygen" / "xml"))}
breathe_default_project = "cxxgpr"
breathe_domain_by_extension = {"hpp": "cpp"}
breathe_default_members = ("members", "undoc-members")
breathe_show_include = False

primary_domain = "cpp"
highlight_language = "cpp"
pygments_style = "friendly"
pygments_dark_style = "monokai"

# --- Theme ----------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"cxxgpr {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "language-switch.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ]
}
html_theme_options = {
    "source_repository": "https://github.com/YUKIKEDA/cxxgpr/",
    "source_branch": "main",
    "source_directory": "docs/en/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#2a5ada",
        "color-brand-content": "#2a5ada",
        "font-stack": "Inter, system-ui, -apple-system, 'Segoe UI', 'Hiragino Sans', 'Yu Gothic UI', sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'Cascadia Code', Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fb0ff",
        "color-brand-content": "#8fb0ff",
    },
}

copybutton_prompt_text = r"^\$ |^PS> "
copybutton_prompt_is_regexp = True

# Doxygen-generated declarations can legitimately repeat across overloads.
suppress_warnings = ["myst.header", "duplicate_declaration.cpp"]
nitpicky = False


def setup(app):  # type: ignore[no-untyped-def]
    """Expose the other language's URL to the sidebar template."""

    def add_context(app, pagename, templatename, context, doctree):  # type: ignore[no-untyped-def]
        lang = app.config.language
        other = "ja" if lang == "en" else "en"
        depth = pagename.count("/")
        context["lang_current"] = lang
        context["lang_other"] = other
        context["lang_other_label"] = "日本語" if other == "ja" else "English"
        context["lang_other_url"] = "../" * (depth + 1) + f"../{other}/{pagename}.html"

    app.connect("html-page-context", add_context)
