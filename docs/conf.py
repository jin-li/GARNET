"""Build the user and methodology manuals from the same Markdown sources."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
project = 'NDgpu'
author = 'NDgpu contributors'
release = '0.1.0'
extensions = ['myst_parser', 'sphinx.ext.mathjax', 'sphinx.ext.autodoc']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
root_doc = 'index'
exclude_patterns = ['_build', 'report.html', '*_plan.md', '*_design.md', 'development_assessments.md']
myst_heading_anchors = 4
myst_enable_extensions = ['amsmath', 'dollarmath', 'colon_fence']
html_theme = 'alabaster'
html_title = 'NDgpu manuals'
html_static_path = []
latex_engine = 'pdflatex'
latex_documents = [
    ('user/index', 'ndgpu-user-manual.tex', 'NDgpu User Manual', author, 'manual', False),
    ('methodology/index', 'ndgpu-methodology-manual.tex', 'NDgpu Software Methodology Manual', author, 'manual', False),
]
latex_elements = {'papersize': 'a4paper', 'pointsize': '11pt', 'fontpkg': r'\usepackage{lmodern}', 'fncychap': '', 'preamble': r'\DeclareUnicodeCharacter{2212}{\ensuremath{-}}'}
