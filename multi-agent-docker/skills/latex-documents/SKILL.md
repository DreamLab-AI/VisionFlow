---
name: LaTeX Documents
description: Professional document preparation with LaTeX - compile papers, presentations, and technical documentation
---

# LaTeX Documents Skill

Comprehensive LaTeX document preparation system with TeX Live toolchain for academic papers, technical documentation, and professional typesetting.

## Capabilities

- Compile LaTeX documents to PDF
- Bibliography management with BibTeX/Biber
- Support for common document classes (article, report, book, beamer)
- Mathematical typesetting with AMS packages
- Figure and table management
- Cross-references and citations
- Multi-file projects with \include and \input
- Bibliography styles (IEEE, ACM, APA, etc.)
- Incremental compilation with latexmk
- Error diagnostics and log parsing

## When to Use This Skill

Use this skill when you need to:
- Write academic papers and research articles
- Create technical documentation
- Generate professional presentations (Beamer)
- Typeset mathematical equations
- Produce publication-quality documents
- Manage bibliographies and citations
- Create multi-chapter books or theses
- Generate consistent formatted output

## Prerequisites

- **TeX Live packages**: texlive-basic, texlive-bin, texlive-binextra, texlive-fontsrecommended, texlive-latexrecommended
- **Bibliography**: biber for biblatex processing
- **Tools**: pdflatex, xelatex, lualatex, latexmk

## Available Commands

### Compilation
- `pdflatex <file>.tex` - Standard LaTeX to PDF
- `xelatex <file>.tex` - Unicode and modern fonts
- `lualatex <file>.tex` - Lua-enhanced LaTeX
- `latexmk -pdf <file>.tex` - Automated build with dependencies

### Bibliography
- `bibtex <file>` - Traditional BibTeX processing
- `biber <file>` - Modern biblatex backend

### Utilities
- `texdoc <package>` - View package documentation
- `kpsewhich <file>` - Locate TeX files

## Instructions

### Creating a Basic Document

1. **Create .tex file** with document structure:
```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}

\title{Document Title}
\author{Author Name}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
Content here...

\end{document}
```

2. **Compile** using:
```bash
pdflatex document.tex
```

### Bibliography Workflow

1. **Create .bib file** (references.bib):
```bibtex
@article{author2025,
  author = {Author, First},
  title = {Paper Title},
  journal = {Journal Name},
  year = {2025}
}
```

2. **In .tex file**:
```latex
\usepackage[backend=biber,style=ieee]{biblatex}
\addbibresource{references.bib}

% In document
\cite{author2025}

% At end
\printbibliography
```

3. **Compile sequence**:
```bash
pdflatex document.tex
biber document
pdflatex document.tex
pdflatex document.tex
```

Or use latexmk:
```bash
latexmk -pdf -bibtex document.tex
```

### Mathematical Typesetting

Common math environments:
```latex
% Inline math
$E = mc^2$

% Display math
\[
  \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
\]

% Numbered equations
\begin{equation}
  \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
\end{equation}

% Aligned equations
\begin{align}
  a &= b + c \\
  d &= e \cdot f
\end{align}
```

### Creating Presentations (Beamer)

```latex
\documentclass{beamer}
\usetheme{Madrid}

\title{Presentation Title}
\author{Your Name}
\date{\today}

\begin{document}

\frame{\titlepage}

\begin{frame}{Frame Title}
  \begin{itemize}
    \item Point 1
    \item Point 2
  \end{itemize}
\end{frame}

\end{document}
```

### Multi-File Projects

Main file (main.tex):
```latex
\documentclass{book}
\begin{document}
\include{chapter1}
\include{chapter2}
\end{document}
```

Chapter files (chapter1.tex):
```latex
\chapter{Introduction}
Content...
```

### Error Handling

Common errors and fixes:
- **Missing package**: Install via `tlmgr install <package>`
- **Undefined control sequence**: Check package imports
- **Missing references**: Run biber/bibtex and recompile
- **File not found**: Check paths and \graphicspath

## Document Classes

Available classes:
- `article` - Journal articles, short papers
- `report` - Technical reports, theses
- `book` - Books, longer documents
- `beamer` - Presentations
- `IEEEtran` - IEEE journal format
- `acmart` - ACM publication format

## Common Packages

### Essential
- `amsmath, amssymb` - Math symbols and environments
- `graphicx` - Include images
- `hyperref` - PDF hyperlinks
- `geometry` - Page layout
- `fancyhdr` - Custom headers/footers

### Bibliography
- `biblatex` - Modern bibliography (use with biber)
- `natbib` - Natural sciences citations

### Tables & Figures
- `booktabs` - Professional tables
- `multirow` - Multi-row cells
- `subcaption` - Subfigures

### Code Listings
- `listings` - Source code formatting
- `minted` - Syntax highlighting (requires Python Pygments)

## Output Formats

- **PDF** - Primary output format
- **DVI** - Intermediate format (convert with dvipdf)
- **PS** - PostScript (convert with dvips)

## Best Practices

1. **Use latexmk** for automated builds
2. **Version control** .tex and .bib files (exclude .aux, .log, .pdf)
3. **Modular structure** for large documents
4. **Consistent formatting** with packages like `cleveref`
5. **Float placement** - Let LaTeX manage figure positions
6. **Cross-references** - Use \label and \ref

## Troubleshooting

### Compilation Issues
- **Check .log file** for detailed errors
- **Clear auxiliary files**: `rm *.aux *.bbl *.blg *.log`
- **Update TeX Live**: `tlmgr update --self --all`

### Bibliography Not Showing
- Ensure `\addbibresource{file.bib}` before \begin{document}
- Run biber: `biber main`
- Check .blg file for biber errors
- Recompile LaTeX twice after biber

### Missing Fonts
- Use xelatex or lualatex for system fonts
- Install font packages: `tlmgr install <font-package>`

## Integration with Jupyter

Export Jupyter notebooks to LaTeX:
```bash
jupyter nbconvert --to latex notebook.ipynb
pdflatex notebook.tex
```

## Related Skills

- **jupyter-notebooks** - Generate LaTeX from notebooks
- **data-visualization** - Create figures for inclusion
- **git** - Version control for documents

## Technical Details

- **TeX engine**: pdfTeX, XeTeX, LuaTeX
- **Distribution**: TeX Live (basic installation)
- **Version**: TeX Live 2024+
- **Additional packages**: Install via `tlmgr`

## Notes

- Basic TeX Live installation (~500MB)
- Full TeX Live is ~7GB (install packages as needed)
- PDF generation typically takes 1-5 seconds
- Multi-pass compilation required for references
- Unicode support via XeLaTeX or LuaLaTeX
- Compatible with Overleaf projects (copy .tex and .bib files)
