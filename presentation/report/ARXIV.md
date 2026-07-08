# arXiv submission package

This directory builds an arXiv-ready source package for the book *After the
Collapse: Mapping the Solution Space for Human–Agent Collaboration* (XeLaTeX +
biber, `main.tex`, 28 chapters, 3 appendices).

Build it with:

```bash
presentation/report/scripts/build-arxiv-package.sh
```

That produces (both under `dist/`, which is git-ignored):

- `dist/arxiv-package/` — the flat source tree arXiv compiles
- `dist/arxiv-<YYYY-MM-DD>.tar.gz` — the tarball you upload

The tarball is the deliverable. Neither it nor the assembled tree is committed;
only this document and the build script are.

## What the package contains

| Item | Why it is there |
|------|-----------------|
| `main.tex` | Top-level source. Its fontspec block is rewritten (see Fonts). |
| `chapters/*.tex`, `appendices/*.tex` | Every `\input`'d file (28 + 3). |
| `main.bbl` | Pre-built bibliography. **arXiv does not run biber**, so the compiled `.bbl` must ship with the source. |
| `images/*` | Only the 23 files actually referenced by `\includegraphics`, all `jpg`/`png`. |
| `fonts/*.otf` | The 12 GNU FreeFont OpenType files the document uses. |
| `00README.json` | Declares the XeLaTeX compiler and the toplevel file. |

Deliberately excluded: the LaTeX aux/log/toc artifacts; `references.bib` (the
`.bbl` is what arXiv uses — see Bibliography); the mermaid `.mmd`, Wardley
`.owm`/`.html`/`.py`, and `diagrams/rendered` / `wardley/rendered` `.svg`
sources (none are referenced by `\includegraphics`; XeLaTeX cannot embed SVG
anyway); and the `notebooklm/` media (audio/video/pptx, ~160 MB, unused by the
book).

## The four arXiv rules and how each is satisfied

1. **No biber on arXiv.** The pre-built `main.bbl` is bundled. The script
   refuses to run if it is missing, and the verification below compiles the
   package with **no biber step at all**.

2. **XeLaTeX is declared.** `00README.json` (verified against the live arXiv
   spec at info.arxiv.org/help/00README.html and .../submit_tex.html) sets
   `process.compiler = "xelatex"` and marks `main.tex` as `toplevel`, and pins
   `texlive_version` to 2025:

   ```json
   {
     "spec_version": 1,
     "process": { "compiler": "xelatex" },
     "sources": [ { "filename": "main.tex", "usage": "toplevel" } ],
     "texlive_version": 2025
   }
   ```

3. **Fonts are bundled, not resolved by name.** See Fonts below — this is the
   single most common reason a XeLaTeX submission that builds locally fails on
   arXiv, and it is closed.

4. **No shell-escape, and only permitted graphics formats.** There is no
   `minted`, no `svg` package, no `epstopdf`, no `\write18` anywhere in the
   sources. Every referenced image is `jpg` or `png` (17 `jpg`, 6 `png`); the
   script aborts if any bundled image is not `pdf`/`png`/`jpg`.

## Fonts (the classic failure mode, closed)

`main.tex` sets `\setmainfont{FreeSerif}`, `\setsansfont{FreeSans}`,
`\setmonofont{FreeMono}` — the GNU FreeFont family. Loaded *by name*, XeLaTeX
resolves these through the compile host's font configuration. That works on
this machine (XeTeX indexes the TeX Live tree) but is exactly what breaks on a
differently-configured arXiv worker; on this host `fc-match FreeSerif` already
falls back to DejaVu, so name resolution is demonstrably fragile.

The build script therefore **bundles the exact GNU FreeFont `.otf` files** into
`fonts/` and rewrites the fontspec block to load them by path
(`Path=./fonts/`, `Extension=.otf`, with explicit Bold/Italic/Oblique members).
The repo `main.tex` is left untouched; only the copy inside `dist/` is
rewritten. Because the bundled files are the same GNU FreeFont metrics XeTeX
was already using, output is byte-for-byte the same layout (339 pages,
confirmed below).

This makes font resolution deterministic and host-independent. Proof: with
`fonts/` removed, the package build hard-fails with *"The font FreeSerif cannot
be found"* — there is no silent fallback to a system font, so the bundled files
are provably the ones used.

**Licence.** GNU FreeFont is distributed under the **GNU GPL v3 with the GNU
Font Exception**. The Font Exception explicitly permits embedding the fonts in,
and distributing them with, a document without imposing the GPL on that
document, and permits redistribution of the font files themselves. Bundling the
12 `.otf` files in the submission is therefore licence-compliant. (Reference:
gnu.org/software/freefont/license.html.)

## Bibliography and the biblatex-version caveat

The document uses `biblatex` with the `biber` backend. The bundled `main.bbl`
was generated locally with:

- **biblatex** `v3.21` (2025/07/10)
- **biber** `2.21 (beta)`
- **`.bbl` format version `3.3`** (this string is in the `.bbl` header)

**Caveat the submitter should understand.** arXiv does not run biber; it feeds
the bundled `.bbl` straight to XeLaTeX, and biblatex reads it at compile time.
biblatex will refuse a `.bbl` whose format version is newer than the biblatex
it ships. `.bbl` format `3.3` requires **biblatex ≥ 3.20**, which is what TeX
Live 2025 ships (3.21). This is why `00README.json` pins `texlive_version:
2025`. As long as arXiv compiles under TeX Live 2025 (the default), the `.bbl`
is compatible. If arXiv ever forced an older TeX Live, the `.bbl` would need
regenerating against that year's biblatex — re-run
`xelatex; biber; xelatex; xelatex` under the matching toolchain and rebuild.

`references.bib` is **not** included. With the `.bbl` present, arXiv uses it and
never consults the `.bib` (per the live submit_tex guidance). The standalone
verification below compiles the full bibliography with no `.bib` in the tree,
confirming it is genuinely unnecessary.

## Size

The assembled tree is **24 MB** (tarball **21 MB**), comfortably under arXiv's
~50 MB practical ceiling, so **no image downscaling was applied**.

| Component | Size | Files |
|-----------|------|-------|
| `images/` | 17 MB | 23 |
| `fonts/` | 6.7 MB | 12 |
| `chapters/` | 640 KB | 28 |
| `main.bbl` | 244 KB | 1 |
| `appendices/` | 60 KB | 3 |
| `main.tex` | 16 KB | 1 |
| `00README.json` | <1 KB | 1 |
| **Total package** | **24 MB** | **69** |
| **Tarball (.tar.gz)** | **21 MB** | — |

The script *does* carry a downscale path (max 1600 px wide, quality 82, applied
only to copies inside `dist/`) that triggers automatically if a future image
set pushes the tree past 45 MB. It is inert today.

## How it was verified

The package was unpacked into a clean directory containing nothing from the repo
tree (no stale `.aux`/`.bcf`), then compiled with XeLaTeX only — **no biber** —
exactly as arXiv does:

```
xelatex -interaction=nonstopmode main.tex   # pass 1 -> 327 pp, exit 0
xelatex -interaction=nonstopmode main.tex   # pass 2 -> 339 pp, exit 0, no rerun requested
xelatex -interaction=nonstopmode main.tex   # pass 3 -> 339 pp, exit 0 (stable)
```

Results:

- **Exit 0** on every pass; **0** lines beginning `!` (no LaTeX errors).
- **339 pages**, matching the repository build exactly.
- **0 undefined citations**, **0 undefined references** in the final log.
- **0 missing graphics**, **0 font-not-found** errors.
- Fonts served from the bundled `./fonts/` only (negative test above).

To reproduce the verification from scratch:

```bash
presentation/report/scripts/build-arxiv-package.sh
mkdir -p /tmp/arxiv-verify && cd /tmp/arxiv-verify
python3 -c "import tarfile; tarfile.open('<repo>/presentation/report/dist/arxiv-<date>.tar.gz').extractall('.')"
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex   # expect: Output written on main.pdf (339 pages)
```

## What the submitter must decide at upload

Two choices are the author's to make in the arXiv web form; the package does not
and cannot pre-decide them:

- **Licence.** arXiv asks you to pick a distribution licence at submission
  (default arXiv non-exclusive licence, or CC-BY / CC-BY-SA / CC0, etc.). This
  is the licence on *your work*, independent of the GNU FreeFont licence on the
  bundled font files. Choose per your intent for the book.
- **Primary category.** This is not a conventional maths/physics paper. Likely
  fits **cs.CY** (Computers and Society) or **cs.HC** (Human–Computer
  Interaction), possibly cross-listed. Pick the category that matches the
  audience you want.
