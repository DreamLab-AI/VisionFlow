#!/usr/bin/env bash
#
# build-arxiv-package.sh — assemble an arXiv-compliant source package for the
# book "After the Collapse" (XeLaTeX + biber).
#
# What it produces (idempotent — safe to re-run):
#   dist/arxiv-package/        the flat source tree arXiv will compile
#   dist/arxiv-<YYYY-MM-DD>.tar.gz   gzipped tarball of that tree
#
# Why the package differs from the repo tree:
#   * arXiv does NOT run biber, so the pre-built main.bbl is bundled.
#   * arXiv resolves \setmainfont by name via its own fontconfig; that is the
#     classic "works locally, fails on arXiv" trap. We bundle the exact GNU
#     FreeFont .otf files and rewrite the fontspec block to load them by path,
#     so the package is self-contained and font-resolution cannot fail.
#   * A 00README.json declares the XeLaTeX compiler and the toplevel file.
#   * Only images actually referenced by \includegraphics are included; the
#     mermaid/wardley/owm sources and orphan renders are left out.
#
# The repo originals are never mutated. Everything is copied into dist/.
#
# Usage: presentation/report/scripts/build-arxiv-package.sh
set -euo pipefail

# ── Locations ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$REPORT_DIR/dist"
PKG_DIR="$DIST_DIR/arxiv-package"
DATE="$(date +%Y-%m-%d)"
TARBALL="$DIST_DIR/arxiv-${DATE}.tar.gz"

# Bundled font source (GNU FreeFont OTF, GPLv3 + font exception — embed OK).
FREEFONT_OTF_DIR="$(dirname "$(find /nix/store -path '*gnu-freefont*/opentype/*/FreeSerif.otf' 2>/dev/null | head -1)")"
if [[ -z "$FREEFONT_OTF_DIR" || ! -f "$FREEFONT_OTF_DIR/FreeSerif.otf" ]]; then
  # Fallback: any FreeSerif.otf on the system.
  FREEFONT_OTF_DIR="$(dirname "$(find / -name 'FreeSerif.otf' 2>/dev/null | head -1)")"
fi
[[ -f "$FREEFONT_OTF_DIR/FreeSerif.otf" ]] || { echo "FATAL: GNU FreeFont .otf files not found" >&2; exit 1; }

# arXiv practical size ceiling; downscale images only if the assembled tree
# would exceed this. 45 MB leaves headroom under arXiv's ~50 MB limit.
SIZE_CEILING_MB=45
MAX_IMG_WIDTH=1600
JPEG_QUALITY=82

echo "==> report dir : $REPORT_DIR"
echo "==> font source: $FREEFONT_OTF_DIR"
echo "==> output     : $PKG_DIR"
echo "==> tarball    : $TARBALL"

# ── Clean slate ──────────────────────────────────────────────────────────────
rm -rf "$PKG_DIR" "$TARBALL"
mkdir -p "$PKG_DIR/chapters" "$PKG_DIR/appendices" "$PKG_DIR/images" "$PKG_DIR/fonts"

# ── 1. main.tex with fontspec rewritten to load bundled fonts by path ────────
python3 - "$REPORT_DIR/main.tex" "$PKG_DIR/main.tex" <<'PY'
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path, encoding="utf-8").read()

original = (r"\setmainfont{FreeSerif}" "\n"
           r"\setsansfont{FreeSans}" "\n"
           r"\setmonofont{FreeMono}")

replacement = r"""% arXiv-self-contained: bundled GNU FreeFont (GPLv3 + font exception),
% loaded by path so arXiv's font resolution cannot fail. See ARXIV.md.
\setmainfont{FreeSerif}[
  Path=./fonts/,
  Extension=.otf,
  UprightFont=*,
  BoldFont=*Bold,
  ItalicFont=*Italic,
  BoldItalicFont=*BoldItalic
]
\setsansfont{FreeSans}[
  Path=./fonts/,
  Extension=.otf,
  UprightFont=*,
  BoldFont=*Bold,
  ItalicFont=*Oblique,
  BoldItalicFont=*BoldOblique
]
\setmonofont{FreeMono}[
  Path=./fonts/,
  Extension=.otf,
  UprightFont=*,
  BoldFont=*Bold,
  ItalicFont=*Oblique,
  BoldItalicFont=*BoldOblique
]"""

if original not in src:
    sys.exit("FATAL: expected fontspec block not found in main.tex; aborting.")
src = src.replace(original, replacement)
open(dst_path, "w", encoding="utf-8").write(src)
print("    main.tex: fontspec block rewritten for bundled fonts")
PY

# ── 2. Chapter / appendix sources (all \input'd files) ───────────────────────
cp "$REPORT_DIR"/chapters/*.tex "$PKG_DIR/chapters/"
cp "$REPORT_DIR"/appendices/*.tex "$PKG_DIR/appendices/"
echo "    copied $(ls "$PKG_DIR"/chapters/*.tex | wc -l) chapters, $(ls "$PKG_DIR"/appendices/*.tex | wc -l) appendices"

# ── 3. Pre-built bibliography (arXiv does not run biber) ─────────────────────
[[ -f "$REPORT_DIR/main.bbl" ]] || { echo "FATAL: main.bbl missing — run xelatex; biber; xelatex; xelatex first" >&2; exit 1; }
cp "$REPORT_DIR/main.bbl" "$PKG_DIR/main.bbl"
echo "    copied main.bbl ($(du -h "$PKG_DIR/main.bbl" | cut -f1))"

# ── 4. Bundle the exact fonts referenced by the fontspec block ───────────────
FONT_FILES=(
  FreeSerif.otf FreeSerifBold.otf FreeSerifItalic.otf FreeSerifBoldItalic.otf
  FreeSans.otf  FreeSansBold.otf  FreeSansOblique.otf FreeSansBoldOblique.otf
  FreeMono.otf  FreeMonoBold.otf  FreeMonoOblique.otf FreeMonoBoldOblique.otf
)
for f in "${FONT_FILES[@]}"; do
  [[ -f "$FREEFONT_OTF_DIR/$f" ]] || { echo "FATAL: font $f not found in $FREEFONT_OTF_DIR" >&2; exit 1; }
  cp "$FREEFONT_OTF_DIR/$f" "$PKG_DIR/fonts/$f"
done
echo "    bundled ${#FONT_FILES[@]} GNU FreeFont .otf files ($(du -sh "$PKG_DIR/fonts" | cut -f1))"

# ── 5. Only the images actually referenced by \includegraphics ───────────────
#     Resolve each basename against the graphicspath roots (images diagrams wardley).
mapfile -t REFS < <(grep -rhoP '\\includegraphics(\[[^]]*\])?\{\K[^}]+' \
  "$REPORT_DIR/main.tex" "$REPORT_DIR/chapters" "$REPORT_DIR/appendices" | sort -u)
missing=0
for img in "${REFS[@]}"; do
  found=""
  for d in images diagrams wardley .; do
    if [[ -f "$REPORT_DIR/$d/$img" ]]; then found="$REPORT_DIR/$d/$img"; break; fi
  done
  if [[ -n "$found" ]]; then
    cp "$found" "$PKG_DIR/images/$img"
  else
    echo "    MISSING image: $img" >&2; missing=1
  fi
done
[[ $missing -eq 0 ]] || { echo "FATAL: unresolved \\includegraphics targets" >&2; exit 1; }
echo "    copied ${#REFS[@]} referenced images ($(du -sh "$PKG_DIR/images" | cut -f1))"

# ── 6. Verify every referenced image is a permitted format (pdf/png/jpg) ─────
while IFS= read -r f; do
  case "${f,,}" in
    *.pdf|*.png|*.jpg|*.jpeg) : ;;
    *) echo "FATAL: non-arXiv image format bundled: $f" >&2; exit 1 ;;
  esac
done < <(find "$PKG_DIR/images" -type f)

# ── 7. 00README.json — declare XeLaTeX + toplevel file (per arXiv spec) ──────
cat > "$PKG_DIR/00README.json" <<'JSON'
{
  "spec_version": 1,
  "process": {
    "compiler": "xelatex"
  },
  "sources": [
    { "filename": "main.tex", "usage": "toplevel" }
  ],
  "texlive_version": 2025
}
JSON
echo "    wrote 00README.json (compiler=xelatex, toplevel=main.tex)"

# ── 8. Optional downscale ONLY if the tree exceeds the ceiling ───────────────
tree_mb() { du -sm "$PKG_DIR" | cut -f1; }
CUR_MB="$(tree_mb)"
if (( CUR_MB > SIZE_CEILING_MB )); then
  echo "==> package ${CUR_MB} MB > ${SIZE_CEILING_MB} MB ceiling — downscaling raster images"
  if command -v magick >/dev/null 2>&1; then MG="magick"; else MG="convert"; fi
  while IFS= read -r img; do
    case "${img,,}" in
      *.png|*.jpg|*.jpeg)
        w="$($MG identify -format '%w' "$img" 2>/dev/null | head -1 || echo 0)"
        if (( w > MAX_IMG_WIDTH )); then
          "$MG" "$img" -resize "${MAX_IMG_WIDTH}>" -quality "$JPEG_QUALITY" "$img"
        fi ;;
    esac
  done < <(find "$PKG_DIR/images" -type f)
  echo "==> after downscale: $(tree_mb) MB"
else
  echo "    package ${CUR_MB} MB is within the ${SIZE_CEILING_MB} MB ceiling — no downscaling"
fi

# ── 9. Tarball (leave in dist/, untracked) ───────────────────────────────────
#     Use python's tarfile so the script does not depend on a system `tar`.
#     Members are stored at the archive root (arXiv unpacks flat), sorted for
#     reproducibility.
python3 - "$PKG_DIR" "$TARBALL" <<'PY'
import os, sys, tarfile
pkg, out = sys.argv[1], sys.argv[2]
paths = []
for root, dirs, files in os.walk(pkg):
    dirs.sort()
    for f in sorted(files):
        full = os.path.join(root, f)
        paths.append((full, os.path.relpath(full, pkg)))
paths.sort(key=lambda p: p[1])
with tarfile.open(out, "w:gz") as tf:
    for full, arc in paths:
        tf.add(full, arcname=arc, recursive=False)
print(f"    tarball: {len(paths)} files -> {out}")
PY
echo ""
echo "==> DONE"
echo "    package : $PKG_DIR ($(du -sh "$PKG_DIR" | cut -f1))"
echo "    tarball : $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo ""
echo "    largest files in package:"
find "$PKG_DIR" -type f -printf '%s\t%p\n' | sort -rn | head -8 \
  | awk -F'\t' '{ printf "      %6.2f MB  %s\n", $1/1048576, $2 }'
