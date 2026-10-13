#!/usr/bin/env bash
# Dream-cycle evaluator: the website build output must exist and contain
# index.html; report page count and size. Checked-in script because the
# annexe ssh dispatch strips nested double quotes from inline entrypoints.
# Exits 1 after printing BUILD-FAIL; the printed marker stays the verdict the
# deploy gate and the dream engine read.
set -u
if cd website/dist 2>/dev/null && test -f index.html; then
  echo "pages: $(find . -name '*.html' | wc -l)  bytes: $(du -sb . | cut -f1)"
  echo BUILD-OK
else
  echo BUILD-FAIL
  exit 1
fi
