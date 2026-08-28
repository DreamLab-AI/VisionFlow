#!/usr/bin/env bash
# Dream-cycle evaluator: the website build output must exist and contain
# index.html; report page count and size. Checked-in script because the
# annexe ssh dispatch strips nested double quotes from inline entrypoints.
set -u
cd website/dist 2>/dev/null && test -f index.html \
  && echo "pages: $(find . -name '*.html' | wc -l)  bytes: $(du -sb . | cut -f1)" \
  && echo BUILD-OK || echo BUILD-FAIL
