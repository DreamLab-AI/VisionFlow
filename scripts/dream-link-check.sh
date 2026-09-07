#!/bin/bash
# dream-link-check.sh — provably-live internal-link/asset integrity check for
# the built static site. Emits a surface-dependent count; non-zero missing
# refs prints LINK-INTEGRITY-FAIL. Used as a dream-cycle evaluator entrypoint.
#
# Two ref families are resolved:
#   1. href/src attributes — ordinary markup links and asset references.
#   2. meta[content] media URLs (og:image, twitter:image, …). These were a
#      shared blind spot: dream-meta-tags-scan.sh COUNTS the tags but never
#      looks at the URL, and this script only ever parsed href/src, so an
#      og:image pointing at a file absent from dist/ passed both gates. The
#      social card is a required asset in website/assets.manifest.json, and
#      this makes that requirement observable from the link surface too.
#
# Social URLs are absolute on the canonical domain (they must be, for the
# crawlers), so "site-relative" here includes same-origin absolute URLs: the
# host is read from dist/CNAME and stripped to a dist-relative path. Genuinely
# off-site URLs are skipped, exactly as they are for href/src.
set -uo pipefail
cd "$(dirname "$0")/../website/dist" 2>/dev/null || { echo "NO-DIST (run build first)"; exit 0; }

site_host=$(tr -d '[:space:]' < CNAME 2>/dev/null) || site_host=''

# Reduce a ref to a dist-relative path, or print nothing if it is off-site.
resolve_ref() {
  local r="${1%%\?*}"; r="${r%%#*}"
  if [ -n "$site_host" ]; then
    case "$r" in
      "https://$site_host"|"http://$site_host") return 0 ;;
      "https://$site_host/"*) printf '%s' "${r#https://$site_host/}"; return 0 ;;
      "http://$site_host/"*)  printf '%s' "${r#http://$site_host/}";  return 0 ;;
    esac
  fi
  case "$r" in
    ''|http:*|https:*|//*|data:*|mailto:*|tel:*|javascript:*) return 0 ;;
    /*) printf '%s' "${r#/}" ;;
    *)  printf '%s' "$r" ;;
  esac
}

miss=0; checked=0
while IFS= read -r f; do
  d=$(dirname "$f")

  # 1. href/src
  for ref in $(grep -oE '(src|href)="[^"#:]+"' "$f" 2>/dev/null | sed -E 's/.*="([^"]+)".*/\1/' | grep -vE '^(https?:|//|mailto:|data:)'); do
    t="${ref%%\?*}"; checked=$((checked+1))
    [ -e "$d/$t" ] || [ -e "./$t" ] || { echo "MISSING: $f -> $ref"; miss=$((miss+1)); }
  done

  # 2. meta[content] media URLs — matched on the whole tag so attribute order
  #    does not matter. Only URL-bearing meta is considered; meta description
  #    content is prose, not a ref.
  for ref in $(grep -oiE '<meta[^>]*(property|name)="(og:image(:secure_url|:url)?|og:video|og:audio|twitter:image|twitter:player)"[^>]*>' "$f" 2>/dev/null \
               | grep -oE 'content="[^"]+"' | sed -E 's/content="([^"]+)"/\1/'); do
    t=$(resolve_ref "$ref")
    [ -n "$t" ] || continue
    checked=$((checked+1))
    [ -e "$d/$t" ] || [ -e "./$t" ] || { echo "MISSING: $f -> $ref"; miss=$((miss+1)); }
  done
done < <(find . -name '*.html')
echo "internal-refs-checked: $checked  missing: $miss"
[ "$miss" -eq 0 ] && echo LINK-INTEGRITY-OK || echo LINK-INTEGRITY-FAIL
