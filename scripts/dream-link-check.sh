#!/bin/bash
# dream-link-check.sh — provably-live internal-link/asset integrity check for
# the built static site. Emits a surface-dependent count; non-zero missing
# refs prints LINK-INTEGRITY-FAIL and exits 1 (the printed marker stays the
# verdict the deploy gate and the dream engine read; the exit code makes the
# contract hold when the script is run on its own). Used as a dream-cycle
# evaluator entrypoint.
#
# Five ref families are resolved:
#   1. href/src attributes — ordinary markup links and asset references.
#   2. meta[content] media URLs (og:image, twitter:image, …). These were a
#      shared blind spot: dream-meta-tags-scan.sh COUNTS the tags but never
#      looks at the URL, and this script only ever parsed href/src, so an
#      og:image pointing at a file absent from dist/ passed both gates. The
#      social card is a required asset in website/assets.manifest.json, and
#      this makes that requirement observable from the link surface too.
#   3. in-page anchors — every href="#id" must resolve to an id in the same
#      shipped document. Family 1 skips refs containing '#' by design, so a
#      nav link to a renamed or removed section passed this gate silently
#      (2026-09-28 finding; that PR never landed). Baseline 0 dangling,
#      operator-verified 2026-10-02, so dangling > 0 now FAILS.
#
#   4. JSON-LD URLs — every same-origin absolute URL inside an
#      application/ld+json block must resolve in dist/.
#      dream-structured-data-scan.sh PARSES the blocks but never looks at
#      the URLs they carry (Organization logo/url, WebSite url, …), so a
#      logo pointing at a file absent from dist/ passed every gate — the
#      same blind spot family 2 closed for meta[content]. Off-site URLs
#      are skipped, exactly as in every other family.
#
#   5. CNAME↔page-host agreement — canonical, og:url and the JSON-LD `url`
#      fields declare this site's own address, and social-card images must
#      be same-site (operator decision 2026-10-09). Families 1-4 treat any
#      host that is not dist/CNAME's as off-site and skip it, so a canonical
#      drifting to the apex host or a card on a CDN passed every gate
#      silently; family 5 makes that drift the failure it is.
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

miss=0; checked=0; anchors=0; dangling=0; hostrefs=0; drift=0
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

  # 3. in-page anchors — same-document only: a fragment never names a file, so
  #    resolution is an id lookup within the referring page itself.
  ids=$(grep -oE 'id="[^"]+"' "$f" 2>/dev/null | sed -E 's/^id="([^"]*)"$/\1/' | sort -u)
  for ref in $(grep -oE 'href="#[^"]+"' "$f" 2>/dev/null | sed -E 's/^href="#([^"]*)"$/\1/'); do
    anchors=$((anchors+1))
    printf '%s\n' "$ids" | grep -qxF -- "$ref" \
      || { echo "DANGLING-ANCHOR: $f -> #$ref"; dangling=$((dangling+1)); }
  done

  # 4. JSON-LD URLs — ld+json payloads carry absolute URLs (logo, url, …).
  #    Same-origin ones must exist in dist/; off-site skipped as everywhere.
  #    dream-structured-data-scan.sh parses these blocks for JSON validity
  #    but never resolves the URLs inside them.
  for ref in $(sed -n '/<script[^>]*application\/ld+json/,/<\/script>/p' "$f" 2>/dev/null \
             | grep -oE '"https?://[^"]+"' | tr -d '"'); do
    t=$(resolve_ref "$ref")
    [ -n "$t" ] || continue
    checked=$((checked+1))
    [ -e "$d/$t" ] || [ -e "./$t" ] || { echo "MISSING: $f -> $ref"; miss=$((miss+1)); }
  done
  # 5. CNAME↔page-host agreement. canonical, og:url and the JSON-LD `url`
  #    fields declare THIS site's address, and social-card images must be
  #    same-site (operator decision 2026-10-09). Families 1-4 treat any host
  #    that is not dist/CNAME's as off-site and skip it, so a canonical
  #    drifting to the apex host or a card on a CDN passed every gate
  #    silently; here the wrong host on a self-referential declaration is
  #    itself the defect.
  if [ -n "$site_host" ]; then
    # canonical link — the page's own preferred URL.
    for ref in $(grep -oE '<link[^>]+rel="canonical"[^>]*>' "$f" 2>/dev/null | grep -oE 'href="[^"]+"' | sed -E 's/href="([^"]+)"/\1/'); do
      hostrefs=$((hostrefs+1))
      case "$ref" in
        "https://$site_host"|"https://$site_host/"*) ;;
        *) echo "HOST-DRIFT: $f -> $ref (canonical; CNAME host is $site_host)"; drift=$((drift+1)) ;;
      esac
    done
    # og:url — the address crawlers are told is canonical.
    for ref in $(grep -oiE '<meta[^>]*(property|name)="og:url"[^>]*>' "$f" 2>/dev/null | grep -oE 'content="[^"]+"' | sed -E 's/content="([^"]+)"/\1/'); do
      hostrefs=$((hostrefs+1))
      case "$ref" in
        "https://$site_host"|"https://$site_host/"*) ;;
        *) echo "HOST-DRIFT: $f -> $ref (og:url; CNAME host is $site_host)"; drift=$((drift+1)) ;;
      esac
    done
    # JSON-LD "url" fields (Organization, WebSite). sameAs, logo and @id are
    # excluded: they may legitimately point elsewhere or carry a fragment.
    for ref in $(sed -n '/<script[^>]*application\/ld+json/,/<\/script>/p' "$f" 2>/dev/null | grep -oE '"url": *"[^"]+"' | sed -E 's/"url": *"([^"]+)"/\1/'); do
      hostrefs=$((hostrefs+1))
      case "$ref" in
        "https://$site_host"|"https://$site_host/"*) ;;
        *) echo "HOST-DRIFT: $f -> $ref (JSON-LD url; CNAME host is $site_host)"; drift=$((drift+1)) ;;
      esac
    done
    # Social-card images: same-site required (operator decision 2026-10-09);
    #    a relative path stays legal and is resolved by family 2 above.
    for ref in $(grep -oiE '<meta[^>]*(property|name)="(og:image(:secure_url|:url)?|twitter:image)"[^>]*>' "$f" 2>/dev/null | grep -oE 'content="[^"]+"' | sed -E 's/content="([^"]+)"/\1/'); do
      hostrefs=$((hostrefs+1))
      case "$ref" in
        "https://$site_host/"*|"http://$site_host/"*) ;;
        https:*|http:*|//*) echo "OFFSITE-IMAGE: $f -> $ref (must be same-site per 2026-10-09)"; drift=$((drift+1)) ;;
      esac
    done
  fi
done < <(find . -name '*.html')
echo "internal-refs-checked: $checked  missing: $miss"
echo "anchors-checked: $anchors  dangling: $dangling"
echo "cname-host-refs: $hostrefs  drift: $drift"
if [ "$miss" -eq 0 ] && [ "$dangling" -eq 0 ] && [ "$drift" -eq 0 ]; then
  echo LINK-INTEGRITY-OK
else
  echo LINK-INTEGRITY-FAIL
  exit 1
fi
