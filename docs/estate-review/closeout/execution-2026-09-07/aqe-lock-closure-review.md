# AQE locked closure review

The unlocked HP build resolved a changed graph and failed its expected hash.
The original lockfile recovered from the working local AQE3.13.12 package binds
all exact dependency integrity values. Reinstalling it with npm ci produced
sha256-di3k986pcOVtEr88BQ9ckHwQXSj6dCFStYj0GSAvfAU=.

Inventory comparison with the local installed package found all164 existing
package manifests byte-identical, with no removed or version-changed package.
The only added manifest is @ruvector/sona-linux-x64-musl0.1.8, already pinned in
the original lock with integrity sha512-9f9ZYzvUuuUrUoZOAtXRfG0ZEDEOJj8+1hnyk7Yf9yj7taocb048tTcKDRrHmAiUBOgt8Nm9Qo+TetvxVnkKYA==.
SONA index.js lines169-190 selects that module only when isMusl() is true;
glibc selects the existing GNU module. This accounts for dependency-install
completeness rather than adopting a floating version. The reviewed new recursive
hash remains a build gate; a repeat build must verify it before activation.
