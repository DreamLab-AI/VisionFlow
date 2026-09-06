---
id: ADR-2003
title: Deploy the website via the GitHub Pages artifact/deploy actions, never a gh-pages branch
date: 2026-08-31
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: cf535f8
owner: jjohare
review_trigger: any change to the deploy mechanism, hosting provider, or custom domain
repo: visionflow
domain: BASELINE-visionflow.md
lineage: reverses legacy docs/archive/adr/ADR-001-website-technology.md Decision 4 (push built dist/ to a gh-pages branch), which ADR-001 self-flagged as superseded in implementation.
---

# ADR-2003 — Deploy the website via the GitHub Pages artifact/deploy actions, never a gh-pages branch

## Context

Legacy ADR-001 D4 chose the classic path: build `dist/` and push it to a `gh-pages`
branch. That needs `contents: write`, a committed or bot-pushed build output, and it
mixes a generated artefact into git history. GitHub's Pages artifact/deploy actions
(OIDC-attested, no branch) are the alternative and were what actually shipped.

## Decision

`deploy.yml` builds on `main`, uploads `website/dist` with
`actions/upload-pages-artifact@v3`, and publishes with `actions/deploy-pages@v4` under
a two-job build→deploy split. The workflow grants exactly `pages: write` +
`id-token: write` (OIDC), keeps `contents: read`, and serialises releases via a
`concurrency: pages` group with `cancel-in-progress: false`. The custom domain
`www.visionflow.info` is emitted into the artefact by `build.sh`, not committed as a
branch `CNAME`. No `gh-pages` branch exists or is written.

## Consequences

- Forecloses branch-push deploy and everything it implies: no `contents: write` on the
  deploy path, no build output in git history, no bot PAT.
- Binds the repo to GitHub Pages' OIDC artifact model — the `github-pages`
  environment, `id-token: write`, and the two named actions. Moving to another host is
  a mechanism change requiring a new ADR, not a config tweak.
- `cancel-in-progress: false` trades deploy latency (releases queue) for never
  publishing a half-built site — an accepted cost.
- The domain lives in one place (`build.sh:16-17`); nothing reconciles it against DNS,
  so a domain change touches only the build script.

## Verification

At `cf535f8`: `deploy.yml:8-11` sets `pages: write`/`id-token: write`/`contents: read`;
`deploy.yml:13-15` the `pages` concurrency group with `cancel-in-progress: false`;
`deploy.yml:29-32` `upload-pages-artifact@v3` path `website/dist`; `deploy.yml:41-43`
`deploy-pages@v4`. No `gh-pages` ref in the repo; `build.sh:16-17` writes the CNAME.

## Closeout extension — 2026-09-04

Retain accepted/complete/live for the configured Pages artefact mechanism. The current workflow builds, uploads website/dist and deploys with needs: build. It does not call the broader package verify sequence. GitHub branch rules, hosted execution, DNS and production serving were not checked.

**Closeout (CP-08/09):** Decide which quality checks must prevent publication and demonstrate rejection of a defective candidate through that gate. Record the published artefact/revision and recovery procedure. Concurrency serialisation does not by itself certify content quality or deployment recovery.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.

## Acceptance progress — 2026-09-05

Retain accepted/complete/live. The artefact/deploy mechanism is unchanged —
still `upload-pages-artifact@v3` plus `deploy-pages@v4`, still no `gh-pages`
branch. What changed is that publication is now gated: the closeout asked which
quality checks must prevent publication, and the answer is now encoded in the
workflow rather than left to other pipelines.

**Implemented.** `.github/workflows/deploy.yml` runs six **blocking** gates
between the build and `upload-pages-artifact`, so a defective candidate never
reaches the deploy job:

| Gate | Check | Rejects |
|---|---|---|
| asset inventory | `website-assets.mjs verify` | a missing or empty required asset |
| build output | `dream-build-check.sh` → `BUILD-OK` | no `dist/index.html` |
| link integrity | `dream-link-check.sh` → `LINK-INTEGRITY-OK` | any unresolved internal ref |
| meta tags | `dream-meta-tags-scan.sh` → `META-SCAN-OK` | missing title/description/canonical/viewport |
| structured data | `dream-structured-data-scan.sh` → `SD-SCAN-OK` | unparseable JSON-LD |
| diagram baseline | `check-diagram-text.js` | invisible text in a committed diagram |

The first five sentinel-based scripts exit 0 even when they fail, so each step
greps its sentinel rather than trusting the exit code — without that the gates
would have been decorative.

The drift counter runs as a seventh, **reported** gate. It reads its truth from
a pinned agentbox checkout (`ref: 89301ec7…`, matching the allowlist pin); where
that sibling checkout is unavailable on the runner the axis cannot be evaluated,
so the step warns and does not block — the partial-source failure mode of
engineering ADR-005 §Decision 2. Where the checkout succeeds it blocks, because
then it is measuring something real. Which of the two happened is recorded, not
inferred.

**Published artefact revision.** `website-assets.mjs verify` is re-run with
`--published-revision "${GITHUB_SHA}"`, and a final step writes every gate
verdict plus `published.{target,revision,workflow_run,ref}` into
`website/build-receipt.json`, which is uploaded as its own workflow artefact.
A published page can therefore be traced to the revision that produced it, the
`tree_sha256` of the exact bytes uploaded, and the checks that admitted it.

**Tests and results.** All six blocking gates were executed locally against a
real build and all pass: `BUILD-OK`; `internal-refs-checked: 9  missing: 0`,
`LINK-INTEGRITY-OK`; `required-missing: 0`, `META-SCAN-OK` (og-tags 6,
twitter-tags 4, up from 3 and 0 before the ADR-2002 work); `json-ld-blocks: 0
parse-errors: 0`, `SD-SCAN-OK`; all 10 diagrams pass the baseline guard;
`ASSET-INVENTORY-OK` at 13/13 required assets. Rejection is demonstrated rather
than asserted: `tests/gates/website-assets.test.sh` cases [3] and [4] remove a
required asset and truncate one to zero bytes, and the inventory gate exits 1
naming the file in both cases — which under this workflow means the candidate
is never uploaded. All four workflow files parse as valid YAML.

Receipts: [gate closeout receipt with per-gate logs](../estate-closeout/2026-09-05/gate-closeout-receipt.json),
[browser receipt](../estate-closeout/2026-09-05/website-browser-receipt.json).

**Remaining.** Nothing here has run on hosted CI — the rewritten workflow is
unexecuted on GitHub Actions, so hosted behaviour, GitHub branch rules, the
Pages environment, DNS and production serving are all still unverified. No
deployment was performed. The recovery procedure the closeout asks for is still
undocumented: the receipt now identifies which revision is live, which is a
precondition for rollback, but the rollback runbook itself does not exist.

Governed paths changed: `.github/workflows/deploy.yml`.
