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
