---
id: ADR-2008
title: Estate health is collected nightly by CI and read, not collected, by the dream cycle
date: 2026-09-06
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: d42eec56157cd7e3ce1a724793116d90c2ddd293
owner: jjohare
review_trigger: a change to the estate roster (scripts/estate-health/roster.json), a change to the snapshot schema, or the dream cycle acquiring credentials of its own
repo: visionflow
domain: BASELINE-visionflow.md
lineage: extends ADR-2006 (the canon owns the cross-repo view) by making that view collected rather than asserted, and replaces the hand-maintained closeout table that ADR-2007's roadmap produced as the living estate view; publishes through the ADR-2003 Pages deploy under the ADR-2002 copy-only build.
---

# ADR-2008 — Estate health is collected nightly by CI and read, not collected, by the dream cycle

## Context

The 2026-09-06 closeout table (`docs/estate-closeout/2026-09-06/closeout-table.md`) was
maintained by hand inside one session. It was accurate at 08:00Z and stale by the evening:
rows 3, 9, 13, 22 and 23 each changed state during the same working day and had to be
rewritten. The estate is fourteen repositories plus six public surfaces and seven registry
entries — more state than a session can re-read, and it moves whether or not anyone is
looking. Two constraints bound any fix. The dream cycle runs on the HP annexe, which holds
no credentials and must stay offline-safe, so it cannot query GitHub. And a push made with
`GITHUB_TOKEN` does not fire `on: push` workflows, so a workflow that commits data cannot
rely on the deploy triggering itself.

## Decision

Estate health is **collected by CI and read by everything else**. A zero-dependency Node
collector (`scripts/estate-health.mjs collect`) queries the roster in
`scripts/estate-health/roster.json` — every repository's default-branch CI state, open PRs,
latest release, head commit and Pages status, plus the public surfaces and the crates.io and
npm entries — and writes one snapshot, `website/static/data/estate-health.json`, against the
`visionflow.estate-health/1` schema. `.github/workflows/estate-health.yml` runs it at 02:30
UTC, commits the snapshot as a bot data commit, and — because a `GITHUB_TOKEN` push fires no
`on: push` trigger — explicitly dispatches `deploy.yml`, so the snapshot reaches
www.visionflow.info through the same publication gates as every other byte of the site. The
`#estate` section renders that committed JSON and nothing else; it performs no live queries.
The 03:00 UTC dream cycle gains a fifth rotation slot, `estate-health`, whose evaluator is
`node scripts/estate-health.mjs check` — an **offline** read of the committed snapshot that
prints one verdict line (`ESTATE-HEALTH-OK`, `ESTATE-HEALTH-STALE` when `generated_at` is
older than 36 hours, or `ESTATE-HEALTH-RED`) and exits non-zero on the latter two. The
binding rule: **a dream night reads the snapshot and forms hypotheses about it; it never
collects one, never acquires a token, and never edits the snapshot by hand.** A red
repository is therefore evidence available to tonight's hypothesis, not a task for it.

## Consequences

- **The hand-maintained table stops being the living view.** The dated closeout tables remain
  what they are — evidence of a particular day's audit — and the nightly snapshot carries the
  current state. A stale closeout row is now a historical record, not a lie.
- **A private repository is unreadable without `ESTATE_READ_TOKEN`.** `visionGraph` is owned
  by `jjohare`, so the workflow's repo-scoped `GITHUB_TOKEN` gets a 404. The collector records
  `readable: false`, `ci.state: "unknown"` and null counts rather than failing the run, and the
  page shows it as unreadable. The snapshot is honest about the hole; it does not close it.
- **A red snapshot still deploys.** The workflow deliberately does not fail on `check`'s
  non-zero exit, and no publication gate in `deploy.yml` consults estate health. The page
  *reports* the estate; it does not gate on it. Making a sibling repository's red CI block
  this site's deploy would hand every sibling a veto over the canon, which ADR-2006 forecloses.
- **The snapshot is a bot data commit under the ordinary gates.** `estate-health[bot]` commits
  one file, and the dispatched deploy runs the same asset-inventory, build-output, link,
  meta-tag, structured-data, diagram and drift gates as a human commit. The bot has no
  privileged path to production.
- **Staleness is itself a finding.** If the workflow stops running, `check` goes `STALE` after
  36 hours and the dream night sees it — the failure mode of a monitor that quietly dies is
  covered by the monitor's own reader, not by trust.
- **Cost.** One more nightly workflow (~5 API calls × 14 repositories), one more required
  asset in `website/assets.manifest.json`, and a roster that must be edited when the estate
  changes. Roster drift is a standing hypothesis surface for the `estate-health` slot.
- **Relation to the neighbouring records.** ADR-2006 says the canon owns the cross-repo view
  and may not assert substrate implementation truth; this record keeps inside that boundary by
  *reporting observed signals* (CI conclusion, release tag, HTTP status) rather than judging
  substrate maturity. ADR-2007's roadmap asked for system evidence over declared status; a
  nightly collected snapshot is that evidence for the estate-surface half of it.

## Verification

At the landing commit (`verified_commit` above, set by the deploy agent):

- `node scripts/estate-health.mjs check` reads the committed
  `website/static/data/estate-health.json` with no network access and prints a single verdict
  line; exit 0 on `ESTATE-HEALTH-OK`, 1 on `ESTATE-HEALTH-STALE` / `ESTATE-HEALTH-RED`. This
  is the exact command `dream.config.json` binds as the `estate-health` evaluator, so the
  dream night's read path is the one tested here.
- `gh workflow run estate-health.yml --ref main`, then `gh run watch`: the run collects a
  snapshot for all fourteen roster repositories, records the verdict in the job summary,
  commits `website/static/data/estate-health.json` as `estate-health[bot]` when it changed,
  and dispatches `deploy.yml`.
- `bash tests/gates/run-all.sh` passes with the new required asset in
  `website/assets.manifest.json` — `website-assets.test.sh` proves the asset gate still goes
  red when the snapshot is missing from `website/dist`.
- The live section: `https://www.visionflow.info/#estate` renders the summary strip, the
  fourteen-repository table, the surfaces and the registry versions from
  `https://www.visionflow.info/data/estate-health.json`, whose `generated_at` matches the
  committed snapshot.
- `node scripts/adr-index-gen.cjs docs/adr` validates this record's frontmatter and regenerates
  the index.

Not verified by the above: that every repository's CI state is *correct* — the collector
reports GitHub's conclusion for the latest run per workflow on the default branch and nothing
more; and that the private `visionGraph` entry is readable, which requires
`ESTATE_READ_TOKEN` to be present as a repository secret.
