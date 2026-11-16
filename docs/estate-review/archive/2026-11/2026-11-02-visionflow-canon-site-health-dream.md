---
title: VisionFlow canon, site, estate health and dream cycle
date: 2026-11-02
status: assessed
type: explanation
---

# VisionFlow canon, site, estate health and dream cycle

All four VisionFlow topics were stamped at `4d361b5`, 133 commits behind the pinned `5f8b1da`. Every one needed a restamp, because cited sources had moved or been regenerated. All four are now restamped to `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083`, and every one of their citations was re-read at that revision.

I assessed 41 panels:

| Disposition | Panels |
|---|---|
| Accurate | 17 |
| Corrected | 22 |
| Accurate with limits | 2 |
| Inaccurate and left unfixed | 0 |

The most important findings are in the ledger tooling. VF-01 described an ADR gate that no longer exists:

- `adr-index-gen.cjs` gained an ADR-2007 evidence-field check, which moved every cited line.
- `adr-ratchet.sh` gained a second rule, an evidence gate that never expires.
- The ratchet's original backlog rule silently expired on 2026-10-20. Since then it only reports.

Separately, VF-04.4 cited the wrong lines for all three estate-health sentinels, and VF-02's `build-receipt.json` citations could never verify at any revision.

## Findings

**F1. The ratchet's backlog rule expired and nothing says so.**
- Trigger: `ADR_RATCHET_UNTIL` defaults to `2026-10-20` (`scripts/adr-ratchet.sh:44`), and `adr-index.yml` never overrides it (`.github/workflows/adr-index.yml:47-49`).
- Consequence: since 2026-10-21 a pull request that adds undecided ADRs passes with only a report (`scripts/adr-ratchet.sh:118-121`). The workflow step is still titled "a new proposal must close a proposed one" (`.github/workflows/adr-index.yml:46`). ADR-2021 landed `proposed` on 2026-10-30 under the reported-only rule. Only the new ADR-2007 evidence gate (`scripts/adr-ratchet.sh:80-112`) still blocks.
- To close: renew or remove the date deliberately, and rename the step to match. Recorded as **Debt** in VF-01.8.

**F2. ADR-2005 contradicts itself.**
- Trigger: frontmatter and index read `accepted / complete / live` (`docs/adr/ADR-2005-drift-counter-allowlist-substrate-sourced.md:6`, `docs/adr/README.md:37`), after the 2026-09-05 pass moved it to complete (`docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:58-59`).
- Consequence: the Consequences text still says implementation is partial because two of four axes are unpoliced (`docs/adr/ADR-2005-drift-counter-allowlist-substrate-sourced.md:46-47`). VF-01.4 and VF-01.10 had repeated the stale "partial".
- To close: amend the paragraph, or reopen the status. Recorded as **Drift** under VF-01.4.

**F3. The `domain` field is not universal.**
- Trigger: VF-01.10's invariant said every record names `BASELINE-visionflow.md` as its `domain`. ADR-2001, ADR-2007 and ADR-2020 have no `domain` key (`docs/adr/ADR-2001-corpus-consolidation.md:1-14`, `docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:1-16`, `docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:1-16`).
- Consequence: the generator never checks the field, so nothing flags the gap.
- To close: add the field, or make the generator require it. The invariant is now a **Drift** marker.

**F4. BASELINE subsections lag the ledger.**
- VF-01.12's invariant permits per-subsection staging "provided every subsection is re-recorded". Four are not:
  - ADR-2010 and ADR-2011 are given as partial/staged and none/inactive (`docs/BASELINE-visionflow.md:229-233`), against complete/staged in the index (`docs/adr/README.md:42-43`).
  - ADR-2014 to ADR-2016 are described as proposed and inactive (`docs/BASELINE-visionflow.md:368`), against ADR-2014 and ADR-2016 accepted/partial/staged (`docs/adr/README.md:46`, `docs/adr/README.md:48`).
  - ADR-2019 is described as inactive (`docs/BASELINE-visionflow.md:391`), against live (`docs/adr/README.md:51`).
  - ADR-2007 is described as a staged proposal (`docs/BASELINE-visionflow.md:404-413`), against complete/live (`docs/adr/README.md:39`).
- To close: re-record those subsections. Added as **Drift** after VF-01.12.

**F5. The roadmap names the wrong owner of sequencing.**
- Trigger: `docs/roadmap.md:11` says ADR-2007 owns estate-wide sequencing. ADR-2007 records that role as overtaken (`docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md:26`), and the roadmap now leads with ADR-2021's proposed business roadmap (`docs/roadmap.md:3`).
- Added as **Drift** after VF-01.2.

**F6. VF-01.6 claimed no supersession edge exists.**
- Trigger: the panel's DIVERGENCE said every record has empty `supersedes`/`superseded_by`. ADR-2020 supersedes ADR-2004, and both sides are recorded (`docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:8`, `docs/adr/README.md:36`, `docs/adr/README.md:52`).
- This was already wrong at `4d361b5`. Corrected in the panel.

**F7. VF-04.4 cited the wrong estate-health sentinel lines.**
- Trigger: the panel cited OK at `:1002` and RED/STALE at `:991`/`:998`. Those lines are the INFO snapshot-age comment and computation. The sentinels are at `scripts/estate-health.mjs:1008`, `:1014` and `:1018`, as VF-03.6 already had.
- `estate-health.mjs` is byte-identical between the two stamps, so this was wrong when written. The citation gate passed it because it checks that a line exists, not what it says. Corrected.

**F8. The meta-tag gate's required set had changed.**
- Trigger: VF-04.4 said og and twitter tags are optional. Since the 10-14 ACCEPT, `og:image` and `twitter:card` are required (`scripts/dream-meta-tags-scan.sh:28-29`).
- Corrected.

**F9. Generated source: `website/build-receipt.json`.**
- Trigger: VF-02 listed this gitignored build output (`website/.gitignore:8`) in `sources:` and cited four of its lines. Its figures (51 files, 60,241,361 bytes) described one local build.
- Consequence: the strict gate failed on the topic in a clean worktree.
- Change: the receipt's fields are now cited where they are written (`scripts/website-assets.mjs:202-240`), the one-build figures are dropped, the file is removed from `sources:` and `website/.gitignore` is added. Recorded as **Drift** in VF-02.
- **Lead action required:** the four VF-02 `website/build-receipt.json` entries in `docs/diagrams/citation-allowlist.json` (`allow[0]`–`allow[3]`) now match nothing. A full-tree strict run fails on stale entries; `--only` skips that check (`scripts/diagram-index-gen.cjs:616-626`). That file is outside my assignment. The four restamps also make `docs/diagrams/VERIFICATION.md` stale for these topics, so `--check-verification` fails until it is regenerated.

**F10. Gate suites, counted wrong.**
- Trigger: VF-02.9 said seven suites. `run-all.sh` runs twelve (`tests/gates/run-all.sh:18-31`); nine were present even at the old stamp.
- The runner and sentinel lines also moved, to `:37` and `:44`. Corrected.

**F11. Collector concurrency is per pool, not estate-wide.**
- Trigger: `CONCURRENCY = 4` is commented "≤ 4 in flight, estate-wide" (`scripts/estate-health.mjs:158`). VF-03.2 repeated that.
- Consequence: three pools run together (`scripts/estate-health.mjs:902-905`), each bounded only by itself (`scripts/estate-health.mjs:229-238`), and each repo worker fires five calls at once (`scripts/estate-health.mjs:532`). The source allows about 28 requests in flight. This is a source reading; no rate-limit failure was observed.
- Corrected, and recorded as **Debt**.

**F12. The estate-health snapshot was regenerated.**
- Trigger: VF-03.11 drew the 2026-10-30 snapshot. At the pin it is 2026-11-01, collected at `0bd6897`: 12 green, 3 red, 1 none, 1 exempt, 1 unreadable (`website/static/data/estate-health.json:3-19`).
- knowledgeGraph moved from exempt to **red**, which shows the rule that a red on HEAD is never exempted (`scripts/estate-health.mjs:112`, `scripts/estate-health.mjs:559`). nostr-rust-forum moved to green.
- Panel and Debt paragraph rewritten.

**F13. Site and record copy is stale.**
- The page dates the augmentation grade to 14 September and calls control and recovery *partial* (`website/static/index.html:580`). The graded table says C2/C3 are `measured` everywhere (`docs/architecture/compatibility-matrix.md:31-32`).
- ADR-2003 still justifies the sentinel greps by "scripts exit 0 even when they fail" (`docs/adr/ADR-2003-pages-artifact-deploy.md:85`). That has been untrue for the build and link scripts since `4159e08`.
- The repository map says seven sidestr crates (`docs/architecture/repository-map.md:18`); the roster probes eight.
- Each is now a **Drift** marker.

**F14. The ledger schema note is half stale.**
- `docs/dream-cycle/LEDGER.md:1` calls the FR6.6 reviewer-column writer at `19463a588` unmerged. In agentbox at the pinned `c57d9c0`, that commit is an ancestor, and `management-api/lib/dream-ledger.js:54-55` declares the columns.
- Deployment is not established by source. Added as **Drift** in VF-04.

**Smaller citation repairs:**
- `index.html` withdrawal text is at `:631`, not `:630`, and the pill is at `:1126`.
- The sidestr site tile no longer states a crate count (`website/static/index.html:1217-1220`).
- `assets.manifest.json:15` is the estate-health entry, not `:13`.
- The C3 row is `compatibility-matrix.md:32`; the longitudinal and re-grade text is at `:26`/`:41`.
- `dream.config.json:78-80` holds the labels.
- `dream-link-check.sh:71` is the MISSING echo.
- `site.spec.js:4-15` lists the sections.
- `deploy.yml:59-68` is the pinned agentbox checkout.
- The README is 163 lines.
- Topic area ranges (VC-01..37, AB-01..30, ES-01..10, OM-00..06) are stale and replaced with `nn`.

## Panel dispositions

| Panel | Disposition | Evidence |
|---|---|---|
| VF-01.2 | corrected | 21 records (`docs/adr/README.md:29`); PRD/DDD rows `docs/README.md:33-35`; roadmap `:11`/`:3`; engineering cut `BASELINE-visionflow.md:272-273`; no `docs/governance/` in tree; `docs/business/` added |
| VF-01.3 | accurate | `PREAMBLE.md:10,12,23`, `BASELINE-visionflow.md:50-51`, `archive/adr/README.md:5`, `adr-status-contract.md:8` |
| VF-01.4 | corrected | enums `adr-index-gen.cjs:43-45`, hard error `:134`; ADR-2005 complete vs body `:46-47`; ADR-2007 `:5-7`, `:14-15` |
| VF-01.6 | corrected | every generator line re-derived (`:28`–`:218`); evidence gate `:39-40`, `:139-150`; ADR-2020↔ADR-2004 edge |
| VF-01.8 | corrected | ratchet `:44,46,64,69,73-75,92-94,109-125`; evidence gate never expires `:31`; workflow lines unchanged |
| VF-01.10 | corrected | ADR-2005 and ADR-2007 now complete/live; ADR-2007 decision at `:26`; `domain` missing on three records |
| VF-01.11 | corrected | uncited area ranges stale; cited lines (`ADR-2006:35,41`, `ADR-2008:76`, `BASELINE:315-321`) accurate |
| VF-01.12 | corrected | README 163 lines; new Drift on subsection lag; other cites accurate |
| VF-01.13 | corrected | `compatibility-matrix.md:31-32`, `:26`, `:41`; remaining ADR-2010/2011/terminology/registry cites accurate |
| VF-01.14 | corrected | `index.html:631`, `:1125-1126`; tile `:1217-1220`; ADR-2012 and registry cites accurate |
| VF-01.15 | accurate-with-limits | ADR-2013 and BASELINE Invariants 8–9 cites hold; the deletions and single-parser claims are sibling-repo facts not checked from this source |
| VF-02.1 | accurate | `build.sh:2,7,17,21,24,27,30,33,40` |
| VF-02.2 | corrected | one-build figures dropped; receipt fields `website-assets.mjs:227-232`; index.html 111,976 B, nostr-contact.js 57,473 B |
| VF-02.3 | accurate | manifest has 17 required, 1 derived, 4 optional groups; `website-assets.mjs:9,129,147,250` |
| VF-02.4 | accurate | `website-assets.mjs:64,78,136,159,177,197,242,252,260,268,271,278` |
| VF-02.5 | corrected | receipt fields `website-assets.mjs:203,205-210,211-226,227-232,235,239-240`; `deploy.yml:155,164,177` |
| VF-02.6 | accurate | `deploy.yml:34-197`; the Jev advisory on `index.html:19` is a false positive (`og:locale en_GB`) |
| VF-02.7 | accurate | `deploy.yml:102,104`; scripts `:12-13`, `:155-156`; test `:36-65`; new Drift on ADR-2003 `:85` |
| VF-02.8 | corrected | pinned ref `deploy.yml:59-68` matches `allowlist.json` `source_pin`; reported path `:138-142` |
| VF-02.9 | corrected | twelve suites `run-all.sh:18-31`, node runner `:37`, sentinel `:44` |
| VF-02.10 | corrected | `site.spec.js:4-15`, `:31-33`; config and fixture cites accurate |
| VF-02.11 | accurate | `website-browser-check.mjs:6,16,18,20,24,48,54`; no call from `deploy.yml` |
| VF-02.12 | accurate | `main.js:908-923`, `mesh-webgl.js:15-16`; Drift on page copy extended |
| VF-03.1 | corrected | 18 repos, 6 surfaces, 22 registries; map `:29-38` includes non-roster rows; area ranges |
| VF-03.2 | corrected | concurrency `estate-health.mjs:902-905`, `:532`; workflow cites accurate. Limit: GITHUB_TOKEN push and dispatch behaviour is GitHub runtime |
| VF-03.3 | accurate | `estate-health.mjs:93,186,312,317,362,379,468,479,518,532,566,588,607,609,622,652,680` |
| VF-03.4 | accurate | `estate-health.mjs:112,166,168,171,357,378,382,418,442,551-560,609,614`; test `:27-32` |
| VF-03.5 | accurate-with-limits | rules `:30,37,42,66,250,262`; snapshot `:10,:16`. The 404 for the cross-owner repo is runtime, confirmed only by the committed snapshot |
| VF-03.6 | accurate | `estate-health.mjs:85,212,942-1018` |
| VF-03.8 | accurate | `estate-health.mjs:148,708,721,738,749,767,788,815` |
| VF-03.9 | corrected | manifest entry `assets.manifest.json:15`; `main.js:587-919`, `index.html:1156,1160` accurate |
| VF-03.11 | corrected | rewritten for the 2026-11-01 snapshot, `estate-health.json:2-19,123,157,178,196,217,371,410,480,491,639,647,658,662` |
| VF-04.1 | accurate | `dream.config.json:2-82`, `README.md:114,139` |
| VF-04.2 | accurate | slots `:6,13,20,27`; the ADR-2008 `:44` doc-drift is still present |
| VF-04.3 | accurate | `dream.config.json:46-58`; `LEDGER.md:8,21`; `dream-asset-refs-scan.sh:7` |
| VF-04.4 | corrected | sentinels `estate-health.mjs:1008,1014,1018`, early STALE `:943-961`; meta required `:24-29`; MISSING `dream-link-check.sh:71` |
| VF-04.5 | corrected | labels `dream.config.json:78-80`. Limit: the draft PR and ledger append are external-engine behaviour |
| VF-04.6 | accurate | `dream.config.json:68-74`; `build.sh:7`; `BASELINE:322` |
| VF-04.8 | corrected | verdict counts re-derived (24 ACCEPT, 11 INCONCLUSIVE, 3 BLOCKED-ENV, 4 REJECT, 5 OPERATOR); schema note drift |
| VF-04.9 | corrected | labels `dream.config.json:78-80`; the rest accurate |
| VF-04.10 | accurate | `dream.config.json:47-59,74`; `deploy.yml:83-138`; `ADR-2009:84` |

## Verification

**Evidence class:** source tracing at the pinned revision only. Every citation in the four topics, Mermaid and prose, was dumped with its text at `4d361b5` and at `5f8b1da`, and read against its claim. Moved lines were re-derived with `git grep -n` and `git show <sha>:<path>`. Prose register citations, which the gate does not resolve, were re-read separately after the final edits. One is deliberately cross-repository: agentbox `management-api/lib/dream-ledger.js:54-55` at `c57d9c0`, under F14. No product tests, CI runs, browser checks or deployments were run or observed.

**Pinned revisions:**
- VisionFlow: `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083`, confirmed in `snapshot-wave2-start.json`.
- agentbox: `c57d9c0508f4`, read for F14 only.

**Gate commands**, run from the worktree root with output captured to a log before the exit code was read:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 01-repo-composition-and-canon.md --check --cite-check --strict-citations --render   # exit 0, 207 citations verified, 11/11 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 02-website-build-and-deploy.md --check --cite-check --strict-citations --render     # exit 0, 133 verified, 0 allowlisted, 12/12 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 03-estate-health.md --check --cite-check --strict-citations --render               # exit 0, 152 verified, 9/9 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 04-dream-cycle-integration.md --check --cite-check --strict-citations --render     # exit 0, 143 verified, 9/9 rendered
```

**Gate history:**
- Before the edits, VF-02 exited 1 on `source path does not exist: website/build-receipt.json`.
- Jev advisories were checked by hand. One was a false positive (`index.html:19`). The others (`website-assets.mjs:211`, `dream.config.json:78`, `dream-link-check.sh:64`, and `estate-health.mjs:991` from the pre-edit run) were real imprecisions and are fixed.

**Not verified:**
- Deployed site content.
- Live estate-health collection.
- Whether the dream engine honours the draft-PR, branch and label contract.
- Whether agentbox's ledger writer is deployed.
- Sibling-repository facts that VF-01.13–VF-01.15 route to.
- The full-tree strict run, which will fail until the lead removes the four stale allowlist entries (F9).
