---
id: DW-08
title: Nightly SEO watch — receipts, advice and proposal boundaries
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/adr/ADR-2010-seo-watch-nightly.md
adrs: [ADR-2010]
sources:
  - ../dreamlab-ai-website/docs/adr/ADR-2010-seo-watch-nightly.md
  - ../dreamlab-ai-website/scripts/seo-watch/crontab
  - ../dreamlab-ai-website/scripts/seo-watch/config.json
  - ../dreamlab-ai-website/scripts/seo-watch/run.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/advise.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/glm.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/safety.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/lib/presence.mjs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/agentbox.toml
verified_commit: {dreamlab-ai-website: b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5, agentbox: 9dfc27b16e82e9786535a55e790710493dd51fdf}
---

## For developers

**Repair follow-up, 2026-11-03:** [DW-11](11-watch-publication-and-inbox-repairs.md)
records the publisher, line-cap and error-scrubbing repairs at their new source
revision. This topic retains the historical observations and anchors below.

Follow the mounted cron through deterministic receipts, optional model advice and the publication gate. The dated estate review reproduces the publisher, accounting, policy and evidence gaps with local fixtures.

## For the business

The running watch gives the site owner nightly discovery checks. Its clean audit rows do not prove search ranking, full-page coverage or a working route from suggested changes to draft pull requests.

## DW-08.1 Supervisor, schedule and state

```mermaid
flowchart TB
    %% Source: ../project/agentbox/flake.nix:308
    A["Manifest seo_watch.enabled<br/>controls baked supervisor stanza"]
    %% Source: ../project/agentbox/flake.nix:3083
    A --> B["supercronic reloads mounted website crontab<br/>runs as devuser with inherited environment"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/crontab:10
    B --> C["05:40 UTC, nonblocking flock<br/>5400-second whole-run timeout"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:141
    C --> D["Audit live pages, then optional advice<br/>and proposal processing"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:35
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:240
    D --> E["Reports, state and ledger outside repo<br/>default workspace/.agentbox/seo-watch"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/crontab:17
    B --> F["Same crontab also runs inbox watch<br/>07:15 UTC, separate lock and state"]
```

**What it shows:** The scheduler is baked while the job and schedule are mounted source. The separate inbox job shares the scheduler switch.
**Why it is this way:** ADR-2010 places this watch alongside the dream engine; overlap locking is not a once-per-date model-call limit.

## DW-08.2 Fetching and deterministic receipts

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:134
    A["Fetch robots, sitemap, llms files<br/>and security.txt concurrently"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:171
    A --> B["Fixed marketing pages plus<br/>least-recently-audited sitemap remainder"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:73
    B --> C["Status, metadata, canonical, JSON-LD<br/>headings, text and alternate checks"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:221
    A --> D["DEBT: robots selects first matching group<br/>repeated groups are not combined"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:154
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:271
    A --> E["DEBT: llms/sitemap origin uses string prefix<br/>lookalike hostname can pass and be fetched"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:43
    A --> F["DRIFT: response-body failure escapes<br/>and can abort the whole audit"]
```

**What it shows:** These are local fetch and content predicates, with narrower RFC and failure handling than the decision claims.
**Why it is this way:** ADR-2010 asks for deterministic live-site evidence before advice; the code separates transport from pure checks but leaves these boundaries open.

## DW-08.3 User-agent, browser and Lighthouse coverage

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:187
    A["First four fixed roster pages<br/>form the configured parity set"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:192
    A --> B["Only successful baseline pages<br/>are fetched as eight crawler agents"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:301
    B --> C["Compare status and main-text hash"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:227
    B --> D["DEBT: report counts planned pages<br/>even when baselines prevent all comparisons"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/audit.mjs:207
    A --> E["Browser visits parity set<br/>Lighthouse rotates three within that set"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:309
    E --> F["Hydration checks heading/title differences<br/>and text-length ratios, not content equality"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/checks.mjs:337
    E --> G["DRIFT: regression uses consecutive readings<br/>ignores dates and includes a five-point drop"]
```

**What it shows:** Raw-agent equality and hydration similarity are different tests. A zero parity finding count does not establish that planned comparisons ran.
**Why it is this way:** ADR-2010 limits nightly browser work, but its consecutive-night language is stronger than the stored-reading comparison.

## DW-08.4 Advice triggers and cost accounting

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:63
    A["Read ledger<br/>DEBT: any parse failure returns empty history"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/advise.mjs:17
    A --> B["Stop if recorded month total is spent<br/>else change, regression, weekly or force trigger"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:174
    B --> C["Fetch origin/main source context<br/>allowlisted and read-only reference files"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:184
    C --> D["GLM call, retry empty reply<br/>with thinking disabled"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:186
    D --> E["DRIFT: record only successful response usage<br/>first empty reply usage is lost"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/advise.mjs:21
    B --> F["DEBT: no persisted per-date reservation<br/>weekly rerun can call again"]
```

**What it shows:** The token budget is a pre-call history check; retries and reruns weaken the one-call and monthly-cap claims.
**Why it is this way:** ADR-2010 trades quiet-night cost for optional advice, but the transport and ledger do not enforce a hard spend ceiling.

## DW-08.5 Patch verification and blocked publication

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:39
    A["Validate paths, headers, line count<br/>and file cooldown"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/advise.mjs:31
    A --> P["DRIFT: crawler policy restriction<br/>is a prompt rule, not a semantic patch gate"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:121
    A --> B["Detached origin/main worktree<br/>share installed dependencies if present"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:143
    B --> C["Git path/application checks<br/>build, lint and tests under cleared environment"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:193
    C --> D["Require exact applied diff<br/>and only expected changed paths"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:288
    D --> E["Create branch, stage patch and report<br/>commit with hooks disabled"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:298
    E --> F["DRIFT: inspect staged names AFTER commit<br/>normal empty index fails expected-file check"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:301
    F -.->|"not reached in local success-to-commit fixture"| G["Push, label and draft PR code<br/>human merge remains required"]
```

**What it shows:** Non-executable text and exact-tree checks bound the model patch, but the normal publisher path rejects after committing.
**Why it is this way:** ADR-2010 requires tested, human-reviewed proposals; the local fixture demonstrates a publication defect without contacting GitHub.

## DW-08.6 Reconciliation, receipts and outcome limits

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:237
    A["Reconcile labelled PRs and owned branches<br/>open PR gate and declined-file cooldown"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:267
    A --> B["DEBT: delete state record even<br/>when remote branch deletion fails"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:190
    C["Provider error retained in advice object"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:241
    C --> D["Report is scrubbed before writing"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:253
    C --> E["DRIFT: ledger serialises raw advice error<br/>error log is also unsanitised"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/presence.mjs:6
    F["Optional weekly Perplexity citation probe"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/presence.mjs:29
    F --> G["DEBT: failed calls count as not cited<br/>in the aggregate denominator"]
    %% Source: ../dreamlab-ai-website/docs/adr/ADR-2010-seo-watch-nightly.md:99
    G --> H["No Search Console or Bing index evidence<br/>citation fraction is not ranking or conversion"]
```

**What it shows:** Review state and retained receipts have failure cases distinct from the live audit. A synthetic provider-key echo reaches the ledger and stderr but not the report.
**Why it is this way:** ADR-2010 requires scrubbing and an outcome metric; the current error and aggregate paths only partly implement those obligations.
