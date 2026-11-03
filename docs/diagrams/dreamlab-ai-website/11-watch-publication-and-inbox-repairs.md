---
id: DW-11
title: Watch publication and inbox repair boundaries
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/adr/ADR-2010-seo-watch-nightly.md
  - ../dreamlab-ai-website/docs/adr/ADR-2011-website-information-identity.md
adrs: [ADR-2010, ADR-2011]
sources:
  - ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/run.mjs
  - ../dreamlab-ai-website/scripts/lib/website-info.mjs
  - ../dreamlab-ai-website/scripts/lib/digest.mjs
  - ../dreamlab-ai-website/scripts/inbox-watch/run.mjs
verified_commit: c3849abb8a2a289fbb013c721dfc0c5544c74b16
---

## For developers

This successor records the 3 November repairs to DW-08.5/08.6 and the registry, privilege and seen-accounting boundaries in DW-09. Historical panels remain evidence about their declared revisions; unrelated debt is not closed here.

## For the business

A verified website suggestion can reach the draft-publication step again. Inbox messages beyond the digest limit remain eligible for a later digest. Source tests do not establish delivery by the deployed nightly jobs.

## DW-11.1 Verify the committed change before publication

```mermaid
flowchart TD
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:153
    A["Git parses patch statistics"] --> B["Enforce changed-line limit"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:289
    B --> C["Apply and verify candidate; commit allowed files and report"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/lib/propose.mjs:312
    C --> D["Read committed HEAD paths with diff-tree"]
    D --> E{"Exact expected file set?"}
    E -->|Yes| F["Continue existing push and draft PR path"]
    E -->|No| G["Refuse publication"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/run.mjs:190
    H["Provider error"] --> I["Scrub before truncation, logging and ledger"]
```

**What it shows:** Publication inspects committed paths after commit. Git's parsed statistics independently enforce the line cap, including hunk content that resembles a diff header. Error sinks receive the scrubbed value.

**Why it is this way:** ADR-2010 requires bounded proposals with retained evidence. The regression uses a scratch repository and local bare remote with a fake GitHub CLI; it sends no real pull request. Other audit, cost and crawler-policy findings in DW-08 remain outside this repair.

## DW-11.2 Confirm registry state and account for displayed messages

```mermaid
flowchart TD
    %% Source: ../dreamlab-ai-website/scripts/lib/website-info.mjs:83
    A["Disclosure response"] --> B{"registered is explicitly true?"}
    B -->|No| C["Report missing registry entry; provisioning may register"]
    B -->|Yes| D["Registry presence confirmed"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:247
    D --> E["Check monitor identity has no extra privilege"]
    %% Source: ../dreamlab-ai-website/scripts/lib/digest.mjs:16
    F["Fresh decoded wraps"] --> G["Render at most 25 messages"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:387
    G --> H{"Digest delivery succeeded?"}
    H -->|Yes| I["Mark only displayed wrap IDs seen"]
    H -->|No| J["Keep messages eligible for retry"]
    I --> K["Overflow remains eligible for next digest"]
```

**What it shows:** Configuration disclosure alone does not prove a D1 registration. Seen state advances after successful delivery for the displayed subset.

**Why it is this way:** ADR-2011's low-privilege identity must survive database resets without confusing ingress configuration with registration. The additive forum `registered` field and this consumer need deployment together. Tests include a 30-message input split over successive digests; live encrypted delivery is separate.