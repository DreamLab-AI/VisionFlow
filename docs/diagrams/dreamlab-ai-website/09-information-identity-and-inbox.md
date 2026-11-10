---
id: DW-09
title: Website information identity — disclosure, monitoring and digest delivery
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/adr/ADR-2011-website-information-identity.md
adrs: [ADR-2011]
sources:
  - ../dreamlab-ai-website/docs/adr/ADR-2011-website-information-identity.md
  - ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs
  - ../dreamlab-ai-website/scripts/lib/website-info.mjs
  - ../dreamlab-ai-website/scripts/inbox-watch/run.mjs
  - ../dreamlab-ai-website/scripts/lib/digest.mjs
  - ../dreamlab-ai-website/scripts/lib/enquiry-topic.mjs
  - ../dreamlab-ai-website/scripts/lib/jev.mjs
  - ../dreamlab-ai-website/scripts/seo-watch/crontab
  - ../dreamlab-ai-website/src/lib/nostr.ts
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/governance_api.rs
verified_commit: {dreamlab-ai-website: f934e26f949f3b95171544ec8ec5bcfd7679f593, nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

**Repair follow-up, 2026-11-03:** [DW-11](11-watch-publication-and-inbox-repairs.md)
records registry-presence, privilege-health and seen/display accounting repairs.
The source stamp below remains historical and unrelated findings remain open.

Separate configured admission, public disclosure, stored registration and delivery acknowledgement. The dated estate audit exercises the actual watcher and provisioner with synthetic encrypted messages and fully mocked transport.

## For the business

The monitor can summarise received enquiry copies without joining private forum zones. Its current health checks and bounded digest do not establish complete enquiry capture, durable registration or that every sender's message was shown to the operator.

## DW-09.1 Configuration disclosure is not a registry receipt

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:105
    A["Read active registry rows"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:70
    A --> B["Merge configured ingress identities<br/>that name an authorising principal"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:78
    B --> C["Active registry row wins for same key and is marked registered<br/>otherwise configuration fills the gap, marked not registered"]
    %% Source: ../dreamlab-ai-website/scripts/lib/website-info.mjs:83
    C --> D["Website classifier separates listed from registered<br/>registered is true only for a registry row"]
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs:97
    D --> E["DRIFT: RESOLVED at dreamlab-ai-website f934e26<br/>provisioner skips registration only when a registry row exists"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:233
    D --> F["DRIFT: RESOLVED at dreamlab-ai-website f934e26<br/>watcher reads the registered flag, so a missing registry row is a problem"]
```

**What it shows:** Public naming still survives a missing database row, but the relay now marks it `registered: false` and both consumers require `registered: true`.
**Why it is this way:** ADR-2011 adopted configuration-backed ingress, and the earlier registry-presence check was repaired by adding the registered flag to the shared response.

## DW-09.2 Provisioning privilege guard versus nightly health

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs:51
    A["Provisioner checks privilege<br/>before publishing kind-0 metadata"]
    %% Source: ../dreamlab-ai-website/scripts/lib/website-info.mjs:100
    A --> B["Require admission and reject<br/>admin flag or nonempty cohort array"]
    %% Source: ../dreamlab-ai-website/scripts/seed/provision-website-info.mjs:115
    B --> C["Repeat privilege check after adoption"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:239
    D["Nightly checks admin, agent and monitor<br/>through checkWhitelist"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:245
    D --> E["Require effective ingress declaration<br/>do not call privilegeProblem"]
    %% Source: ../dreamlab-ai-website/scripts/lib/website-info.mjs:120
    E --> F["DEBT: privileged monitor can pass<br/>nightly admission health"]
```

**What it shows:** The setup guard and recurring health predicate are different. A healthy nightly receipt does not certify the monitor's lack of privilege.
**Why it is this way:** ADR-2011 requires a labelled identity with no cohorts or admin rights; the current watcher verifies routing admission rather than that full invariant.

## DW-09.3 Enquiry copies and the scope of the probe

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/src/lib/nostr.ts:342
    A["Primary submission accepted by relay"]
    %% Source: ../dreamlab-ai-website/src/lib/nostr.ts:352
    A --> B["Best-effort separate monitor wrap<br/>missing/equal key is a no-op"]
    %% Source: ../dreamlab-ai-website/src/lib/nostr.ts:363
    B --> C["Copy failure does not change<br/>visitor-facing primary result"]
    %% Source: ../dreamlab-ai-website/scripts/seo-watch/crontab:17
    D["07:15 UTC watcher, flock<br/>300-second timeout"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:299
    D --> E["Publish synthetic wrap to monitor<br/>also performed in dry-run mode"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:316
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:187
    E --> F["Read monitor inbox and verify<br/>seal signature and rumor author binding"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:255
    D --> G["Bundle substring checks for keys<br/>are not a browser form-submission test"]
```

**What it shows:** The synthetic probe tests the monitor path, while actual visitor copies remain best effort. A dry run suppresses digest delivery and state writes, not the probe.
**Why it is this way:** The relay isolates each recipient's inbox, so ADR-2011 relies on extra encrypted copies rather than reading the admin's copy.

## DW-09.4 Pagination, classification and digest retirement

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:49
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:137
    A["Three-day window<br/>500 wraps per page, at most ten pages"]
    %% Source: ../dreamlab-ai-website/scripts/lib/digest.mjs:81
    A --> B["DEBT: next cursor is oldest second minus one<br/>unseen ties at the boundary can be skipped"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:334
    B --> C["Mark only probe wraps seen at classification<br/>other IDs wait for the digest"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:200
    C --> D["DEBT: any content containing probe phrase<br/>is omitted as a synthetic probe"]
    %% Source: ../dreamlab-ai-website/scripts/lib/digest.mjs:25
    C --> E["Sanitised digest itemises first 25<br/>remaining count is a summary line"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:377
    E --> F["Relay acknowledgement permits state save<br/>rejected digest leaves seen state unsaved"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:387
    F --> G["DEBT: RESOLVED at dreamlab-ai-website f934e26<br/>only itemised IDs are marked seen, the rest roll into the next digest"]
```

**What it shows:** Bounds prevent unbounded processing, and messages past the 25-item cap stay unseen so the next digest carries them.
**Why it is this way:** The watcher deduplicates received wraps and bounds one digest; its state now marks only itemised messages seen after an acknowledged delivery.

## DW-09.5 Plaintext topic annotation and retained evidence

```mermaid
flowchart TB
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:214
    A["Decrypted enquiry text<br/>up to 1500 characters for topic annotation"]
    %% Source: ../dreamlab-ai-website/scripts/lib/enquiry-topic.mjs:86
    A --> B["Optional DREAMLAB_JEV switch<br/>off by default"]
    %% Source: ../dreamlab-ai-website/scripts/lib/enquiry-topic.mjs:30
    B --> C["Local-only restriction applies only<br/>when TOPIC_LOCAL_ONLY is set to 1"]
    %% Source: ../dreamlab-ai-website/scripts/lib/jev.mjs:38
    C --> D["Explicit endpoint, then configured proxy"]
    %% Source: ../dreamlab-ai-website/scripts/lib/jev.mjs:59
    D --> E["Otherwise vendor endpoint with API key<br/>plaintext enquiry is request state"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:369
    A --> F["Dry-run prints composed digest<br/>including enquiry excerpts"]
    %% Source: ../dreamlab-ai-website/scripts/inbox-watch/run.mjs:345
    A --> G["Normal status retains counts and problems<br/>seen state is not a complete message archive"]
```

**What it shows:** Recipient encryption ends where the monitor decrypts. Optional annotation and dry-run output have their own plaintext destinations.
**Why it is this way:** The topic feature annotates operator mail without changing delivery; its executable default is broader than the module's old local-only header comment.
