---
id: DW-07
title: Forum provisioning, seeding, embeddings, backup and migration tooling
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
adrs: [ADR-2005]
sources:
  - ../dreamlab-ai-website/scripts/seed-forum.mjs
  - ../dreamlab-ai-website/scripts/seed-semantic.mjs
  - ../dreamlab-ai-website/scripts/assign-cohorts.mjs
  - ../dreamlab-ai-website/scripts/seed/seed-forum-zones.mjs
  - ../dreamlab-ai-website/scripts/seed/provision-carol-agent.mjs
  - ../dreamlab-ai-website/scripts/seed/provision-junkiejarvis-relay.mjs
  - ../dreamlab-ai-website/scripts/seed/whitelist-admin-recipient.mjs
  - ../dreamlab-ai-website/scripts/seed/probe42.mjs
  - ../dreamlab-ai-website/scripts/seed/probe-admin42.mjs
  - ../dreamlab-ai-website/scripts/seed/cleanup-qa-foreign.mjs
  - ../dreamlab-ai-website/scripts/seed/reseed-minimoonoir.mjs
  - ../dreamlab-ai-website/scripts/embeddings/batch-embedding-sync.ts
  - ../dreamlab-ai-website/scripts/embeddings/mcp-embedding-sync.ts
  - ../dreamlab-ai-website/scripts/embeddings/postgres-embedding-sync.ts
  - ../dreamlab-ai-website/scripts/backup/forum-backup.sh
  - ../dreamlab-ai-website/scripts/backup/crontab
  - ../dreamlab-ai-website/scripts/keybase-migration/server.mjs
  - ../dreamlab-ai-website/tests/rvf-validation/test-rvf-dreamlab.mjs
  - ../dreamlab-ai-website/tests/rvf-validation/test-search-api-e2e.mjs
  - ../dreamlab-ai-website/.gitignore
  - ../dreamlab-ai-website/README.md
  - ../dreamlab-ai-website/forum-config/deploy/search-worker.wrangler.toml
verified_commit: 9a3dd8830
---

## DW-07.1 Two generations of forum-provisioning scripts
```mermaid
flowchart TB
    ROOT["scripts/ root-level, 2026-03-01<br/>seed-forum.mjs:1-9, seed-semantic.mjs:1-7,<br/>assign-cohorts.mjs:1-9 — early prototype tier"]
    SEED["scripts/seed/, 31 files, 2026-06-10 onward<br/>seed-forum-zones.mjs:1-6 — current operational toolkit"]
    ROOT -->|"superseded by"| SEED
    SEED --> PROV["Provisioning: seed-forum-zones, provision-carol-agent[-key],<br/>provision-junkiejarvis-relay, provision-mirror-child,<br/>whitelist-admin-recipient, grant-agent-pod-read,<br/>add-section, seed-extra-sections, reseed-sections,<br/>reseed-minimoonoir"]
    SEED --> PROBE["Diagnostics: probe42, probe-admin42, probe-calendar,<br/>probe-etag, probe-family, probe-kind0,<br/>probe-panel-response, probe-users"]
    SEED --> QA["QA/dev: acs-live-test, nip07-shim-qa, browser-login,<br/>test-jj-calendar, test-junkiejarvis,<br/>test-website-chat-roundtrip, notif-followon,<br/>dm-carol-agent, list-channels, publish-foreign-reply,<br/>cleanup-qa-foreign, find-admin-key"]
```
- DIVERGENCE from the audit's count: `scripts/seed/` holds **31** files (verified `find scripts/seed -type f | wc -l`), not the 37 the wave-2 brief cites; the three root-level seeders (`seed-forum.mjs`, `seed-semantic.mjs`, `assign-cohorts.mjs`) bring the total provisioning-script count to **34**. None of the excess 3 the brief implies were found under any plausible additional path.
- The root-level tier and `scripts/seed/` are not two views of the same thing — they are chronologically distinct generations (git-log dates above) that happen to share a directory prefix; DW-07.2 details why this matters operationally.

## DW-07.2 Root-level seeders — a hardcoded key and a broken import path
```mermaid
flowchart TB
    KEY["ADMIN_PRIVKEY_HEX, a plaintext secp256k1 private key<br/>hardcoded in source"] --> F1["seed-forum.mjs:20"]
    KEY --> F2["seed-semantic.mjs:17"]
    KEY --> F3["assign-cohorts.mjs:17"]
    IMPORT["imports from an absolute path outside this repo:<br/>/home/devuser/workspace/project2/community-forum/node_modules/..."] --> F1
    IMPORT --> F2
    IMPORT --> F3
    F1 --> RESULT["none of the three can run in a fresh checkout;<br/>the referenced path does not exist in this repo or its siblings"]
```
- INVARIANT VIOLATION: `CLAUDE.md`'s own Security Rules state "NEVER hardcode API keys, secrets, or credentials in source files" (`CLAUDE.md` Security Rules section) — these three files do exactly that, unconditionally, not behind a `--dev`/env-gated fallback.
- Whether the embedded key is still live cannot be determined from this repo alone; it is not one of the pubkeys enumerated in `dreamlab.toml [admin]`/`[[agents]]` (DW-03.3), so it appears to be a standalone test identity rather than the operator's admin key — but a plaintext private key checked into git history is a credential-hygiene defect regardless of scope.
- Contrast with `scripts/seed/`'s current tier (DW-07.3): every script there reads its signing key from an external `.env` at request time, never embeds one.

## DW-07.3 `scripts/seed/` — keys sourced externally, not embedded
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator, node scripts/seed/<script>.mjs
    participant ENV as agentbox/.env<br/>EXTERNAL, see AB-*
    participant KEYS as scripts/seed/.test-keys.json<br/>gitignored, .gitignore:77
    participant RELAY as dreamlab-nostr-relay<br/>wss://…workers.dev
    OP->>ENV: readFileSync agentbox/.env<br/>provision-junkiejarvis-relay.mjs:4, probe-admin42.mjs:4
    ENV-->>OP: match JUNKIEJARVIS_PRIVKEY_HEX / AGENTBOX_PRIVKEY_HEX
    OP->>KEYS: readFileSync .test-keys.json<br/>provision-carol-agent.mjs:5, probe42.mjs:4
    KEYS-->>OP: per-role test privkeys (friends-carol, ...)
    OP->>RELAY: finalizeEvent + publish, NIP-98 admin auth where required
```
- `scripts/seed/probes/` is also gitignored (`.gitignore:78`) — ad hoc probe output/scratch state never lands in the tree.
- `whitelist-admin-recipient.mjs:1-6` documents its own no-op status as of 2026-07-14: the interim contact-DM recipient is already whitelisted, so the script is a safety check pending ADR-041 Decision 5's key-split cutover (cross-reference DW-03.4).
- `reseed-minimoonoir.mjs:1-6` records an in-repo migration note: the `friends`→`minimoonoir` zone-id rename orphaned `friends-*` channels (kind-40 section tags are immutable), requiring a delete-and-recreate rather than an in-place rename — direct evidence for the "Friends zone does not exist" DOC-DRIFT already flagged in DW-03.1.

## DW-07.4 `scripts/embeddings/*` — a different pipeline than README's semantic-search claim
```mermaid
flowchart LR
    README["README.md:194 'Semantic search' claim:<br/>Workers AI bge-small-en-v1.5 over R2 RVF"] -.->|"implemented by"| KITWORKER["search-worker [ai] binding<br/>search-worker.wrangler.toml:11-12, see DW-03.10<br/>upstream kit code, not in this repo"]
    BATCH["batch-embedding-sync.ts<br/>header: 'Batch Embedding Sync for RuVector PostgreSQL'"] --> MCP["claude-flow MCP embeddings_generate tool<br/>via direct library import"]
    PG["postgres-embedding-sync.ts"] --> RUVECTOR["RuVector Postgres — agentbox memory store<br/>EXTERNAL, unrelated repo concern"]
    RVFTEST["tests/rvf-validation/test-rvf-dreamlab.mjs:1-9<br/>'matching DreamLab's all-MiniLM-L6-v2 model output'"] --> RVFLIB["@ruvector/rvf Node SDK, local tmp files<br/>not Workers AI, not R2"]
```
- DOC-DRIFT: `README.md:194` markets semantic search as `bge-small-en-v1.5` embeddings over an R2-backed RVF store. Nothing in `scripts/embeddings/*` or `tests/rvf-validation/*` builds that pipeline: `batch-embedding-sync.ts` and `mcp-embedding-sync.ts` sync **RuVector Postgres memory entries** via the **claude-flow MCP** toolchain (agentbox's own memory system, not this site's search index), and the two `tests/rvf-validation/*` suites validate the `@ruvector/rvf` Node SDK against a **384-dim `all-MiniLM-L6-v2`** model over local JSON/tmp files — a different model and a different store than the README's `bge-small-en-v1.5`/R2 claim. The feature README:194 describes is real (search-worker's Workers AI binding, DW-03.10), but its build/index path is upstream-kit code this repo does not vendor; the embeddings tooling that does live in this repo is a separate, unrelated concern that happens to share the word "embedding".
- This resolves the audit's "index-build path has no home" gap: the path exists, just not for the feature the auditor was looking for it in service of.

## DW-07.5 Nightly Cloudflare-to-NAS backup
```mermaid
flowchart TB
    CRON["supercronic, agentbox [program:forum-backup-cron]<br/>crontab:6, 03:10 UTC nightly"] --> SCRIPT["forum-backup.sh<br/>forum-backup.sh:1-19"]
    SCRIPT --> AUTH["CLOUDFLARE_API_TOKEN / ACCOUNT_ID<br/>from env or agentbox/.env, forum-backup.sh:21-22<br/>EXTERNAL, see AB-*"]
    SCRIPT --> D1["D1 dreamlab-relay, dreamlab-auth<br/>critical, forum-backup.sh:26-29"]
    SCRIPT --> R2["R2 dreamlab-pods (critical), dreamlab-vectors (nice-to-have)<br/>forum-backup.sh:30"]
    SCRIPT --> KV["KV SESSIONS/POD_META/SEARCH_CONFIG<br/>low priority, rebuildable, forum-backup.sh:31-35"]
    D1 & R2 & KV --> DEST["DEST_ROOT, default /mnt/dell/shared/backups/dreamlab-forum<br/>forum-backup.sh:21, KEEP_NIGHTS=14"]
```
- INVARIANT: missing Cloudflare credentials exit the script with code 2 rather than silently succeeding, "so the cron surface shows the gap instead of silently succeeding" (`forum-backup.sh:14`).
- Content-sensitivity note in the script header: kind-1059 DMs are E2E encrypted at rest, but kind-40/42/0 events are signed plaintext, so "the NAS copy is readable content — keep it on the trusted segment only" (`forum-backup.sh:16-18`).
- This is the only backup/restore path for the forum found in this repo — there is no corresponding restore script, only the raw D1/R2/KV export.

## DW-07.6 `keybase-migration/` — a standalone local-only importer
```mermaid
flowchart TB
    OPERATOR["operator runs locally:<br/>cd scripts/keybase-migration && npm install && npm start<br/>server.mjs:3-4"] --> SRV["http.createServer, binds 127.0.0.1 only<br/>server.mjs:6-8,23"]
    SRV --> KB["lib/keybase.mjs — reads a Keybase paper key"]
    SRV --> CONV["lib/convert.mjs::buildPlan — Keybase to Nostr identity mapping"]
    SRV --> NOSTR["lib/nostr.mjs — publishes migrated identity to the relay"]
    SRV --> D1LIB["lib/d1.mjs — writes migration records to D1"]
    SRV --> WORK["work/ (gitignored)<br/>generated USER keys ARE persisted: work/keys.json<br/>server.mjs:9-10 'that is the deliverable'"]
```
- Security posture stated in its own header: the keybase paper key and the relay admin key "live in process memory for the duration of a request/job and are never logged or written to disk" (`server.mjs:6-8`); by contrast, generated per-user output keys are deliberately persisted to `work/keys.json` as the tool's actual deliverable, handed to the friend being migrated.
- This is a fully standalone Node app (own `package.json`/`package-lock.json`) nested inside the main repo's `scripts/` tree, not wired into any CI workflow or the main `npm` scripts (DW-02.3) — it is operator-run, out-of-band tooling.

## DW-07.7 `tests/rvf-validation/` — RVF library validation, not a forum test
```mermaid
flowchart LR
    T1["test-rvf-dreamlab.mjs:1-9<br/>413 lines, full lifecycle test"] --> LIB["@ruvector/rvf Node SDK<br/>384-dim, cosine metric"]
    T2["test-search-api-e2e.mjs:1-14<br/>258 lines"] --> SIM["simulates 500 embeddings,<br/>ingest to mock R2 (local JSON file),<br/>compares JSON brute-force vs RVF HNSW"]
    LIB --> DIM["DIM=384 test-rvf-dreamlab.mjs:19,<br/>METRIC=cosine test-rvf-dreamlab.mjs:20<br/>matches all-MiniLM-L6-v2, not bge-small-en-v1.5"]
```
- Both suites run as plain Node scripts (`node test-rvf-dreamlab.mjs`), not through Playwright or Vitest — they are not wired into `playwright.config.ts`'s `testDir` discovery (DW-05.5) or any `npm test` script (DW-02.3), and are absent from every CI workflow (DW-05.1-05.4). They validate the `@ruvector/rvf` library itself against a simulated workload, not this repo's live search-worker.
- Together with DW-07.4, these confirm the audit's "MISSING" finding was really two gaps in one: the *feature* README:194 describes has no in-repo build path (it's upstream-kit code), and the *tests* named after it validate unrelated tooling.
