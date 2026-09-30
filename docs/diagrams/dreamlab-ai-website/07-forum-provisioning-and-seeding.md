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
  - ../dreamlab-ai-website/scripts/keybase-migration/package.json
  - ../dreamlab-ai-website/tests/rvf-validation/test-rvf-dreamlab.mjs
  - ../dreamlab-ai-website/tests/rvf-validation/test-search-api-e2e.mjs
  - ../dreamlab-ai-website/.gitignore
  - ../dreamlab-ai-website/README.md
  - ../dreamlab-ai-website/forum-config/deploy/search-worker.wrangler.toml
verified_commit: 9b8ea495da80aaa5b45795af916bda4470467481
---

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
- Contrast with `scripts/seed/`'s current tier (DW-07.3): every script there reads its signing key from an external `.env` at request time, never embeds one — the two tiers are chronologically distinct generations, not two views of the same thing, and `scripts/seed/` holds **31** files (verified `find scripts/seed -type f | wc -l`), bringing the total provisioning-script count to **34**, not the 37 an earlier audit brief cited.

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
- `scripts/seed/probes/` is also gitignored (`.gitignore:78`) — ad hoc probe output/scratch state never lands in the tree. `whitelist-admin-recipient.mjs:1-6` documents its own no-op status as of 2026-07-14: the interim contact-DM recipient is already whitelisted, so the script is a safety check pending ADR-041 Decision 5's key-split cutover (cross-reference DW-03.4).
- `reseed-minimoonoir.mjs:1-6` records an in-repo migration note: the `friends`→`minimoonoir` zone-id rename orphaned `friends-*` channels (kind-40 section tags are immutable), requiring a delete-and-recreate rather than an in-place rename — direct evidence for the "Friends zone does not exist" DOC-DRIFT already flagged in DW-03.1.
- Contrast with the standalone `scripts/keybase-migration/` tool (`server.mjs:7-10`): the keybase paper key and relay admin key it handles live in process memory for a request/job's duration and are never logged or written to disk, but the generated per-user output keys it produces ARE deliberately persisted to `work/keys.json` (gitignored) — that persisted file is the tool's actual deliverable, handed to the friend being migrated.

## DW-07.4 `scripts/embeddings/*` — a different pipeline than README's semantic-search claim
```mermaid
flowchart LR
    README["README.md:203 'Semantic search' claim:<br/>Workers AI bge-small-en-v1.5 over R2 RVF"] -.->|"implemented by"| KITWORKER["search-worker [ai] binding<br/>search-worker.wrangler.toml:11-12, see DW-03.10<br/>upstream kit code, not in this repo"]
    BATCH["batch-embedding-sync.ts<br/>header: 'Batch Embedding Sync for RuVector PostgreSQL'"] --> MCP["claude-flow MCP embeddings_generate tool<br/>via direct library import"]
    PG["postgres-embedding-sync.ts"] --> RUVECTOR["RuVector Postgres — agentbox memory store<br/>EXTERNAL, unrelated repo concern"]
    RVFTEST["tests/rvf-validation/test-rvf-dreamlab.mjs:1-9<br/>'matching DreamLab's all-MiniLM-L6-v2 model output'"] --> RVFLIB["@ruvector/rvf Node SDK, local tmp files<br/>not Workers AI, not R2"]
```
- DOC-DRIFT: `README.md:203` markets semantic search as `bge-small-en-v1.5` embeddings over an R2-backed RVF store. Nothing in `scripts/embeddings/*` or `tests/rvf-validation/*` builds that pipeline: `batch-embedding-sync.ts` and `mcp-embedding-sync.ts` sync **RuVector Postgres memory entries** via the **claude-flow MCP** toolchain (agentbox's own memory system, not this site's search index), and the two `tests/rvf-validation/*` suites validate the `@ruvector/rvf` Node SDK against a **384-dim `all-MiniLM-L6-v2`** model over local JSON/tmp files — a different model and a different store than the README's `bge-small-en-v1.5`/R2 claim. The feature README:203 describes is real (search-worker's Workers AI binding, DW-03.10), but its build/index path is upstream-kit code this repo does not vendor; the embeddings tooling that does live in this repo is a separate, unrelated concern that happens to share the word "embedding".
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

## DW-07.7 `tests/rvf-validation/` — RVF library validation, not a forum test
```mermaid
flowchart LR
    T1["test-rvf-dreamlab.mjs:1-9<br/>413 lines, full lifecycle test"] --> LIB["@ruvector/rvf Node SDK<br/>384-dim, cosine metric"]
    T2["test-search-api-e2e.mjs:1-14<br/>258 lines"] --> SIM["simulates 500 embeddings,<br/>ingest to mock R2 (local JSON file),<br/>compares JSON brute-force vs RVF HNSW"]
    LIB --> DIM["DIM=384 test-rvf-dreamlab.mjs:19,<br/>METRIC=cosine test-rvf-dreamlab.mjs:20<br/>matches all-MiniLM-L6-v2, not bge-small-en-v1.5"]
```
- Both suites run as plain Node scripts (`node test-rvf-dreamlab.mjs`), not through Playwright or Vitest — they are not wired into `playwright.config.ts`'s `testDir` discovery (DW-05) or any `npm test` script (DW-02.3), and are absent from every CI workflow (DW-05.1-05.4). They validate the `@ruvector/rvf` library itself against a simulated workload, not this repo's live search-worker.
- Together with DW-07.4, these confirm the audit's "MISSING" finding was really two gaps in one: the *feature* README:203 describes has no in-repo build path (it's upstream-kit code), and the *tests* named after it validate unrelated tooling.
- `scripts/keybase-migration/` (own `package.json:1-4`) is likewise a fully standalone Node app nested inside the main repo's `scripts/` tree, not wired into any CI workflow or the main `npm` scripts (DW-02.3) — a local-only Keybase-to-Nostr identity importer (`http.createServer`, binds `127.0.0.1` only, `server.mjs:1-5`), operator-run and out-of-band, the same shape of untested/unintegrated tooling as the RVF suites above.
