---
title: DreamLab site and forum-config diagrams
date: 2026-11-02
status: assessed
type: explanation
---

# DreamLab site and forum-config diagrams

All 47 panels in topics DW-01 to DW-07 were assessed by source tracing at dreamlab-ai-website `0e157280` (and nostr-rust-forum `c39f41f1` for the kit citations in DW-04). Results: 24 accurate, 16 corrected, 7 accurate-with-limits, 0 inaccurate-unfixed. The most important finding is that the deploy gate does not enforce what the overlay's own Rust validator calls required: `REQUIRED_AUTH_SECRETS` lists four auth-worker secrets and its test is described as a fail-closed guard, but no workflow calls it, and the bash step that does gate the deploy requires only two of the four (DW-03.8, F1). Three other mechanism claims were wrong and are now fixed: which D1 database each worker binds (DW-03.10), the claim that ADR-2016 was never written when it is a kit ADR (DW-06.6, D15), and the claim that request/reply correlation in Talk-to-AI had not been built (DW-04.3). Topics 01 and 03–07 were edited and restamped. Topic 02 was not edited and keeps its stamp. Topics 08–10 were not touched.

## Findings

**F1. The auth-worker secret validator is test-only and disagrees with the deploy gate (Tension, DW-03.8, DW-04.6, DW-05.1).**
- *Trigger:* deploying the auth worker without `NATIVE_POD_URL` or `NATIVE_POD_ADMIN_KEY`.
- *Consequence:* the deploy succeeds with a warning. The Rust constant says both secrets are required and that `NATIVE_POD_URL` is "gated here so a fresh deploy without it fails fast" (`forum-config/src/deploy_config.rs:67-78`). The placeholder test checks that the placeholders are still present; it does not fail when they are (`deploy_config.rs:271-296`).
- *Evidence:* the gate is a bash step that requires `PRF_SERVER_SECRET` and `ADMIN_PUBKEYS` and only warns on the `NATIVE_POD_*` pair (`.github/workflows/workers-deploy.yml:301-337`). A grep for `validate_required_secrets` and `validate_deploy_dir` at the pinned revision finds them only in `deploy_config.rs` tests and in docs. The repo's own closeout already records this (`CLOSEOUT-SECURITY-AUDIT.md:148-152`).
- *Product-doc drift:* `set-worker-secrets.yml:8` says "All four are validated by the workers-deploy.yml pre-deploy gate".
- *To close:* either call the validator from the deploy job, or delete it and treat the bash step as canonical. Then align the header comment.

**F2. Wrong D1 binding claims (corrected, DW-03.10, DW-03.11).**
- *What the topic said:* the relay worker's `DB` and the other workers' `REPLAY_DB` were one physical database.
- *What the config says:* `dreamlab-auth` (id `e3981999…`) is bound as `DB` by the auth worker and as `REPLAY_DB` by the pod, search and relay workers (`auth-worker.wrangler.toml:14-17`, `pod-worker.wrangler.toml:35-38`, `search-worker.wrangler.toml:25-28`, `relay-worker.wrangler.toml:93-96`). The relay's own `DB` is `dreamlab-relay` (`relay-worker.wrangler.toml:74-77`), which the auth and pod workers bind as `RELAY_DB`.
- DW-03.11 also confused the `dreamlab-relay` D1 database with the `NostrRelayDO` Durable Object's storage. The relay D1 schema lives in kit code. This is now an **Open** item.
- A second **Open** item: `whitelist` is created in `dreamlab-auth` (`forum-config/deploy/migrations/001_init.sql:266`), while `SECURITY_OVERVIEW.md:269` reads `whitelist.is_admin` from the relay D1.
- *To close:* trace the kit's admin resolution at the deployed `KIT_REF`.

**F3. ADR-2016 exists in the kit (corrected, DW-06.6 I7, DW-06.9 D15).**
- *What the topic said:* `dreamlab.toml:34,149` cite an ADR that "was never written".
- *What exists:* the cited record is `nostr-rust-forum/docs/adr/ADR-2016-end-to-end-encrypted-zones.md` (accepted, `implementation_status: complete`), present at both `9c36b60` and `c39f41f`. Topic 06 now cites it, and its `verified_commit` became a per-repository map.
- The underlying breach of IDENTITY-zones invariant 3 ("only zone3 is encrypted", `docs/IDENTITY-zones.md:180-181`) is still real, because zones 2–4 are encrypted (`forum-config/dreamlab.toml:123,137,148`).
- *To close:* revise IDENTITY-zones invariant 3 and point it at kit ADR-2016.

**F4. Talk-to-AI correlation is built on the client (corrected, DW-04.3).**
- The client mints its own request id per question and binds replies to it (`src/components/AIChatFab.tsx:51-56,435-441`).
- Only protocol-level correlation, an `e` tag on the agent's reply, is missing. That is on the agent side, outside this repo.

**F5. The pin guard runs the same evaluator as `pin-check` (corrected, DW-05.4).**
- *What the topic said:* `kit-pin-guard.yml` is a narrower check than `ci.yml`'s `pin-check`.
- *What the code does:* the wrapper runs `node scripts/pin-parity.mjs` (`scripts/dream-kit-pin-guard.sh:27-28`). Without `--skip-actions`, that evaluator also sweeps Action SHA pins (`scripts/pin-parity.mjs:57-60`).
- The two gates differ only in triggers and annotation style.

**F6. A topic-03 citation passed the gate while pointing at the wrong file (corrected).**
- Bare `README.md:69` and `README.md:99-101` resolved by suffix to `forum-config/README.md`, the only README in `sources:`. The lines there ("### Key management", a mermaid fragment) do not support the claim. The root README lines do.
- The root `README.md` was added to `sources:`, and an exact match now wins.
- *Lesson for the gate:* a passing suffix match is not proof that the citation supports the claim.

**F7. Smaller corrections.**
- DW-01.6 arrows had the `cp -r kit/dist/*` merge on the React edge. It belongs on the forum edge (`.github/workflows/deploy.yml:270-273`).
- DW-03.9 said `provision-kv` substitutes the KV id. In fact it resolves or creates the namespace (`workers-deploy.yml:74-84`), and the per-worker job runs `sed` (`:231-237`).
- DW-04 anchors moved from `nostr.ts:225` (a comment close) to `:226`, and from `:213-216` to `:207-213`. The kit deletion anchor moved from `nip_handlers.rs:2468` to `:2453-2457`.
- DW-04.6 cross-referenced the wrong panel.
- DW-06.7 D8, DW-06.9 D10 and D15 pointed at the wrong panels or carried stale text. The D9/D11 note named stale SHAs (`BASELINE-architecture.md:6` now declares `e02c95d0`).
- DW-07.2 named an old stamp.
- DW-03.3 now cites `forum-config/README.md:47` for "trust 3". It also records a **Drift**: `forum-config/README.md:18` says "Three trust domains", while its own diagram (`:22-38`) and `IDENTITY-zones.md:86-87` say four.
- The DW-07.8 Debt now cites the 25-item cap (`scripts/lib/digest.mjs:18-19`) and the seen-marking (`scripts/inbox-watch/run.mjs:327-328`).

**Carried drifts re-verified, still open in the product docs:**
- `README.md:65` names kit `8a5cf81` against pins at `9c36b60`.
- `BASELINE-architecture.md:117` names `KIT_REF` `ea7d95e…`.
- The History table still marks `931898a` as current (`docs/architecture/kit-compatibility-record.md:355`).
- `IDENTITY-zones.md:89` says "Seven agents", but there are eleven `[[agents]]`.
- `CLAUDE.md:14` still says "dual-SPA".
- The System One credential check skips the auth worker (`workers-deploy.yml:287-288`).

## Panel dispositions

| Panel | Disposition | Evidence |
|---|---|---|
| DW-01.1 | accurate | `BASELINE-architecture.md:35-41`; `forum-config/Cargo.toml:6-9,50-53`; `dreamlab.toml:320` |
| DW-01.2 | accurate | `deploy.yml:244,335`; `README.md:54`; `CLAUDE.md:14`; `BASELINE-architecture.md:140-142` |
| DW-01.3 | accurate-with-limits | `deploy.yml:405-412,428-439,442`; `workers.rs:155-172`. The DNS non-resolution is a 2026-06-09 observation carried in a comment (`deploy.yml:49-56`), not re-checked |
| DW-01.5 | accurate | All four pin sites agree on `9c36b60`: `deploy.yml:130`, `workers-deploy.yml:44`, `rust-ci.yml:20`, `kit-compatibility-record.md:30-31`. Drifts at `README.md:65` and `kit-compatibility-record.md:355` confirmed |
| DW-01.6 | corrected | Merge edge moved to the forum arrow (`deploy.yml:270-273`). Gate condition `deploy.yml:147-149` and env payload `:90-107` confirmed |
| DW-01.7 | accurate-with-limits | `deploy.yml:291-301,367-397,399-402`; `public/spa-redirect.js`. That GitHub Pages serves `404.html` for unknown paths is host behaviour |
| DW-01.8 | accurate | `dreamlab.toml:244,264,275-277`; `CLAUDE.md:16-17` |
| DW-02.1 | accurate | `src/App.tsx:10-24,31-81`; `src/prerender.tsx:34`; `vite.config.ts:61-70` |
| DW-02.3 | accurate | `package.json:21-32`; `scripts/prerender.mjs:51,62,143-158`; `check-discovery.mjs:32`; `check-claims.mjs:2-3` |
| DW-02.4 | accurate-with-limits | `vite.config.ts:20-45`. The claim that the production build has no equivalent guard rests on static hosting, not on a test |
| DW-02.5 | accurate | `index.html:18,64-67,80,85-88`; `vite.config.ts:16-48` (two plugins only); `prerender.mjs:43-47,94-127,157` |
| DW-02.7 | accurate | `Contact.tsx:47-62,456-457,482-517`; `nostr.ts:352` |
| DW-03.1 | corrected | Root `README.md` added to `sources:` so `README.md:69,99-101` resolve to the file the claim is about. Text otherwise confirmed (`dreamlab.toml:33-37,101-151`; `branding.rs:17,28,40-42,54-71`) |
| DW-03.3 | corrected | Trust-3 citation and trust-domain **Drift** added. Eleven agents and three test users confirmed (`dreamlab.toml:337-412,443-454`) |
| DW-03.4 | accurate | `dreamlab.toml:39-72`; `deploy.yml:62-73` |
| DW-03.5 | accurate | Seven `compare()` groups and their site counts (`config-mirrors.mjs:161-311`); `:187-193` |
| DW-03.6 | accurate | `dreamlab.toml:21,27,33-41`; wrangler lines; `check-config-mirrors.mjs:15,27-35`; `config-mirrors.mjs:337-506` |
| DW-03.7 | accurate | `config-mirrors.mjs:41-47,109,315-326`; `test-and-lint.yml:80-83` |
| DW-03.8 | corrected | F1 rewired as a **Tension**. The TypeSafe receipt (run 41028875471) is historical and not re-checked |
| DW-03.9 | corrected | KV substitution wording (F7). Binding line ranges confirmed |
| DW-03.10 | corrected | F2 edges and invariant |
| DW-03.11 | corrected | 24 tables and 476 lines confirmed (`001_init.sql`). D1/DO conflation fixed; two **Open** items (F2) |
| DW-04.1 | accurate | `nostr.ts:31,98,279-281,684-686,849-851`; zero occurrences of "NIP-98" in `nostr.ts`; `AUTHENTICATION.md:200`; `IDENTITY-zones.md:46-56,162-164` |
| DW-04.3 | corrected | F4 |
| DW-04.4 | corrected | Anchors (F7). Lazy AUTH and retry confirmed (`nostr.ts:260-330`) |
| DW-04.5 | accurate-with-limits | Kit `nip_handlers.rs:1087-1153,2453-2480` at `c39f41f` (unchanged from `9c36b60`). The Worker deploy receipt is historical |
| DW-04.6 | corrected | Cross-reference and gate attribution fixed. The layered resolution rests on `SECURITY_OVERVIEW.md:259-272`, a doc about kit code not read here |
| DW-04.8 | accurate-with-limits | Values match `SECURITY_OVERVIEW.md:201-242`. Enforcement is kit code not traced here |
| DW-04.9 | accurate | `deploy.yml:80`; `dreamlab.toml:407-412,435-440`; `website-info.mjs:100-110`; `provision-website-info.mjs:97` |
| DW-05.1 | corrected | Secret-validation sentence and product-doc **Drift** (F1). Jobs, triggers and the aggregator confirmed (`ci.yml:11-63,169-186,225,266-310`) |
| DW-05.2 | accurate | `rust-ci.yml:4-9,20,39-62,84-93,118,142`; `test-and-lint.yml:20-30` |
| DW-05.3 | accurate | `test-and-lint.yml:1-17,40,88-91,142-177`; `deploy.yml:147-149`; `workers-deploy.yml:64-66,110`; `BASELINE-architecture.md:191` |
| DW-05.4 | corrected | F5 |
| DW-05.8 | accurate | `pin-parity.mjs` (lib) `:47,99-174,190,284-307,370-378`; tests and fixture lines confirmed |
| DW-06.1 | accurate | `generate-workshop-list.mjs:5-124`; `generate-testimonials.mjs:11-25`; `CLAUDE.md:86-87` |
| DW-06.2 | accurate | `generate-workshop-list.mjs:40-46,73-112`; `content-generators.mjs:7-11,88-99` |
| DW-06.5 | accurate | `docs/api` and `docs/security` file lists match `git ls-tree`; `MODERATION_API.md:9,15` |
| DW-06.6 | corrected | F3 |
| DW-06.7 | corrected | D8 cross-reference. D1–D7 confirmed |
| DW-06.9 | corrected | D10, D15 and the SHA note (F3, F7). D12 to D14 confirmed (`3baf2fa4` is a commit; embeddings headers name RuVector Postgres and all-MiniLM-L6-v2) |
| DW-06.8 | accurate | `claims.mjs:28-56,89-142`; register holds 3 features and 2 claims; `README.md:65,125` |
| DW-07.2 | corrected | Stamp reference. 37 files under `scripts/seed` at `0e157280`; `CLOSEOUT-SECURITY-AUDIT.md:113-116,300` |
| DW-07.3 | accurate | Seed-script lines; `.gitignore:77-78`; `server.mjs:7-10`; `reseed-minimoonoir.mjs:1-6` |
| DW-07.4 | accurate-with-limits | Embeddings headers read; `search-worker.wrangler.toml:11-12,33-36`; `SYSTEM-ONE.md:3-5,13-16`. Run 41028875471 is not re-checked |
| DW-07.5 | accurate | `scripts/backup/crontab:6`; `forum-backup.sh:1-35` |
| DW-07.7 | accurate | Line counts 413 and 258; `test-rvf-dreamlab.mjs:1-20`; `test-search-api-e2e.mjs:1-14` |
| DW-07.8 | accurate-with-limits | `seo-watch/crontab:17`; `run.mjs:4-11,35,187,288-356,376-380`; `digest.mjs:18-19`. The probe-phrase suppression is reproduced only in the 2026-11-01 dated review, not here |

## Verification

**Evidence class.** Source tracing at pinned revisions, through `git show` and `git grep` only.
- dreamlab-ai-website: `0e1572801d03a1af6d8bc8861f928da087d69eed`.
- nostr-rust-forum: `c39f41f104b114d6a3fa29013daaf6fa58b45080`, for DW-04.5 and the ADR-2016 attribution in DW-06.

No tests, builds, deployments or live endpoints were run.

**Why no line numbers moved on restamp.**
- `git diff --stat a67cb2a8 0e157280` touches only `docs/dream-cycle/LEDGER.md`.
- `git diff 9c36b603 c39f41f1 -- crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs` is empty.
- Every new or moved anchor was taken from `git grep -n` or a numbered `git show` at the pinned revision.

**Restamps.**
- Topics 01, 03, 05 and 07: `verified_commit` set to dreamlab-ai-website `0e157280…`.
- Topics 04 and 06: a per-repository map {dreamlab-ai-website `0e157280…`, nostr-rust-forum `c39f41f1…`}.
- Topic 02 was not edited and keeps `a67cb2a8…`, which is byte-identical for every file it cites.

**Gate.** Each of the following was run from the worktree root and exited 0, with 0 cite-check warnings:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only <topic> --check --cite-check --strict-citations --render
```

| Topic | Citations verified at the declared revision |
|---|---|
| 01 | 25 |
| 02 | 48 |
| 03 | 113 |
| 04 | 42 |
| 05 | 26 |
| 06 | 46 |
| 07 | 26 |

**Limits.**
- The gate counts appear to cover only citations inside diagram blocks. Topic 02's prose cites `deploy.yml`, which is not in its `sources:`, and still produced no warning. The gate also resolves by suffix (F6), so a pass does not show that a citation supports its claim. Every prose anchor in these topics, including each one rewritten here, was checked by hand at the pinned revision.
- Six topics were restamped, so the generated `VERIFICATION.md` and `COVERAGE.md` now lag the declared revisions until the lead regenerates them.
- Not verified:
  - kit behaviour behind `SECURITY_OVERVIEW.md` limits and admin resolution;
  - which D1 the kit reads `whitelist.is_admin` from;
  - Cloudflare secret state and Worker run 41028875471;
  - DNS for the branded worker domains;
  - GitHub Pages 404 handling.
