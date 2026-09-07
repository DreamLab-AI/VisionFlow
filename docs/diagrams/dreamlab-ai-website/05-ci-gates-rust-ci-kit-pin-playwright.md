---
id: DW-05
title: CI gates — rust-ci, KIT_REF pin guard, playwright, config mirrors
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2005]
sources:
  - ../dreamlab-ai-website/.github/workflows/ci.yml
  - ../dreamlab-ai-website/.github/workflows/test-and-lint.yml
  - ../dreamlab-ai-website/.github/workflows/rust-ci.yml
  - ../dreamlab-ai-website/.github/workflows/kit-pin-guard.yml
  - ../dreamlab-ai-website/.github/workflows/set-worker-secrets.yml
  - ../dreamlab-ai-website/.github/workflows/deploy.yml
  - ../dreamlab-ai-website/.github/workflows/workers-deploy.yml
  - ../dreamlab-ai-website/playwright.config.ts
verified_commit: 9a3dd8830
---

## DW-05.1 `ci.yml` — the ten-job PR/push gate and its aggregator
```mermaid
flowchart TB
    TRIG["push/PR to main, paths: src/**, forum-config/**,<br/>.github/workflows/**, scripts/**, dream.config.json<br/>ci.yml:14-51 — deliberately broad, see note"] --> NL["node-lint: eslint"]
    TRIG --> NT["node-test: vitest run"]
    TRIG --> NB["node-build: npm run build, stub VITE_* env"]
    TRIG --> NA["node-audit: npm audit high, continue-on-error"]
    TRIG --> PC["pin-check: scripts/pin-parity.mjs --github"]
    TRIG --> EC["endpoint-check: scripts/check-effective-endpoints.mjs --github"]
    TRIG --> CM["config-mirrors: scripts/check-config-mirrors.mjs --github, see DW-03.5"]
    TRIG --> RF["rust-fmt: cargo fmt forum-config --check"]
    TRIG --> RC["rust-clippy: cargo clippy forum-config -D warnings"]
    TRIG --> RT["rust-test: cargo test forum-config"]
    NL & NT & NB & NA & PC & EC & CM & RF & RC & RT --> PASS["ci-pass aggregator<br/>ci.yml:260-303 — single required status check"]
    PASS -->|any != success| BLOCK["::error:: one or more required jobs did not succeed"]
```
- `ci.yml:14-30` deliberately enumerates `.github/workflows/**` and `scripts/**` as trigger paths, not just the three `KIT_REF` pin sites: pin-check sweeps ALL workflow files for tag-pinned actions, so any workflow edit can break it — narrower path filters previously let `kit-pin-guard.yml` carry an unpinned `actions/checkout@v4` without re-running the gate that rejects it (2026-09-06 incident, `ci.yml:23-27` comment).
- `rust-clippy` here is `-D warnings` (blocking, `ci.yml:` rust-clippy job) — unlike `rust-ci.yml`'s clippy job against the kit, which is explicitly advisory (DW-05.2).

## DW-05.2 `rust-ci.yml` — manual-only gate against the upstream kit
```mermaid
flowchart LR
    TRIG["workflow_dispatch only<br/>rust-ci.yml:8-9<br/>kit source lives in a separate repo"] --> CLONE["clone nostr-rust-forum at KIT_REF<br/>rust-ci.yml:21"]
    CLONE --> FMT["fmt: cargo fmt kit --check"]
    CLONE --> CLIPPY["clippy (advisory): warnings reported, not enforced<br/>rust-ci.yml:40-63"]
    CLONE --> TN["test-native: 5 kit crates<br/>-p nostr-bbs-{core,auth-worker,pod-worker,preview-worker,search-worker}<br/>+ relay-worker --features test-exports<br/>rust-ci.yml:85-94"]
    CLONE --> TW["test-wasm: cargo test --target wasm32-unknown-unknown<br/>-p nostr-bbs-core --no-run<br/>rust-ci.yml:119"]
    CLONE --> CWC["check-wasm-client: cargo check<br/>-p nostr-bbs-forum-client, wasm32 target<br/>rust-ci.yml:143"]
```
- `rust-ci.yml:4-6` states the division of labour explicitly: "Triggered by workflow_dispatch (manual) since kit source lives in a separate repo. The test-and-lint.yml workflow covers forum-config/ on every push/PR automatically" — `forum-config/` gets automatic gating (DW-05.3); the kit itself only gets this manual, advisory-clippy check.

## DW-05.3 `test-and-lint.yml` — the reusable pre-deploy gate, fixed 2026-09-05
```mermaid
sequenceDiagram
    autonumber
    participant CALLER as deploy.yml / workers-deploy.yml<br/>uses: workflow_call
    participant GATE as gate job<br/>test-and-lint.yml:41
    participant STEPS as pins, endpoints, mirrors,<br/>vitest, react_build, rust_fmt, rust_test<br/>each `if: always()`, id-tracked
    participant SUM as Summary step<br/>test-and-lint.yml:145-181
    CALLER->>GATE: workflow_call
    GATE->>STEPS: run every gate step regardless of prior failure
    STEPS-->>SUM: steps.<id>.outcome for each
    SUM->>SUM: FAILED = any outcome != success<br/>test-and-lint.yml:165-171
    SUM-->>CALLER: passed=true only if FAILED is empty<br/>test-and-lint.yml:174,179
    CALLER->>CALLER: needs.gate.outputs.passed == 'true'<br/>required in addition to needs:[gate]
```
- INVARIANT: `test-and-lint.yml:1-17` documents two closed defects (ADR-2002/2003 closeout, 2026-09-05): (1) it previously ran no unit tests at all — Vitest lived only in the separate `ci.yml`, which `deploy.yml` does not depend on, so a red suite never blocked a deploy; (2) the final step wrote `passed=true` unconditionally under `if: always()`, so the output was true even when an earlier step had failed.
- `deploy.yml:111-117` and `workers-deploy.yml:62-66` both check `needs.gate.outputs.passed == 'true'` explicitly, not merely `needs: [gate]` — the comment notes this is deliberate: requiring the gate's own verdict means a future change that makes a gate step non-blocking cannot quietly re-open the publication path.
- DOC-DRIFT: `docs/BASELINE-architecture.md:173` (estate closeout, dated 2026-09-04) claims "CI has a Vitest/pin/admin aggregator, while deployment uses a separate reusable gate without Vitest" — that description predates the fix `test-and-lint.yml`'s own header documents as landing 2026-09-05 (one day later); as of this verified commit, `test-and-lint.yml` DOES run Vitest (`id: vitest`, line 89-92) and its `passed` output is a real aggregation, not a constant.

## DW-05.4 `kit-pin-guard.yml` — pin-parity as its own gate
```mermaid
sequenceDiagram
    autonumber
    participant T as trigger: PR, push to main, workflow_dispatch<br/>kit-pin-guard.yml:3-7
    participant J as pin-parity job<br/>kit-pin-guard.yml:13
    participant S as scripts/dream-kit-pin-guard.sh
    T->>J: checkout, setup-node 20
    J->>S: bash scripts/dream-kit-pin-guard.sh
    S-->>J: stdout containing PIN-DRIFT or PIN-PARITY-OK
    alt output contains PIN-DRIFT
        J->>J: ::error:: align KIT_REF in deploy.yml/workers-deploy.yml/rust-ci.yml,<br/>crate versions in forum-config/Cargo.toml,<br/>CANONICAL_* in kit-compatibility-record.md<br/>exit 1
    else no PIN-PARITY-OK token found
        J->>J: ::error:: pin-parity emitted no verdict token, exit 1
    end
```
- This is a narrower, faster check than `ci.yml`'s `pin-check` job (which runs the fuller `scripts/pin-parity.mjs --github`, also covering Action SHA pinning across all workflows) — `kit-pin-guard.yml` delegates to a plain-ESM shell+node script with no deps, pinned to Node 20 specifically because it "delegates to scripts/pin-parity.mjs" (kit-pin-guard.yml:18-19 comment).

## DW-05.5 Playwright / E2E surface
```mermaid
flowchart LR
    CFG["playwright.config.ts<br/>testDir ./tests, timeout 120s, retries 0"] --> CHROMIUM["Nix-pinned chromium executablePath<br/>fallback if PLAYWRIGHT_CHROMIUM_PATH unset"]
    CFG --> SPECS["tests/forum-smoke.spec.ts<br/>tests/login-deep.spec.ts"]
    CFG --> PROBES["tests/agentbox-relay-test.mjs<br/>tests/nostr-integration.mjs<br/>Node integration probes, not Playwright specs"]
    SPECS --> CORS["forum-smoke.spec.ts:404-410<br/>asserts workers return scoped CORS<br/>cited by BASELINE-architecture.md:86"]
```
- Playwright/E2E specs are NOT wired into `ci.yml`, `test-and-lint.yml`, or any gate job in this tree — `CLAUDE.md:63` documents `npx playwright test` as a manual command requiring a running deployment (`playwright.config.ts` comment), consistent with `package.json`'s `test:e2e` script existing outside the `predev`/`prebuild` and CI-invoked script set.

## DW-05.6 `set-worker-secrets.yml` — the one-shot operator secret push
```mermaid
flowchart TB
    TRIG["workflow_dispatch only<br/>set-worker-secrets.yml:12-13"] --> S1["NATIVE_POD_URL -> dreamlab-auth-api"]
    TRIG --> S2["NATIVE_POD_ADMIN_KEY -> dreamlab-auth-api"]
    TRIG --> S3["PRF_SERVER_SECRET -> dreamlab-auth-api"]
    TRIG --> S4["ADMIN_PUBKEYS -> dreamlab-auth-api"]
    S1 & S2 & S3 & S4 --> API["CF Workers Secrets API PUT<br/>jq-built JSON body, idempotent<br/>set-worker-secrets.yml:20-38"]
    API -.->|validated at deploy time by| GATE["workers-deploy.yml 'Validate required auth-worker secrets are set'<br/>see DW-03.7"]
```
- `set-worker-secrets.yml:1-9` states the pipeline never generates these four values — they are read from GitHub repo secrets and pushed as-is; `PUT` on the CF secrets endpoint is idempotent, so the workflow is safe to re-run.
