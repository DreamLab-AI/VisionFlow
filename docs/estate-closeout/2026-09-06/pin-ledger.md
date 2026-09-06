# VisionFlow estate — pinned-inheritance ledger

Compiled 2026-09-06. All local checkouts were `git fetch`ed during this run; "current value"
for a sibling means that sibling's live `origin` HEAD (or crates.io `max_version` /
GitHub tag HEAD where a Nix/Cargo pin targets a release). Assessment key:
**CURRENT** = pin equals target's current value · **HELD-INTENTIONALLY** = pin is behind but a
cited doc says so on purpose · **DRIFTED** = pin is behind with no such citation · **MUTABLE** =
the pin site does not pin at all (floating tag/branch/`:latest`).

Live HEADs at time of writing (all local checkouts equal their `origin` tracking branch except
dream-machine, which is 1 commit **ahead**, unpushed):

| Repo | HEAD |
|---|---|
| VisionFlow | `8cf1a1bf9e4ef2ef98ee6c7bf56ef361aa8d0304` |
| VisionClaw (`project`) | `2d87c3b81c21ed9cb1a116cbb84b73d8e96dba2f` |
| agentbox (`project/agentbox`) | `eb7794b17892d9032b4f9ce0ed22ea0d9dac3497` |
| loom | `c86f462f41d43c56d12292ffc12ae72bef6bf799` |
| nostr-rust-forum | `486ec5aa0eac8741a35adc4ccd4b4aaf12871012` |
| solid-pod-rs | `40f160c69b0539a1a7e0a9b8d6d49331aae144fa` |
| dreamlab-ai-website | `000a40e43956ff534bc8ac198c2b8c75d7f7b517` |
| prose-sanitiser | `e6d456d0294db03b7bd00e69dda461bfb8ac7d04` |
| diagram-ir | `16e325489df518dd3f951f090cb3f5f07929aebf` |
| knowledgeGraph | `7bbf0aae2b60a63c2714b750e3b381bce6be7ad8` |
| dream-machine (dream-engine) | `7c30573a2d73c8fa4c67a43042d7c0b204eefa13` (local; +1 unpushed vs `origin/main` `e1a0acde…`) |
| visionGraph | `fabcdbcc9fd67951087a9492fb45780bd379134b` |
| dreamlab-cumbria | `15fc366f16c3566ce04476fa299cec1174d17dfe` |

---

## 1. Git submodules

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `project/.gitmodules` + `project` submodule index → `agentbox` | `DreamLab-AI/agentbox` | `eb7794b17892d9032b4f9ce0ed22ea0d9dac3497` (branch `archive/feature-high-perf-networking-850-geb7794b17`) | `eb7794b17892d9032b4f9ce0ed22ea0d9dac3497` | 0 | **CURRENT** — the working checkout at `project/agentbox` is itself on this exact commit. |

No other repo in the estate carries a git submodule (`git submodule status` empty in VisionFlow, loom, nostr-rust-forum, solid-pod-rs, dreamlab-ai-website, prose-sanitiser, diagram-ir, knowledgeGraph, dream-machine, dreamlab-cumbria).

---

## 2. Nix pins (`project/agentbox/flake.nix`, `flake.lock`, `lib/*.nix`)

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `flake.nix:18` `inputs.aoe.url` | `DreamLab-AI/agentbox-of-empires` | `d615b8c8` (short) | `d615b8c829028e272b492ea1726e8f490bc1479c` = current `main` HEAD | 0 | **CURRENT** |
| `flake.nix:37` `inputs.codexPlugin.url` | `openai/codex-plugin-cc` | `db52e28f4d9ded852ab3942cea316258ae4ef346` | same = current `main` HEAD | 0 | **CURRENT** |
| `lib/solid-pod-rs.nix:52,55` `version`/`rev` | `DreamLab-AI/solid-pod-rs` | `v0.5.0-alpha.3` / `87b35a1b32f9789e296ebbf7277b9ecc01657c42` | latest tag `v0.5.0-alpha.8`; crates.io `solid-pod-rs` max_version `0.5.0-alpha.8` | **58 commits / 5 tagged releases behind** (alpha.3→5→6→7→8) | **DRIFTED** — no doc found citing this as intentional; the file's own comment block only narrates up to alpha.3/0.5.0-alpha.3 and gives no "hold" rationale. |
| `lib/diagram-ir.nix:54-55` `rev = "v${version}"` (0.1.0) | `DreamLab-AI/diagram-ir` | `v0.1.0` → `095a2d849fd5ccebea3ccaa1961f65b0a1ad2140` | HEAD is 2 untagged commits past `v0.1.0` (only tag in repo); crates.io max_version `0.1.0` | 2 commits, 0 releases | **CURRENT** (tracks the only cut release; crates.io agrees) |
| `lib/prose-sanitiser.nix:56,70` `version`/`rev` | `DreamLab-AI/prose-sanitiser` | `v0.1.1` → `696626baa18bb7d1ef5c34870c057870afe009d0` | HEAD is 1 untagged commit past `v0.1.1` (latest tag); crates.io max_version `0.1.1` | 1 commit, 0 releases | **CURRENT** |
| `lib/nagual-qe.nix:46,52,58` `rev`/hashes | `proffesor-for-testing/nagual-qe` (non-DreamLab) | `b3f7a12609e3cb0a59e53bc6d6a82c08d46b0d05` | same = current `master` HEAD | 0 | **CURRENT** |
| `lib/dream-engine.nix` | *not a cross-repo pin* — builds the in-tree `services/dream-engine` crate vendored inside agentbox itself; `homepage` even points at `github.com/DreamLab-AI/agentbox`, not `dream-engine`/`dream-machine`. | n/a | n/a | n/a | **N/A** — agentbox does **not** Nix-pin the standalone `DreamLab-AI/dream-engine` repo at all; it ships a separately-maintained fork of the same idea in-tree. Worth flagging to the queen as a duplication risk, not a pin-drift. |
| *(no pin found)* | `DreamLab-AI/loom` | — | — | — | **N/A** — the Loom is consumed only as an HTTP endpoint (`agentbox.toml` `http://192.168.2.132:8084/v1`), never as a Nix/source input. No pin site exists to drift. |

---

## 3. Cargo path/git deps between estate repos

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `project/agentbox/services/nostr-pod-bridge/Cargo.toml:21` | `nostr-rust-forum` crate `nostr-bbs-core` | `path = "../../../../nostr-rust-forum/crates/nostr-bbs-core"` (unversioned path dep, resolves to whatever the sibling checkout on disk contains) | sibling checkout HEAD `486ec5a…` | n/a (floating by construction) | **MUTABLE** — path deps have no rev; agentbox's `nostr-pod-bridge` compiles against whatever `nostr-rust-forum` happens to be checked out at, which is a *different* source of truth from the Nix-pinned `solid-pod-rs` rev two lines below. |
| `project/agentbox/services/nostr-pod-bridge/Cargo.toml:22,65` | `solid-pod-rs` crates `solid-pod-rs-nostr`, `solid-pod-rs` | `path = "../../../../solid-pod-rs/crates/…"` | sibling checkout HEAD `40f160c…` (past `v0.5.0-alpha.8`) | n/a | **MUTABLE**, and **inconsistent with §2**: the Nix-built `solid-pod-rs-server` binary (`lib/solid-pod-rs.nix`) is fetched at tag `v0.5.0-alpha.3`, while any local/dev cargo build of `nostr-pod-bridge` compiles against the live sibling checkout (currently 5 releases newer). Two different `solid-pod-rs` snapshots can end up in the same agentbox image depending on build path. |
| `loom/Cargo.toml:60` `[workspace.dependencies].ruvector-core` | `ruvnet/ruvector` (not in the requested repo list, but present at `/home/devuser/workspace/ruvector`) | `path = "../ruvector/crates/ruvector-core"`, branch `report/agentbox-field-2026-07` @ `677b2475409c50cb964be8a3b848da2952390535` | same (unchanged since the 2026-09-05 draft manifest — see §6) | 0 (checkout unmoved) | **MUTABLE by construction / CURRENT-by-luck** — no rev pin exists; happens to still match because nobody has moved the sibling checkout since 2026-09-05. |
| `project/xr-client/rust/Cargo.toml`, `project/crates/*/Cargo.toml` | intra-repo (`visionclaw-domain`, `visionclaw-xr-presence`, etc.) | `path = "../…"` | n/a | n/a | **N/A** — internal to VisionClaw's own workspace, not a cross-repo estate pin. |
| `solid-pod-rs/crates/*/Cargo.toml` (8 members) | intra-repo, e.g. `solid-pod-rs = { version = "0.5.0-alpha.8", path = "../solid-pod-rs", … }` | version string tracks the workspace's own current version | n/a | n/a | **N/A/CURRENT** — internal workspace path+version pins, all correctly reading `0.5.0-alpha.8` (the repo's own HEAD version), no cross-repo relevance. |
| `nostr-rust-forum`, `prose-sanitiser`, `diagram-ir`, `dreamlab-ai-website` (root/service Cargo.toml) | — | no `git = "https://github.com/DreamLab-AI` and no `path = "../"` cross-repo entries found | — | — | No sites. |

---

## 4. GitHub Actions pins (`uses:`)

Counted every real (non-`node_modules`, non-vendored-third-party) workflow file per repo.
`project/agentbox/skills/echoloop`, `.../ruvector-catalog`, `project/scripts/whelk-rs`, and
`project/voice-stack/unmute` (461 workflow files!) are **vendored third-party trees**, excluded
here as noise, not estate pin sites.

| Repo | Mutable-tag `uses:` | SHA-pinned `uses:` | Enforces SHA pinning? |
|---|---|---|---|
| dreamlab-ai-website | 2 | 57 | **Yes** — near-total SHA pinning (matches CLAUDE.md's note). The 2 mutable outliers are worth a follow-up (not itemised here to keep this table short — see raw dump). |
| solid-pod-rs | 0 | 33 | **Yes** — fully SHA-pinned. |
| project (VisionClaw root + client) | 42+8 | 0 | **No** — 100% mutable tags (`actions/checkout@v4` etc.) across `ci.yml`, `docs-ci.yml`, `ontology-publish.yml`, `xr-godot-ci.yml`, `client/benchmarks.yml`. |
| project/agentbox | 55 | 5 | **Mixed/inconsistent** — only `deepsec.yml` SHA-pins its 5 actions (`actions/checkout@11d5960a…` etc., all annotated `# v4`); the other 13 agentbox workflows (`ci.yml`, `release.yml`, `manifest-validate.yml`, `contract-tests.yml`, `invariants.yml`, `build-multi-arch.yml`, `image-scan.yml`, `secret-scan.yml`, `shellcheck.yml`, `tui-tests.yml`, `flake-check.yml`, `nix-flake-update.yml`, `ontology-publish.yml`) use mutable tags. |
| VisionFlow | 19 | 0 | No |
| nostr-rust-forum | 27 | 0 | No |
| prose-sanitiser | 7 | 0 | No |
| diagram-ir | 7 | 0 | No |
| knowledgeGraph | 3 | 0 | No |
| dream-machine | 16 | 0 | No |
| loom | 0 workflows exist at all | 0 | **N/A — loom has no `.github/workflows` directory**, i.e. no CI gate of any kind, let alone a pinning policy. Flag this to the queen as a gap, distinct from a drifted pin. |
| dreamlab-cumbria | (workflows only exist nested inside vendored tools `tools/fossflow`, `infrastructure/…/web-simulation`) | — | N/A — no first-party workflow at the repo root. |

**Assessment per row:** every **MUTABLE** cell above is exactly that verdict — a mutable-tag pin
site, by definition not pinned to a resolvable commit, so "current vs target" doesn't apply; the
finding *is* the mutability. dreamlab-ai-website and solid-pod-rs's SHA rows are **CURRENT** by
construction (each SHA is a real, resolvable commit of the named action; no further drift check
applies to a pinned action version itself, only to whether a newer release exists — out of scope
here). Full per-line dump (`grep -n uses:` across all 281 matched lines, classified) was produced
during research and can be regenerated from the same commands if the queen wants the exhaustive list.

---

## 5. Website kit pin (`dreamlab-ai-website` ↔ `nostr-rust-forum`)

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `.github/workflows/deploy.yml:98` `KIT_REF` | `nostr-rust-forum` | `a7544687b4d1c09807862d749b27f8c8da307a12` | HEAD `486ec5a…` | 5 commits behind HEAD; 3 commits **past** the `v1.0.0-beta.9` tag (`90ffe74…`) | **HELD-INTENTIONALLY** |
| `.github/workflows/workers-deploy.yml:44` `KIT_REF` | same | same | same | same | **HELD-INTENTIONALLY** |
| `.github/workflows/rust-ci.yml:21` `KIT_REF` | same | same | same | same | **HELD-INTENTIONALLY** |
| `docs/architecture/kit-compatibility-record.md:30-31` `CANONICAL_KIT_SHA`/`CANONICAL_KIT_VERSION` | same | `a7544687…` / `1.0.0-beta.9` | same | same | **HELD-INTENTIONALLY** — this file *is* the citation: it states the pin deliberately advances 3 commits past the `v1.0.0-beta.9` tag to include commit `a754468` ("relay: indexed tag lookups via trigger-maintained `event_tags` table — D1 free-tier fix"), and documents `ci.yml`'s `pin-check` job as enforcing all four sites (+ `forum-config/Cargo.toml`) stay byte-identical. |
| `forum-config/Cargo.toml:49-52` `nostr-bbs-{core,config,mesh,rate-limit} = "=1.0.0-beta.9"` | crates.io `nostr-bbs-*` crates | `=1.0.0-beta.9` (exact pin) | crates.io max_version `nostr-bbs-core` = `1.0.0-beta.9` (verified); `kit-compatibility-record.md`'s "Resolved kit packages" block records the exact registry checksums | 0 | **CURRENT** — and the version string agrees with `CANONICAL_KIT_VERSION` above, so the whole five-site lockstep (`KIT_REF`×3 + `CANONICAL_KIT_SHA` + Cargo exact-pins) is internally consistent right now. |

No drift here — this is the healthiest pin cluster in the estate: five independent sites, one
enforced by a CI `pin-check` job, all agreeing, with the one intentional lag explicitly justified
in a citable doc.

---

## 6. VisionFlow cross-repo pins

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `scripts/drift-counter/allowlist.json:6` `source_pin.revision` | `DreamLab-AI/agentbox` | `eb7794b17892d9032b4f9ce0ed22ea0d9dac3497` | `eb7794b1…` (agentbox HEAD) | 0 | **CURRENT** |
| `.github/workflows/drift-counter.yml:61` `ref:` | same | same | same | 0 | **CURRENT** — and identical to the allowlist's pin, exactly as the allowlist's own doc string demands ("keep the two in step"). Three-/four-way agreement across allowlist.json, drift-counter.yml, the submodule pin in §1, and agentbox's own HEAD. |
| `.github/workflows/fixture-drift.yml` `env.CANONICAL_REPO` | `DreamLab-AI/VisionClaw` `tests/fixtures/` | **no `ref:`** — checks out VisionClaw's default branch at run time | n/a | n/a | **MUTABLE by design**, and the workflow's own header comment admits the gate has been a "green no-op" historically and, as of the 2026-09-05 revision, now *fails* rather than passing vacuously when the cross-repo checkout can't be provisioned (no deploy key/PAT wired yet). Not a drifted pin — there is no pin, and the file says so. |
| `docs/estate-closeout/2026-09-05/release-manifest.local-draft.json` (repositories[].head, generated `2026-09-05T14:18:04Z`, `status: "local-draft"`) | 13 sibling repos | snapshot HEADs as of 2026-09-05 | current HEADs (this run) | See table below | **DRIFTED-by-age** (informational — the file is explicitly `local-draft`/`not-compared`, not an enforced pin; nothing currently checks it against live state) |
| `website/assets.manifest.json` | — | staged **site assets** (CSS/JS/images), not sibling-repo revisions | — | — | Out of scope — not a cross-repo pin site despite the name. |

Draft-manifest staleness (recorded 2026-09-05 14:18 vs. HEAD now):

| Repo | Manifest-recorded HEAD | Current HEAD | Commits since |
|---|---|---|---|
| VisionFlow | `8cf1a1b…` | `8cf1a1b…` | 0 |
| VisionClaw (`project`) | `b00c28a0…` | `2d87c3b8…` | **21** |
| agentbox | `89301ec7…` | `eb7794b1…` | **21** |
| solid-pod-rs | `d6ac7f51…` | `40f160c6…` | 1 |
| nostr-rust-forum | `f18b471e…` | `486ec5aa…` | 2 |
| dreamlab-ai-website | `7e243741…` | `000a40e4…` | 2 |
| loom | `8cdef36b…` | `c86f462f…` | 1 |
| knowledgeGraph | `7bbf0aae…` | `7bbf0aae…` | 0 |
| visionGraph | `fabcdbcc…` | `fabcdbcc…` | 0 |
| dream-engine (`dream-machine`) | `7c305 73a…` | `7c30573a…` | 0 (local; origin has moved sideways via unpushed local commit, see header) |
| WasmVOWL / logseq(`project4`) / ruvector / RuView (not in the team-lead's repo list, but present on disk and referenced by this manifest) | — | — | all 0 — unchanged since the draft |

Read: the draft release manifest is one day stale specifically for the actively-worked estate-closeout repos (VisionClaw +21, agentbox +21 commits — both had heavy activity today, 2026-09-06) and still accurate for the quieter ones. It is explicitly a **local, uncommitted draft** (`status: "local-draft"`, `fixtures.status: "not-compared"`), so this is not a broken enforced gate — it's a snapshot nobody has re-run today.

---

## 7. Docker/compose image pins, `agentbox.toml` integrations, Loom model identity

| Pin site (file:line) | Pins target | Pinned value | Target current | Delta | Assessment |
|---|---|---|---|---|---|
| `project/agentbox/docker-compose.yml:11` + `agentbox.toml:376` (`integrations.ruvector_external.image`) | `ruvnet/ruvector-postgres` (Docker Hub) | `2.0.5@sha256:7fb09d43…` | not checked against registry (no Hub API call made; comment in-file states bump path is `./agentbox.sh ruvector update` only, rehearsed against a `pg_basebackup` snapshot) | — | **HELD-INTENTIONALLY** — tag+digest pinned deliberately (comment: "A floating `:latest` here silently drifts… the running sidecar was one image release behind before this pin"), with a documented, gated bump procedure. |
| `project/docker-compose.unified.yml:247`, `project/agentbox/docker-compose.solid-pods.yml:26` (cloudflared) | `cloudflare/cloudflared` | `latest@sha256:6b599ca3…` (unified.yml, digest-pinned) vs **`latest` with no digest at all** (solid-pods.yml:26) | n/a | — | **Inconsistent within agentbox itself**: `docker-compose.unified.yml` pins cloudflared by digest (effectively immutable despite the `latest` tag label); `docker-compose.solid-pods.yml` uses bare `cloudflare/cloudflared:latest` with **no digest** → genuinely **MUTABLE**, will silently pull whatever Cloudflare publishes next, for the exact same service the other compose file goes to lengths to pin. |
| `project/docker-compose.unified.yml:290` `image: ${LOOM_IMAGE:-loom:rust}` | `loom` build | local tag `loom:rust`, no registry/digest | n/a | — | **MUTABLE** — floating local build tag, not reproducible across hosts. |
| `project/agentbox/docker-compose.android.yml:22` | `redroid/redroid` | `13-mtg` | n/a | — | **MUTABLE** (not a DreamLab estate repo; noted for completeness only). |
| `agentbox.toml:459-510` `[integrations.solid_pod_rs]` | `solid-pod-rs` runtime integration (not a version pin — a config/endpoint binding) | `base_url = https://pods-native.dreamlab-ai.com`, `port 8484`, feature flags (`enable_did_nostr=true`, `sign_requests=false` **by declared design**, per ADR-2064/2078 — signing key not yet provisioned) | — | — | **HELD-INTENTIONALLY**, cited in-line to ADR-2064/2078 ("declared unsigned DELIBERATELY … until ADR-2078 provisions the signer"). |
| `agentbox.toml:131-160` `[sovereign_mesh.relay]` | nostr relay | `bind = "127.0.0.1"`, `port 7777`, `expose = false` (loopback-only; LAN/WAN reached only via the separate `forum_relay_url` WSS endpoint to the Cloudflare Worker relay) | — | — | **CURRENT/as-designed** — matches CLAUDE.md's "relay (loopback vs LAN)" framing exactly: the local relay is loopback-bound; the LAN/WAN-facing relay is the external `wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev`. |
| `agentbox.toml:675,1707` `[skills.ontology.condense].endpoint` / `loom_url` | Ontology Loom façade | `http://192.168.2.132:8084/v1`, `model = "qwen3.8-27B"` | — | — | Endpoint URL matches project CLAUDE.md and `loom/docs/QWEN3.8-CONNECTION.md`'s port/host claims (`:8084` façade, `:8085` raw model, `10.10.10.1`⇄`192.168.2.132` DNAT). **Model *name*, however, is DRIFTED** — see next row. |
| project `CLAUDE.md` claim ("Deployed model today: **Qwen3.8-27B**… runs inside the Loom stack as the `loom-model` container on `:8085`") **and** `agentbox.toml`'s `model = "qwen3.8-27B"` | `loom` deployed model identity | plain "Qwen3.8-27B" | `loom/docs/QWEN3.8-CONNECTION.md` **Addendum (2026-08-18)**: "the deployed variant is now **Heretic abliterated Qwen3.8-27B Q8_0**" — serving `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` at Q8_0 on the same `:8085` façade/port | 1 undocumented model swap (2026-08-18) not reflected upstream | **DRIFTED** — both the container-env CLAUDE.md and agentbox.toml's ontology-condense config still name the vanilla base model; loom's own connection doc (dated after cutover) records a different, uncensored/abliterated variant now actually being served behind the identical URL/port. Functionally invisible (same façade contract) but a provenance/trust-boundary discrepancy worth flagging given the Loom's stated role as "the email privacy system" / ontology-grounding trust boundary. |

---

## 8. Open PRs / dream branches

| PR | Repo | Branch | State | Ahead/behind `main` | Assessment |
|---|---|---|---|---|---|
| #1-6 (dependabot) | `DreamLab-AI/dream-engine` | `dependabot/github_actions/…` ×5, `dependabot/npm_and_yarn/dev-dependencies-…` | OPEN, unmerged since 2026-08-16 | not measured (bump-only diffs against whatever `main` was on 2026-08-16) | **DRIFTED** — all 6 have sat open ~3 weeks; each is a routine action-version/dev-dep bump (`configure-pages` 5→6, `deploy-pages` 4→5, `setup-node` 4→7, `upload-pages-artifact` 3→5, `codeql-action` 3→4, plus a grouped npm dev-deps bump) and none has been merged, so `dream-engine`'s own CI still runs the *old* pinned action versions in production despite Dependabot having already found the new ones. |
| #10 | `DreamLab-AI/dream-engine` | `dream/ledger-signals-2026-09-06` | DRAFT | **+2 / -0** vs `main` | **CURRENT** — branch is strictly ahead of `main` (unmerged additions only), not behind. |
| #4 | `DreamLab-AI/agentbox` | `dream/sovereign-mesh-2026-09-06` | DRAFT | **+1 / -2** vs `main` | **DRIFTED** — the branch is 2 commits behind current `main` as well as carrying 1 unmerged commit; needs a rebase/merge before it's mergeable cleanly. |

---

## Summary of DRIFTED and MUTABLE findings (for the queen)

**DRIFTED (needs a decision — bump, or document as held):**
1. **`project/agentbox/lib/solid-pod-rs.nix`** pins `solid-pod-rs` at tag `v0.5.0-alpha.3` (rev `87b35a1b`) — the sibling checkout, its own crates' internal version pins, and crates.io are all already at `v0.5.0-alpha.8`. **58 commits / 5 releases behind, no documented hold.** This is the single largest pure version-drift in the estate.
2. **`docs/estate-closeout/2026-09-05/release-manifest.local-draft.json`** is a day stale for VisionClaw (+21 commits) and agentbox (+21 commits) since it was generated; harmless because it's an explicit local draft never enforced, but will mislead anyone reading it as current.
3. **`DreamLab-AI/dream-engine` PRs #1-#6 (dependabot)** — six routine CI/dep bumps open unmerged for ~3 weeks.
4. **`DreamLab-AI/agentbox` PR #4** (`dream/sovereign-mesh-2026-09-06`) is 2 commits behind `main`.
5. **Loom deployed-model identity** — `project/CLAUDE.md` and `agentbox.toml` both still say plain "Qwen3.8-27B"; loom's own `docs/QWEN3.8-CONNECTION.md` addendum (2026-08-18) records the actually-served model as the Heretic-abliterated variant. Same endpoint/port, different model — a silent swap not reflected in the two consumer-facing docs.
6. **agentbox's `nostr-pod-bridge` Cargo path-deps** (`../../../../solid-pod-rs/…`, `../../../../nostr-rust-forum/…`) pull from whatever the sibling checkout on disk happens to be, which for `solid-pod-rs` is *already 5 releases newer* than the Nix-pinned `v0.5.0-alpha.3` used to build the standalone `solid-pod-rs-server` binary — two different snapshots of the same upstream inside one agentbox build.

**MUTABLE (no pin exists at all, by design or by omission):**
1. **GitHub Actions**: VisionFlow, VisionClaw (`project`), nostr-rust-forum, prose-sanitiser, diagram-ir, knowledgeGraph, dream-machine, and 13 of agentbox's 14 workflows use floating major-version tags (`@v4` etc.), not 40-hex SHAs. Only `dreamlab-ai-website` (57/59 lines) and `solid-pod-rs` (33/33) enforce SHA pinning estate-wide; agentbox enforces it in exactly one workflow (`deepsec.yml`).
2. **`loom` has zero `.github/workflows`** — no CI pin policy exists because no CI exists.
3. **`project/agentbox/docker-compose.solid-pods.yml:26`**: `cloudflare/cloudflared:latest` with no digest, while the sibling `docker-compose.unified.yml` pins the *same* image by digest — an internal inconsistency, not just an unpinned image.
4. **`project/docker-compose.unified.yml:290`**: `${LOOM_IMAGE:-loom:rust}` — floating local build tag for the Loom image, no registry/digest at all.
5. **`.github/workflows/fixture-drift.yml`** (VisionFlow) checks out `DreamLab-AI/VisionClaw`'s default branch with no `ref:` — by design (documented in-file as still blocked on missing deploy-key/PAT credentials), so this is a known/self-declared gap rather than a silent one.
6. **`loom/Cargo.toml`**'s `ruvector-core` path-dep and `agentbox`'s two `nostr-pod-bridge` path-deps (item 6 above) have no rev pin of any kind — they track whatever is on disk.

**Healthiest cluster:** the website-kit pin (§5) — five sites (`deploy.yml`, `workers-deploy.yml`, `rust-ci.yml` `KIT_REF`s + `kit-compatibility-record.md` `CANONICAL_KIT_SHA`/`VERSION` + `forum-config/Cargo.toml` exact crate pins) all agree, one lag is explicitly cited as intentional, and a CI `pin-check` job (`ci.yml`) enforces the lockstep on every push. The VisionFlow↔agentbox drift-counter pin (§6, first two rows) is the second-healthiest: allowlist.json and the CI workflow's `ref:` agree byte-for-byte and both match agentbox's actual HEAD.
