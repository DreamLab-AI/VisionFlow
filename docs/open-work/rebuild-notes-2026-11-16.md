---
title: Rebuild notes for the unmerged branches of 15 and 16 November 2026
status: live
date: 2026-11-16
type: reference
---

# Rebuild notes: unmerged branches of 15 and 16 November 2026

Seventeen branches hold finished work that is committed locally and not merged, pushed or running. This page says, for each, what it changes when it lands and what has to be rebuilt, set or run for the change to take effect. Item status is kept in the [open-work register](README.md); this page is the landing order and the rebuild steps.

Test results are as each engineer reported them. No suite was re-run by the coordinator, and no branch has had a line-by-line review.

## Order

Merge within a repository in the order listed. Across repositories, the forum's contract branch goes first, because agentbox vendors its files.

| Step | Repository | Branches, in order |
|---|---|---|
| 1 | nostr-rust-forum | `feat/governance-contract-v1`, then `integrate/phase-0-1` |
| 2 | solid-pod-rs | `fix/estate-review-2026-11-15` |
| 3 | agentbox | `fix/estate-review-contracts`, `fix/estate-review-hardening`, `feat/governance-contract-v1` without its last commit, `fix/journal-boundary-and-review-leg` |
| 4 | VisionClaw | `fix/estate-review-2026-11-15`, `feat/golden-thread-2026-11-15` without its last commit, `fix/governance-builders-to-contract`, `feat/store-backup-and-restore` |
| 5 | loom | `fix/estate-review-2026-11-15`, `feat/nightly-promotion` |
| 6 | dream-engine | `fix/estate-review-2026-11-15` |
| 7 | VisionFlow | `fix/estate-review-2026-11-15` |

**Two commits are held and must not be merged:** VisionClaw `86c6227f2` and agentbox `52358a1a7` (decider attribution fails closed). Each is the last commit on its branch. VisionClaw's can land once the bridge forwards the signed decision event; agentbox's is to be rewritten. Merging either as it stands stops human decisions being attributed.

Expect conflicts in two places: agentbox `flake.nix` and `agentbox.toml` (three branches touch them), and decision-record headers in VisionClaw and agentbox (several branches re-stamped `verified_commit`). For a header conflict take the later stamp, then run that repository's `scripts/adr-index-gen` with `--check` after committing.

## nostr-rust-forum

| Branch | Changes | To take effect |
|---|---|---|
| `feat/governance-contract-v1` (`da0eee5`) | Golden files for the governance events under `contracts/governance/v1/`, checked by a test | Merge. No rebuild; nothing at runtime changes |
| `integrate/phase-0-1` (`2316ebe`) | Upgrade Phases 0 to 4 merged with main, including the inbox-key attestation fix; three relay migrations renumbered to 0012 to 0014; blank `audited_by` fields removed. 3,384 tests reported passing | Merge; choose the core crate's next version (the branch's notes say beta.19, which already shipped); deploy the workers; run `wrangler d1 migrations apply` for the relay database. The new features are flagged off until switched on. The README's interop table still says "on the upgrade branch" and needs rewording |

## solid-pod-rs

| Branch | Changes | To take effect |
|---|---|---|
| `fix/estate-review-2026-11-15` (`f68d3e4`) | Token-age check now rejects stale and future tokens; proxy no longer forwards the upstream credential across origins; `/.well-known/apps` omits manifests an anonymous caller cannot read; constant-time compare uses `subtle` | Merge, then cut a release. Consumers pin the crate by version (VisionClaw at alpha.22, forum and website at alpha.20; agentbox by sibling path), so nothing changes for them until the release and a pin bump. Changelog: tokens more than 60 seconds off are now refused |

## agentbox

All four need the image rebuilt (the Nix build, from the host) and the container recreated.

| Branch | Changes | Settings and steps |
|---|---|---|
| `fix/estate-review-contracts` (`2c1f85706`) | Agent-event feed sent in the envelope VisionClaw accepts, with a credential; task routes return the fields VisionClaw requires; relay consumer admits nobody when its allowlist is empty | Set `AGENTBOX_HOST_WS_TOKEN` to a VisionClaw session token, or the feed stays off (it logs that it is disabled). The flake change that passes the allowlist is unbuilt |
| `fix/estate-review-hardening` (`08b24eeab`) | Seven sidecar ports publish on `${AGENTBOX_SIDECAR_BIND:-127.0.0.1}`; judge backend local by default; `dependabot.yml` | Recreate the browser, GUI-tools and XR sidecars. Set `AGENTBOX_SIDECAR_BIND` to a LAN address only if another machine needs them, and add the variable to `.env.example` (a deny rule blocked that edit). Other containers on the same Docker network can still reach the sidecars |
| `feat/governance-contract-v1` (hold `52358a1a7`) | One contract module read from the vendored manifest; producer contract test; elevation publisher reports when it is not configured | Merge after the forum's contract branch. The flake change that vendors the manifest is unbuilt. Eight producer mismatches are recorded as known failures |
| `fix/journal-boundary-and-review-leg` (`8ceceb06e`) | Journal writer appends one whole record per write; verifier accepts declared boundaries; denial-count script fixed; Gemini audit leg replaced by Codex | After the rebuild, copy `docs/reference/audit-journal/boundaries.json` to `~/workspace/events/boundaries.json`; the journal then verifies with two declared boundaries. The Codex leg logs a free skip until `codex` is on the cron path. Keep it off until it is confined and its spend capped (register items `CODEX-LEG-CONFINE`, `JOURNAL-LOCK`) |

## VisionClaw

All four need the dev stack rebuilt from the host shell: `./scripts/launch.sh up dev --no-prompt` for source changes; `./scripts/launch.sh rebuild dev` where a branch changes dependencies (the backup branch adds a crate and a `rusqlite` feature).

| Branch | Changes | Settings and steps |
|---|---|---|
| `fix/estate-review-2026-11-15` (`eafa7d3e4`) | `VISIONCLAW_DEV_MODE` defaults to 0; dev ports 3001 and 4000 bind `${VISIONCLAW_BIND:-127.0.0.1}`; `dependabot.yml`; governance kinds taken from the forum crate | For the headset loop add `VISIONCLAW_BIND=<the headset-facing interface address>` and, if wanted, `VISIONCLAW_DEV_MODE=1` to `.env`. Remove `ALLOW_INSECURE_DEFAULTS=1` from the host `.env`, or the bypass survives the merge. In the same change: mark ADR-2039 accepted with its verified commit and supersede ADR-2108 (owner ruling, 16 November) |
| `feat/golden-thread-2026-11-15` (hold `86c6227f2`) | `stopped` task state; `.agentic-qe` untracked; forum validator run over events VisionClaw builds; `regenerate` replays approved write-backs; decide refuses in production when no forum client is configured | No settings. Replay restores approvals as they were attributed at the time |
| `fix/governance-builders-to-contract` (`11134cc41`) | Proposals built through the forum's own type; decision-elevation cases use the `knowledge_enrichment` category | Stacked on the branch above without its held commit. The voice proposal's event id changes. Open: which risk tier and task properties each builder declares |
| `feat/store-backup-and-restore` (`eb112ba01`) | `visionclaw-backup` crate (backup, verify, restore); periodic in-server Oxigraph checkpoint | Full rebuild. Nothing runs until `ONTOLOGY_CHECKPOINT_DIR` is set and a volume is added for it. Build `visionclaw-backup` on the host (the image ships only the server); mount `/mnt/dell` there; install the example units from `scripts/systemd/` and enable `visionclaw-backup.timer`; run `verify`, then time one real restore. A failed boot checkpoint is now a warning, not an aborted start |

Already on main and waiting only for a rebuild: the three `*_JEV_ALLOW_CLOUD` defaults are 0. After the rebuild, private-content items take their non-judge fallback unless the backend is declared local.

## loom

| Branch | Changes | To take effect |
|---|---|---|
| `fix/estate-review-2026-11-15` (`27c038f`) | Read-only SPARQL guard ignores keywords inside literals, IRIs and comments | Merge and rebuild the image on the Loom's host |
| `feat/nightly-promotion` (`e458827`) | `promote_generation` with a promotion log; nightly build with the visionGraph publish flags; start-up fallback to a retained generation; `/health` reports promotion and fallback | On the Loom's host, as written in `docs/published-generation-reload.md` on the branch: pull; build the three release binaries; run `install -d ~/.local/state/loom/bundles` before recreating the container; rebuild the image; install the nightly units; run the nightly once and the reload once; enable both timers and lingering. About 0.9 GB per retained bundle, five kept |

## dream-engine

| Branch | Changes | To take effect |
|---|---|---|
| `fix/estate-review-2026-11-15` (`6b45ad4`) | Auto-merge is pinned to the reviewed commit | Merge and release as 0.5.0: `DraftPr` gains a public field, which breaks callers. The engine that runs nightly is agentbox's in-tree copy, so the running behaviour changes only when agentbox's pin moves and its image is rebuilt |

## VisionFlow

| Branch | Changes | To take effect |
|---|---|---|
| `fix/estate-review-2026-11-15` (`f4a5db5`) | A stale estate-health snapshot names its cause; the nightly fetches before checking | Merge. Takes effect at the next nightly |

## Not on any branch

- visionGraph's attribution is already on its main; the third-party notices page is still to write.
- `FsPaymentStore` stays until agentbox's `/v1/wallet` exists.
- The upgrade rehearsal (`scripts/activation/flake-update-rehearsal.host.sh`) runs from the host, then the flake-update branch can merge.
