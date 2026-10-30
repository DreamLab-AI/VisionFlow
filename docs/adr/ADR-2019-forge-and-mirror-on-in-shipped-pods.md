---
id: ADR-2019
title: The solid-pod-rs forge and mirror are on in shipped pods; open edit, mainnet and any follow list stay opt-in
date: 2026-10-20
decision_status: accepted
implementation_status: partial
activation_status: live
supersedes: []
superseded_by: []
verified_commit: 7d55032
verified_paths: []
owner: jjohare
review_trigger: the shipped agentbox pod sets FORGE_MIRROR_FOLLOW so the kind-30618 mirror starts (completes this record); VisionClaw gains a `/forge` adapter; the owner changes the open-edit, mainnet or follow-list defaults; JSS plugins/forge changes its contract
repo: visionflow
domain: BASELINE-visionflow.md
lineage: "Amends ADR-2018 decision bullet 4 (forge and mirror default-off in every consumer). Companion to solid-pod-rs ADR-2010 (the implementation record) and consumer ADRs in agentbox and VisionClaw."
---

# ADR-2019 — The solid-pod-rs forge and mirror are on in shipped pods; open edit, mainnet and any follow list stay opt-in

## Context
- ADR-2018 said the `forge` and `mirror` features stay default-off in every consumer until the owner enables them in that consumer's own ADR. On 2026-10-19 the owner decided to enable them, and to have the Rust crates reach feature parity with the JSS `plugins/forge` plugin first.
- The Rust forge now covers that plugin (solid-pod-rs checklist rows 245 to 264), the mirror's accepted risks were closed, and an independent security review's findings were fixed. solid-pod-rs ADR-2010 records the implementation.
- Consumers differ. Agentbox builds `solid-pod-rs-server` as a service from a pinned source revision. VisionClaw does not run that server: it embeds the core library behind its own `/solid` handler, so a version bump alone gives it neither the CORS fix nor the forge. The forum's Worker tier cannot run git.

## Decision
- Shipped agentbox pods build the server with `forge-full` and start the forge and the kind-30618 mirror at boot. VisionClaw hosts the forge only after a `/forge` adapter exists there, decided in VisionClaw's own ADR. The forum is unchanged.
- "On" means available and started, not acting. The mirror follows nothing until relays and `<pubkey>/<repo>` entries are configured; the shipped follow list is the operator's own key and nothing external. Announce publishes only to configured relays.
- These stay opt-in and unset in every shipped default: `open_edit` (JSS `openEdit` skips authorisation, so any caller can edit any repository), `allow_mainnet` (the default chain is a testnet; the forge never broadcasts or spends), `private_repos`, and any external follow entry.
- ADR-2018 bullet 4 is superseded by this record. Its other bullets stand: a parity claim names its comparator, and a consumer takes forge features only through a solid-pod-rs release.

## Consequences
- Agentbox gains relay egress and a remote-influenced `git fetch` in its default build, inert until configured. Its manifest and a source-revision bump are agentbox's own ADR and an image rebuild from the host shell.
- The compatibility matrix and the README state the parity figure from the checklist script (264 rows, 99.2% strict), with the comparator split unchanged.
- Cost: the owner now carries the configuration of the follow list and relays per deployment; there is no shipped default that follows anyone else's repositories.

## Verification
Status is partial and inactive: nothing is released or built by a consumer yet. Established against base `4027542` (VisionFlow) and solid-pod-rs `8dd5dc1` by `scripts/parity-check.sh` (264 rows) and solid-pod-rs ADR-2010, which carries the test and review evidence.

## Close-out 2026-10-30

`activation_status` inactive → live for the forge; `implementation_status` stays partial because the kind-30618 mirror does not start in the shipped pod. Evidence executed today:

- Release: crates.io `solid-pod-rs-server` 0.5.0-alpha.18 (published 2026-10-29) lists the `forge-full`, `forge-mirror` and `forge-announce` features. Agentbox pins solid-pod-rs `76ab7aab` (tag `v0.5.0-alpha.18`) in `lib/solid-pod-rs.nix:38,54` and adds `forge-full` by default in `flake.nix` (`solidPodRsExtraFeatures`, `(solidPodRsCfg.forge or {}).enable or true`).
- Runtime: the running agentbox pod is `/nix/store/…-solid-pod-rs-server-0.5.0-alpha.18/bin/solid-pod-rs-server`. `/var/log/solid-pod.log` records `forge: started` at 2026-10-30T14:22:46Z, and `GET http://127.0.0.1:8484/forge` answers 200 with the forge page.
- Opt-in defaults hold: the pod's environment carries `FORGE_OPEN_EDIT=0`, `FORGE_ALLOW_MAINNET=0` and `FORGE_PRIVATE_REPOS=0`, with chain `txbt4`. Announce emission is logged `DISABLED (set announce_relays to opt in)`.
- Gap: `FORGE_MIRROR_FOLLOW` is empty in the pod environment. solid-pod-rs `crates/solid-pod-rs-server/src/lib.rs:4965-4981` (`mirror_config`) returns `None` without a follow list, and `:5022-5027` starts the mirror only on `Some`. `grep -c "mirror on kind-30618" /var/log/solid-pod.log` returns 0. The Decision's shipped follow list (the operator's own key) is therefore not wired, so the mirror is compiled in but not started. Closing this is an agentbox change (populate the follow list in the pod launcher), not a VisionFlow one.
