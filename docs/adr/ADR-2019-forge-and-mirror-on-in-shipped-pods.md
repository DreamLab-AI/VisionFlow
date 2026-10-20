---
id: ADR-2019
title: The solid-pod-rs forge and mirror are on in shipped pods; open edit, mainnet and any follow list stay opt-in
date: 2026-10-20
decision_status: accepted
implementation_status: partial
activation_status: inactive
supersedes: []
superseded_by: []
verified_commit: 4027542
verified_paths: []
owner: jjohare
review_trigger: solid-pod-rs publishes a release carrying `forge-full`; agentbox or VisionClaw builds it; the owner changes the open-edit, mainnet or follow-list defaults; JSS plugins/forge changes its contract
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
