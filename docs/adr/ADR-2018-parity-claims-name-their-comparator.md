---
id: ADR-2018
title: A parity claim names its comparator; the JSS plugins forge is a separate comparator, and solid-pod-rs forge features reach consumers only through a release
date: 2026-10-19
decision_status: accepted
implementation_status: partial
activation_status: inactive
supersedes: []
superseded_by: []
verified_commit: 587ea80
verified_paths: []
owner: jjohare
review_trigger: solid-pod-rs publishes a release that carries the forge `gitmark` module or the `mirror` feature; any consumer enables the `forge` or `mirror` feature; the JSS plugins repository moves its forge contract
repo: visionflow
domain: BASELINE-visionflow.md
lineage: "Companion to solid-pod-rs ADR-2009 (the implementation record). VisionFlow owns only the cross-repo claim and the adoption rule; the code is in solid-pod-rs."
---

# ADR-2018 — A parity claim names its comparator; the JSS plugins forge is a separate comparator, and solid-pod-rs forge features reach consumers only through a release

## Context
- The estate states solid-pod-rs parity with JavaScriptSolidServer in several places (README ledger and Mermaid, the Pod row of the compatibility matrix, the solid-pod-rs diagrams), with figures from "~96%" to "~98%" that name no comparator and no row count.
- JSS core moved from `e7e0525` (package `0.0.220`) to `8d68ceb` (`0.0.221`) on 2026-10-18. That delta is three behaviours: a wider CORS expose list, the core git lane deferring to plugin-owned prefixes, and one root WebSocket registration.
- The forge and gitmark work the maintainer has been shipping lives in a different repository, `JavaScriptSolidServer/plugins` (`forge/`, `63a16cb`, 2026-10-19), which the core tree does not contain. Core parity says nothing about it.
- solid-pod-rs-forge ships Phases 0 to 3 of that plugin. It has no NIP-34 event builder, so anchor tags and the kind-30618 mirror need new work, and the mirror is relay egress plus a remote-influenced `git fetch`.
- Consumers pin solid-pod-rs from crates.io with caret ranges (owner decision 2026-10-18). `0.5.0-alpha.15` is released and predates this work; the forge `gitmark` module, the CORS list change and the `mirror` feature sit in solid-pod-rs `[Unreleased]` and are published nowhere.

## Decision
- A parity figure in canon names its comparator (repository, commit, package version) and its tracked row count, and is sourced from solid-pod-rs `PARITY-CHECKLIST.md` through `scripts/parity-check.sh`, never restated from memory. The current figure is 97.3% strict over 244 rows, against JSS `8d68ceb` and, for forge only, `plugins@63a16cb`.
- The JSS `plugins` repository is a second comparator, bounded to the commits a section lists. Canon does not infer full forge parity from it, and a plugin ported to Rust is not claimed on a consumer until that consumer builds a release that contains it with the feature on.
- Forge features reach consumers only through a solid-pod-rs release. VisionClaw, agentbox and the forum take them by their existing caret ranges after a release; none pins a git revision or a path to get them early.
- The `forge` feature and, above all, the `mirror` feature stay default-off in every consumer until the owner enables them in that consumer's own ADR. `mirror` opens relay sockets and runs `git fetch` against configured remotes, so enabling it is an egress decision, not a version bump.
- Agentbox's native pod is the tier that runs `solid-pod-rs-server` as a service; VisionClaw links it as a library with default features off, and the forum's Worker tier cannot run git. Any of them could enable the forge, and none does today. Any change to its `solid-pod-rs-server` feature list is an agentbox ADR and an image rebuild from the host shell, not an edit made from here.

## Consequences
- The README Mermaid node and the compatibility matrix Pod row now carry the sourced figure and the comparator split. Two rendered diagram nodes still say 97.7% (`docs/diagrams/solid-pod-rs/09` line 224 and `10` line 207): the render-source gate compares each topic with its committed render, so they change only with a renderer pass, which needs the sidecar and is not done here.
- The estate-health snapshot can later read the figure instead of a hand-typed number; until then this record is the rule and the checklist is the source.
- Cost: the row count and percentage move whenever the checklist does, so canon prose that quotes them needs the same sweep each time. The forum's ADR-2015 comparator line is refreshed in this change for that reason.
- Follow-on, not done here: a solid-pod-rs release; an agentbox ADR to enable `forge`; an owner decision on `mirror`.

## Verification
Status is partial and inactive. Established against base `587ea80` (VisionFlow, edits uncommitted at the time of writing) by reading the edited canon files against solid-pod-rs `scripts/parity-check.sh` output (244 rows, 218/224 = 97.3%) and `PARITY-CHECKLIST.md` §24 and §25. The solid-pod-rs changes the claim rests on are uncommitted in that repository's working tree at the time of writing (base `f382039`); they are recorded in its ADR-2009, which carries the evidence for the code.
