# Estate closeout execution — 2026-09-07

This executes the [source audit](2026-09-07-estate-audit.md) and [sprint](closeout/2026-09-07-sprint.md). The [master register](../../../project/docs/TODO-unified.md) owns the current disposition of the 61 distinct items open at execution start. Source implementation, deployment and system acceptance remain separate. RuView is excluded at the user's direction.

The execution repairs demonstrated defects in both ontology publishers, trust and governance transactions, release authentication, GPU analytics, identity boot, retrieval verification, recovery and diagram tooling. It also corrects stale task premises and records deliberate disabled profiles. Detailed changes, original failures, passing tests and remaining boundaries are in the [VisionClaw](closeout/2026-09-07-execution-visionclaw.md), [Agentbox/Loom](closeout/2026-09-07-execution-agentbox.md), [federation/publishers](closeout/2026-09-07-execution-federation.md) and [canon/workflow](closeout/2026-09-07-execution-canon.md) reports.

The reconciled register contains **38 closed, 21 blocked and two excluded items**. No item remains merely marked in progress: the HP build finished, its rollout failed readiness, and rollback restored the original healthy runtime. The remaining rows identify source, authority, deployment, device or observation problems.

## Browser verification

The original [seven-surface receipt](evidence/execution-2026-09-07/browser-final/receipt.json) recorded 34 passing checks, but its explorer mobile/search assertions were later found too weak. It remains historical evidence, not final acceptance of those assertions. The corrected harness uses native CDP input, checks positive search results and compares actual document width to the requested375px. It disables cache only in its own tabs and waits for observer callbacks. The [published three-surface pass](evidence/execution-2026-09-07/browser-published-observer-ready/receipt.json) verifies the canon, explorer and notes without injected CSS. A subsequent complete-estate pass exposed a Dream homepage table overflow; commit `a6c560a` scopes the ledger-only overflow rule and is published. The [final seven-surface run](evidence/execution-2026-09-07/browser-estate-final-v2/receipt.json) passes **35/35 checks** without CSS injection. The local VisionFlow candidate also passes the [26-check WebGL/reduced-motion/fallback suite](evidence/execution-2026-09-07/visionflow-browser/website-browser-receipt.json).

The notes failure was a reproducible mobile overflow. Early three-second readiness checks also ran before notes/forum had loaded; later readiness polling corrects that interpretation. Original and intermediate receipts remain available. No authenticated human-decision, phone or headset journey is inferred from these public browser checks.

## Native XR visual upgrade

The [Godot visual upgrade](closeout/2026-09-07-xr-visual-upgrade.md) adds the spatial environment, graph material/targeting improvements, shared HUD/radial styling and operator comfort controls. Actual Godot screenshots and the integrated native suite supplement the public browser checks; they do not close headset or Android acceptance.

## Evidence and publication boundary

The [execution evidence](evidence/execution-2026-09-07/) and [Agentbox/runtime evidence](closeout/execution-2026-09-07/) retain raw logs, source identities, negative controls and runtime probes. The [ADR graph](evidence/execution-2026-09-07/adr-graph.json) has repository-qualified keys and typed references, preserving lineage separately from supersession. Historical counts and audit receipts describe their dated snapshots; implementation changes do not rewrite those snapshots.

Commits use explicit paths. Pre-existing changes required by tested implementations are documented in their lane reports; unrelated corpus deletions, authoring state, credentials, local databases and temporary files are preserved. Another editor continues changing diagrams, so current source hashes and final validation results are required in addition to commit labels. No force push or destructive history cleanup is part of this execution.

## Problems that prevent complete system acceptance

The live Loom service does not yet expose the new identity contract and reports mismatched graph/semantic generations. VisionClaw and ComfyUI runtime endpoints are unavailable from this container. Custody migration and the scoped private-repository CI credential are not supplied. Phone/headset journeys and the defined multi-day recall observation window cannot be replaced by fixtures. Historical compound decision routing is now reconciled in the [disposition annex](closeout/2026-09-07-historical-disposition-annex.md). Current diagram validation is repeated after the XR implementation changes.

The supported disabled profile remains explicit for mobile/relay exposure and other held surfaces. Reopening one requires its relevant admission, custody and runtime evidence. Source closures do not turn those profiles on.
