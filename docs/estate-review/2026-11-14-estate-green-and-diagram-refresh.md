# Estate health repairs and diagram refresh, 14 November 2026

This is a progress receipt, not completion of the estate-wide alignment.

## Website and CI

The live health data generated at **2026-11-14 22:24:48 UTC** reports 21 repositories: **17 green, zero red, one without a qualifying current run, one exempt and two unreadable**. All six checked web surfaces respond successfully. This snapshot predates any follow-on pull request from the diagram refresh.

- [Forum PR 80](https://github.com/DreamLab-AI/nostr-rust-forum/pull/80) updates SQLite and rustix past the reported advisories and adapts the affected API calls. All twelve PR checks passed. Main CI [43646244735](https://github.com/DreamLab-AI/nostr-rust-forum/actions/runs/43646244735) and Security Audit [43646244775](https://github.com/DreamLab-AI/nostr-rust-forum/actions/runs/43646244775) pass at `7e1f036`.
- [Agentbox PR 48](https://github.com/DreamLab-AI/agentbox/pull/48) fixes the CAD assertion, explicit ComfyUI loopback binding, session test readiness and non-secret environment classification. It reconciles 52 stale or invalid ADRs against their governed source changes. All 161 ADRs and their index pass, with decision, implementation and activation axes preserved. Thirteen applicable PR checks and six triggered main workflows pass at merge `e458b308`. The path-filtered [Manifest validate run 43647747255](https://github.com/DreamLab-AI/agentbox/actions/runs/43647747255) was dispatched separately and passes.
- The [website refresh 43647926396](https://github.com/DreamLab-AI/VisionFlow/actions/runs/43647926396) passes; the published JSON was fetched separately to confirm the result.

The remaining rows have distinct causes. VisionClaw has Actions disabled and no qualifying run at its current head. The collector cannot read the private visionGraph and visiontouch-rs repositories with its current token. WasmVOWL has an intentional frozen-repository exemption. These states have not been relabelled green.

## Diagram source reconciliation

This batch updates 45 topics: `AB-23`, `AB-33`, `AB-38`, `AB-40`, `AB-43`, `DE-01`, `DE-02`, `DE-03`, `DE-04`, `DE-05`, `DE-06`, `DE-11`, `DE-13`, `ES-03`, `ES-90`, `NF-01`, `NF-04`, `NF-07`, `NF-09`, `NF-10`, `NF-12`, `OM-09`, `SP-01`, `SP-02`, `SP-04`, `SP-05`, `SP-07`, `SP-08`, `SP-09`, `SP-10`, `SP-15`, `VC-39`, `VC-42`, `VC-43`, `VC-45`, `VK-01`, `VK-02`, `VK-03`, `VK-05`, `VW-01`, `VW-02`, `VW-03`, `VW-04`, `VW-05`, `VW-06`. Changes were checked against the fetched default-branch revisions, except explicitly retained consumer or historical scopes.

| Area | Reviewed change and resulting documentation |
| --- | --- |
| Forum | SQLite dependency and NIP-19 API changes; source-derived function anchors and the forum consumer lockfile entry. |
| Dream engine and prose tools | Dependency/API maintenance and version metadata; four weak anchors corrected by reading the named definitions and statements. |
| Solid Pod | Alpha.22 release, security lockfile changes, the dated SQLite exception, moved MRC20 function, refund/route anchors and consumer pins. Agentbox source selects alpha.22; the declared VisionClaw requirement remains alpha.21. |
| Visioncraft | Input and expectation ceilings, validation of never-started process receipts, and the affected type/validation anchors. |
| VOWL | Plain-array loading, its validation-before-replacement boundary, changed API anchors and existing feature/builder citation corrections. One new panel records the additive loader. |
| Obsidian | Plugin 0.5.0 and engine v0.8.1 pins; per-block query channels, stable subscriptions, unload guards, bounded model chunk loop, cancellation after review, and reporting a persisted write as applied when indexing fails. |
| Agentbox | Unchanged relay/media implementations against updated governing documents, current sidestr binary source pin, and dream archive/review evidence boundaries. |
| Codescape consumer | VisionClaw `a6d05ed` still pins Codescape `e3b76f6`; four consumer topics retain that producer revision rather than claiming the newer standalone implementation is deployed. |

The full gate passes: **269 topics, 2,041 rendered diagrams and 33,292 resolved citations**, with zero warnings, zero working-tree fallbacks and zero unresolvable citations. Five pre-existing citations to an ignored LaTeX log remain explicitly allowlisted. Semantic review advisories on the changed areas were read against source and either corrected or retained with the supported scope; this is not a claim that a sampled model review proves every diagram's meaning.

Source symbols and exact document passages supplied the revised coordinates. No citation was updated from a diff offset. Historical runtime/test receipts retain their original dates and revisions. This batch runs diagram checks and rendering; it does not repeat the old live GPU, desktop, model, deployment or formal-checker acceptance runs.

## Work still open

The current-source audit still identifies substantial Agentbox and VisionClaw coverage to reconcile. Its raw change list also includes intentional old scopes: OM-10 links to the certificate repair in OM-13, SM-17 links to the successor review-pack contract, and the Codescape consumer topics describe a pinned submodule. NF-28 describes the explicitly unmerged upgrade branch. KG-08 refers to the retained-explorer repair now published in knowledgeGraph PR 5; the follow-up receipt below records its checks. A changed source list is an audit queue, not proof that each historical diagram should be restamped.

The 1 November semantic audit and its later handoff remain historical receipts. This progress record does not promote them to a fresh whole-estate verification. The outstanding CI access/Actions questions, remaining diagram source review and pending local update review prevent close-out.


## Follow-up: published repairs and source alignment

The live health data generated at **2026-11-14 22:38:53 UTC** still reports **17 green, zero red, one without a qualifying run, one exempt and two unreadable**, with six of six surfaces up. [Refresh 43649153768](https://github.com/DreamLab-AI/VisionFlow/actions/runs/43649153768) passes; the served JSON was fetched independently.

[knowledgeGraph PR 5](https://github.com/DreamLab-AI/knowledgeGraph/pull/5) publishes the retained explorer repair and reconciliation at `7815bbc`. All 26 worker tests pass, including a twenty-round transfer regression with both slots occupied. Replacing only the implementation with the old source makes both transfer regressions fail. The main [ADR gate 43648905242](https://github.com/DreamLab-AI/knowledgeGraph/actions/runs/43648905242) passes. KG-07 now closes the publication finding while retaining the unresolved renderer lifecycle decision; KG-08 keeps its original repair revision, now an ancestor of main.

This follow-up aligns nineteen topics: AB-37/42, DW-10, ES-09/10/11, KG-06/07/08 and VC-03/06/13/18/23/24/28/39/41/59. It updates Agentbox manifest, schema, hash-helper and paired Solid alpha.22 source anchors; reviews the additive XR quality-filter constructor and unchanged analytics contract; records the now-merged negative H1 result; and updates the defined XR CI engine to Godot 4.8-beta2 with its archive digest check. It explicitly records that repository Actions are disabled. The full corpus still passes with 269 topics, 2,041 rendered diagrams and 33,293 resolved citations, zero fallbacks or unresolvable citations, and the five existing LaTeX-log exceptions.

The source-change queue has 78 topics after the additional VisionClaw ADR publication and its diagram reconciliation. This includes historical and pinned-consumer scopes described above. It remains an audit queue; neither this batch nor the passing structure/citation gate claims completion of the remaining source review.

[VisionClaw PR 42](https://github.com/DreamLab-AI/VisionClaw/pull/42), merged at `2e7ecc2e7`, reconciles the eight stale ADRs and records H1 in ADR-2174. All 162 ledger records pass locally; all six numeric H1 verdicts match their committed thresholds; the existing production dev-auth gate passes locally. The experiment was not rerun, and repository Actions remain disabled. Historical activation receipts stay dated. The updated diagram batch reads the new ADRs without treating their source review as fresh deployment acceptance.

## Follow-up: settlement, daemon and snapshot boundaries

A further thirteen topics are aligned: AB-07/25/31/32, DW-08, SR-01, VC-17/35/50 and VF-02/03/06/08. The settlement proposal still distinguishes candidate authority rules from implemented testnet components. Its governance passages, manifest gates, schema fields and sealed-chain workflow anchors were read at Agentbox `e458b308`. Historical runtime receipts and the retained sidestr repair revision remain dated. The daemon catalogue now includes the Python supervisor launcher, the podcast job's locking and corpus-tool PATH, and the gated CAD service. The ontology scheduler, website cron and voice switch retain their source-supported behaviour at their revised coordinates.

VF-03.11 now names a dated 14 November 22:38 UTC snapshot: seventeen green repositories, zero red, one without a qualifying run, one exempt and two unreadable. It explains why the offline checker can return `ESTATE-HEALTH-OK` while access and CI gaps remain. The separately fetched 22:49:29 snapshot has the same counts; [refresh 43650062669](https://github.com/DreamLab-AI/VisionFlow/actions/runs/43650062669) passes. This panel is a pinned example, not a claim that its timestamp is always the website's latest.

All 2,041 diagrams rendered successfully, followed by an eleven-diagram render after the daemon changes. The final strict whole-corpus check reports 269 topics and 33,290 resolved citations, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is now 65 topics, including the retained historical and consumer scopes. Remaining source review and the CI access questions are still open.

## Follow-up: XR connection and rendering behaviour

Twenty-one further topics now read VisionClaw `2e7ecc2`: VC-14/27/32/36/37/38/39/40/43/44/45/46/47/48/51/52/53/54/57, VT-07 and the VisionClaw portion of ES-01. The four Codescape consumer topics retain producer `e3b76f6`, which is still the submodule revision at this VisionClaw head.

The overview and connection diagrams distinguish a completed upgrade from a live graph stream. Read progress is measured in bytes, the default watchdog sends pings every three seconds and expires after ten seconds of read silence, and large text parsing refreshes that clock on completion. This is a pump watchdog, not a timeout around every awaited operation. Graph retries shorten their nominal delay by up to twenty percent jitter; the first applied position frame resets the failure count. Resume redials only an open, down connection that is not already dialling. Manual reconnect releases held nodes before closing the old socket. The new VC-44.4 draws these boundaries.

The quality toggle applies a local filter immediately and sends the corresponding server update. The local filter takes precedence over received settings, survives store clear/reconnect, and restores the received filter when switched off. The obsolete claim that filter acknowledgements are unreachable is removed. The visual-budget topic drops the unused edge-ranking path. The engine and deployment descriptions now distinguish the sole 4.8-beta2 runtime and versioned exports from historical 4.6.1 measurements. This review did not rerun a headset, GUT or hardware acceptance session.

The full gate passes: **269 topics, 2,042 rendered diagrams and 33,303 resolved citations**, zero warnings, fallbacks or unresolvable citations, and the same five ignored-log exceptions. Generated indexes and verification metadata are refreshed. The source-change audit has no remaining VisionClaw rows. Its fifty topics / fifty-one repository rows comprise forty Agentbox rows, three VisionFlow rows and eight historical or pinned-consumer rows elsewhere; some Agentbox rows are also deliberately historical. These remain a review queue, not fifty established defects. CI access and activation questions are unchanged.

## Follow-up: session migration, memory observations and governance scope

Six topics now read Agentbox `e458b308`: AB-20/21/39, ES-05/07 and VK-05. AB-39.6 draws the new registered legacy-worktree adoption before destination creation: a foreign session defers migration, the seed's old pane can be closed, and the rename repairs Git and submodule paths while retaining files. The previous private-tmux probes remain dated; the unchanged delivery and API lifetime findings are not closed by this migration.

ES-07.11 separates returned search hits from whole-value retrieves. Their counters use a bounded asynchronous queue and a side table, leaving indexed memory rows untouched. Evidence references remain disabled in the manifest; the implemented path validates references, replaces them on rewrite and withdraws them on deletion without claiming to recall summaries or external copies. AB-20 shows those write/sweep boundaries and the independently gated memory schema. The reference document now agrees on the enforced true-recall floor of 102/120 and distinguishes the 107/120 target. The answer-quality experiment remains a dated, separate receipt, with consolidation inconclusive.

ES-05 records the operator-delegated sweep exception from ADR-2197 and keeps the engine's automatic merging off. Its signer panel now distinguishes implemented optional-key support from independent provisioning. VK-05 corrects a mistaken equivalence between Agentbox's pinned VisionCraft 0.3.0 consumer and the newer standalone upstream review; this does not update the consumer pin or claim image activation.

The full strict/render gate passes **269 topics, 2,044 diagrams and 33,342 resolved citations**, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is forty-five topics / forty-six repository rows: thirty-four Agentbox, four VisionFlow and eight historical or pinned-consumer rows elsewhere. The queue, CI access questions and pending update review remain open. No memory maintenance, recall experiment, provider acceptance, host rebuild or operator tmux switch was performed for this documentation batch.

## Follow-up: management API, adapters and identity references

Six further topics now read Agentbox `e458b308`: AB-03/04/11/26 and ES-02/03. The management API description drops the retired port-9095 AoE service. Boot, identity, compression, validation and settlement references now point to the exact current source passages. The compression panel distinguishes an enabled manifest from live acceptance. Its unused pod/bead compression switches remain recorded. The validation diagram now stops on an error instead of drawing a failed validation into boot.

AB-04.17 shows the external Postgres adapter's atomic delete-and-withdraw statement: it deletes an agentbox-owned entry and replaces matching evidence references in surviving rows with timestamped withdrawal markers. Withdrawal requires a deletion and works even after evidence tracking is disabled. The adapter returns `deleted` from the deletion count, not the SQL statement's row count. This does not recall prose or external copies. The auth and event-forwarding paths are otherwise unchanged; their existing limitations remain open. The settlement amendments are source/test updates and preserve the recorded activation boundaries.

The full strict/render gate passes **269 topics, 2,045 diagrams and 33,353 resolved citations**, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is thirty-nine topics / forty repository rows: twenty-eight Agentbox, four VisionFlow and eight historical or pinned-consumer rows elsewhere. The remaining source review and CI access/Actions questions still prevent close-out. This documentation review did not run a payment, provider or custody acceptance rehearsal.

## Follow-up: custody, pod versions and retained EVM state

Ten topics are reconciled: AB-13/14/16/24/34/36, ES-04/06/08 and SR-03. Agentbox references now read `e458b308`. The custody catalogue records twelve role secrets, thirty-six devuser secrets and 704 non-secret names, and includes both identity-port and session groups. The producer overview recognises the implemented chain URN while retaining the missing node, mirror and asset registrations. Native session provisioning has replaced the old AoE wording; the egress register's older review remains dated. These source checks do not activate role isolation or repeat custody acceptance.

ES-08 distinguishes Agentbox's Solid-pod alpha.22 pin from VisionClaw's alpha.21 pin. Its declared pod source passages are unchanged across those pins. The historical deployment receipts remain historical, and an empty configured mirror list does not establish the state of a running pod.

SR-03 now separates the old audit findings from the implemented repairs at `60259`. Withdrawal matching consumes distinct matching outputs. Snapshot loading checks retained worlds' code hashes, map keys and required code presence before installing state. Time, receipt metadata and external commitments remain outside that validation; retained histories and the verdict-cache limitation remain documented. No EVM experiment or chain activation was performed.

The full strict/render gate passes **269 topics, 2,045 diagrams and 33,369 resolved citations**, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is twenty-nine topics / thirty repository rows: eighteen Agentbox, four VisionFlow and eight historical or pinned-consumer rows elsewhere. Remaining source review and the CI access/Actions questions are still open.

## Follow-up: image composition, sidecars and MCP catalogue

Five topics now read Agentbox `e458b308`: AB-01/06/09/23/30. The build overview uses the actual RuVector 0.3.18 pin and records the older tarball comment as drift. It distinguishes the local QGIS gate from the sidecar proxy, names the imported Blender package, and shows eighteen enabled System One consumers with skill intent disabled. The build command includes security evidence and `--keep-going`. The GUI sidecar now builds from the repository root with its explicit Dockerfile; compose mount references and CLI dispatch anchors follow the current source.

The MCP registry contains twenty-four entries: nine projector-managed, three bespoke and twelve reference entries. Its removed reference entries no longer appear in the overview. The memory panel records returned-hit counting after output shaping. The dream-engine topic distinguishes its draft publication from the external merge decision and shows the operator-delegated sweep exception, including exclusions, protected paths and the prohibition on fabricating human review evidence. The supporting-service anchors are refreshed; their existing behavioural limitations remain recorded.

The full strict/render gate passes **269 topics, 2,045 diagrams and 33,386 resolved citations**, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is twenty-four topics / twenty-five repository rows: thirteen Agentbox, four VisionFlow and eight historical or pinned-consumer rows elsewhere. Source review and CI access/Actions questions remain open. No image rebuild, sidecar restart or provider acceptance run was performed for this batch.

## Follow-up: manifest catalogue and CI boundaries — 15 November

AB-05 and ES-09/10 now read Agentbox `e458b308`. The catalogue census is 129 entries: fourteen surfaces and 115 modules. The overview includes Bend, local QGIS, CAD and its dependent slicer, counts seventeen artefact probes with two required for readiness, and removes the retired Codex MCP probe. It distinguishes the disabled skill-intent consumer, the historical manifest fixture and the compiled setup-server requirement. Vault and Ontomark share the corpus CLI gate.

The CI overview names the generalised operator-tmux guard and the CAD launcher and supervised-Python checks. Their image-dependent behaviour remains outside a plain runner's static evidence. The custody overview records the archived AoE profile files and removed mount, with host-volume removal still a separate operator step after a rebuild. Source declarations are kept separate from fresh image or custody acceptance.

The full strict/render gate passes **269 topics, 2,045 diagrams and 33,394 resolved citations**, with zero warnings, fallbacks or unresolvable citations and the same five ignored-log exceptions. The source-change queue is twenty-one topics / twenty-two repository rows: ten Agentbox, four VisionFlow and eight historical or pinned-consumer rows elsewhere. Remaining source review and CI access/Actions questions remain open.
