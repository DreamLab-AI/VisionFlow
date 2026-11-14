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
