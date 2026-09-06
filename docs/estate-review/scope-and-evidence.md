---
title: Estate scope and evidence method
status: in-progress
date: 2026-09-04
type: reference
---

# Estate scope and evidence method

## Establish identity before interpreting architecture

The [repository map](../architecture/repository-map.md) lists six repositories. The current [root README](../../README.md) also links Loom, knowledgeGraph and dream-engine; it describes WasmVOWL as the corpus explorer. The [closeout](../closeout/final-design.md) includes RuView and RuVector slices. These sources define an initial investigation set, not a census of everything in the container.

There are many unrelated projects, experimental worktrees and generated directories under the workspace. Directory count would overstate the estate. Conversely, restricting the review to the old map would omit the production of the ontology, its delivery to models, and the self-improvement mechanism now central to the vision.

## Located repositories

Paths below are relative to `/home/devuser/workspace`. Commit prefixes identify the initial 2026-09-04 observation; full hashes and tracked-change counts for the initial set are in the [snapshot](evidence/snapshot.json); the added authoring repository is recorded in [knowledge receipts](evidence/knowledge-snapshot.json) and [lineage](evidence/knowledge-boundaries.json). A hash identifies the committed base, **not all bytes in a dirty worktree**. Individual source hashes preserve the inspected evidence where recorded.

| Component | Local path | Observed base | Inclusion basis |
|---|---|---|---|
| VisionFlow | `VisionFlow` | `8cf1a1bf9e4e` | Canon, public account and coordination mechanisms |
| VisionClaw | `project` | `b00c28a0d766` | Runtime knowledge, governance and embodiment substrate |
| agentbox | `project/agentbox` | `dd020a8ad79f` | Agent runtime; `agentbox` resolves to this same directory |
| solid-pod-rs | `solid-pod-rs` | `d6ac7f510868` | Sovereign storage and shared protocol foundation |
| nostr-rust-forum | `nostr-rust-forum` | `f18b471e499b` | Human decision surface and relay kit |
| dreamlab-ai-website | `dreamlab-ai-website` | `7e243741c8ea` | Branded deployment and operator surface |
| Loom | `loom` | `8cdef36bb571` | Context assembly and model-facing grounding |
| knowledgeGraph | `knowledgeGraph` | `7bbf0aae2b60` | Corpus, compilation pipeline and publication |
| Logseq history | `logseq` → `project4` | `4f233f321097` | Historical publisher retained for pre-split provenance |
| visionGraph | `visionGraph` | `fabcdbcc9fd67` | Current authored vault and publisher, bound by agentbox manifest |
| WasmVOWL | `WasmVOWL` | `36105cc3ad04` | Separate explorer source to compare with the corpus's embedded copy |
| dream-engine | `dream-machine` | `7c30573a2d73` | Local checkout's origin identifies DreamLab's dream-engine fork |
| RuVector | `ruvector` | `677b2475409c` | Memory/vector dependency; Loom directly names its core crate by sibling path |
| RuView | `RuView` | `b48ab7dada50` | Sensing extension included in the earlier closeout |

RuView is on `closeout/2026-07-03`, RuVector on `report/agentbox-field-2026-07`, and WasmVOWL on `master`; Logseq reports `obsidian`; the other listed checkouts report `main`. This matters when comparing local fixes with release claims. Local branch names do not prove that commits were merged or deployed. VisionClaw's `origin` lookup returned no value; its local path is established, but this pass does not assert a remote-tracking relationship.

The directories `logseq-publisher` and `logseq-publisher-rust` did not resolve a Git HEAD during initial inspection. Their names alone do not establish separate current repositories. The [knowledge-production trace](knowledge-production.md) established the extended Logseq pipeline; the subsequent [vault-transition trace](authored-vault-transition.md) identifies visionGraph as its current successor. Embedded explorer source is distinct from the standalone WasmVOWL checkout. The legacy publisher directories still require an explicit historical/dependency disposition. Upstream projects cited for lineage or industry comparison, such as Buzz and JSS, are not automatically DreamLab-owned components; inspect vendored or locally consumed code where it affects the estate's behaviour.

## Evidence rules

1. **Intent:** PRDs, ADRs and book chapters establish the desired behaviour and its rationale. They do not prove that behaviour executes.
2. **Implementation:** read the path from entry point through decision, mutation and acknowledgement. A named struct, route or tool is weaker evidence than a wired call path.
3. **Verification:** distinguish source inspection, a local executable probe, a test using doubles, a real service interaction, and a cross-repository runtime journey. Preserve failing results too.
4. **Delivery:** a local commit or successful build does not establish deployment. Release and deployment assertions need their own pinned evidence.
5. **Absence:** scope negative findings to the search and paths inspected. “Not found in this path” is different from “does not exist anywhere”.
6. **Inference:** identify architectural judgements and proposed priorities as analysis. They must not acquire the status of measured results through repetition.

The [book's canon chapter](../../presentation/report/chapters/13-visionflow-canon.tex) sets out the historical/planned/scaffolded/standalone/integrated/federation-verified/released vocabulary. This review preserves those distinctions. It records evidence depth separately so that, for example, a historical claim of integration can be revisited without being silently promoted by a fresh date.

## Reproduction and limits

Run `python3 docs/estate-review/evidence/collect.py` from the VisionFlow checkout. It records repository identities and selected SHA-256 source hashes, runs the local count checker, and copies the four dream evaluators into a temporary fixture with deliberately invalid content. It writes a new snapshot beside the collector. Preserve the previous snapshot if comparing observations across dates.

The collector's success means it collected receipts, not that the checks passed. Read each recorded exit code and output. It does not start services, build sibling repositories, query production, inspect credentials, or claim federation verification. Dirty counts exclude untracked files; zero is not a certification of a clean checkout. Source hashes provide change detection, not archived file contents.
