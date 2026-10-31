---
title: Market comparison and audit evidence gaps
status: in-progress
date: 2026-10-31
type: explanation
---

# Market comparison and audit evidence gaps

This is a bounded continuation of the estate review at VisionFlow `4025263`, not an estate-wide completion certificate. Product documentation was searched on 31 October 2026; competitor features below are documented capabilities, not independently tested performance. No pricing, market-share or superiority claim is established.

## Position and competing user choices

VisionFlow's current README presents a composition of knowledge authoring, reasoning, agent execution, human review and self-owned data. The most plausible differentiated offer is a traceable, governed knowledge-change workflow spanning these components. That is an inference about product direction, not proof of customer demand or of a fully integrated deployment.

| User job and alternative | Documented capability | Consequence for VisionFlow |
|---|---|---|
| Ask questions of team documents: Open WebUI | Upload collections, attach knowledge to conversations, retrieve passages and inspect source references. Processing location depends on configured providers. [Knowledge documentation](https://docs.openwebui.com/features/workspace/knowledge/) | Local models and document chat alone provide little differentiation. Compare onboarding time, permission handling and source correction with Loom plus the estate UI. |
| Self-host a workspace assistant: AnythingLLM | Docker deployment supports single/multiple users, local models, retrieval and agents. [Installation](https://docs.anythingllm.com/installation-docker/overview) | A user should be able to reach a useful first answer without installing every estate repository. Measure installation and recovery effort, rather than inferring ease from architecture. |
| Build agents that resume after human review: LangGraph | Checkpointers retain thread state; stores retain cross-thread information. In-memory checkpointers lose state on restart. [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Human review and memory are established competing capabilities. Test the estate's distinctive signed decision, authorised application and durable receipt together, including restart and duplicate delivery. |
| Answer questions across a graph: Microsoft GraphRAG | Local search combines extracted graph data and text; global search uses community reports; basic vector search supplies a comparison. [Query overview](https://microsoft.github.io/graphrag/query/overview/) | Evaluate formal ontology grounding against both graph and vector baselines. Graph structure itself is not evidence of better answers. |
| Construct and query a graph in application code: Neo4j GraphRAG | Official Python tooling supports graph construction and retrieval. [Package documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/index.html) | Explain when reviewed vocabulary and reasoning justify the authoring burden. Compare correction cost as well as retrieval quality. |

These alternatives address different layers. A framework comparison cannot establish that a turnkey application is harder to operate; a graph library comparison cannot establish equivalent identity or governance guarantees. The table identifies concrete substitutes for user jobs, not interchangeable architectures. Solid interoperability, ontology authoring tools and spatial collaboration products still need their own comparison pass.

## Utility priorities and falsifiable acceptance

The following priorities are recommendations. Numerical thresholds should be agreed before evaluation; none is a measured result.

1. **Ship a small governed knowledge journey.** Let a research or engineering team import a bounded corpus, ask a sourced question, propose a correction, review the exact diff, apply it and retrieve the new version. Record corpus and policy versions and show the resulting receipt. Candidate components: Ontomark/visionGraph, Loom, agentbox, forum and storage. Acceptance must include rejection, stale approval, duplicate delivery and restart between approval and application.
2. **Offer an independent grounding entry point.** Package Loom with a reproducible sample corpus and a minimal consumer. Record time to first useful answer, operator interventions, deletion propagation, source fidelity and resource usage. Compare with Open WebUI/AnythingLLM on the same documents and model.
3. **Make reasoning benefits measurable.** Use held-out questions covering entity lookup, multi-hop relations, contradictions and unanswerable requests. Compare basic vector retrieval, a graph baseline and ontology-assisted retrieval with identical model/context budgets. Report failures and authoring/indexing costs; evaluate on an external corpus as well as the synthetic estate testbed.
4. **Make authority legible.** Show who proposed, who approved, what capability was used, what bytes changed, and whether the receipt survived recovery. A shared identifier or an approval badge alone is insufficient. Use permission revocation and replay attempts as acceptance cases.
5. **Test spatial interaction as an optional benefit.** Compare desktop and XR on the same graph comprehension/correction tasks. Measure completion, mistakes, fatigue and accessibility. Keep headset installation outside the first document-answer journey until the benefit is demonstrated.

A credible initial audience is teams with recurring, reviewable knowledge changes and a reason to retain control of their data. Consumer general-purpose chat and broad enterprise replacement are less defensible positioning until the narrower workflow demonstrates reliability and lower total effort. Validate this hypothesis through observed pilot tasks, not feature-count surveys.

## Audit of the audit machinery

| Finding | Evidence at 4025263 | Impact and next acceptance |
|---|---|---|
| Snapshot omits governing-only files | `scripts/capture-diagram-source-snapshot.cjs:25` iterates `meta.sources`; lines 27–29 skip RuView; lines 32–35 hash files but only mark directories | Stable snapshots cover only included bytes. Include governing paths in a collector repair; prove a governing-only edit changes the snapshot. Retain explicit directory and excluded-repository limits. |
| Render parity wording overclaims provenance | `docs/diagrams/visionflow/06-diagram-tooling.md` VF-06.14 and `scripts/check-diagram-render-sources.py:38-45` | Equality with an `.mmd` input, an SVG hash and a width check do not prove that SVG was rendered from that input. A fresh successful render is stronger evidence. |
| ADR and diagram status disagree | VF-06.1 NOTE4 describes outstanding Pele surfaces; `docs/adr/ADR-2020-pele-renders-every-estate-diagram.md`, Close-out 2026-10-31, records seven-surface local pin agreement and 1,526 renders | Distinguish the diagram's historical revision from current local implementation. The ADR still does not establish running-image activation. A dated correction was added to VF-06; full-topic semantic refresh remains open. |
| Triage is not semantic coverage | Existing untracked `docs/review/2026-10-31-triage.md` lists 115/115 topics for re-authoring, predominantly model timeouts or cap exclusions | Preserve this as a queue, not proof that every topic is wrong or examined. Review sources and governing documents directly. The untracked triage was not modified. |
| Scope table is historical | `scope-and-evidence.md` lists fourteen September repository identities; current root README additionally names Ontomark, its plugin, sidestr-rs and Codescape | Rebuild a deduplicated current scope ledger from explicit links and consumed paths before claiming estate coverage. Do not sweep unrelated workspace repositories. |

## Verification and remaining work

The snapshot collector was read in full. Its file has no diff from VF-06's declared `4d361b58` through `4025263`, so the bounded source finding applies at both revisions. The collector was not executed and no product code was changed.

VF-06's fifteen diagrams passed the local Pele render and structural check after the wording correction:

```
node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --render --only visionflow/06-
parsed 1 topic files, 15 mermaid diagrams
rendered 15/15 diagrams via pele
```

`--no-source-paths` deliberately makes this a structural/render check, not a citation audit. The repository generator rejects the skill's newer `--report` flag with exit 2; rerunning with supported flags passed. This tooling-version difference must be accounted for in the full audit.

The remaining objective includes current repository/ADR inventories and section dispositions, semantic review of every diagram, per-repository source and adversarial checks, broader product comparisons, and consolidated priorities. Earlier audit chapters and ADR closeout receipts must be revalidated before being described as current. This pass neither certifies the whole estate nor requires all implementation gaps to be repaired before the assessment can finish.
