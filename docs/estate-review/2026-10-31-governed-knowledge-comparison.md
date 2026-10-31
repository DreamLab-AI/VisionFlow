---
title: Governed knowledge alternatives and a user evaluation protocol
status: in-progress
date: 2026-10-31
type: explanation
---

# Governed knowledge alternatives and a user evaluation protocol

VisionFlow needs to demonstrate a useful, reliable change workflow before its breadth becomes a persuasive product argument. Graph memory, ontology-guided extraction, provenance and human review already appear in alternatives' documentation. The opportunity assessed here is a narrower combination: a portable authored corpus, review of a precise proposed change, and a durable, independently inspectable result. This is a positioning hypothesis, not a finding of market uniqueness or product readiness.

This extends the [first market comparison](2026-10-31-market-and-audit-gaps.md), which covered document assistants and graph/agent frameworks. Primary documentation was consulted on 31 October 2026. The four alternatives below were not installed or benchmarked. Documentation establishes an advertised contract; it does not establish correct execution, comparative quality, operating cost or absence of other features. Moving documentation URLs must be pinned again when an evaluation starts.

## Compare the job and the delivered system

| Alternative | Documented capability relevant to this review | Useful comparison with VisionFlow | Boundary of this evidence |
|---|---|---|---|
| Graphiti | Incremental temporal graphs built from episodes, with hybrid retrieval; domain entity and edge types defined through Pydantic models. [Overview](https://help.getzep.com/graphiti/getting-started/welcome), [custom types, v4](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types/). | Maintaining changing facts for an agent, retrieving current evidence and assessing domain modelling effort. | A framework is a component baseline. Typed extraction alone does not establish equivalence to OWL reasoning, a human approval workflow or a complete product. Those equivalences were not tested. |
| Cognee | An API sequence adds data, builds a knowledge graph and searches it. Dataset endpoints expose deletion, requiring the dataset's delete permission. [API reference](https://docs.cognee.ai/api-reference/introduction). | Time and effort to ingest, retrieve, correct and remove a team's source material. | API availability does not prove complete removal from every derived store, cache or answer. Measure this explicitly; do not infer a defect from the lack of an executed test here. |
| TrustGraph | Ontology RAG selects relevant ontology components to guide extraction. Its explanation guide traces retrieval events and graph edges towards source text using provenance relationships. [Retrieval architecture](https://docs.trustgraph.ai/architecture/retrieval.html), [explanation guide](https://docs.trustgraph.ai/guides/building/explainable-ai.html). | Ontology authoring effort, answer quality and the usefulness of source explanations. | Traceability and source location do not by themselves establish that a source entails a claim. No vendor quality or latency figures are adopted here. |
| TopQuadrant TQ Data Foundation | Isolated working copies, authorised review/approval and committing changes to production collections. Workflows can include multiple collections, and templates govern transitions. [Workflow guide](https://docs.topquadrant.com/latest/user_guide/workflows/index.html). | How effectively people inspect and approve knowledge changes, including changes spanning several objects. | A documented multi-collection commit does not establish crash atomicity. Direct editing also exists for sufficiently privileged users; this is not evidence that every write requires review. [Reference-data guide](https://docs.topquadrant.com/latest/quick_start_guides/edg_reference_data_management/index.html). |

Deployment scope matters. Graphiti explicitly describes its reference MCP/REST servers as non-production, without authentication or authorisation; a production service must provide those controls, or users can consider Zep. Count the integration work in a framework evaluation. Do not attribute that reference-server limitation to the managed Zep product. [Graphiti repository, reference servers](https://github.com/getzep/graphiti#reference-mcp-and-rest-servers).

Graphs, ontologies, provenance and approval screens offer limited differentiation on their own. This comparison does not establish that any listed alternative reproduces the estate's exact design or that VisionFlow should replace its components with one of them.

## What the estate evidence changes

These are findings from the preceding pinned local audits, not newly executed competitor results or observations of deployed services.

| User expectation | Local evidence | Consequence for a pilot |
|---|---|---|
| A quoted source supports the proposed fact | A positive `requires` relation survived checks when the exact source said it did **not** require the object. [Ingest audit](2026-10-31-external-source-ingest.md). | Score source entailment separately from citation presence. Include negation, attribution, conditions and conflicting versions. |
| The reviewed bundle remains identifiable | Endpoint changes reused a bundle ID; a budget-exhausted rerun removed a previous manifest. [Ingest audit](2026-10-31-external-source-ingest.md). | Preserve the exact review input, configuration and output. An identifier must resolve to the artefact the reviewer saw. |
| An expired request cannot change the corpus | A signed consumer fixture accepted an expired proposal. Closing the forum row does not itself veto that consumer. [Ingress audit](2026-10-31-governance-ingress.md). | Judge expiry at the mutation boundary as well as in the UI. Keep this local result separate from a full deployed bypass claim. |
| Restart preserves pending work | A checkpoint included an unfinished event; restoring it suppressed redispatch. [Ingress audit](2026-10-31-governance-ingress.md). | Inject a restart with one completed and one pending decision; inspect both effects and receipts. |
| A stable concept keeps its identity | Proposal construction and direct apply admitted a changed resource IRI. [Construction audit](2026-10-31-proposal-construction.md). | Enforce identity continuity or an explicit migration workflow before presenting stable identity as a guarantee. |
| One group decision can reach one coherent result | The producer signed grouped proposals that the real adapter refused before writer invocation. [Construction audit](2026-10-31-proposal-construction.md). | Either support the group through application/recovery, or declare the scope unsupported before requesting review. Splitting it silently changes the decision. |

These failures belong near the beginning of a demonstration, where they can change its design. Adding more connectors or a richer graph display would not resolve them. This is a prioritisation judgement derived from the observed workflow boundaries.

## A bounded pilot with a falsifiable benefit

Engineering and research teams that repeatedly change shared definitions are a plausible audience, particularly when they need to explain why a change was accepted. Give pilot participants a concrete task: import a small authorised corpus, answer a sourced question, correct an outdated relationship, review the exact change, apply it, restart the service and retrieve the corrected answer with its decision record.

Use three comparisons rather than one league table. Compare document lookup with the document-assistant baselines in the first report; compare changing graph facts with Graphiti/Cognee/TrustGraph; compare review and maintenance work with an established governance workflow and a simple Git review plus document-search baseline. Eligibility follows the job: a retrieval library without a review UI is not automatically a failed governance product. Record the additional application work needed to make it serve that job.

The [machine-readable evaluation specification](evidence/2026-10-31/governed-knowledge-evaluation.json) defines ten cases, synthetic source text, expected outcomes, evidence to retain and links to prior local counterexamples. All cases are `not_run`. It is a protocol for a future comparative run, not an executable benchmark or a claim that those systems passed or failed.

Before executing it:

1. Freeze each candidate's version, deployment, adapters, configuration and capability declaration. Record which controls are native and which were added for the pilot. Preserve corpus bytes and hashes, model/embedding identifiers and prompts. Exclude private estate material from external services.
2. Separate deterministic control tests from stochastic answer tests. Use the same corpus, held-out questions and budgets where configurable. Report a product's fixed/default model as a separate configuration when matching is impossible. Record repeated trials and seeds where available; do not select the best answer after seeing the results.
3. Have reviewers score answer correctness and source entailment against a prepared answer key, blind to the candidate where practical. Counterbalance task order for human workflow comparisons. Report disagreements, failures and unsupported cases alongside successes.
4. Record active author/reviewer/operator minutes, elapsed time, interventions, token use, actual billed cost where available and retained storage. Keep one-off integration effort separate from recurring task effort. Missing billing data is unknown cost, not zero cost.
5. Retain the request, exact reviewed artefact, approval, before/after state, retrieval evidence and recovery outcome together. Redact secrets without discarding identifiers needed to connect the records. Report component-only tests separately from the whole journey.

Measure utility primarily through **active human minutes per correctly completed, durable knowledge correction**, with completion rate reported beside it. Count time spent on failed attempts in the numerator; report the ratio as undefined when no corrections succeed. A fast wrong answer or lost approval is not a successful task. Source entailment, false acceptance, unsupported scope and operator effort remain separate measures, so a composite score cannot conceal a broken control.

For the deterministic pilot fixtures, unauthorised mutation, an accepted expired approval, lost pending work and a partial group reported as complete are hard failures. Unsupported scope identified before review is a safe refusal but does not fulfil the grouped-edit job. Numerical quality, latency and effort thresholds must be declared with pilot participants before outcomes are known; this review invents no adoption or performance target.

## Development choices and remaining coverage

Prioritise one recoverable correction journey with consistent identity, scope and expiry enforcement. Make evidence understandable: show the supporting passage, the actual proposed relation, what changed since review, and whether application completed. Then measure whether domain modelling and independent checks save enough review effort to justify their authoring and operating costs. Portability should be tested by exporting and reconstructing an inspectable corpus and decision trail; a file format or signed identifier alone is insufficient proof of that outcome.

Defer a broad enterprise-replacement claim until the bounded pilot demonstrates recurring value. Spatial interaction remains a separate usability hypothesis requiring its own desktop comparison. Formal-checker activation, Solid interoperability, deployment operations, accessibility, commercial terms and the wider competitive field remain open. This pass adds a sourced product assessment and a concrete evaluation specification; it closes no unexamined ADR or diagram topic and makes no claim that the estate audit is complete.

Validation in this pass covers JSON structure, case/source identifiers, local evidence links, report navigation and whitespace. No competitor benchmark, live approval, production write or product-code test was run for this documentation-only change.
