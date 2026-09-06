---
title: Agent grounding and governance boundaries
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Agent grounding and governance boundaries

Agentbox implements much of the connective work that turns the estate into an agent environment: it routes ontology reads, builds proposals, classifies consequential actions, receives signed decisions and proxies approved enrichment decisions. Its strongest design choices are explicit failure handling, shared policy helpers and observable receipts. The gaps appear where a mode change, cache hit or asynchronous hand-off changes the effective contract.

This chapter follows those paths in the current checkout. It does not yet assess the entire Nix container, memory service, process lifecycle, payments, privacy filtering or skill estate. [Receipts](evidence/agent-snapshot.json) preserve source hashes, test results and [synthetic probes](evidence/agent-boundary-probes.cjs). Tests used temporary files and fake relay/signature dependencies; they establish local logic, not live federation or cryptographic interoperability.

## Grounding is routed through several different contracts

[ontology-bridge.js](../../../project/agentbox/mcp/servers/ontology-bridge.js) exposes tools for ontology health, discovery, class access, graph queries, validation, proposal and bounded retrieval. Most remote tools target VisionClaw. The shared [retrieval library](../../../project/agentbox/mcp/servers/lib/ontology-retrieval.js) instead chooses Loom when `LOOM_FACADE_URL` is configured, unless a VisionClaw transport is explicitly injected.

The Loom path calls `/loom/search` for seeds and `/loom/sparql` for outgoing and child relationships. It does **not** call Loom's `/loom/scaffold` or chat endpoint. Agentbox serialises the returned terms into its own bounded Turtle context. Consequently, Loom's scaffold confidence policy, verbatim response behaviour and model-answer benchmark are not evidence for this agent-tool path. It needs its own retrieval and grounding checks.

The VisionClaw expansion path selects asserted or inferred named graphs according to the request. The Loom expansion helper queries its merged graph and does not consume the requested provenance selector, yet the shared result carries the request's `provenance` label. That label is therefore a requested scope, not proof that the Loom helper isolated asserted from inferred statements.

**Assessment:** presenting one tool name is useful for agents, but backend substitution must preserve the properties callers rely on: provenance scope, generation identity, domain filters, budgets and error meaning. If a backend cannot preserve a property, the result should state that limitation explicitly.

## Budget and domain checks can be bypassed by the cache

The library applies maturity and domain filters to seeds, expands relationships and clamps the output to a model-tier budget. Its existing **20 Node tests pass**, including direct budget checks, TTL behaviour, expansion and telemetry. Separate probes found two gaps in the composed path.

A request for an AI-domain result with a 1,000-token override produced an 830-token context. Repeating the query and tier with domain `robotics` and a 50-token override returned the same AI seed and all 830 tokens with `cache_hit: true`. The cache key includes query, tier, depth, mode, provenance and full-mode flag, but excludes domain and `max_tokens`; a hit returns the stored response before either constraint is reconsidered.

A second fixture made seed retrieval succeed and expansion throw. The library correctly continued with seed context and recorded an expansion failure through telemetry, but the returned result still said `degraded: false` and contained no error. A consumer inspecting only the response cannot distinguish successful expansion from that fallback.

These are executable helper-level findings. They do not imply that all deployed calls hit the problematic sequence. Include constraint changes in cache tests, and either key the cache by the complete effective request or cache raw evidence and reapply filters and budgets per call. Return the actual retrieval stage reached, while retaining telemetry for diagnosis.

## Local ontology mode changes both visibility and mutation semantics

The [local backend](../../../project/agentbox/mcp/servers/lib/ontology-local.js) reads `Class` fences from the configured Markdown corpus. It does not check the Page publication flag. A synthetic private class was visible to `classGet`; this can be legitimate for a trusted authoring tool, but it is a different scope from Loom's published-ontology projection. It also enumerates only top-level Markdown files. A synthetic namespaced class was missed, whereas the current [visionGraph publisher](../../../visionGraph/pipeline/jsonld_parser.py) deliberately traverses namespaces.

Local mutations are direct file edits. `axiomAdd` changes the Class block, adds a local-edit provenance breadcrumb, ensures frontmatter and writes the file. It does not call the remote proposal gate, reasoner, approval consumer or PR creator. Its axiom-name mapping is also a simplification: for example, `DisjointWith` becomes `contrastsWith`, and `SomeValuesFrom` becomes `requires`. Those authoring relationships must not be assumed to preserve the full meaning of the named OWL operations.

The normal remote `ontology_axiom_add` path refuses direct load by default through [ontology-propose.js](../../../project/agentbox/mcp/servers/ontology-propose.js). However, setting `AGENTBOX_ONTOLOGY_LOCAL=1` causes the bridge to dispatch locally **before** reaching that guard. The probe established that the remote descriptor was guarded while the actual local helper changed the temporary Markdown file. The dispatch order is source-inspected; this was not a full MCP transport test.

The exact activation condition matters. With local mode off and direct load disabled, the remote axiom guard returns a policy error without a network call, so a network outage does not automatically bypass that guard. Other network-error fallbacks and forced local mode still need per-tool compatibility checks. In particular, local `propose` expects subject/object-style axiom arguments, unlike the normal create/amend tool schema.

**Assessment:** keep authoring fallback and governed ontology contribution distinct in the tool contract. A development mode may intentionally permit direct edits, but agents and humans should be able to recognise that authority change before invoking the operation. The current “only sanctioned route” description is insufficient across modes.

## Remote proposal handling has advanced beyond earlier gap descriptions

The agent helper builds a tagged create/amend payload for `/api/ontology-agent/propose`. Inspection of the current [VisionClaw DTO](../../../project/crates/visionclaw-domain/src/types/ontology_tools.rs) found the corresponding `ProposeInput` shape. The [handler](../../../project/src/handlers/ontology_agent_handler.rs) binds agent and user identity to the authenticated principal rather than accepting those body fields as authority. It also accepts optional idempotency and signature-envelope inputs.

The inspected create path in [OntologyMutationService](../../../project/src/services/ontology_mutation_service.rs) performs signature preconditions, payload hashing/idempotency reservation, a write-ahead intent, conflict checks and a call to `WhelkInferenceEngine::check_axiom_set`. After those gates it generates Markdown and attempts a GitHub PR. A failed PR creation yields a staged result; a successful one yields `PRCreated`. The returned ACSP gate remains pending.

This is materially more than an empty proposal scaffold. It also differs from the shorthand “human approval → PR”: this path creates the PR as the review projection, while approval is still pending. It commits its local intent/idempotency receipt before that approval, so “committed” in this layer must not be read as a merged ontology change. Full durability, retry-after-failure, amend semantics and eventual promotion remain to be traced in the VisionClaw chapter.

The helper currently does not expose the receiver's optional idempotency/signature-envelope fields in its MCP tool schema. Default server behaviour can still accept its request, but a deployment requiring signed envelopes needs a compatible producer path. A helper test alone does not establish that stricter deployment mode works.

## The authority gate separates action class from resource permission

[authority.js](../../../project/agentbox/management-api/lib/authority.js) classifies actions as recoverable, zero-tolerance or escalation-required. Unclassified actions escalate. Recoverable actions proceed; the others require publication of an action request and a verified, correlated approving response. The checked [manifest](../../../project/agentbox/agentbox.toml) enables this gate and classifies broker writeback as zero-tolerance.

This is a useful separation from pod ACLs: permission to access a resource does not settle whether a particular action should proceed. The classification table, however, only controls callers that actually invoke the gate. It cannot constrain the local Markdown helper merely because an ontology-write action is listed in the manifest.

At [server startup](../../../project/agentbox/management-api/server.js), the preferred implementation is [authority-consumer.js](../../../project/agentbox/management-api/lib/authority-consumer.js). It publishes signed requests, subscribes to decisions, checks the shared [approver allowlist](../../../project/agentbox/management-api/lib/authz.js), verifies signatures, matches correlation keys and tracks open/decided requests in memory. The simpler [decision waiter](../../../project/agentbox/management-api/lib/governance-decision-waiter.js) supports the relay-consumer fallback. Neither the existence of a waiter nor a successful signature check alone establishes end-to-end authorisation; inspect the selected consumer and its caller together.

## A real approval can arrive too early for the waiter

The synthetic authority-consumer fixture published a request, delivered an allowlisted decision through the consumer, then registered `awaitDecision`. The consumer recorded the request as decided, but the later wait returned `null` on timeout. `awaitDecision` registers only in the pending map; it does not consult the decided cache to replay an already received decision.

The gate calls publication before awaiting a decision, so this ordering is relevant to the hand-off. The test used injected verification and a fake bridge, establishing the ordering bug without claiming a real relay reproduced it. The failure denies rather than authorises the action, but it can leave an operator with a recorded approval and a denied operation.

The selected authority/proposal suites report **35 passing Jest tests**. Their fake transports provide useful checks for off-allowlist rejection, normal decision delivery and duplicate local decisions. Add an early-arrival case and a restart/recovery contract; in-memory decided/open maps are not evidence of durable workflow recovery.

## Human decisions have more than one signing front door

The [approvals route](../../../project/agentbox/management-api/routes/approvals.js) implements `/v1/approvals/:id/decide`. It requires NIP-98 authentication and an allowlisted approver, then asks the consumer to sign and publish kind 31403 with the operator delegation key. It also checks that the request is pending and rejects duplicate/in-flight decisions.

This is a concrete second signing front door alongside the forum, explicitly described as such in the source. It contradicts an unqualified public claim that the forum is the only place a human decision is signed. The deeper invariant can survive: human intent is authenticated and represented by a signed decision, while multiple interfaces can initiate that process. The canon should distinguish the human HTTP principal, the delegation key that signs the event and the resource actor that later applies it.

## Decision receipt is not application receipt

The [relay consumer](../../../project/agentbox/mcp/nostr-bridge/relay-consumer.js) checks event signatures, ingress policy and recipient addressing before persistence and dispatch. It sends governance decisions to the orchestrator adapter and notifies the decision waiter. The [local process adapter](../../../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js) can deliver JSON to a matched agent's stdin or persist it for later pickup. That is delivery evidence; it is not an acknowledgement that the requested domain action completed.

The [broker bridge](../../../project/agentbox/management-api/routes/broker-bridge.js) has a more explicit application boundary. It guards the decision before proxying it to VisionClaw, sends broker attribution, and reads `attributed`, `writeback_triggered` and `writeback_committed`. An attributed, triggered writeback that failed to commit returns an error rather than a successful closure. This corrects an older gap description that said the bridge ignored committed state; that description should not be repeated as current fact.

The remaining system trace is from the human's signed event through this proxy into the authoritative mutation and final acknowledgement. Preserve all three identities and distinguish requested, approved, delivered, applied and published states. The code already expresses several of these distinctions; the cross-repository account should carry them through consistently.

## Cross-model consultation and acceptance

Historical agentbox ADR-037 D4 specifies a different-family consultant for closure verification. The [actual helper probe](evidence/consultant-diversity-probe.cjs) establishes three bounded behaviours: an unknown producer paired with codex reports anti_fox_ok true, a known same-family pair reports false, and a candidate pool containing only the producer family returns no selector result. [Receipt](evidence/consultant-diversity-probe.json) records source hashes. These calls neither invoke a model nor validate a closure.

Unknown identity is not proof of diversity. The helper compares its resolved unknown token with a known family; it cannot establish the actual producer's lineage. The registry maps consultant names to fixed families, while the response envelope separately carries the reported model. Family selection and actual executed-model attestation therefore need reconciliation when providers, configuration or aliases change.

Current consultant-base accepts an optional producer_family. After invocation it returns an ok envelope and, when that input is present, adds a verification record. A same-family result logs a warning but remains a successful consultation. In a bounded search of agentbox JavaScript/CJS, selectVerifier call sites occur in tests; no production orchestration caller was established. This does not prove callers in other languages or external harnesses are absent. It does mean the helper alone cannot support a claim of mechanically enforced cross-family closure.

Consultant errors are logged and rethrown; a completed call supplies free-form response text, citations and execution metadata rather than a required typed closure verdict. The consultation URN hashes consultant plus question, so repeated questions with different context or model/results share that name. It is a grouping identifier, not a unique receipt binding a candidate revision to a particular verification result.

CP-01/03/05/07/09 requires a visible production dispatch/admission path, known producer and actual verifier identities, an explicit no-independent-verifier outcome and a candidate/context/result-bound receipt. Distinguish successful invocation, independent review and an accepted closure. Test failure, inconclusive response, disagreement, changed candidate/context, repeated question and provider-model drift. Do not promote a warning-only envelope into an acceptance gate. No paid call, provider request or real verification decision ran in this pass.

## Voice speaker, target and mandate scope

Historical agentbox ADR-037 D7 has a current route and production dispatcher wiring. [Nineteen existing tests pass](evidence/voice-authority-snapshot.json), covering deterministic transcript mapping and route outcomes with fake signature verification/dispatch and agent-event auth disabled in those route fixtures. This supports the tested producer behaviour, not full request authentication, actual signing or a target's execution.

The route parses the submitted mandate, verifies its event signature and checks its revoked flag/expiry. When agent-event authentication supplies a DID, it reconciles the mandate grantee with that speaker; otherwise it uses the grantee as speaker identity. It separately normalises actor_did into the target key. Naming a target is distinct from proving authority over it.

In the inspected path, recordFromSignedMandate validates kind, canonical URN and issuer/agent shapes. Neither that helper nor the route compares the signed event pubkey to record.issuer. The route does not apply the mandate container/modes to the target actor or parsed operation. isMandateActive examines the submitted record rather than fetching a latest revocation state. Signature validity, issuer authority, resource scope and revocation freshness therefore remain separate acceptance obligations. This source finding does not establish an unauthorised action on a deployed system; global admission and downstream execution checks need their own trace.

The producer publishes a signed ACSP ActionRequest addressed to the actor, then emits a visual agent-action envelope carrying the request ID. It returns dispatched:true after a dispatcher result containing a string ID. That is request-dispatch evidence, not an actor application acknowledgement or human approval. A failure after publication but before the response also needs retry/idempotency handling. Raw transcript and speaker/target fields enter the request, so recipient and retention policy must cover the actual published content.

CP-01/03/04/05/06/08 requires issuer-signature binding, accepted authority semantics for target/action scope, current revocation checks and authenticated speaker reconciliation through the full server. Test wrong issuer, unrelated resource, read-only grant for mutation, stale revoked grant, replay, unavailable dispatcher and post-publication failure. Carry request → decision if required → applied/rejected receipt to the caller and render state. Preserve the distinction between a producer requesting an action and the action having happened. No real voice, relay or scene operation ran.
