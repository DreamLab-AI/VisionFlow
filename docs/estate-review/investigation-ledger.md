---
title: Estate investigation ledger
status: in-progress
date: 2026-09-04
type: reference
---

# Estate investigation ledger

This ledger separates verified findings from investigation questions. It supplements existing gap registers; it does not close or reopen their entries without tracing the current implementation. Evidence for the first findings is in [canon and verification](canon-and-verification.md) and the [execution snapshot](evidence/snapshot.json).

## Established findings

| ID | Finding and consequence | Evidence strength | Next evidence or completion criterion |
|---|---|---|---|
| ER-001 | Public estate scope exceeds the old repository map and release generator | Source comparison | Trace all directly linked components and consumed dependencies; make reviewed coverage explicit |
| ER-002 | Local count gate fails for prose counts and archived ADR paths | Executed checker, exit 1 | Distinguish historical/live count sites, update ownership and location mapping, rerun against pinned sources |
| ER-003 | Fixture workflow can succeed without a cross-repository comparison | Workflow inspection | Demonstrate a real compared revision set and injected mismatch rejection |
| ER-004 | Four dream evaluators emit failure text but return zero; engine sends it to an LLM without a deterministic rejection gate | Isolated executable probes plus service caller trace | Typed evaluator contract and rejection through the complete cycle |
| ER-005 | Deployment workflow builds and publishes without invoking the local browser verification sequence | Workflow and package inspection | Establish actual publication policy and whether external checks gate it |
| ER-006 | Local repository names and aliases can mislead scope: agentbox is one checkout at two paths; dream-machine is the DreamLab dream-engine fork | Filesystem resolution and Git inspection | Preserve canonical identity throughout further dependency tracing |
| ER-007 | Extracted knowledgeGraph and Logseq publish different artefact sets; Logseq is a thirteenth repository identity | Source trace and fresh extracted build | Pin authoring/extraction/publication revisions and define reproducible consumer exports |
| ER-008 | Malformed JSON-LD is omitted before validation; string `"false"` is publishable | Synthetic probes in both pipelines | Input census diagnostics and strict boolean publication validation |
| ER-009 | Private ancestor identifiers/labels enter public derived Logseq outputs | Synthetic closure → API/scaffold/Turtle probe | Define and test inference visibility across every export |
| ER-010 | Zero source individuals differs from five generated RDF maturity individuals; display/RDF counts have different denominators | Fresh build and RDF census | Scope prose claims to their measured projection |
| ER-011 | Asserted RDF normalises URNs to HTTP; inferred RDF retains source URNs | Source and synthetic identity probe | Test joins in the actual consumer and choose one mapping contract |
| ER-012 | Loom loads content into memory but reports disk generation; mirror replaces files sequentially | Source inspection, not runtime reproduction | Two-generation activation/crash/reload test with explicit served-bundle identity |
| ER-013 | Agent retrieval cache omits domain and token override, returning an AI seed/830 tokens for a robotics/50-token follow-up | Executable helper probe | Reapply per-call constraints or key the complete effective request |
| ER-014 | Expansion failure returns seed context with `degraded: false` | Executable helper probe | Carry degradation stage to consumers as well as telemetry |
| ER-015 | Forced local ontology dispatch precedes remote direct-load guard; local helper edits Markdown directly and misses namespace pages | Source dispatch trace and temporary-corpus helper probe | Separate direct authoring from governed mutation; align namespace/provenance contracts |
| ER-016 | Early valid approval is recorded as decided but a later waiter times out | Fake-transport/verification probe | Register before publication or replay decided state; define durable recovery |
| ER-017 | Agentbox dashboard has an authenticated, allowlisted signing front door; forum-only claim is overbroad | Source route and boot wiring | Reconcile human/delegation/signing identities across canon and interfaces |
| ER-018 | visionGraph is the current corpus; old maps and a governing-document reference lag the split | Manifest/path/repository trace | Include current source, distinguish archived lineage and publication/distribution revisions |
| ER-019 | Current producer tests: 57 pass, one `_misc` cardinality assertion fails; validation has one duplicate-IRI error | Existing suite and aggregate census | Resolve corpus/test contract using intended policy; verify exact deployed revision separately |
| ER-020 | Dream service evaluates baseline before model-generated patch; persists accepted patch without rerunning checks | Source execution-order trace | Bind ACCEPT to a tested candidate tree, or label proposal-only execution |
| ER-021 | Dream parser accepts negated fallback text and explicit acceptance despite failure evidence | Actual module with synthetic reports | Strict typed verdict and required-evaluator veto |
| ER-022 | Dream archive, witness and patch base read HEAD separately | Source trace | Freeze a run manifest and pass it through dispatch and persistence |
| ER-023 | Alphabetical dream roster cap has no fairness rotation; restart forgets last run date | Source trace, no live reproduction | Local restart/over-cap simulation and durable run identity |
| ER-024 | Native and edge pod consumers pin different versions; agentbox also applies a provenance-path patch | Manifests, lockfiles and Nix source | Map fixes and enabled features to actual deployed binaries |
| ER-025 | Native malformed specific ACL falls back to broader grant; edge source also collapses read/parse failures into misses | Actual native helper probe and edge source trace | Distinguish missing, invalid and unavailable policy |
| ER-026 | Edge successful private/ACL responses can receive public cache directives | Handler/helper source trace | Verify private read and revocation through delivery/cache path |
| ER-027 | Native replay LRU accepts an evicted ID within TTL; content write success precedes provenance completion | Actual cache probe and provenance caller source | Capacity/restart contract plus partial-provenance repair |
| ER-028 | Forum relay OK precedes decision projection; projection writes are separate and errors can be ignored | [Source trace; 47 domain tests](forum-decisions.md) | Transactional or reconciled projection and distinct accepted/applied receipts |
| ER-029 | Proxy tests pass with runtime dependencies; last-good daemon token survives credential-file removal | [45 assertions and token-reader source](runtime-ingress.md) | Define revocation across file removal, daemon restart and browser sessions |
| ER-030 | Full sync clears graph families before fetching/reconstructing a replacement | [Source trace](visionclaw-data-runtime.md) | Stage and activate one generation; inject partial fetch and restart failures |
| ER-031 | Provenance inserts can leave partial records; backup success can omit requested files | [Emitter source and temporary WAL backup probe](visionclaw-data-runtime.md) | Atomic or repaired provenance, required backup-set validation and full restore |
| ER-032 | Website deploy gate differs from CI aggregator; pin parity does not inspect resolved library contents | [Workflow/source inspection and passing parity check](commercial-surfaces.md) | Bind effective checks, libraries, clients and workers to one release receipt |
| ER-033 | Chat tier selection carries no visitor authority proof; late replies lack request correlation | [Component/transport source; 97 local tests](commercial-surfaces.md) | Decide tier contract and prove correlated authorised replies |
| ER-034 | Agent-event session identity is not passed to frame attribution processing | [Handler/processor source](rendered-state.md) | Bind sender to claimed agent or specify trusted delegation |
| ER-035 | XR skips V5 sequence values; old action events can overwrite done/idle with working | [Decoder/store source; 218 library tests](rendered-state.md) | Explicit freshness, precedence and expiry tested across real delivery |
| ER-036 | Hover and query execution exist beyond older XR governing claims; 0x44 remains staged | [Source inspection](rendered-state.md) | Current scene, denial/error and headset receipts; live presence integration |
| ER-037 | RuView has a server UDP path despite stubbed MAT adapters; server header offsets disagree with firmware | [Actual extracted-parser probe; 100 hardware tests](sensing-extension.md) | Canonical codec and firmware-to-server fixtures, then physical capture validation |
| ER-038 | RuView model label can survive heuristic fallback; no named estate consumer found in inspected paths | [Source and bounded consumer search](sensing-extension.md) | Per-frame source/inference truth and explicit adopted integration |
| ER-039 | Memory embedding failure can store NULL vectors or preserve old vectors with replacement values | [Production factory with mock dependencies; ten tests](shared-memory.md) | Coherent value/vector repair or reject failed embeddings |
| ER-040 | TTL-only memory defaults to semantic, while sweep deletes episodic; reads omit expiry filters | [Metadata/query source and isolated factory probe](shared-memory.md) | Choose and test visibility expiry, cleanup, protected scope and recovery |
| ER-041 | 176 RuView candidates are vendored RuVector records: 161 identical and 15 divergent | [Path/SHA256 comparison](closeout/adr-lineage.md) | Trace divergence and consumed relevance without treating copies as adopted decisions |
| ER-042 | Forum trust sweep counts planned changes and OFFSET can skip rows after demotion | [Source and isolated query fixture](forum-decisions.md#identity-and-trust-decision-closeout) | Stable pagination, committed outcomes and recoverable state/audit receipts |
| ER-043 | Loom accepts stored vector settings that violate its cosine/384 contract | [Actual local adapter probe](consumed-vector-storage.md) | Validate effective configuration/model and bind artefact generation before readiness |
| ER-044 | Provider wrappers accept wrong hosts containing an expected substring | [Actual wrappers with stub CLI](runtime-egress-and-profiles.md) | Parsed endpoint policy and negative launch fixtures |
| ER-045 | Live mirror and digest have separate raw-content/provider and gating paths | [Dry-run sentinel and Rust source](runtime-egress-and-profiles.md) | Explicit content, recipient, off-switch and retention policy per path |
| ER-046 | MCP projector retains deleted definitions and stale state on malformed registry while exiting zero | [Actual projector fixtures](configuration-projection.md) | Ownership history, schema validation and persisted/loaded state receipts |
| ER-047 | Adapter connection timeout continues startup; off replacement can fail | [Server control flow](adapter-dispatch.md) | Explicit per-slot readiness and lifecycle fault injection |
| ER-048 | Privacy wrapper coverage is method/value scoped and encoding is separate | [Actual wrapper with injected redactor](adapter-dispatch.md) | Method/field coverage, schema preservation and complete route tests |
| ER-049 | Skill lint accepts empty references directories and body-only metadata | [Actual lint fixtures](capability-instructions-and-enforcement.md) | Typed frontmatter, entry-context budget and baked-revision evidence |
| ER-050 | Paired identifier helpers agree on tested hashes but differ on bead crossing; precomputed Rust addresses accept malformed suffixes | [Actual two-language fixture](federation-identifiers.md) | Exact grammar, supported-kind agreement and paired CI/recovery evidence |
| ER-051 | Vault-disabled resolution retains a legacy corpus override; Notes launch checks binary presence | [Resolver fixture and launcher source](authored-vault-transition.md#runtime-path-overrides-and-notes-launch) | Precedence/off-state matrix, relocation and editor recovery |
| ER-052 | W066 permits consumer-before-producer manifests; hybrid helper can reuse retained aggregates with capture off | [Validator and mocked-consumer fixtures](learning-evidence.md#producer-ordering-is-advisory) | Enforced admission, retained-corpus policy, freshness and restart/override matrix |
| ER-053 | Reaper confirmation checks launcher shape; Hermes Stop checks PID existence; delivered SIGTERM is not observed exit | [Process source and four native tests](process-lifecycle.md) | Process-instance binding, registry reconciliation and recoverable shutdown outcomes |
| ER-054 | GPU wrapper now configures graphics beyond CUDA-only ADR scope; existing backend gate skips and uses old dispatch calls | [GPU source and skipped-test receipt](rendered-state.md#gpu-packaging-and-runtime-boundary) | Explicit graphics decision, current locked evaluation and separate compute/presentation receipts |
| ER-055 | Port gate rejects block public mapping but accepts nested flow equivalents | [Actual gate fixtures](runtime-ingress.md#port-gate-syntax-and-exposure-coverage) | Structured effective-config audit, complete input inventory and listener/authority receipts |
| ER-056 | Bridge allowlist guards inbox consumption after relay store/broadcast; standalone empty-list config omits whitelist | [Cross-repository source and three helper tests](runtime-ingress.md#relay-admission-versus-inbox-authorisation) | Backend-specific admission, subscriber visibility and durable delivery/recovery |
| ER-057 | ADR validator can pass a stale index; reader entry page retained pre-archive authority/routes | [Generator fixture and navigation review](canon-and-verification.md#decision-register-and-reader-navigation) | Semantic re-verification, generated-index comparison and historical reader routes |
| ER-058 | Custody proposal lacks confirmed lifecycle owners; static bearer and archive-integrity checks do not provide expiry or recovery | [Provisional register and source review](runtime-ingress.md#custody-and-revocation-acceptance) | Confirmed custody, bounded revocation and protected recovery receipts |
| ER-059 | Eight service manifests declare licence split but lack promised adjacent texts/READMEs; extraction changes inventory | [Package collector](configuration-projection.md#service-package-and-release-metadata) | Package ownership, notices, archive/dependency and release identity receipts |
| ER-060 | Replay cache has bounded process-local claims; body binding and downstream retry semantics remain separate | [Extracted helper and call-site source](runtime-ingress.md#visionclaw-replay-and-operation-boundaries) | Combined clock/restart tests, route body policy and idempotent mutation recovery |
| ER-061 | Initial and position outputs filter public/owner metadata; metadata authority and client transition coverage remain unverified | [Six domain tests and current handlers](rendered-state.md#visibility-defaults-and-output-coverage) | Trusted metadata, all-output matrix and client-state transition receipts |
| ER-062 | Role mutations consume earlier caller authority; removal restores fallback; report mode bypasses gate denials | [Role and route source trace](role-authority.md) | Caller revocation boundary, durable transitions and effective route/mode receipts |
| ER-063 | Non-debug dev-auth builds retain bypass; proposed release assertion was cited as guaranteed | [Nine actual extracted build cases](role-authority.md#development-bypass-and-release-identity) | Shipped feature identity, profile boot rejection and full transport acceptance |
| ER-064 | Related security ADRs overstated report-mode expiry/release exclusion and completeness of four-flag profiles | [Source reconciliation](role-authority.md#profile-claims-and-effective-policy) | Unified effective-policy matrix and release-bound boot acceptance |
| ER-065 | Converter accepts two page paths mapping to one destination; dry-run can write an explicit report | [86 native tests and actual CLI fixtures](authored-vault-transition.md#converter-collision-and-dry-run-boundaries) | Collision planning, full accounting, path policy and recovery before promotion |
| ER-066 | Class markers accept non-string scalars; local fallback metadata bypasses inclusion check | [56 vault tests and consumer source](authored-vault-transition.md#inclusion-typing-and-local-fallback) | Typed formal-data exception, fallback scope and migration/visibility/publication receipts |
| ER-067 | Settings rename has distinct alias/migration mechanisms; helper success does not certify live persistence or peer registry IDs | [Eleven tests and settings source](configuration-projection.md#knowledge-settings-migration) | Shared compatibility matrix, persisted restart and named retirement release |
| ER-068 | Direct provenance/legacy mint sites remain outside typed-only claims; canonical naming is separate from signed authority | [Mint-site source and prior hash parity](federation-identifiers.md#mint-site-coverage-and-proof-of-identity) | Complete mint/lookup inventory, signed-session policy and governed migration |
| ER-069 | Browser signing migration trigger is reached while server session acceptance remains; bridge correlation is not delegated authority | [Source and eleven mocked interceptor tests](role-authority.md#request-realms-and-deferred-delegation) | Consumer retirement census, session lifecycle and explicit delegated-authority journey |
| ER-070 | Current host SimParams layouts match at 53 offsets, but same-size swap passes original guard; typed overflow logging does not cover untyped encoder | [Extracted layout probe and dispatch/encoder source](rendered-state.md#simulation-layout-and-force-authority) | Actual CUDA/module parity, force transitions and all-class capacity/mapping evidence |
| ER-071 | Missing nvcc bypasses fallback; invalid nonempty output passes the PTX phase; fixed-width rewrite and runtime selection need separate acceptance | [Six isolated build cases and loader source](rendered-state.md#ptx-build-acceptance-and-loaded-artefact-identity) | Provenance-bound module selection, actual driver/symbol/ABI and native-link receipts |
| ER-072 | HUD press-mode coverage is incomplete; the existing hierarchy predicate test rejects a currently accepted label | [Constructor source and extracted existing-test failure](rendered-state.md#xr-control-coverage-and-hierarchy-semantics) | Full control/runtime coverage, producer semantics and reconciled layout tests |
| ER-073 | ACSP consumers retain state and drop event/request identifiers at the decision adapter; pending removal precedes persistence | [Consumer source trace](forum-decisions.md#visionclaw-acsp-consumption-and-recovery) | Durable authority, full correlation and failure/restart journey receipts |
| ER-074 | Workspace extraction remains partial; four GPU supervisors use direct context sends plus bus publication, with discarded delivery results | [Manifest and supervision source](rendered-state.md#gpu-supervision-and-context-delivery) | Responsibility/dependency gates, acknowledged generations and failure/recovery receipts |
| ER-075 | Development wrapper skips builds for newer crate CUDA/manifests; four stale baselines remain despite operative extension coverage | [Three timestamp fixtures](configuration-projection.md#development-restart-and-build-input-coverage), [validator receipt](evidence/dev-docs-closeout.json) | Complete input/artefact identity and semantic baseline/index reconciliation |
| ER-076 | VisionClaw historical routing finds 43 explicit lineage mentions among 137 records; compound predecessors retain independent obligations | [Historical reconciliation](historical-decision-reconciliation.md) | Section-level successor, retirement or deferral with source evidence |
| ER-077 | Liveness fired applies one recent server-SHA observation to both canary kinds; failure transitions also fire | [Observer source review](liveness-observation.md) | Typed outcomes, producer revision, continuity policy and promotion-consumer trace |
| ER-078 | KPI summary reads persist two sequential snapshots and can fire on empty inputs; count confidence and bounded lineage do not prove outcome quality | [KPI source-to-panel trace](kpi-outcomes.md) | Metric semantics, capture health, run/lineage completeness and freshness evidence |
| ER-079 | D1 checker accepts count-only roster and HTTP-success fired:false reply; dashboard latch precedes observation acknowledgement | [Three mocked checker cases and consumer source](liveness-observation.md#consumer-evidence-and-promotion-scope) | Journey-specific predicates, durable receipts/retry and explicit canon promotion evidence |
| ER-080 | Agentbox histories include 46 records without exact operative-lineage mentions; compound and journal/scheduling proposals retain independent obligations | [Historical reconciliation](historical-decision-reconciliation.md#agentbox-routing-and-compound-obligations) | Consultation/voice traces and explicit journal/scheduling adoption evidence |
| ER-081 | Unknown producer passes diversity check; consultant envelope warns rather than rejects same-family verification | [Three helper assertions and envelope source](agent-grounding-and-governance.md#cross-model-consultation-and-acceptance) | Known identities, production dispatch/admission and candidate-bound outcome receipts |
| ER-082 | Voice producer validates submitted mandate activity/signature but lacks issuer-key and target/resource-mode comparison in the inspected route | [Source and 19 existing mocked tests](agent-grounding-and-governance.md#voice-speaker-target-and-mandate-scope) | Full authority/revocation and request-to-application receipts |
| ER-083 | Failed local log write still advances journal state; retry reports duplicate and citation check accepts changed content | [Actual temporary journal/adapter fixture](shared-memory.md#execution-journal-durability-and-reconstruction) | Durable append contract, recovery and content-bound model provenance |
| ER-084 | Dream validator admits empty, inline and missing-script evaluator configurations; selected deep has no admission-time evaluator association | [Four extracted validation cases](self-improvement.md#evaluator-readiness-before-scheduling) | Per-deep target readiness and deterministic handoff before scheduling |
| ER-085 | Local catalog interface/ranking differ from ADR proposals; referenced benchmark harness absent in checkout | [Catalog source review](catalog-decisions.md) | Reconcile contracts, data and reproducible recommendation evidence |
| ER-086 | Forum page replay omits store tombstones; local message projection does not remove events deleted from store | [Sprint consumer review](forum-decisions.md#forum-navigation-counts-and-cold-entry) | Browser/relay deletion, cold-entry and recovery journeys |
| ER-087 | Harness audit treats undeclared-planned controls as source-backed and duplicate edges inflate coverage | [Actual script fixtures](engineering-governance.md) | Source resolution, distinct coverage and actual enforcement evidence |
| ER-088 | Loom selects sibling RuVector source; RuView lock records registry packages, while vendored ADR histories differ | [Upstream source identity](upstream-decision-scope.md) | Consumed-package and feature-bound adoption/acceptance |
| ER-089 | RuView seven integration helpers exist; selected consumers and advertised incremental/bounded-memory benefits remain unproved | [Signal/MAT source assessment](sensing-extension.md#signal-and-mat-integration-helper-availability-versus-execution) | Selected-path, retention and numerical/performance evidence |
| ER-090 | RuView training integration completion mixes active antenna attention with uncalled/optional helpers | [Training source assessment](sensing-extension.md#training-integration-and-model-evidence) | Caller, assignment, gradient and measured model/resource acceptance |
| ER-091 | Real-data training CLI validates synthetically; cross-domain evaluator imputes missing experiment metrics | [Dataset and metric assessment](sensing-extension.md#dataset-separation-and-evaluation-meaning) | Explicit holdout/data identity and observed metric evidence |
| ER-092 | Rapid adaptation succeeds below configured readiness with unchanged weights; integrated MERIDIAN path remains unverified | [Native adaptation probe](sensing-extension.md#cross-environment-adaptation-and-readiness) | Enforced calibration admission and installed-model evaluation |
| ER-093 | Unified trace joins by identity alone; source presence and canary fire do not establish complete authorised task provenance | [Source inspection](joined-provenance-trace.md) | Correlated task, resource access, capture completeness and bounded consistent reads |
| ER-094 | CTC producer attaches turn totals per Bash step; forwarding works but complete DAG aggregation/display is not established | [17 tests and five helper assertions](transaction-cost-accounting.md) | Agreed cost units, deduplication, complete capture and actual consumer receipt |
| ER-095 | Selected failure classification works, but malformed requests are untagged and trajectory/wire modes disagree; complete metrics producer remains unverified | [16 tests and six route assertions](failure-telemetry.md) | Complete source census, authoritative field and durable deduplicated metrics |
| ER-096 | Decision API and relay-derived UI history expose different evidence; authenticated access is not case-scoped and signing context is incomplete | [Source review](decision-history.md) | Stable authorised history, retained request context and correlated applied receipts |
| ER-097 | Public agent disclosure has sixteen source mounts, but one-shot active-only lookup conflates missing disclosure with human authorship and loses revoked history | [Source and mount census](agent-disclosure.md) | Freshness/error states, historical principal binding and rendered surface coverage |
| ER-098 | Pocket references resolve only to the newest retained in-memory match; canonical naming does not establish a durable decision record | [Nine tests and five route assertions](pocket-provenance.md) | Durable exact-record binding, authorised lookup and actual phone reconstruction |
| ER-099 | Server fingerprint retrieval is brute-force; mismatched dimensions can match at zero distance and live HNSW/fusion acceptance is unverified | [Five native assertions](sensing-extension.md#fingerprint-retrieval-and-hnsw-acceptance) | Typed versioned vectors, selected live consumer and labelled scale/fallback evaluation |
| ER-100 | RVF witness JSON is implemented, but training hash labels are derived metadata and proposed chain verification lacks completeness anchors | [Source/design assessment](sensing-extension.md#witness-segments-and-audit-completeness) | Actual content identity, independent head expectations and durable producer coverage |
| ER-101 | Server model signatures are unenforced; firmware signature gate computes a public-input hash rather than authenticating a signer | [Seven-source assessment](sensing-extension.md#secure-sensing-claims-and-model-admission) | Real primitive and trust-policy enforcement across all loaders and configurations |
| ER-102 | Multistatic consensus is signal agreement, not replicated state; unique-device participation and partition contracts remain unverified | [Source/design review](sensing-extension.md#multi-device-agreement-and-replicated-state) | Authenticated membership, tentative/committed semantics and recovery tests |
| ER-103 | ESP32 WASM runtime exists, but zero capabilities allow all and post-return timing does not establish hard execution limits or portable offline lifecycle | [Source assessment](sensing-extension.md#edge-runtime-implementation-and-isolation-boundaries) | Per-callback isolation, explicit capability policy and target-profile persistence tests |
| ER-104 | SONA reports convergence without samples and accepts incompatible profiles; complete feedback/safety lifecycle remains unverified | [20 native tests and six assertions](sensing-extension.md#sona-feedback-admission-and-profile-lifecycle) | Candidate-bound evaluation, typed profiles and durable rollback/installation |
| ER-105 | Spatial GNN arithmetic is implemented, but three proposed consumer modes and online improvement remain unverified | [28 native tests and mode assessment](sensing-extension.md#gnn-mode-coverage-and-learning-evidence) | Trained-state installation, mode-specific consumers and held-out feedback evaluation |
| ER-106 | Foundational Rust ADRs lag the fifteen-member workspace and actual algorithm/backend selection | [Manifest/source assessment](sensing-extension.md#rust-workspace-and-foundational-contracts) | Selected target coverage, numerical parity and actual provider/backend execution |
| ER-107 | Sensing UI labels and retained observations do not establish hardware type or freshness; original startup gating has drifted | [Source and freshness assessment](sensing-extension.md#sensing-ui-source-and-freshness-contract) | Browser transitions, selected-server routing and hardware/model provenance |
| ER-108 | Survivor tracking helpers exist, but scan/query integration and confirmation/event semantics remain incomplete | [Tracking assessment](sensing-extension.md#survivor-tracking-and-operational-integration) | Authoritative identity/counts, matching objective, persisted transitions and selected runtime acceptance |
| ER-109 | Mobile screens mix proxy/demo data with transport status; ADR platform and acceptance claims exceed current gates | [Mobile assessment](mobile-sensing.md) | Thirty-one scoped criteria, authoritative tracking and selected native/web/server evidence |
| ER-110 | Training UI/handlers exist but are not mounted by declared entrypoints; algorithm labels and UI contracts diverge | [Training operation assessment](sensing-extension.md#training-ui-and-model-operation-contracts) | Fourteen implementation items, job/data identity and browser-to-inference evidence |
| ER-111 | macOS Rust adapter, named Swift helper and server dispatch do not form the proposed scan path | [Platform assessment](sensing-extension.md#macos-helper-and-runtime-contract) | Versioned protocol/identity, selected runtime and nineteen platform verification rows |
| ER-112 | Python mock isolation has improved, but synthetic replay and warning-only CI do not establish proof of physical acquisition | [Proof replay assessment](sensing-extension.md#python-proof-replay-and-mock-boundaries) | Entry-point mock gates, locked build, rejecting CI and provenance-bound capture acceptance |
| ER-113 | Deferred roadmap planner conflates observed and predicted state; heuristic and priority claims have counterexamples | [Planning design assessment](sensing-extension.md#roadmap-planning-design-and-evidence-state) | Catalogue reconciliation, search correctness, authorised execution and six performance budgets |
| ER-114 | Rust primary-backend migration commands and topology differ from declared package targets and current routers | [Migration assessment](sensing-extension.md#rust-primary-backend-migration) | Selected binary/features, API/model parity, measured target properties and retirement/rollback |
| ER-115 | CRV facade implementation does not establish the proposed six-stage runtime or cross-room identity journey | [CRV assessment](sensing-extension.md#crv-stage-facade-and-identity-evidence) | Thirty-seven scoped criteria, selected dependency/caller and measured identity evidence |
| ER-116 | RuView strategy supersession overstates full realisation and absorption of storage, learning, security, consensus and edge obligations | [Surviving domains](historical-decision-reconciliation.md#ruview-strategy-supersession-and-surviving-domains) | Explicit domain disposition, dependency/consumer identity and measured adopted-path acceptance |
| ER-117 | RVF packaging does not establish three cognitive-container contracts, transactional persistence or cross-target interoperability | [Container lifecycle](sensing-extension.md#cognitive-container-contract-and-durable-lifecycle) | Versioned consumer fixtures, durable recovery, branch/replay acceptance and measured storage/latency |
| ER-118 | Commodity RSSI components permit confident absence from empty features; declared capabilities do not establish freshness, CLI delivery or captured accuracy | [Commodity sensing](sensing-extension.md#commodity-sensing-capabilities-and-observation-readiness) | Observation readiness/source identity, reconciled rules, captured proof bundle and supported installation path |
| ER-119 | Six signal modules are standalone helpers; geometry and signed-velocity claims exceed the inspected estimator semantics | [Algorithm adoption](sensing-extension.md#signal-algorithm-semantics-and-runtime-adoption) | Consumed adapters, finite configuration, geometry/direction contracts and captured benchmark evidence |
| ER-120 | Contrastive pretraining uses a narrower loss than ADR-024; EWC proxy/accessors do not establish an applied fine-tuning constraint | [Objective boundary](sensing-extension.md#contrastive-training-objective-and-consolidation-boundary) | Ratified objective, batch/gradient evidence and real consolidation/joint-training consumers; wider ADR review pending |
| ER-121 | Embedding augmentation differs from its physical contract; merged LoRA is applied again and ordinary restore omits adapters | [Augmentation/state](sensing-extension.md#embedding-augmentation-and-projection-state) | Label-preserving augmentation, merge/output invariants and provenance-bound model round trips; wider ADR review pending |
| ER-122 | Embedding quantisation check measures vector rounding with an incorrect tied-rank formula; deployment counts understate optional pose encoder | [Quantisation evidence](sensing-extension.md#embedding-quantisation-and-deployment-evidence) | Actual quantised-model comparison, valid rank metric and measured scoped target budgets; wider ADR review pending |
| ER-123 | AETHER orchestration and acceptance definitions remain incomplete, including unattainable normalised-variance gates and global mining | [Cumulative ADR-024](sensing-extension.md#aether-cumulative-closeout-contract) | Corrected criteria, explicit phase/future disposition and real index/adaptation/data/target acceptance |

None of these is a new assertion that the entire estate is broken. Each describes a specific boundary that can be checked and improved.

## Remaining investigation by level

| Level | Investigation | Evidence required |
|---|---|---|
| Vision and value | Compare promises of judgement, sovereignty, grounding and legibility with an actual operator journey | Intended outcomes, implemented interactions, outcome measurements and explicit limits |
| Knowledge production | Pipeline and Logseq publisher trace complete for this pass; review content quality, enrichment promotion, legacy publishers and explorer in depth | Existing build/test/probe receipts plus pending promotion and consumer evidence |
| Grounding and evaluation | Initial Loom composition/retrieval/generation source trace written; verify runtime activation, agent consumers and benchmark controls | Router probes, dataset/generation identity, raw evaluation results, negative controls and scoring |
| Runtime knowledge | Trace VisionClaw ingestion, reasoner invocation, mutation fencing, persistence and query APIs | Wired call paths, ownership of state, tests and recoverable failure behaviour |
| Agent execution | Trace agentbox spawn identity, tool permissions, memory embeddings, privacy and session lifecycle | Implementation paths, configuration defaults and focused tests |
| Authority and storage | Trace signing, NIP-98, ACLs, delegation, replay and revocation across native and edge pod tiers | Producer/consumer agreement, denied-request tests, mutation receipts and recovery procedures |
| Governance | Follow a proposal from agent to forum decision to applied/rejected mutation and acknowledgement | Common identifiers, real signature verification, idempotency, timeout and retry behaviour |
| Human surfaces | Inspect forum, commercial overlay, desktop and XR representations | Actual components and routes, signing boundary, stale/error handling, accessibility and operator journey |
| Embodiment | Follow action ingestion into graph identity and rendered beam; investigate voice scope, XR and deferred physics | Source wiring, binary/schema contract, tests and targeted runtime receipts |
| Memory and sensing extensions | Examine only consumed RuVector paths first; distinguish RuView simulation from hardware acquisition | Dependency pins, feature selection, persistence tests, adapter source and measured hardware evidence if present |
| Self-improvement | Toolkit/service trace and tests complete for this pass; candidate evaluation gap established | Full cycle simulation, deployed receipts, restart recovery and roster fairness |
| Operations and delivery | Compare build/deploy prerequisites, observability, secrets lifecycle, backups and release pins | Configuration plus verified restore/recovery and deployment receipts where feasible |
| Corpus coherence | Reconcile findings with book, baseline, historical closeout, registers and public pages | Claim-to-source links, dated supersession and navigation checks |

## Next working sequence

The next evidence gaps are consumed RuVector internals and learning/privacy, browser/GPU and explorer paths, and full reasoner/proposal transactions. Cross-repository acceptance must then connect the established component findings into actual author-to-answer, decision-to-mutation, data-recovery, rendered-action, learning and release journeys. Continue operative ADR amendments and historical/upstream lineage reconciliation alongside those traces. The inventory is coverage evidence, not a completion score.

## Completion audit to perform at the end

The review is complete only when all linked estate repositories have an explained inclusion or exclusion, each included component has a source-grounded account of intent, implementation and gaps, and the critical cross-repository journeys have been traced. It must cover organisational value, user experience, data/semantics, code/contracts, trust/governance, operations and evidence quality.

Every consequential claim must point to an inspected source or receipt and state its verification level. Where live proof is unavailable, distinguish the limitation from a confirmed implementation defect. Finally verify the new section's navigation, source references and factual consistency against the current worktrees, and reconcile new findings with the older corpus without erasing its history.

## Documentation verification for this pass

Local file-target checking resolved 68 links across this section and its parent index, with no missing targets. `git diff --check` reported no whitespace errors. Unicode inspection reported zero suspicious characters in each of the five new Markdown documents. The prose scanner returned stylistic findings; these were reviewed editorially rather than treated as factual errors or mechanically rewritten. Its dash-density reports included Markdown delimiters. No browser, service, cross-repository runtime or production check was performed.

## Second pass: knowledge-production evidence

The previous turn was progress: it created the initial review and verified canon findings. This pass also made progress: added a full temporary knowledgeGraph build, 13 passing extracted-pipeline tests, 52 passing Logseq tests, input/publication/closure probes, Logseq-to-publisher lineage, and initial Loom generation/delivery source analysis. Receipts are in `evidence/knowledge-snapshot.json` and `evidence/knowledge-boundaries.json`; the generated corpus was temporary and was not published. The broad review remains in progress.

Second-pass documentation verification: all local file targets resolve after adding the knowledge chapters and evidence links; whitespace validation passes. Both new chapters passed invisible-character inspection. The stylistic scanner reported article-led paragraph openings and one vocabulary preference; these were reviewed as editorial suggestions, not treated as correctness failures. knowledgeGraph remains free of tracked worktree changes after the build and tests.

## Third pass: agent boundaries and current corpus

The previous turn was progress, producing the knowledge and Loom chapters with verified receipts. This pass adds agent grounding/governance and current-vault chapters. Agent tests pass (20 Node + 35 Jest); current visionGraph tests report 57 passed and one failed corpus-size assertion. Synthetic probes reproduce cache constraint loss, hidden expansion degradation, direct local-authoring behaviour, namespace omission and the early-approval race. The current producer also reproduces the previously identified publication/identity boundary cases. Remaining core work includes full VisionClaw, pod/forum/deployment traces, Loom runtime/benchmark evidence, renderer/XR, runtime/memory/sensing and dream-engine. No blocker or completion claim is recorded.

## Fourth pass: self-improvement semantics

Added [self-improvement](self-improvement.md) with the toolkit/service distinction, baseline-versus-candidate execution order, evaluator failure handling, parser probes, witness scope, draft promotion, memory adapters and scheduler/recovery limits. Existing suites pass: 78 Rust and 125 TypeScript tests. The npm wrapper initially failed writing metadata with ENOSPC; direct invocation of the installed runner succeeded. No remote cycle or external publication ran. This is further substantive progress, not completion of the estate review.

## Fifth pass: pod authority and storage

The previous pass made progress on dream-engine semantics. This pass adds [storage and authority](storage-and-authority.md): consumer version differences, native/edge policy and replay, private-response caching, filesystem integrity and provenance. All 46 selected existing tests pass. Temporary-crate probes reproduce malformed-policy inheritance and capacity-evicted replay acceptance. No external service was mutated. Full cross-tier grant/mutate/revoke/recover remains open, as do the other major runtime and human-surface investigations.

## Sixth pass: forum evidence and ADR closeout expansion

The previous storage pass was substantive progress. This pass traced forum signing, relay admission/OK, case projection and domain transitions; 47 governance tests pass. The user then extended the active task to upgrading and extending all ADR docs across the estate into a complete-system closeout roadmap. The new [roadmap](closeout/README.md), candidate inventory/collector and proposed VisionFlow ADR-2007 establish that work. The baseline and generated ADR index were updated together. Candidate classification is provisional; record-level evidence amendments across repository packs are not complete. Preserve the original all-level implementation review alongside this expanded deliverable.

## Seventh pass: ingress verification and operative ADR amendments

The previous pass made progress on the ADR roadmap and first cross-repository extensions. This pass adds runtime ingress analysis, a 45-assertion local proxy test with genuine signature paths (initial three skips resolved through installed dependencies), refreshed source verification for four identity ADRs, and a substantive partial-status correction for governed ontology writes. Governing documents were amended alongside the records. Remaining validator errors stay open; no running-image or production claim was inferred from source tests.

## Eighth pass: central graph persistence and recovery

The previous pass made substantive ingress/ADR progress. This pass adds VisionClaw persistence analysis, shared-store setup, destructive multi-step full sync, scoped derived fences, incomplete-provenance risk and an actual WAL backup probe. Four VisionClaw ADRs and both governing documents receive closeout extensions; provenance completeness is correctly partial. The full runtime/render/reasoner review and estate-wide record upgrade remain active.

## Ninth pass: commercial surfaces and release boundaries

Added the commercial chapter and source-hash receipt. All 97 existing Vitest tests pass; pin parity and the eight-record ADR validator pass. Extended all eight operative website ADRs and both governing documents with scoped acceptance conditions. CI and release gates differ; tier selection does not transmit visitor authority; late replies lack question correlation at the client boundary. These are source/local-test findings, not live deployment evidence. Archive lineage and the remaining estate packs remain open.

## Tenth pass: action and rendered state

The previous commercial pass was progress. This pass adds an XR source trace and 218 passing library tests, corrects stale hover/query documentation, and extends ADR-2018/2019/2020/2034/2036 plus protocol/XR governing documents. Sequence values are discarded by XR; action status lacks freshness comparison; authenticated session identity is not passed to frame attribution. These are source findings, not live reproductions. Browser/GPU, full XR journeys and the remaining ADR corpus stay open.

## Eleventh pass: RuView sensing paths

The previous XR pass was progress. This pass inspects distinct MAT/server/hardware paths, runs 100 passing hardware-crate tests and reproduces the server parser header mismatch using the actual extracted function. Adds sensing chapter/receipts, corrects overbroad README claims, and extends RuView ADR-018/023/028/035. No named consumer was found in inspected VisionClaw source/manifests or agentbox manifest/flake; integration remains unestablished rather than assumed. Upstream-reference search was used for orientation, local checkout for conclusions. Physical measurement and remaining ADR ownership/lineage remain open.

## Twelfth pass: shared memory contracts

The sensing pass was progress. This pass adds shared-memory source analysis, ten passing mock factory tests and isolated embedding/TTL probes. ADR-2014 becomes partial for its unimplemented searchable-write guarantee; ADR-2018/2019 and LEARNING-memory gain acceptance conditions. No real DB mutation or recall benchmark ran. Remaining work includes consumed RuVector internals, learning/privacy paths, operational recovery and the wider ADR corpus.

## Thirteenth pass: corpus coherence and vendored lineage

The memory pass was progress. This pass reconciles chapter navigation, coverage and the established-findings register through ER-041, replacing stale first-pass coverage claims. A new reproducible lineage collector compares 176 RuView vendored ADR candidates with standalone RuVector: 161 byte-identical, 15 divergent. All remain visible; no semantic equivalence, upstream adoption or completion is inferred. Full ADR upgrades and system acceptance remain open.

## Fourteenth pass: standalone ontology explorer

The lineage pass was progress. Added WasmVOWL source/test evidence: 47 native Rust tests pass; frontend 19 pass/60 fail, including 32 graph-store Map-plugin failures. Hook nodes/edges schema differs from Rust class/classes parser. Added proposed ADR-001 and README qualification; publisher copies are distinct and not condemned by standalone results. Full tree lineage, real WASM/browser, accessibility and measured performance remain open.

## Fifteenth pass: learning producer and privacy

The explorer pass was progress. Added learning chapter/receipts, 27 passing helper tests and actual-redactor synthetic probes exposing quoted/JSON coverage gaps. ADR-2015 privacy guarantee becomes partial; ADR-2016 and governing document gain outcome/recovery conditions. Missing pg-module watermark handling differs from query-exception retry. No real transcripts or database mutations were used. Full learning-loop evidence remains open.

## Eighteenth pass: Loom decisions and generation tests

The learning pass was progress. Extended all four Loom ADR candidates and RUST-ARCHITECTURE with generation activation, consumer diagnostics, benchmark and release-identity conditions. Default-feature workspace library tests pass; a focused generation suite is recorded in the new receipt. The inventory now also recognises prefixed ADR filenames, retaining the agentbox cross-link stub as support material. Its archive/operative lineage and stale consumer claim were corrected. Live profile parity, full bundle activation and measured benchmark acceptance remain open.

## Seventeenth pass: extracted producer decision pack

The Loom pass was progress. Extended knowledgeGraph's four operative ADRs and governing baseline using verified pipeline findings and current-vault lineage. Rechecked count/validation workflow separation and corrected inventory classification of two ADR-named ontology pages. Identity preservation and release acceptance remain open; the count tripwire is not removed.

## Eighteenth pass: dream decision reconciliation

The producer pass was progress. Extended all three dream-machine decision candidates plus agentbox ADR-2024 and its governing document. Repeated parser probes match the previous source hashes and reproduce ACCEPT after failure/negated text. Configured recall bands are now explicitly distinguished from an enforced candidate gate; human merging remains required. Inner-loop acceptance is a prerequisite for the fork's proposed outer-loop optimisation.

## Nineteenth pass: forum operative decisions and trust sweep

The dream pass was progress. Extended all nine forum operative ADRs and both governing documents. Added 22 passing native key tests and a source-extracted SQLite pagination fixture: 200 of 400 eligible rows remain after the offset advances over the shrinking result set. Source also ignores trust/audit write errors before returning the planned level. ADR-2006 becomes partial; live deployment and cross-service journeys remain unverified.

## Twentieth pass: governance contract and historical routing

The forum trust pass was progress. Added proposed forum ADR-2010 with durable signed/accepted/projected/received/applied receipt stages and failure/recovery acceptance conditions. Updated its governing baseline and routing preamble. A companion maps all 24 frozen records, including corrected routes to the three sprint canonicals, without editing archives. Six prior governance source hashes match; no new live or runtime evidence is implied. Historical implementation verification remains open.

## Twenty-first pass: current vault decision pack

The governance/history pass was progress. Added proposed visionGraph ADR-VG-001/002 and publication governing contract, corrected the README vault-format owner link, and revalidated 18 earlier source hashes. Preserved existing user corpus deletions. No new pipeline/runtime test or deployment receipt is implied; inclusion and generation acceptance remain open.

## Twenty-second pass: historical authoring decisions

The current-vault pass was progress. Added a six-record Logseq decision companion with current responsibility and specific acceptance requirements, preserving archived files. Inspected baseline-aware IRI gate semantics and corrected the classification of two rendered ontology pages. No historical status or component test is promoted to complete-system evidence.

## Twenty-third pass: remaining shared pod decisions

The historical-vault pass was progress. Extended the six remaining pod ADRs and governing baseline; all seven operative records now carry scoped acceptance. Revalidated prior pod source hashes and inspected OIDC discovery, git feature defaults and single-endpoint configuration. No runtime, standards-status or external anchoring certification is implied.

## Twenty-fourth pass: consumed RuVector artefact contract

The pod pass was progress. Traced the actual local RuVector dependency, redb storage and index ordering. A temporary actual-adapter probe returns ready for Euclidean and wrong-width artefacts; Euclidean score conversion differs from cosine and the width mismatch fails only on query. Added review/receipt and Loom ADR-137/138/governing qualifications. Temporary example removed; no shared DB or deployed service mutation.

## Twenty-fifth pass: runtime profiles and session egress

The vector pass was progress. Actual wrappers with stub executables accept wrong-host substrings; live-hook dry-run preserves an invented password sentinel when configured and skips for off/no-identity. Rust digest source separately sends flattened input to its provider. Added chapter/probes and ADR-2007/2026 with governing qualifications. No real credentials, transcripts, providers or relay messages were used.

## Twenty-sixth pass: configuration projection

The egress pass was progress. Actual projector fixtures distinguish successful gate removal from registry deletion, malformed input and missing requirements. Added chapter/probe and ADR-2003/2008/2031 governing qualifications. ADR-2008 becomes partial; no actual MCP config, server, image or running process changed.

## Twenty-seventh pass: adapter lifecycle and dispatch

The projection pass was progress. Added actual privacy-wrapper fixtures and source lifecycle review. Narrowed ADR-2004/2005 to partial for timeout/replacement and universal redaction/encoding guarantees, retaining the accepted design. No real persistence, network or live startup faults were exercised.

## Twenty-eighth pass: capability instructions and limits

The dispatch pass was progress. Inspected orchestration-only tree-search and its runtime/config search scope, and reproduced two lint acceptance gaps. Added review/receipts and ADR-2020/2021 governing qualifications. Neither agent orchestration nor provider spending nor image rebuild was performed. Runtime cap and off-state acceptance remain open.

## Twenty-ninth pass: federation identifier parity

The capability pass was progress. Executed the actual Rust URI module and JS bridge against shared invented inputs: five hashes agree, bead mapping differs and malformed precomputed suffixes pass Rust KG constructor/parser. Added review/receipt and three ADR amendments plus the missing agentbox protocol governing document. No live ingest or persistence ran; cross-repository CI remains open.

## Thirtieth pass: architectural synthesis and completion audit

The identifier pass was progress. Integrated newer chapters into the reading table, refreshed stale repository coverage and replaced first-pass future claims in the architectural argument with actual findings. Added a deliverable audit separating assessment completion from implementing the closeout roadmap; all-record review and other assessment requirements remain explicitly open. No new runtime evidence is claimed.

## Thirty-first pass: vault runtime and Notes

The synthesis pass was progress. Added actual resolver fixture and launcher/consumer source evidence; extended agentbox ADR-2028/2029 and baseline. Syntax checks pass, but legacy override and binary fallback need explicit off-state semantics. No user corpus, real terminal or recovery database changed.

## Thirty-second pass: learning producer ordering

The vault runtime pass was progress. Five actual-validator fixtures establish advisory W066 semantics; two actual hybrid-factory fixtures with injected dependencies apply retained-aggregate bonuses while recording is off. ADR-2017 becomes partial and its governing learning document gains admission and corpus-lifecycle acceptance. No external memory, real transcript, route deployment or process configuration changed.

## Thirty-third pass: runtime process lifecycle

The learning-ordering pass was progress. Reviewed shared daemon discovery, reaper registry/eligibility/signalling and Hermes start/stop paths. Four targeted existing native tests pass. ADR-2032 becomes partial against its all-signalling-tools rule; baseline and roadmap gain instance identity and shutdown acceptance. No live daemon discovery, launch, signal or PID-file mutation ran.

## Thirty-fourth pass: GPU wrapper scope

The process lifecycle pass was progress. Extended agentbox ADR-2006 and baseline against current wrapper, backend and package-selection source. Preserved historical CUDA implementation status while recording the reached graphics review trigger. Existing backend test exits 77 without Nix; its old dispatcher calls require maintenance. Added eight source hashes and scoped runtime acceptance. No build, GPU workload or render ran.

## Thirty-fifth pass: compose exposure gate

The GPU pass was progress. Current port gate passes; unchanged-script temporary fixtures reject a block public mapping but accept nested flow equivalents. Extended ADR-2013 to partial and its ingress governing document with parser, deployment-input and listener acceptance. No Docker evaluation, service launch or port binding ran. Relay ADR-2012 source inspection began but its reconciliation remains open.

## Thirty-sixth pass: relay admission boundary

The port-gate pass was progress. Traced the bridge through consumed relay WebSocket and typestate ingestion; allowlisting applies later at inbox consumption. Three native helper tests pass. ADR-2012 becomes partial, with governing and roadmap qualifications for both backends. Standalone empty-policy runtime remains unverified. No relay launch, event send or pod write ran.

## Thirty-seventh pass: decision navigation and index semantics

The relay pass was progress. Extended ADR-2001, corrected its template guidance and repaired exact archive-target links and authority framing in the agentbox entry page. Actual validator still reports six stale records; an isolated stale-index fixture passes check mode. No archive or verification baseline was rewritten. Full historical routing and index regeneration remain open.

## Thirty-eighth pass: custody register foundation

The navigation pass was progress. Extended ADR-2027 and its governing surface with seven provisional credential roles and concrete lifecycle acceptance. Source review qualifies bearer scope/expiry, ambient SSH identity and backup integrity. Full-policy none/inactive remains; no credential or backup contents were accessed and no lifecycle operation ran.

## Thirty-ninth pass: service release metadata

The custody pass was progress. Inventoried eight remaining local service packages and adjacent files. ADR-2030 becomes partial for missing promised licence texts/READMEs; governing document and subtree notice qualify the packaging account. All 32 operative agentbox records now carry closeout extensions, while six stale baselines, historical decisions and implementation acceptance remain open. No registry, publication or legal compatibility assessment ran.

## Fortieth pass: VisionClaw replay boundary

The service-package pass was progress. Extended VisionClaw ADR-2002 and security governing surface. Six actual extracted helper assertions pass; source traces optional payload binding and token consumption before user resolution. Complete/live remains scoped to the cache decision. No signature, HTTP, mutex-race or deployed test ran; combined boundary and mutation recovery remain open.

## Forty-first pass: VisionClaw visibility

The replay pass was progress. Six native domain filter tests pass; current source filters both initial graph and position stream, superseding older missing-initial-filter accounts. Extended ADR-2003 and security governing surface with metadata authority, output coverage and client transition acceptance. No socket, browser or metadata mutation ran.

## Forty-second pass: VisionClaw role authority

The visibility pass was progress. Traced caller/target transaction boundaries, fallback after explicit-role removal and central gate exceptions/report mode. Extended ADR-2010 to partial and qualified ADR-2011, with governing document and new authority chapter. No database, route or race execution ran; acceptance remains explicit.

## Forty-third pass: development features and release controls

The role pass was progress. Nine extracted-helper build cases confirm feature/debug/variable behaviour. Extended ADR-2037/2038/2039 and corrected a proposed CI assertion cited as guaranteed. Full image/profile/transport acceptance remains open; no server or headset ran.

## Forty-fourth pass: profile dependency reconciliation

The build-feature pass was progress. Extended ADR-2012/2026/2027; corrected report-mode release/expiry and paired-environment statements, retaining scoped implemented controls and partial profile status. Prior nine-case source hashes match. No new runtime test or deployment ran.

## Forty-fifth pass: vault conversion boundaries

The profile pass was progress. Ran 70 unit and 16 integration converter tests, then two actual CLI fixtures. Mixed legacy/folder names collide with successful exit and one output body; explicit dry-run report writes. ADR-2042 becomes partial, governing format and roadmap gain collision/recovery acceptance. Source fixtures remain intact; no real or in-place corpus conversion ran. ADR-2040 inclusion review remains open.

## Forty-sixth pass: vault inclusion and fallback

The converter pass was progress. Ran 56 domain vault tests and traced metadata parsing, GitHub authored-page inclusion and startup local fallback. Extended ADR-2014 lineage and ADR-2040 to partial for typed/shared-reader guarantees, with governing format qualifications. No actual corpus scan, sync, graph ingest or public disclosure was exercised.

## Forty-seventh pass: knowledge settings compatibility

The inclusion pass was progress. Three Rust alias tests and eight client migration tests pass. Extended ADR-2041 and vault governing surface with persisted/patch/transport/rollback acceptance, preserving scoped complete/staged status. No live storage or settings mutation ran.

## Forty-eighth pass: identifier mint sites

The settings pass was progress. Revalidated three prior helper source hashes and inspected direct provenance/legacy mint sites plus DID proof checking. Extended ADR-2021 to partial and qualified ADR-2022, with governing taxonomy and roadmap acceptance. No persistent ID, named graph or signed challenge was changed or exercised.

## Forty-ninth pass: request realms and deferred delegation

The identifier pass was progress. Extended VisionClaw ADR-2009/2013 and identity governing surface. Current client signing supersedes the historical interceptor-dependence claim; server legacy credentials remain. Eleven mocked interceptor tests pass using the installed runner after npm's launcher failed with ENOSPC. No live authentication, revocation, relay or federation deployment ran. Full estate assessment remains unfinished.

## Fiftieth pass: simulation and compact wire contracts

The request-realm pass was progress. Extended VisionClaw ADR-2024/2028/2029 and GPU/identifier governing surfaces. Extracted Rust/C++ structs agree on 53 offsets; a same-size field swap still passes the original size assertion. Traced final force dispatch and the untyped wire overflow branch. No CUDA compilation, GPU tick, over-range frame or deployed client ran. Full estate assessment remains unfinished.

## Fifty-first pass: PTX acquisition and compatibility

The simulation pass was progress. Executed six fixtures through the unchanged extracted PTX build phase and traced runtime selection/validation. Extended ADR-2030 to partial, corrected missing-nvcc/build-guarantee prose and updated GPU governance and roadmap. No CUDA compiler, native phase, driver load or kernel launch ran. Full estate assessment remains unfinished.

## Fifty-second pass: XR controls and hierarchy

The PTX pass was progress. Extended ADR-2032/2033/2035, setting HUD implementation partial, and recorded ADR-2031 tombstone disposition. Eleven constructor sites contain three explicit press-mode assignments; the extracted existing hierarchy test fails against the unchanged predicate. Updated XR governance and roadmap. No Godot, headset, full actor suite or GPU layout ran. Full estate assessment remains unfinished.

## Fifty-third pass: ACSP consumers

The XR pass was progress. Extended VisionClaw ADR-2006 to partial and qualified baseline claims about stateless approval and retained-kernel integration. Traced subscription, response projection, pending-map consumption and proposal inbox storage. No live event, actor race, failed write or PR ran. Full estate assessment remains unfinished.

## Fifty-fourth pass: extraction and GPU supervision

The ACSP pass was progress. Extended ADR-2005/2007 and baseline, preserving partial extraction and marking supervision partial against broadcast/isolation claims. Recorded current manifest and actor counts; traced direct context distribution and restart boundaries. No build timing, actor failure or GPU recovery ran. Full estate assessment remains unfinished.

## Fifty-fifth pass: development startup and operative corpus

The supervision pass was progress. Extended ADR-2001/2008, completing all 42 operative numbered dispositions. Three extracted timestamp cases expose missed crate CUDA/manifest edits; actual validator still fails on four stale records. Updated baseline and review/roadmap. No Cargo build/clean, Docker or service operation ran. Historical/upstream reconciliation and overall assessment remain unfinished.

## Fifty-sixth pass: historical obligations

The operative-pack pass was progress. Generated routes and hashes for 137 VisionClaw historical candidates, preserving uncertainty in exact-number matching. Reviewed ADR-090/110 scope and separated ADR-130's six decisions, with liveness/KPI source reconciliation still open. Frozen records remain unchanged. Historical and upstream semantic assessment remains unfinished.

## Fifty-seventh pass: liveness observer

The historical mapping pass was progress. Traced ADR-130 D3 into durable registration, HTTP/relay observations, SHA/freshness status and KG watchdog transitions. Added an observer chapter and updated historical disposition. No request, relay event, storage write or watchdog execution ran. KPI and promotion-consumer reconciliation remain open; the full assessment is unfinished.

## Fifty-eighth pass: KPI outcomes

The liveness pass was progress. Traced historical ADR-130 D5 through event tap, two computed metrics, transactional per-metric lineage, sequential summary persistence and client polling. Added KPI chapter and updated historical disposition; two metrics remain explicitly deferred. No event, database mutation, HTTP request or browser ran. Full estate assessment remains unfinished.

## Fifty-ninth pass: liveness consumers

The KPI pass was progress. Ran the actual D1 script with fake curl: empty roster exits 2, count-only and fired:false receipt cases exit 0. Traced dashboard status/latch handling and archived canon promotion rule; no automatic promoter was established in bounded searches. No network, beam, browser or real promotion ran. Full assessment remains unfinished.

## Sixtieth pass: agentbox historical obligations

The consumer pass was progress. Generated routes/hashes for all 72 archived agentbox records, separated ADR-037 sections and preserved ADR-057/072 proposal acceptance. Added a local companion without editing frozen records. No runtime, journal write or scheduling action ran. Semantic historical/upstream assessment remains unfinished.

## Sixty-first pass: independent consultation

The agentbox history pass was progress. Executed three pure diversity helper cases and traced the optional warning-only envelope. Production selector wiring was not established in the bounded search; invocation success is distinct from closure acceptance. Updated historical D4 route and roadmap. No consultant/provider call ran. Full assessment remains unfinished.

## Sixty-second pass: voice authority

The consultation pass was progress. Ran 19 existing voice tests with explicit Jest node/root configuration and inspected mandate, target and dispatch boundaries. Updated historical D7 and roadmap with issuer/scope/revocation and applied-receipt requirements. No full-server auth, real signature/relay, microphone or scene ran. Full assessment remains unfinished.

## Sixty-third pass: execution journal

The voice pass was progress. Composed actual journal/local events adapter over a temporary invalid log directory. Five assertions confirm failed persistence still advances journal/idempotency state, notifies a subscriber and permits an in-range citation for changed text. Updated proposal assessment and roadmap; no production record or model call ran. Full assessment remains unfinished.

## Sixty-fourth pass: evaluator admission

The journal pass was progress. Executed four cases against the extracted unchanged DreamConfig validator, retaining the narrow Darwin rejection while showing broader readiness is not enforced there. Traced slot selection and later evaluator dispatch; updated proposed ADR-072 assessment. No command, SSH, model or night ran. Full estate assessment remains unfinished.

## Sixty-fifth pass: roadmap consolidation and assessment audit

The evaluator pass was progress. Consolidated accumulated findings into six ordered delivery stages, added new chapters to the reading table and refreshed the requirement audit while preserving pass-30 history. Current coverage is 862 candidates/117 markers; 392 remaining decision candidates are in RuVector/RuView. This is not completion: semantic historical/imported disposition and remaining claim reconciliation remain open. No runtime operation ran.


## Sixty-sixth pass: VisionFlow operative obligations

Extended all seven operative VisionFlow records with scoped evidence and CP-01/08/09 closeout conditions while preserving historical verification and status axes. The current validator accepts seven records; the browserless committed-baseline check passes ten diagrams. Added source hashes, qualified the governing baseline and replaced stale unexamined-consumer wording with the later engine assessment. No build, browser, hosted CI or deployment ran. Engineering, imported and historical semantic coverage remains open.


## Sixty-seventh pass: catalog decisions and support classification

The previous turn was verified progress. Reviewed nine catalog ADRs against the current instructional entry point, TypeScript search/data/proposal paths and benchmark report. Added individual closeout extensions preserving original status; qualified the historical report and local README. Classified three BHIL templates and one worked example as support without changing their teaching contents. Updated the inventory collector to recognise dated extensions beyond September 4. Source-only evidence: no search/model/swarm/build/benchmark ran. Whole-estate historical/imported and claim review remains incomplete.


## Sixty-eighth pass: forum sprint consumers

The previous turn made verified progress. Traced all three canonical sprint ADRs into current client path, count and bootstrap code; preserved archived stubs. Added seven source hashes and per-record acceptance extensions. Credited implemented mechanisms and identified page-replay tombstone and local-projection reconciliation gaps. No browser, relay, auth or service-worker execution ran. Whole-estate assessment remains unfinished.


## Sixty-ninth pass: engineering decisions

The forum pass made verified progress. Reviewed all 17 sections across engineering ADR-004/005 and extended both records. Current audit and two synthetic fixtures executed the unchanged script: missing sources pass, duplicate pairings produce 200%. Added schema/workflow/source scope and linked existing authority findings. No hosted CI, provisioning, precedent or mandate operation ran. Upstream/historical semantic review and system claims remain open.


## Seventieth pass: upstream pack and consumer identity

The engineering pass made verified progress. Generated 393-decision/17-family routes, captured nine registry package/version identities and full diffs for 15 divergent ADR pairs. Classified three ADR-authoring agent definitions as support. Linked consumed-source distinctions and prioritised semantic review without declaring family grouping complete. No build, package resolution, deployed or hardware operation ran. Upstream/historical semantic and system-claim review remains open.


## Seventy-first pass: RuView signal integration

The upstream mapping pass made verified progress. Traced ADR-017’s seven helpers and exact-symbol references, distinguishing availability from runtime selection. Identified graph rebuild, measurement accumulation, solver fallback and append-only compressed-buffer scope. Extended the ADR and roadmap; no algorithm or hardware test ran. Remaining upstream/historical semantic and system-claim assessment is open.


## Seventy-second pass: RuView training integration

The signal pass made verified progress. Reviewed ADR-016’s five integration claims against training source and symbol references; qualified its blanket historical completion table. Captured six hashes and caller-search evidence. No compilation, gradient, model training or benchmark ran. Remaining upstream/historical dispositions and system claims are open.


## Seventy-third pass: dataset separation and metrics

The training-integration pass made verified progress. Reviewed ADR-015’s five phases and traced real-data CLI validation to synthetic samples. Compiled unchanged eval.rs in a temporary harness; six assertions confirm imputation and empty-input defaults. Extended the ADR and roadmap. No dataset download, training, hardware or held-out evaluation ran. Whole-estate historical/upstream and system-claim assessment remains open.


## Seventy-fourth pass: MERIDIAN adaptation

The dataset pass made verified progress. Reviewed ADR-027’s seven phases and source wiring. Four assertions over unchanged rapid_adapt.rs show one frame accepted despite a 200-frame readiness minimum, with zero loss and initial weights. Extended the proposed ADR and roadmap; no real model installation, calibration or training ran. Whole-estate upstream/historical and system-claim assessment remains open.

## Seventy-fifth pass: canon claims and reading structure

The adaptation pass made verified progress. Reconciled eleven canon-chapter assertions against scoped implementation reviews, retained chapter/source hashes and connected the argument to delivery acceptance. Added sensing evidence navigation and removed stale initial-only framing. No runtime or external literature verification ran; other book chapters and remaining historical/upstream assessment remain open.

## Seventy-sixth pass: published roadmap commitments

The canon pass made verified progress. Mapped all eighteen book roadmap commitments to current evidence and closeout packages, retaining explicit unexamined consumers and measurement/falsification obligations. Qualified same-family refusal, Mesh Velocity and hardware-only XR closure accounts without rewriting dated history. No runtime, external pilot or scoring exercise ran. Full assessment remains open.

## Seventy-seventh pass: joined provenance consumer

Traced the registered HTTP route through service, both repository reads and API RBAC. Recorded six source hashes and ER-093. Qualified book commitment 11 and extended ADR-2016 with its consumer acceptance obligations. No runtime or existing tests executed; upstream and historical semantic review remains open.

## Seventy-eighth pass: transaction-cost accounting

Traced recorder, emitter, schema, capture and available consumer references. Seventeen existing utility/route tests and five synthetic helper assertions passed. Added ER-094, eleven source hashes and producer/receiver semantic obligations; qualified book commitment 3 and extended trajectory ADR-2015. No live workflow or displayed CTC was verified. Upstream and historical assessment remains open.

## Seventy-ninth pass: failure telemetry boundaries

Reviewed classifier, recorder, publisher, route, receiver and metrics consumer. Sixteen existing tests and six in-process route assertions passed; captured twelve source hashes and ER-095. Qualified book commitment 5 and trajectory ADR-2015. QE and independent-runtime census, live delivery and rendered counts remain open alongside upstream/historical assessment.

## Eightieth pass: decision history and context

Traced the registered decision read API, NIP-98 gate, D1 query/shape, client event history and signed response context. Added ER-096 and six source hashes, qualified book commitment 17 and extended forum ADR-2010. This was source-only; signed requests, D1, browser and complete history recovery remain untested. Upstream/historical assessment remains open.

## Eighty-first pass: agent disclosure

Traced public disclosure endpoint, admin registration provenance, one-shot client cache and sixteen literal badge mounts. Added ER-097 and a source/mount receipt, qualified book commitment 13 and extended forum ADR-2010 with display boundaries. No browser/runtime acceptance ran; complete author-surface census and historical attribution remain open.

## Eighty-second pass: pocket provenance

Verified mirror composition with nine existing tests and lookup/eviction with five in-process assertions. Six source hashes support ER-098. Updated book commitment 9, ADR-2026 and closeout requirements. No message was sent; phone, deployed access control and durable producer binding remain unverified. Upstream/historical assessment remains open.

## Eighty-third pass: fingerprint retrieval

Reviewed RuView ADR-004 and all four migration phases. Added source-qualified extension, differentiated three retrieval implementations and ran five assertions against unchanged extracted server code. Recorded ER-099 and source hashes. No full server/model/sensor or scale benchmark ran; upstream and historical review remains open.

## Eighty-fourth pass: witness chain and audit completeness

Reviewed RuView ADR-010 across all eight proposed event types, actual witness segment producers/readers and illustrative verification/persistence design. Added ER-100 and five source hashes. Preserved Deferred while qualifying stale incidental-only wording. No runtime, training, cryptographic or regulatory validation ran. Remaining upstream/historical assessment is open.

## Eighty-fifth pass: secure sensing and model admission

Reviewed all five ADR-007 protection layers and actual server/firmware loading boundaries. Added ER-101, seven hashes and bounded PQ symbol search; retained Deferred with explicit primitive and deployment requirements. No cryptographic, device, network or compliance test ran. Remaining upstream/historical assessment stays open.

## Eighty-sixth pass: multi-device coordination

Reviewed ADR-008 against all five coordination requirements and current multistatic helper. Added ER-102, three hashes and bounded search output. Deferred status retained; no distributed algorithm or sensor run. Upstream/historical assessment remains open.

## Eighty-seventh pass: edge-runtime lifecycle

Reviewed ADR-009 across four profiles and concrete later firmware runtime. Added ER-103, four hashes and bounded portable-interface search. Preserved Deferred while qualifying no-code wording. No device/browser/WASM execution ran; upstream/historical assessment remains open.

## Eighty-eighth pass: SONA adaptation lifecycle

Reviewed ADR-005 triggers, feedback, safety, profiles and four validation steps. Twenty unchanged native tests and six assertions passed. Added ER-104 and receipt; preserved partial status. No real model/room or deployment was evaluated. Upstream/historical assessment remains open.

## Eighty-ninth pass: GNN mode coverage

Reviewed ADR-006 across three modes, spatial implementation and proposed online learning. Twenty-eight unchanged native tests pass; recorded ER-105 and four hashes. Partial status retained. No actual CSI/model/server acceptance ran; upstream/historical assessment remains open.

## Ninetieth pass: sensing review navigation

Added a reproducible 45-record local RuView map, separate from vendored copies, with direct existing assessment links. The generated count corrects the preliminary manual estimate: 15 extended decisions and 30 awaiting dedicated review. Expanded sensing navigation across all current evidence sections. No new implementation finding or ADR marker added; remaining semantic review is explicit.

## Ninety-first pass: foundational Rust ADR pack

Extended all three Rust workspace ADRs after manifest and source review. Recorded fifteen members, excluded device crate, partial quality-guided phase path and unbound ONNX provider options. Added ER-106 and receipt. No build/model/target acceptance ran; upstream/historical assessment remains open.

## Ninety-second pass: sensing UI provenance

Reviewed all five ADR-019 commitments against current client handlers. Four isolated Node assertions passed; seven hashes support ER-107. Preserved Accepted and added source/freshness acceptance conditions. No browser, socket, hardware or model run occurred. Upstream/historical assessment remains open.

## Ninety-third pass: survivor lifecycle

Reviewed ADR-026 components, three integration steps and five events. Ten unchanged lifecycle/Kalman tests and two extracted assignment assertions passed; six hashes support ER-108. Documented scan/query integration gap, matching objective, tentative visibility, raw position and rescue retention semantics. Accepted retained; no complete MAT or field acceptance ran. Upstream/historical assessment remains open.

## Ninety-fourth pass: mobile sensing companion

Reviewed ADR-034 phases, five screens, 31 acceptance criteria and five future increments. Five isolated WebSocket assertions pass; receipt records selected source hashes, 25 placeholder tests and six empty Maestro files. Added a dedicated mobile chapter and qualified consumed-ADR relationships. No native/browser/full-suite/typecheck/field acceptance ran; upstream/historical assessment remains open.

## Ninety-fifth pass: training UI contracts

Reviewed all fourteen ADR-036 implementation items. Thirteen source hashes and two isolated model-service assertions support ER-110. Qualified spec-only wording while preserving Deferred; documented unmounted handlers, linear signal-teacher objectives, request shapes and lifecycle semantics. No live training/server/browser acceptance ran; upstream/historical assessment remains open.

## Ninety-sixth pass: macOS sensing bridge

Reviewed ADR-025 principles, helper modes, pipeline reuse, nineteen verification rows and five future items. Eight hashes support ER-111. Credited separate Python connected-AP path while recording Rust/helper protocol mismatch and missing main dispatch. Partial status retained; no macOS/Swift/hardware or external research validation ran. Upstream/historical assessment remains open.

## Ninety-seventh pass: review navigation maintenance

Updated the main reading order to include engineering, catalog, mobile and upstream assessments and the local sensing decision map. Replaced the stale four-extension RuView summary with an evidence-area description and generated coverage link. Added a reproducible structural checker for inline local links, internal heading/HTML anchors and incoming page links. This does not establish external links, rendered documentation or semantic completion. No new ADR extension or runtime finding was added; upstream/historical assessment remains open.

## Ninety-eighth pass: Python proof replay

Reviewed ADR-011's five decisions, twelve concrete change rows and combined acceptance claim. Seventeen hashes and a failed default replay (NumPy absent) support ER-112. Credited mock-boundary/parser improvements while distinguishing synthetic provenance, warning-only CI and unverified build gates. No successful replay/build/hardware acceptance ran; upstream/historical assessment remains open.

## Ninety-ninth pass: roadmap planner design

Reviewed ADR-038 state/actions/goals, search methods, modules, integrations, commands and budgets. Seven proposed module paths are absent. Recorded heuristic and PageRank counterexamples in ER-113 and a source/design receipt. Deferred retained; no planner/CLI/swarm execution ran. Upstream/historical assessment remains open.

## Hundredth pass: Rust migration contract

Reviewed ADR-020 phases, twelve replacement mappings, commands and migration steps. Seven hashes establish fifteen members, absent MAT ONNX feature/binary and separate router/listener composition. Accepted retained; no Cargo/model/target acceptance ran. Upstream/historical assessment remains open.

## Hundred-and-first pass: CRV stage contracts

Reviewed ADR-033's four phases, stage/dependency mappings and 37 criteria. Four hashes and bounded bridge search support ER-115. Qualified Implemented to the facade, distinguishing two-stage frame method, separate later calls, defaults, zero timestamps and room-key convergence. No suite/benchmark/hardware/identity acceptance ran; upstream/historical assessment remains open.

## Hundred-and-second pass: integration strategy lineage

Reconciled ADR-002 against its successors, eight rollout domains, six limitations, additional attention/branching and five mitigations. Six source hashes support ER-116. Preserved Superseded and historical notes with a prominent evidence qualification, corrected a workspace ADR link and retained dedicated ADR-003 review as pending. No runtime/publication/benchmark acceptance ran; upstream/historical assessment remains open.

## Hundred-and-third pass: cognitive container lifecycle

Reviewed ADR-003 types, adapter, lifecycle, storage and performance claims. Seven source hashes, bounded symbol search and two arithmetic assertions support ER-117. Deferred retained with existing packaging credited. No Cargo/device/failure-injection/benchmark acceptance ran; upstream/historical assessment remains open.

## Hundred-and-fourth pass: commodity sensing

Reviewed ADR-013 input/extraction/classification contracts, capabilities, deployment comparison, proof artifacts and module/test claims. Eight source hashes and four isolated AST assertions support ER-118. Preserved Accepted with Implemented qualified to components. No numerical/full-suite/live collector/installed CLI or accuracy acceptance ran; upstream/historical assessment remains open.

## Hundred-and-fifth pass: signal algorithm semantics

Reviewed ADR-014 six algorithms, four per-module promises, five benefits and three costs. Seven hashes, bounded caller search and three isolated compiled Fresnel assertions support ER-119. Accepted retained; no full crate/FFT/captured-data/benchmark acceptance ran. Upstream/historical assessment remains open.

## Hundred-and-sixth pass: contrastive objective boundary

Made a partial ADR-024 assessment of projection, loss and consolidation. Four hashes and three unchanged-function compiled assertions support ER-120. Added an evidence note without a complete extension marker; all remaining ADR requirements retain pending review. No full training or dataset evaluation ran.

## Hundred-and-seventh pass: embedding augmentation and state

Continued partial ADR-024 review across seven augmentation proposals and projection/LoRA persistence. Seven hashes and six compiled assertions support ER-121. Recorded merge output change, adapter omission and amplitude-scaling phase surrogate. No full suite/training/RVF/device round trip ran; remaining ADR requirements remain pending.

## Hundred-and-eighth pass: embedding quantisation and budgets

Continued partial ADR-024 review through quantisation, parameter budgets, eleven performance targets and two deployment proposals. Seven hashes and six compiled assertions support ER-122. No full model inference, target benchmark or full ADR completion is claimed.

## Hundred-and-ninth pass: cumulative AETHER closeout

Completed ADR-024 documentation review across all seven phases, 33 named tests, four indices, consequences/risks and five future directions. Added requirement receipt and three compiled mining assertions. Corrected unattainable variance acceptance interpretation. Partially Implemented retained; no full runtime/quality/target acceptance is claimed.
