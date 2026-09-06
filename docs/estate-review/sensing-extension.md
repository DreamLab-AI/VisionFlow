---
title: RuView sensing extension and evidence boundaries
status: in-progress
date: 2026-09-04
type: explanation
---

# RuView sensing extension and evidence boundaries

RuView could give the estate physical presence and activity inputs without a camera. The inspected checkout contains signal processing, firmware, transport, simulation and inference-related components, but these do not form a verified VisionClaw sensing journey. Searches found no named RuView/wifi-densepose consumer in VisionClaw's inspected source/manifests or agentbox's main manifest/flake. Treat this as an evaluated extension until a consumed revision, adapter and authority contract are identified. The [receipt](evidence/ruview-snapshot.json) records search scope, source hashes and tests.

## Reading the evidence

The [local decision map](closeout/sensing-decisions.md) lists all 45 RuView decisions and their current review state. The [upstream map](closeout/upstream-packs.md) separately retains vendored RuVector decisions. Neither map treats an extension marker as implementation acceptance.

| Question | Evidence section |
|---|---|
| Has the primary-backend migration reached a deployable contract? | [Rust migration](#rust-primary-backend-migration) |
| What do the foundational Rust decisions currently cover? | [Workspace, signal libraries and backends](#rust-workspace-and-foundational-contracts) |
| What does the mobile companion actually consume? | [Mobile screens and 31 acceptance criteria](mobile-sensing.md) |
| Can a viewer distinguish hardware, simulation and stale observations? | [Sensing UI provenance](#sensing-ui-source-and-freshness-contract) |
| What does commodity RSSI sensing establish? | [Capabilities and observation readiness](#commodity-sensing-capabilities-and-observation-readiness) |
| What is the macOS acquisition path? | [CoreWLAN bridge](#macos-helper-and-runtime-contract) |
| Does the CRV facade establish six-stage sensing and room identity? | [CRV stage contracts](#crv-stage-facade-and-identity-evidence) |
| Could an automated roadmap planner safely select closeout work? | [Planning design](#roadmap-planning-design-and-evidence-state) |
| What does proof replay establish? | [Python determinism and mock boundaries](#python-proof-replay-and-mock-boundaries) |
| Which input and output paths exist? | [Runtime paths](#separate-paths-separate-maturity), [codec mismatch](#firmware-and-server-parser-disagree), [model labels](#model-labels-and-observation-quality) |
| Does embedding quantisation validation establish deployable model fidelity? | [Quantisation and deployment evidence](#embedding-quantisation-and-deployment-evidence) |
| Do embedding augmentations and saved adapters preserve their contracts? | [Augmentation and projection state](#embedding-augmentation-and-projection-state) |
| How does the full AETHER decision close out? | [Cumulative requirements](#aether-cumulative-closeout-contract) |
| Does contrastive pretraining optimise the stated objective? | [Training objective and consolidation](#contrastive-training-objective-and-consolidation-boundary) |
| Can the UI deliver a trained and activated model? | [Training UI contracts](#training-ui-and-model-operation-contracts) |
| What do the six original advanced signal algorithms establish? | [Algorithm semantics and adoption](#signal-algorithm-semantics-and-runtime-adoption) |
| Are declared algorithms used by consumers? | [Signal/MAT integration](#signal-and-mat-integration-helper-availability-versus-execution), [training integration](#training-integration-and-model-evidence), [GNN modes](#gnn-mode-coverage-and-learning-evidence) |
| Do retrieval and evaluation establish useful predictions? | [Fingerprint retrieval](#fingerprint-retrieval-and-hnsw-acceptance), [dataset separation](#dataset-separation-and-evaluation-meaning) |
| Does tracking preserve identity and confirmed counts? | [Survivor lifecycle](#survivor-tracking-and-operational-integration) |
| Does adaptation change a validated, recoverable model? | [Rapid adaptation](#cross-environment-adaptation-and-readiness), [SONA lifecycle](#sona-feedback-admission-and-profile-lifecycle) |
| Which container capabilities and durability guarantees exist? | [Container lifecycle](#cognitive-container-contract-and-durable-lifecycle) |
| What establishes integrity, authority and complete history? | [Witness/audit completeness](#witness-segments-and-audit-completeness), [secure model admission](#secure-sensing-claims-and-model-admission) |
| What survives device boundaries and offline operation? | [Multi-device coordination](#multi-device-agreement-and-replicated-state), [edge-runtime lifecycle](#edge-runtime-implementation-and-isolation-boundaries) |

Read these as evidence layers: a helper test, selected runtime path, trained-model evaluation and complete estate journey prove different things. The [closeout conditions](#closeout-conditions) apply across the sections; later findings qualify the initial assessment rather than silently replacing its historical evidence.

## Separate paths, separate maturity

The disaster-response `wifi-densepose-mat` [hardware adapter](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-mat/src/integration/hardware_adapter.rs) returns unimplemented errors for ESP32, Intel, Atheros, UDP and PCAP paths; its simulated device produces data. Its [neural adapter](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-mat/src/integration/neural_adapter.rs) returns success from `load_models` while deliberately keeping `is_loaded` false and using rule-based classification. The log warns about this fallback. A successful method return therefore does not prove a model was loaded.

The separate [sensing server](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/main.rs) does implement UDP reception and ESP32 packet parsing. Auto-selection probes for a packet for two seconds, then Windows WiFi, then simulation. Missing a device during this startup probe can select simulation; the selected background task is not itself proof of ongoing source discovery. Packet-source labels describe the selected path, not validated measurement accuracy.

The README previously generalised the MAT adapter limitation to the entire repository and said export wrote placeholders. Both statements are corrected: another receive path exists, and `--export-rvf` currently refuses to write a model and exits 1. Neither correction establishes hardware or trained-model validation.

## Firmware and server parser disagree

The [firmware encoder](../../../RuView/firmware/esp32-csi-node/main/csi_collector.c) and [hardware library parser](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-hardware/src/esp32_parser.rs) use a 20-byte header with a two-byte subcarrier count, four-byte frequency, sequence at offset 12, RSSI at 16 and noise at 17. The sensing server's private parser instead reads a one-byte count, two-byte frequency, sequence at 10, RSSI at 14 and noise at 15.

The [reproducible probe](evidence/ruview-parser-probe.py) compiles the actual extracted server parser and supplies synthetic bytes in the firmware layout. [Observed result](evidence/ruview-parser-probe.json): sequence **42 → 2,752,512**, RSSI **−45 → 0**, noise **−95 → 0**. The 64-subcarrier I/Q amplitudes still decode, which could conceal the header error in a moving visualisation. This is a deterministic parser mismatch, not a physical-device experiment.

All **100 hardware-crate library tests pass**, including parser and loopback UDP tests. They cover a different parser from the sensing server's private implementation. This is why a green component suite cannot close the firmware-to-consumer contract.

## Model labels and observation quality

The sensing server sets `model_loaded` when `ProgressiveLoader::new` succeeds, even if loading layer A returns an error. Its pose output labels the mode `model_inference` from that flag, but can fall back to `derive_pose_from_sensing` when keypoints are absent. Container availability, trained-weight provenance and the per-frame inference actually used need separate reporting. This source pass does not establish trained model accuracy.

The [RVF reader](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_container.rs) explicitly documents CRC integrity without cryptographic signature verification or enforcement of declared capability flags. A container flag is therefore not an enforced loading permission. No real-world sensing or physiological-accuracy claim is established by these local tests.

## Closeout conditions

CP-01/06 must first identify whether RuView is an estate-operated component or an external dependency, and locate the actual consumer before claiming integration. An adopted path needs a pinned revision, schema and timestamp mapping, sensor identity, authorised collection context, retention/deletion rules and explicit simulated/heuristic/model provenance carried into graph data and UI.

CP-06/08 needs one canonical codec exercised from firmware fixture through the real server receiver, malformed/truncated packet checks, correct source-loss behaviour, per-source timing and uncertainty, and a versioned physical capture with independently recorded ground truth. Any trained-model claim additionally needs actual weight hashes, training/evaluation provenance and a test proving fallback frames retain honest labels. Keep simulation and physical acceptance receipts distinct.

The initial four RuView ADR extensions scope historical runtime claims; later sections assess training, dataset and adaptation decisions. Together they record dependencies and evidence limits; they do not adopt upstream decisions on behalf of maintainers or certify deployment. The remaining upstream and vendored ADRs need explicit lineage and ownership disposition in the [estate roadmap](closeout/README.md).

## Signal and MAT integration: helper availability versus execution

RuView ADR-017 names seven RuVector integration points. [Current source and caller-search evidence](evidence/ruview-signal-integration.json) confirms corresponding helpers in signal/MAT and a parallel integration crate. The bounded exact-symbol search across Rust crates primarily finds definitions, exports and tests; it does not establish their activation in the main sensing flow. Aliased calls, external consumers and deployed profiles require separate tracing. No algorithm test, sensor or hardware run occurred in this pass.

| Integration | Implemented source mechanism | Remaining obligation |
|---|---|---|
| Subcarrier mincut | Rebuilds a pairwise graph, computes an exact partition and selects the higher-mean side | Establish consumer selection, retained incremental state and measured benefit over existing variance/sensitivity sorting |
| Spectrogram gating | Separate attention/mincut gating helper exists | Bind dimensions/layout to the real STFT consumer and evaluate downstream model quality |
| BVP attention | Calls scaled-dot-product attention; errors fall back to a sensitivity-weighted sum | Record selected/fallback path and compare actual velocity-estimation outcomes |
| Fresnel geometry | Separate solver helper exists | Trace the caller, numerical validity and geometry-estimation error |
| TDoA triangulation | Feature-gated helper accumulates normal equations, tries Neumann and falls back to a direct 2×2 solve | Main triangulator still calls its existing least-squares method; establish selection and calibrated localisation evidence |
| Breathing compression | Feature-gated wrapper appends encoded frames and exposes flush/decode | Integrate detector consumption, retention, flush visibility and bounded memory/accuracy measurements |
| Heartbeat compression | Per-bin compressed append buffers and decoded recent-band power | Verify retained-window semantics and quantitative error through the actual consumer |

The mincut helper constructs pairwise edges on every invocation. A dynamic-capable library does not make this wrapper incremental. Likewise, the TDoA helper processes every measurement before solving the fixed-size system: fixed solve dimension does not make the complete operation constant in measurement count. Invalid AP indices fall back to reference coordinates in the inspected helper; input rejection and numerical residuals need explicit acceptance criteria.

Both compressed wrapper implementations retain encoded vectors without an eviction path in the inspected types. Selecting recent decoded samples is not bounded storage retention. Push synchronises access timestamps with the current frame count; the ADR's wall-clock hot/warm/cold claims need verification against the actual compressor policy and input cadence. No measured 50–75% saving, increased zone capacity or signal-accuracy improvement is established by wrapper presence.

CP-01/06/07/08/09 requires selected-path receipts for all seven integrations, explicit fallback reporting, duplicate-wrapper ownership, representative before/after measurements and negative input/recovery tests. Preserve ADR-017's accepted design status while qualifying implementation and activation. Its historical package-availability discussion is dated; use the current lockfile and consumed-source map for source identity rather than treating older “non-existent crate” wording as current package information.

## Training integration and model evidence

RuView ADR-016 marks five training integrations complete and always-on. The [current source receipt](evidence/ruview-training-integration.json) supports a narrower, mixed account. This pass inspected the training crate and exact-symbol references; it did not compile, train a model, test gradients or benchmark prediction quality.

| Integration | Current evidence | Closeout obligation |
|---|---|---|
| Dynamic person matching | DynamicPersonMatcher exists; assignment_mincut constructs a fresh matcher per call | Establish retained state across frames and valid one-to-one minimum-cost assignments against an independent oracle |
| Antenna mincut attention | ModalityTranslator calls apply_antenna_attention before its learned FC branches | Verify numerical layout, training semantics, device-transfer cost and held-out benefit |
| CSI compression | CompressedCsiBuffer exists; exact-symbol uses in the inspected training source are its definition/implementation and tests | Connect storage to MmFiDataset or qualify as optional helper; measure peak memory and reconstruction error |
| Sparse interpolation | Sparse solver helper exists; dataset retains the ordinary interpolation path | Establish explicit algorithm selection, fallback/residual reporting and numerical comparison |
| Spatial attention | apply_spatial_attention is marked allow(dead_code), describes future use and has no caller in the inspected training source | Integrate into the intended heads or retain inactive status; measure model and training effects |

The mincut wrapper extracts crossing prediction-to-ground-truth edges; that alone does not prove assignment cardinality, uniqueness or minimum total cost. Its insert/delete methods discard returned errors. Availability of graph operations is not a verified dynamic tracking lifecycle. The retained Hungarian path remains relevant to deterministic evaluation; do not infer metric equivalence from both methods returning index pairs.

The antenna bridge moves tensors to CPU vectors, runs the Rust attention kernel and constructs fresh tensors from returned values before the FC branches. This is a host computation boundary requiring explicit gradient semantics and performance evidence. It does not show that downstream FC weights cannot train; nor does the uncalled spatial helper establish a trained spatial-attention decoder. Any future insertion after learned features needs appropriate gradient verification.

CompressedCsiBuffer simulates access history while encoding an input array. This does not establish live recency policy, transparent dataset use or the ADR's 50–75% end-to-end memory reduction and larger batch capacity. The sparse solver's iteration claim also needs whole-operation measurement, including matrix construction and fallback, rather than inheriting a library complexity label.

CP-01/06/07/08/09 requires per-integration implementation and activation status, a defined default training path, input/data/split/model identities, assignment and gradient checks, and measured accuracy/memory/latency comparisons. Keep ADR-016 Accepted as a design choice while qualifying its historical completion table. A declared integration is not a trained-model acceptance receipt.

## Dataset separation and evaluation meaning

ADR-015 requires an MM-Fi subject holdout (33–40), a secondary Wi-Pose loader, teacher-generated DensePose labels, model training and a deterministic synthetic proof. These are distinct evidence obligations. Current source and [six native metric assertions](evidence/ruview-evaluation-probe.json) establish the following limits; no dataset download, model training or held-out evaluation ran.

| ADR phase | Current evidence and disposition |
|---|---|
| MM-Fi loader and holdout | Loader discovers subject/action directories and ordinary interpolation exists. The real-data CLI passes all discovered samples to training, then constructs synthetic validation samples; it does not implement the specified 33–40 holdout. Externally curated directories could change input composition, but are not a recorded subject-split contract here. |
| Wi-Pose | No WiPoseDataset implementation was identified in the inspected training dataset module. Retain the secondary-loader obligation; this bounded finding is not a repository-wide absence claim. |
| Teacher labels | The inspected trainer passes None for the optional DensePose/transfer inputs in its loss call. Availability of a multi-term loss does not establish teacher-label generation, cache provenance or use in training. |
| Training and checkpoint selection | Trainer evaluates the supplied validation dataset. Under the CLI's real-data path that dataset is synthetic, so best validation PCK is pipeline-verification evidence, not held-out human/environment accuracy. |
| Synthetic proof | Proof source describes fixed-seed loss decrease and expected weight-hash comparison, with missing-hash skip semantics. This pass did not execute it. Even a successful synthetic proof would not certify generalisation to real observations. |

The cross-domain evaluator introduces additional interpretation risks. When domain 2 is absent, few_shot_mpjpe is the mean of source and cross-domain errors; missing domain 3 uses the cross-domain error as hardware error. adaptation_speedup is an error ratio, despite its field description referring to labelled-sample savings. Cross-domain aggregation includes all nonzero domains, including adaptation/hardware labels. These computed fields must not be presented as independently measured experiments.

The unchanged evaluator compiled successfully in a temporary native harness. Six assertions confirm that errors of 1 and 3 for domains 0/1 yield imputed few-shot error 2, hardware error 3 and adaptation_speedup 1.5 without adaptation or hardware observations. Empty coordinate vectors passed to mpjpe with 17 joints return zero through missing-coordinate defaults. Empty evaluation returns cross-domain error zero and gap ratio one. These are isolated metric behaviours, not evidence that a real report used such inputs.

CP-01/06/07/08/09 requires a revision-bound subject/environment/device split manifest, an independent test set, explicit missing-data results, dimension/unit validation and separate observed versus imputed metrics. Match claims to actual sample counts and experiment definitions. Synthetic pipeline proof, real validation accuracy, adaptation efficiency and hardware transfer must have distinct receipts.

## Cross-environment adaptation and readiness

RuView ADR-027 (MERIDIAN) remains Proposed. It describes a seven-phase route from hardware normalisation to domain-adversarial training, geometry conditioning, virtual augmentation, rapid adaptation, evaluation and deployment. Source contains corresponding normalisation and vector-based factorisation/geometry/augmentation/adaptation primitives. Their existence does not establish the proposed integrated training and serving path.

The [source and native probe receipt](evidence/ruview-adaptation-probe.json) captures a bounded Rust caller search. The inspected trainer, dataset and model do not wire the named domain factoriser, gradient-reversal layer, geometry conditioning or rapid-adaptation components into the main path. A standalone backward method that negates a supplied vector is not an autograd connection. Likewise, generating geometry-conditioned vectors does not prove the pose model consumes them. External or aliased consumers remain outside this bounded finding.

RapidAdaptation has a capped frame buffer and exposes is_ready against min_calibration_frames. Its adapt method checks for an empty buffer and zero rank, but does not enforce that configured minimum. It starts a fresh weight vector at 0.01 and updates it using manually written contrastive/entropy steps. Its result has no binding to a base model, target layer set, environment identity or held-out outcome. Those associations would have to be supplied by a caller.

Four assertions against unchanged rapid_adapt.rs compiled in a temporary native harness confirm that one two-value frame under a 200-frame requirement is not ready, yet adapt succeeds, reports one frame used and zero contrastive loss, and returns unchanged initial weights. This demonstrates a helper admission gap, not a deployed adaptation failure. No weights were installed into a model, and no hardware calibration or training ran.

| MERIDIAN phase | Current disposition and required acceptance |
|---|---|
| Hardware normalisation | Module/export exists; verify selected dataset and serving callers, input validity, phase treatment and measured per-frame budget |
| Factorisation and adversarial training | Standalone vector primitives exist; prove loss/gradient reversal in the actual optimiser path and held-out environment behaviour |
| Geometry conditioning | Geometry/FiLM helpers exist; bind measured geometry and trained weights to the consumed pose representation |
| Virtual augmentation | Augmentation implementation exists; verify training selection, realism and improvement on independently held-out environments |
| Rapid adaptation | Helper optimisation exists; enforce readiness and validate derivatives, model/layer identity, install/rollback and post-adaptation accuracy |
| Cross-domain evaluation | Metric helpers exist with previously reproduced missing-data imputation; require observed experiments, true sample-efficiency measurement and explicit absent results |
| RVF deployment | Container availability alone does not prove MERIDIAN installation; require model/geometry/adapter compatibility, loaded-state receipts and recovery |

CP-01/06/07/08/09 requires a complete calibration-to-installed-model journey. The proposal's ten-second calibration, under-five-second adaptation, within-15%-of-baseline accuracy and greater-than-fivefold sample-efficiency targets remain acceptance targets, not results. The [dataset/evaluation review](#dataset-separation-and-evaluation-meaning) explains why an error ratio or imputed metric cannot close those targets.

## Fingerprint retrieval and HNSW acceptance

[RuView ADR-004](../../../RuView/docs/adr/ADR-004-hnsw-vector-search-fingerprinting.md) proposes primary HNSW retrieval, human/empty-pattern fusion, confirmed-observation learning and a four-stage migration away from threshold-only detection. Its historical status says partially realised through ADR-024. The current [server fingerprint index](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/embedding.rs) implements a useful cosine-search interface with four index types, but explicitly uses brute force. Every query scores all entries and sorts the full result set before truncating. That does not implement the ADR's HNSW graph, quantisation, hyperbolic distance or stated latency/recall expectations.

The [server CLI builder](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/main.rs) constructs twenty synthetic CSI windows, extracts fingerprints, runs a sample query and returns. It does not demonstrate a saved production index or a live human/empty classifier adopting the result. A separate [WiFi-scan matcher](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-wifiscan/src/pipeline/fingerprint_matcher.rs) linearly compares templates and checks input dimensions. A separate [edge MicroHnsw](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-wasm-edge/src/spt_micro_hnsw.rs) declares a single-layer graph with at most 64 vectors of dimension eight. Credit those distinct implementations; neither establishes the server ADR's 329-dimensional, million-vector performance contract.

The [native probe](evidence/ruview-fingerprint-probe.json) compiles the unchanged cosine function and fingerprint-index block in isolation. Five assertions verify empty-index anomaly handling, zero top-k, acceptance of a NaN-containing entry, and a dimension mismatch that receives zero cosine distance: query `[1]` compared with stored `[1,100]` uses only the common prefix and is treated as non-anomalous at threshold 0.1. This is an actual helper result, not a model-quality or production-input finding. Input dimension, finite-value and embedding-version checks must precede meaningful retrieval claims. Stored timestamps and anomaly flags do not themselves implement pruning or drift exclusion in this search path.

| ADR-004 migration phase | Current disposition | Closeout evidence |
|---|---|---|
| 1 Parallel HNSW and threshold logging | Brute-force helper and synthetic builder exist; live paired HNSW/threshold path not established | Select the actual implementation and record both outputs on the same real observations |
| 2 Labelled fusion-weight A/B test | No executed comparison in this review | Held-out labels, fixed splits, baselines and uncertainty with matching feature/model identity |
| 3 Increase similarity weight towards 0.7 | Proposed rollout value is not an observed deployment setting | Versioned rollout, measured false-positive/negative effects and rollback |
| 4 Threshold fallback for cold start | Empty helper reports anomaly; complete classifier fallback remains unverified | Empty/missing/corrupt/stale index and incompatible-vector tests through the selected live consumer |

CP-03/06/07/08 should establish feature dimension and model generation, durable index identity, confirmed-label admission, drift/retention policy and end-to-end caller activation. Benchmark recall against exact search at declared vector counts and hardware before adopting the ADR's speed/memory numbers. Re-identification and cross-environment benefit require their own evaluation; a named index type or compatible interface is insufficient. No full server, sensor, trained model, scale benchmark or external research validation ran in this pass.

## Witness segments and audit completeness

[RuView ADR-010](../../../RuView/docs/adr/ADR-010-witness-chains-audit-trail-integrity.md) remains deferred. Its proposed signed, cross-attested operational chain is distinct from the implemented RVF witness segment. [Source receipt](evidence/ruview-witness-snapshot.json) records five hashes and a bounded exact-symbol search; no training, cryptographic verification, container roundtrip or runtime test ran.

The [container builder](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_container.rs) serialises a caller-provided `training_hash` and metrics as JSON in segment 0x0A. Its reader parses that JSON; `has_witness` reports segment presence. Existing roundtrip tests assert preservation of supplied values, not independent training or device attestation. [rvf_pipeline](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_pipeline.rs) forwards optional training provenance. These are implemented paths, so describing all witness references as merely incidental understates the code.

The [training API](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/training_api.rs) formats `training_hash` as `sha256:` followed by two sixteen-digit hexadecimal quantities: weight-vector length and a scaled best-PCK value. It does not hash model weights, dataset, code or training transcript at this expression. Different training runs with the same length and quantised score receive the same value. This is a source-derived limitation, not a reproduced training run. Neither this string nor the container's CRC-derived integrity fields establish signed provenance or an independently verified content digest.

### Proposed event coverage

| ADR event | Current witness-chain disposition | Closeout evidence |
|---|---|---|
| ChainInit | No named operational chain implementation found in bounded Rust search | Provisioned identity, configuration/firmware binding and durable genesis |
| HumanDetected | No demonstrated detection-to-chain producer | Expected detection census and durable append acknowledgement |
| TriageDecision | No demonstrated signed decision chain | Exact input/output and actor binding; separate audit integrity from decision validity |
| DetectionCorrected | No demonstrated correction lineage | Link correction to original event without erasing history |
| ModelAdapted | Training metadata segment exists, not this append-only event chain | Actual before/after model digests and accepted adaptation receipt |
| ZoneScanCompleted | No demonstrated completion attestation | Coverage definition, missing observations and signed closure |
| CrossAttestation | No demonstrated peer anchor exchange | Independent peer identity, latest-head expectations and disconnected recovery |
| OperatorAction | No demonstrated operator-action chain | Authenticated action and correlated applied/rejected result |

The search for the ADR's `WitnessChainStore`, `WitnessEntry`, `WitnessedEvent`, `append_witness` and `verify_chain` found no exact implementation in the Rust crates. That does not assert absence of every external or differently named audit facility. It means those proposed interfaces and event producers are not established by this source review.

### Design requirements before implementation

Even the ADR's illustrative verifier needs a stronger completeness contract. It checks links between supplied adjacent entries and validates signatures only when present. It does not require scheduled anchors, an expected genesis/head or an externally known event count. A valid prefix can therefore remain internally consistent after suffix removal; an empty supplied chain also performs no checks. Treat these as specification gaps, not bugs in a deployed verifier.

The illustrative append routine advances the in-memory chain before container persistence. Define rollback or durable reconciliation on append failure and process restart, plus contiguous indices, device boot epochs, key rotation/revocation and atomic export boundaries. Cross-device attestations need trusted freshness and peer independence, not just mutually repeated claims.

CP-04/06/07/08 should choose whether this deferred capability is required for the accepted sensing scope, bind every required producer and replace descriptive training labels with verifiable content identities. Test modification, insertion, omitted anchors, truncation, missing events, reordered entries, forked heads, compromised/revoked keys and offline recovery against independent expectations. Benchmark overhead on the chosen hardware and actual record sizes. The ADR's legal/regulatory and operational claims remain unevaluated proposals; this technical review supplies no compliance or decision-safety certification.

## Secure-sensing claims and model admission

[RuView ADR-007](../../../RuView/docs/adr/ADR-007-post-quantum-cryptography-secure-sensing.md) remains deferred. A bounded source search found no selected post-quantum algorithm or proposed hybrid-interface references in Rust/TOML/Python/C/header files under rust-port, firmware and v1, excluding vendor. [Seven source hashes](evidence/ruview-secure-sensing-snapshot.json) establish this pass. That search does not assess external dependencies, deployed proxies or every possible implementation spelling.

The [server container reader](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_container.rs) checks segment structure and CRC-derived fields, explicitly warns that signatures and capabilities are not enforced, then returns a loaded reader. The [pipeline builder](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_pipeline.rs) writes an empty crypto segment. An empty segment is not an implemented signature, and a warning is not rejection of an unauthenticated model.

The [firmware upload path](../../../RuView/firmware/esp32-csi-node/main/wasm_upload.c) has a compile-time `CONFIG_WASM_VERIFY_SIGNATURE` gate. When compiled in, it rejects missing NVS keys and absent or failed signatures before loading RVF WASM; the raw-WASM branch also checks this configuration. Those admission branches are useful, but their verification primitive must be assessed separately.

The [firmware verifier](../../../RuView/firmware/esp32-csi-node/main/rvf_parser.c) calls ordinary SHA-256 init/update/final operations over the supplied 32-byte public key and signed region, then compares the resulting digest to the first 32 bytes of the signature field. Despite its Ed25519/HMAC comments and success log, this path calls neither an Ed25519 verifier nor HMAC. A digest derived solely from public data does not establish possession of a signing secret. This is a source-level authentication defect; no upload, device exploit or hardware execution was performed. The compile flag and NVS key check therefore cannot be treated as proof of signed-publisher admission.

| ADR protection layer | Current disposition | Closeout evidence |
|---|---|---|
| Model integrity | Server crypto placeholder; optional firmware gate uses a public-input digest | Real signature verification, trusted publisher policy, signed content/metadata scope and rejection of unsigned/tampered/unknown-key modules |
| Data at rest | Reviewed RVF builder serialises payloads; proposed CsiEncryptor interface not found | Selected authenticated-encryption path, key/nonce lifecycle, restoration and explicit plaintext boundaries |
| Data in transit | Proposed PQ TLS/mTLS path not established by this source review | Actual endpoint/proxy/peer configuration, negotiated protection, authenticated device identity and failure tests |
| Audit trail | Witness JSON exists; signed chain remains deferred | [Independent head/anchor and producer completeness](#witness-segments-and-audit-completeness) |
| Device identity | NVS stores a configured WASM public key; this is not device attestation or a rotation lifecycle | Provisioning trust, secret custody, key rotation/revocation, boot identity and auditable recovery |

The ADR's proposed encrypted-search path also lacks a specified consumer algorithm: its encryptor returns ciphertext while the search example assumes useful distances. The reviewed [fingerprint implementation](#fingerprint-retrieval-and-hnsw-acceptance) takes plaintext floating-point vectors. Define where authorised decryption occurs or supply an explicit alternative search design and leakage model; the illustrative snippets do not establish that bridge. No standards, quantum timeline, clinical, legal or compliance assertion is validated by this code assessment.

CP-04/06/08 must ratify the actual threat model and deployment boundaries, select maintained primitives and verify complete loader enforcement before exposing a secure-model claim. Test signature omission, modified payload and metadata, wrong/revoked keys, replay/rollback and both RVF/raw-WASM paths under each build configuration. Bind results to the exact firmware, server and model artefacts. Benchmark the selected implementation before adopting the ADR's overhead numbers. Preserving Deferred prevents placeholder functionality from becoming an assurance claim while these requirements remain open.

## Multi-device agreement and replicated state

[RuView ADR-008](../../../RuView/docs/adr/ADR-008-distributed-consensus-multi-ap.md) remains deferred. Its coordination proposal combines replicated decisions, causal clocks and disconnected local updates. The current Rust search returns “consensus” only as a reference amplitude used in [multistatic fusion](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs), not the proposed Raft/CRDT interfaces. [Three source hashes and search output](evidence/ruview-consensus-snapshot.json) establish the bounded result. No distributed process, partition, sensor or algorithm test ran.

The fuser validates timestamp spread, extracts each frame's first channel and checks amplitude dimensions. It weights contributions by cosine similarity to their mean. These are useful signal-processing operations, not proof that nodes agreed on an authoritative survivor registry, membership or model version. A one-frame input receives coherence 1.0 by construction, so that value alone cannot establish independent corroboration.

The reviewed `fuse` body does not check unique `node_id` values. `active_nodes` counts usable channel contributions, and node positions are assigned by array index with origin as the default. Frames without channels are skipped during amplitude extraction while the returned `node_frames` preserves the original input. Identity-to-position alignment and duplicate/missing-channel handling therefore need explicit caller contracts. The configuration's `min_nodes` and `enable_person_separation` fields do not govern this body; their names do not establish admission or person separation here. These are source-level limits, not reproduced live fusion defects.

| ADR coordination requirement | Current disposition | Closeout evidence |
|---|---|---|
| Consistent survivor registry | Replicated-log and merge interfaces not established in current crate search | Authenticated membership, durable ordered operations and explicit tentative versus committed observations |
| Coordinated scanning | Proposed leader assignments lack a verified replicated consumer | Ownership epochs, duplicate assignment handling and partition/rejoin reconciliation |
| Model synchronisation | Proposed accumulated deltas do not establish compatible application | Base-model/adapter identity, ordering, deduplication, validation and rollback |
| Clock synchronisation | Fuser checks physical timestamp spread; this does not implement clock synchronisation | Clock source/error bounds, boot epochs and separate causal/physical-time contracts |
| Partition tolerance | Proposed CRDT merge remains unverified | Concurrent conflicting updates, restart, retained history and deterministic reconciliation |

Before implementation, reconcile the proposal's authoritative connected log with locally accepted disconnected changes: which operations may proceed without quorum, what remains tentative, and how reconnection resolves conflicts without silently claiming a global decision was already committed. The survivor-register sketch and merge table describe different collection structures; choose the actual identity, deletion/correction and merge semantics. “All adaptations valuable” is not an admission rule for model deltas. The proposed urgency merge also needs an explicit ordered domain and conflict examples before acceptance; this review does not validate operational triage policy.

CP-04/06/07/08 should bind nodes to independent authenticated devices, reject duplicate/replayed observations and define degraded single-device output. Verify the actual selected fusion caller and frame/position contract separately from replicated state. Run disconnection, delayed delivery, repeated operations, membership changes, lost logs and incompatible model updates against the chosen consistency policy. The four deployment-size scenarios and latency figures remain design targets until tested on the selected topology; feature-level agreement cannot close distributed-consensus acceptance.

## Edge-runtime implementation and isolation boundaries

[RuView ADR-009](../../../RuView/docs/adr/ADR-009-rvf-wasm-runtime-edge-deployment.md) remains deferred for its proposed portable, self-contained RVF runtime. Its blanket no-implementation description needs qualification: [ESP32 firmware](../../../RuView/firmware/esp32-csi-node/main/wasm_runtime.c) provides a WASM3 module runtime, and the [Rust edge crate](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-wasm-edge/src/lib.rs) defines sensing modules for its `csi` host imports. These later ADR-040 paths are concrete but do not implement all of ADR-009's browser/mobile/field promises. [Four source hashes](evidence/ruview-edge-runtime-snapshot.json) record a source-only review; no WASM build, device, browser or runtime test ran.

The firmware loader rejects null/empty and oversized binaries, limits slots and uses fixed arenas when allocated. RVF upload parses the container, optionally applies the [signature gate already reviewed](#secure-sensing-claims-and-model-admission), loads the WASM payload, applies manifest metadata and starts it. These admission/allocation checks deserve credit independently of signature validity and actual execution isolation.

Host imports consult a capability mask for phase, amplitude, variance, vitals, events, logs and history. However, `slot_has_cap` grants all capabilities when the mask is zero. That sentinel is described as raw-WASM compatibility, but the helper does not distinguish raw modules from a manifest explicitly supplying zero. The manifest's empty capability set therefore cannot be assumed to deny every import. Define raw/development policy separately from an explicit deny-all manifest.

Frame execution calls `m3_CallV` synchronously and measures elapsed time afterwards. Successful calls exceeding the budget increment a fault counter and stop after ten; the counter resets only below half-budget, not on every under-budget frame. The “consecutive faults” log is therefore narrower than the actual reset policy. A module that never returns cannot reach this post-call budget check. No instruction-fuel or interruption mechanism appears in this reviewed body; other platform watchdog behaviour was not established here. Initialisation and timer callbacks also invoke the module without this frame-budget sequence. Test each callback independently before claiming a hard per-module time limit.

| ADR profile | Current evidence | Required closeout |
|---|---|---|
| Browser | Proposed WifiDensePoseEdge/IndexedDB interfaces not found by bounded Rust/JS/TS symbol search | Actual container load, offline inference, quota/error recovery and persisted reload |
| IoT | Concrete ESP32 WASM3 loader and sensing modules; not proof of proposed complete model/index package | Signed admission, capability semantics, bounded callbacks, memory limits and reboot persistence |
| Mobile | No target WebView journey established in this review | Background suspension, storage lifecycle, model compatibility and offline recovery on each accepted platform |
| Field | Proposed full container depends on open retrieval, adaptation, audit and consensus contracts | Accepted feature scope, durable local state and tested disconnected/rejoin lifecycle |

The ADR's boot/sense/sync/sleep lifecycle must be verified as a whole: authenticating and loading the artefact, processing frames under enforced limits, persisting model/index state, reconciling updates and recovering after power loss. The reviewed runtime's live module slots and host imports do not establish that complete container persistence path. Its quoted boot time, bundle sizes and quantisation losses remain unevaluated targets.

CP-03/04/06/07/08 should select the required deployment profiles, bind container/firmware/module versions and test real resource enforcement, zero/unknown capability masks, missing imports, trapping and non-returning modules, callback failures, storage exhaustion and restart. Preserve deliberate platform differences rather than treating one working module loader as evidence that one container runs everywhere. The deferred decision can be revised only against that chosen implementation and its full target-client evidence.

## SONA feedback, admission and profile lifecycle

[RuView ADR-005](../../../RuView/docs/adr/ADR-005-sona-self-learning-pose-estimation.md) remains partially realised. The [SONA module](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/sona.rs) implements LoRA arithmetic, EWC regularisation, environment statistics and temporal-loss helpers. Its unchanged standalone module passes twenty native tests. Six additional assertions establish empty-input and incompatible-profile behaviour; the [receipt](evidence/ruview-sona-probe.json) includes exact probe source and output. No real model, new room, persistent profile or deployment was exercised.

`adapt` optimises linear mean-squared loss over supplied features and targets plus an EWC penalty. It does not itself collect the ADR's four feedback sources. Temporal consistency has a separate helper and configuration weight, but that weight is not applied in this loop; skeleton plausibility, multi-view loss and stable-confidence loss are not computed there. A targeted Rust search finds SonaAdapter references in its definition and tests, not an established production caller. This bounded evidence does not exclude external or differently named consumers.

| Proposed trigger | Current evidence boundary | Required acceptance |
|---|---|---|
| Confidence drop | No complete trigger-to-adapt consumer established | Calibrated score, window and sample identity |
| Distribution drift | EnvironmentDetector helper exists; proposed KL-trigger pipeline unverified | Measured drift definition, threshold and invocation |
| New environment | Retrieval and adaptation are separately reviewed helpers | Compatible fingerprint/model identity and calibration admission |
| Periodic | Complete scheduled lifecycle unverified | Durable cadence, cooldown, restart and duplicate suppression |
| Manual | Callable helper does not prove an authorised API workflow | Scoped request, selected model and applied/rejected receipt |

Empty samples return unchanged parameters, zero steps, zero loss and convergence true, without incrementing adaptation count. The native probe verifies this behaviour. Consumers must distinguish no data from an evaluated converged candidate. On nonempty input, loss is evaluated before each update; an iteration-limit exit can report pre-update loss alongside post-update parameters. Re-evaluate the returned candidate before promotion.

`save_profile` constructs an in-memory struct containing name, LoRA matrices, Fisher values, reference parameters and count. `load_profile` clones these without model identity, configuration, dimension or finite-value validation. A native probe loaded a four-parameter profile into a two-parameter adapter: its matrix length became four while its declared parameter count remained two. This proves inconsistent state is accepted at the helper boundary, not a subsequent production panic. The [RVF pipeline](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/rvf_pipeline.rs)'s optional SONA matrix segments do not establish the complete profile's restart roundtrip.

The ADR's five safety controls—step-norm bound, validation rollback threshold, checkpoint retention, consecutive-rollback shutdown and cooldown—need explicit implementation and caller evidence. Maximum optimisation steps and EWC regularisation do not replace these controls. The [rapid-adaptation review](#cross-environment-adaptation-and-readiness) and [witness review](#witness-segments-and-audit-completeness) also qualify automatic new-room and recorded-adaptation claims.

CP-03/06/07/08 should bind feedback provenance, base/candidate model and environment identity; reject incompatible profiles and distinguish no-op, converged, rejected and installed states. Test persistence, rollback and forgetting across sequential environments. All four original validation steps remain open: labelled cross-environment evaluation, paired static/adapted deployment, rapid-change forgetting tests and latency on accepted targets. Native helper tests cannot supply those measurements or establish the claimed absence of forgetting.

## GNN mode coverage and learning evidence

[RuView ADR-006](../../../RuView/docs/adr/ADR-006-gnn-enhanced-csi-pattern-recognition.md) remains partially realised. The [graph module](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/graph_transformer.rs) implements a COCO body graph, graph convolutions and a CSI-to-pose forward path: embedded antenna features feed keypoint cross-attention, a GNN stack and coordinate/confidence heads. The `embed` path exposes post-GNN features for the [fingerprint extractor](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/embedding.rs). This is concrete computation, not merely a declared graph interface.

Twenty-eight unchanged standalone native tests pass, including graph, attention and weight-roundtrip cases. The [receipt](evidence/ruview-gnn-probe.json) records output, four hashes and an exact search for the ADR's three mode types. The tests use local arithmetic and synthetic inputs; no trained model, actual CSI, full server or hardware ran. Neither native output shape nor parameter roundtrip establishes pose quality or runtime activation.

| Proposed integration mode | Current evidence | Closeout route |
|---|---|---|
| Query-time refinement | Skeleton GNN and embeddings exist; current fingerprint search scans cosine distances without this reranker. GnnQueryRefiner type not found in bounded Rust search | Connect the actual retrieval candidate graph, train/load compatible weights and compare reranked recall with baseline |
| Temporal sequence recognition | Proposed TemporalPatternRecognizer type not found by the same search; CSI-to-pose forward alone does not supply a persistent activity sequence recogniser | Trace timestamped window/edge lifecycle, bounded retention, labels and actual multi-frame activity output |
| Multi-person disentanglement | Proposed MultiPersonDisentangler type not found; the skeleton forward path is not evidence of learned person-specific CSI decomposition | Verify selected separation/tracking consumer, person count, identity continuity and occlusion/overlap cases |

The exact-symbol search is a navigation bound, not proof that no alternative implementation or external consumer exists. The [multistatic assessment](#multi-device-agreement-and-replicated-state) likewise separates feature fusion from person separation and device agreement. Map any alternative consumer explicitly before treating it as satisfying one of these three modes.

The ADR's learning loop says each query improves weights using temporal, multi-AP and physical-plausibility feedback. The inspected forward/embed methods borrow the model immutably and do not update weights. Weight setters and serialisation permit external training/loading, but they do not establish this online feedback loop. Confirmed outcomes, model generation, candidate validation and rollback need caller evidence, as detailed in the [SONA lifecycle](#sona-feedback-admission-and-profile-lifecycle). The historical domain-adversarial extension also retains [ADR-027's integration limits](#cross-environment-adaptation-and-readiness).

CP-03/06/07/08 should retain the working spatial model path while defining each required mode's actual graph, producer, trained state and consumer. Bind inference to a verified model artefact rather than constructor-initialised parameters; prove installation separately from training. Evaluate query refinement, temporal classification and multi-person separation on held-out observations with explicit baselines and error measures. Test feedback admission, incompatible graph/model shapes, restart and regression rollback. The ADR's per-mode latency budget and accuracy improvements remain unevaluated until measured on those selected paths.

## Rust workspace and foundational contracts

The three [Rust workspace ADRs](../../../RuView/rust-port/wifi-densepose-rs/docs/adr/ADR-001-workspace-structure.md) preserve Accepted decisions about modularity, signal libraries and inference backends. Their current implementation scope is recorded in the [manifest/source receipt](evidence/ruview-workspace-snapshot.json): all fifteen member manifests were parsed, with no Cargo dependency resolution, compilation or target execution.

### Workspace structure

The original nine members remain: core, signal, nn, api, db, config, hardware, wasm and cli. Six additional members are mat, train, sensing-server, wifiscan, vitals and ruvector. The separate wasm-edge crate is explicitly excluded. Workspace-wide success therefore cannot establish device-WASM acceptance. Much package metadata inherits the shared workspace version; modular crates do not by themselves prove independent release/version management.

CP-01/06/08 should preserve the original responsibilities while documenting the added runtime/training paths and their shared interfaces. In particular, the older neural/API crates and sensing-server's own model/serving code are distinct ownership paths. Bind release manifests and test matrices to all selected members and the separately built device target; validate dependency/feature combinations rather than inferring isolation from directory structure.

### Signal library and algorithm selection

[Signal ADR-002](../../../RuView/rust-port/wifi-densepose-rs/docs/adr/ADR-002-signal-processing.md)'s ndarray, rustfft, num-complex and num-traits choices remain in the signal manifest. The crate also depends on graph, attention, solver and temporal-analysis packages. Availability of the original pure-Rust libraries does not prove the entire expanded dependency graph builds for every target.

| Named implementation | Current evidence | Closeout requirement |
|---|---|---|
| Phase sanitisation | Standard, Custom, Itoh and QualityGuided branches exist | Compare each method against numerical fixtures; qualify QualityGuided's unused quality ordering |
| CSI processing | Amplitude/phase processing, Hamming window and confidence smoothing functions exist | Trace the selected consumer and verify dimensions, window parameters and smoothing semantics |
| Feature extraction | Feature module exists alongside later signal integrations | Validate Doppler/PSD/amplitude/phase/correlation outputs against known signals and actual consumer configuration |
| Motion detection | Variance and adaptive-threshold path exists; adaptive mode defaults off | Test the enabled and default paths with labelled stationary/motion/noise inputs |

The [phase sanitizer](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-signal/src/phase_sanitizer.rs)'s quality-guided branch computes `_quality` but does not use that map for traversal order; its own comments defer a full implementation. Treat the selectable name as a partial algorithm, not completed quality-guided equivalence. The [CSI processor](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-signal/src/csi_processor.rs)'s exposed smoothing is confidence smoothing; the broader ADR wording needs an explicit signal-stage contract. CP-03/06/08 should establish numerical parity and target-build evidence before adopting performance and WASM claims.

### Neural backend selection

[Inference ADR-003](../../../RuView/rust-port/wifi-densepose-rs/docs/adr/ADR-003-neural-network-inference.md) proposes ONNX by default plus tch and Candle alternatives. The manifest retains those optional dependencies. In the inspected neural source, Backend implementations are OnnxBackend and MockBackend; a bounded file/implementation search did not establish tch/Candle implementations. Dependency selection is not an executable backend.

The [ONNX session](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-nn/src/onnx.rs) genuinely loads models from file or bytes and delegates inference to the session. Both constructors accept `_options` without applying them to Session::builder. GPU settings exposed by the builder therefore do not establish provider selection at this boundary. Current `cuda` and `tensorrt` crate features enable `onnx`, unlike the ADR's proposed direct ort provider-feature forwarding. This does not establish that every deployment is CPU-only; it leaves the selected execution provider unverified.

CP-06/07/08 should implement or explicitly defer each alternative, apply validated options, expose the selected provider and prove a known model's inference on each supported backend. Distinguish load success, warmup, actual device execution, model provenance and output quality. The existing sensing-server spatial-model tests cover a different implementation; they cannot certify these backends. No trained model or GPU ran in this pass.

## Sensing UI source and freshness contract

[ADR-019](../../../RuView/docs/adr/ADR-019-sensing-only-ui-mode.md) remains Accepted as an architectural choice. Its original startup and deployment description does not match all current paths. Seven source hashes and four passing isolated assertions are retained in the [receipt](evidence/ruview-sensing-ui-probe.json). The probe imports the unchanged service in Node and stubs status fetch; it does not render a browser, open a WebSocket or validate a device or model.

| Original commitment | Current evidence | Closeout requirement |
|---|---|---|
| Independent sensing transport | `sensing.service.js` now derives `/ws/sensing` from page origin and fetches `/api/v1/status`; the ADR's direct `:8765` client description is historical | Document selected server/proxy profile and demonstrate deployment without the full DensePose stack |
| Automatic sensing-only startup and API suppression | The bounded UI search finds `sensingOnlyMode` only initialised false in the detector; `ApiService.request()` has no such gate and `app.js` initialises Dashboard and Hardware when their containers exist | Decide whether to retain, replace or retire this mode; test startup and supported tabs with backend absent |
| Suppressed health polls | `health.service.js` sets `_backendUnavailable` on initial failure and subsequent interval callbacks return early | Define recovery and demonstrate discovery after a backend becomes available |
| Gaussian signal and body display | The renderer uses signal values, classification presence/confidence and breathing-band power; its body blob modulates random splat sizes with a wall-clock sinusoid | Label illustrative geometry and heuristic confidence; establish independent evidence before treating it as a measured pose or breathing rate |
| Explicit fallback | The service and tab distinguish server simulation, client simulation and reconnecting; fallback starts after five scheduled failures, with a twenty-attempt ceiling, or immediately on constructor failure | Align prose with actual thresholds and prove the complete outage/recovery sequence |

The current banners improve on the original ADR's small simulated badge. Nevertheless, `_detectServerSource()` assigns `live` when the status request fails or returns non-OK, without requiring a hardware frame. `_applyServerSource('wifi')` also maps to `live`, whose banner says **LIVE — ESP32 HARDWARE**. These assertions establish label behaviour, not that WiFi input is synthetic. Unknown source strings map to server simulation, conflating unknown with known synthetic input.

The service retains `_serverSource` across reconnects. In the isolated transition from ESP32 to reconnecting to connected, another ESP32 frame leaves the label reconnecting because the raw string has not changed. A later status response may correct it; the asynchronous status request has no connection-generation guard, so acceptance must also cover delayed responses from an earlier connection. The probe does not claim a browser race was observed.

`onData()` immediately replays `_lastMessage`, including after disconnection, and the same RSSI history receives all sources. The tab's state callback changes labels without clearing the rendered observation. No age check appears in these inspected handlers. Retaining a last observation can be useful, but it requires visible age, source transitions and invalidation policy. A source badge alone cannot establish freshness, calibrated confidence or trained inference. See the separate [model-path assessment](#model-labels-and-observation-quality).

CP-01/06/09 closeout requires an agreed source envelope separating hardware type, transport state, inference method and observation time; browser assertions for unknown source, failed status, stale/reordered status, reconnect with unchanged source, server/client simulation, normal closure and late subscription; and a real selected-server demonstration with packet/model provenance. RuView UI and sensing maintainers are proposed accountable roles. Estate acceptance additionally needs the consumed VisionClaw adapter and display, which remain unestablished.

## Survivor tracking and operational integration

[ADR-026](../../../RuView/docs/adr/ADR-026-survivor-track-lifecycle.md) specifies Kalman continuity, fingerprint re-identification, lifecycle control, assignment, three `DisasterResponse` integration steps and five domain events. All four tracking modules exist. Six lifecycle and four Kalman tests pass unchanged under `rustc --edition=2021 --test`; two additional assertions exercise unchanged extracted assignment functions. The [receipt](evidence/ruview-tracking-snapshot.json) retains six source hashes and complete test output. This is helper evidence; the complete MAT crate, field sensing and operator workflow were not executed.

| Decision component | Current implementation | Required acceptance |
|---|---|---|
| Constant-velocity Kalman filter and gating | Six-state filter, covariance and Mahalanobis gate exist; four module tests pass. Tracker updates its Kalman state but writes the raw observation into `Survivor::update_location()` | Establish which position consumers display; evaluate continuity, uncertainty and invalid/non-finite inputs using a defined coordinate frame and time source |
| Weighted fingerprint re-identification | Four configured weights exist; heartbeat absence redistributes weight. Distance adds weighted absolute feature deltas and normalised spatial Euclidean distance, rather than taking a joint weighted Euclidean norm. Missing location defaults to origin; missing breathing defaults to zero | Resolve metric wording and missing-data semantics; evaluate false merges/splits and zone transitions with labelled observations |
| Tentative/Active/Lost/Terminated/Rescued lifecycle | Six state tests pass. Birth constructor counts the first hit; tentative miss terminates; lost expiry uses `Instant`, independently of prediction `dt_secs` | Define confirmation visibility, observation time versus processing time, replay/restart semantics and configuration bounds |
| Association and birth | Small problems use `hungarian_assign`; larger ones use greedy costs. Small-problem helper builds unweighted admissible adjacency and an augmenting-path matching, ignoring relative finite costs | Resolve maximum-cardinality versus minimum-cost objective, deterministic ties and crossing identities; validate both size branches |
| `DisasterResponse` field, scan update and survivor query | Tracker field and accessors exist. `scan_cycle()` still calls `event.record_detection()` directly; `survivors()` still reads the event list | Connect detection observations, tracker results and the authoritative survivor query, or revise the proposed integration explicitly |

The assignment probe supplies costs `[[1,100],[100,1]]`. The unchanged small-problem helper returns the crossed assignment with total cost 200, while the diagonal costs 2. It establishes the distinction from minimum-cost Hungarian matching; it does not evaluate the full tracker on measured trajectories. Positional matching in the inspected aggregate does not filter by zone, while the special no-position matching branch does. Lost-track matching checks age and fingerprint distance; it likewise has no explicit zone filter. Resolve whether coordinates span zones before assuming that either behaviour is correct.

### Confirmation, events and retention

The [event enum](../../../RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-mat/src/domain/events.rs) declares all five proposed events, but declaration does not prove publication. The inspected scan cycle emits detection/zone/alert events without converting tracker results into tracking events.

| Proposed event | Current aggregate result or transition | Closeout condition |
|---|---|---|
| `TrackBorn` on Tentative → Active | `born_track_ids` is populated at first-observation construction; `active_tracks()` and `active_count()` include Tentative | Separate candidate birth from confirmed survivor admission; demonstrate single-spike rejection in the actual query and alert consumer |
| `TrackLost` | Active-to-Lost transitions populate `lost_track_ids` | Publish and persist once, with stable track/survivor identity and observation cause |
| `TrackReidentified` | Age-qualified fingerprint match populates result and restores Active | Demonstrate identity/history continuity and bounded false merges through the actual consumer |
| `TrackTerminated` | Tentative miss or lost expiry populates result; timed cleanup retains terminal tracks for less than sixty seconds when timestamped | Establish durable history and explicit query semantics after removal |
| `TrackRescued` | `mark_rescued()` can call unconditional lifecycle rescue for any found state; update reports every rescued track on every tick. Rescue does not set `terminated_at`, so its `None` branch retains it | Define allowed predecessor states, operator authority, idempotence, event deduplication and retention |

The ADR's claim that Terminated is unrecoverable needs qualification: `hit()` preserves terminal states, but `rescue()` overwrites them. Its promised smoothed trajectory likewise needs a consumer contract because the embedded survivor receives raw positions. These are source findings, not an observed operator failure. The stated 60–80% duplicate reduction, field tuning and triage benefits are unverified source claims; this review provides no clinical, rescue-performance or external-study validation.

CP-01/05/06/09 should sequence this work as: agree identity, confirmation, time, zone and event contracts; integrate the tracker into the chosen MAT path; verify exact transition and assignment cases; demonstrate recorded sensor scenarios and operator actions; then establish any estate adapter and acceptance journey. MAT tracking and response maintainers are proposed accountable roles. Hardware availability, inference validity and disaster-response suitability remain separate evidence obligations.

## Training UI and model operation contracts

[ADR-036](../../../RuView/docs/adr/ADR-036-rvf-training-pipeline-ui.md) retains Deferred. Its blanket “spec-only” and “UI is not implemented” wording needs qualification: `TrainingPanel.js`, `ModelPanel.js`, services and Rust recording/training/model handlers exist; `app.js` imports and constructs the panels. However, the declared binary is `src/main.rs`, and neither it nor `src/lib.rs` declares those three handler modules or imports their route factories. The inspected main router exposes singular `/api/v1/model/...` inspection/SONA paths, rather than the panels' plural `/api/v1/models` management and `/api/v1/train` paths. Existing files do not establish a reachable training API in this target.

The [receipt](evidence/ruview-training-ui-snapshot.json) records thirteen source hashes, an explicit non-ignored entrypoint search and two passing isolated model-service assertions. No server, model, training job or browser was run. Other deployment targets were not exhaustively tested.

| ADR implementation item | Existing evidence and remaining disposition |
|---|---|
| 1.1 Recording API | Handler routes cover start/stop/list/download/delete. `RecordedFrame` contains timestamp, subcarriers, RSSI, noise and feature JSON. Demonstrate tick-to-durable-record wiring, source identity, environment metadata and any optional camera-label association; handler code alone does not supply it |
| 1.2 Contrastive pretraining | Pretrain handler delegates to `real_training_loop` with a type label. This loop fits linear weights to deterministic signal-derived teacher targets, not the specified NT-Xent/VICReg embedding objective. Implement or explicitly replace the objective and evaluate collapse/retrieval |
| 2.1 Dataset integration | API loop reads recording IDs and falls back to live frame history when none load; it does not select MM-Fi/Wi-Pose through the proposed common dataset interface. Keep the separate [dataset assessment](#dataset-separation-and-evaluation-meaning), and require explicit dataset identity, labels and missing-ID failure semantics |
| 2.2 Training API | Start/stop/status/progress handlers exist, with real gradient updates in the standalone implementation. Wire routes and validate request shapes, job identity, cancellation, concurrency and progress meaning before claiming a reachable workflow |
| 2.3 RVF export | Loop writes best linear weights and metadata/witness segments. Verify successful file creation, model indexing, download and round-trip inference. The existing [witness assessment](#witness-segments-and-audit-completeness) qualifies its training-hash claim |
| 3.1 LoRA fine-tuning | LoRA handler sets configuration names but calls the same newly initialised full linear model loop. Its rank appears in the request/response, not a rank-constrained adapter update in that loop | 
| 3.2 Profile switching | Activation handler validates profile membership and assigns `active_lora_profile`; demonstrate actual adapter installation and changed inference with base weights preserved |
| 4.1 Model panel | Library/load/unload/profile controls exist. Require contract-level response validation and active-model provenance, plus inspector and browser action coverage |
| 4.2 Training dashboard | Recording/config/progress/chart/completion views exist. Require reachable service flow, per-job metrics, export/download and correct stopped/failed/completed distinctions |
| 4.3 Live enhancements | Selector, LoRA and split-view controls exist. The inspected split toggle adds overlay labels; it does not itself prove two independent inference streams on identical inputs. Verify confidence/trail consumers and an actual paired comparison |
| 4.4 Settings | Model/training fields exist. Prove persisted values reach the selected runtime, including device/provider, directories and any camera option; form presence is insufficient |
| 4.5 Dark mode | Panels include dark styling. Browser visual/accessibility acceptance remains unexecuted |
| 5.1 Inference wiring | The API trainer exports a linear feature-to-51-coordinate model, distinct from the specified ONNX `[1,T*links,56]` to `[1,17,4]` route. Require one selected input/model/output contract, smoothing and source-labelled inference through its actual consumer |
| 5.2 Progressive loading | Existing progressive container layers are assessed elsewhere; no timed A/B/C UI-to-inference transition was demonstrated. Measure readiness from usable computation, not a load flag |

### Training identity and metric meaning

`compute_teacher_targets()` derives pose targets from subcarrier amplitude statistics and motion; it does not read camera pose labels. The loop fits a linear model with mini-batch gradient updates and weight decay. A numerical fit is useful calibration evidence, but the resulting PCK against that teacher does not establish independent human-limb accuracy. Feature means/stds are calculated across the whole feature matrix before the 80/20 split, so held-out feature distribution informs preprocessing. The windows and split also need session/person/environment separation before a generalisation claim.

Supervised, pretrain and LoRA routes all call this loop with different type strings. `pretrained_rvf` and `lora_profile` are configured but not used to initialise that loop's weights; LoRA rank is not its update constraint. Accordingly, these names cannot substitute for three algorithm-specific acceptance runs. Insufficient recordings can select live history, and export metadata's `simulated: false` cannot by itself establish measured training data.

### UI requests, completion and cancellation

`TrainingPanel._launchTraining()` always wraps hyperparameters inside `config`. The supervised request expects that shape, but pretraining expects top-level epochs/LR, and LoRA requires top-level `base_model_id` and `profile_name`. The panel supplies `base_model`/`profile_name` inside `config`; these are not the same contract. This is a source comparison, not a recorded live HTTP rejection.

The panel displays Completed whenever training is inactive and its loss array is nonempty; it does not require a completed terminal phase in those render branches. The stop handler clears the shared active flag and reports stopping without awaiting task termination. New-start admission and old-task cleanup therefore need job-specific cancellation evidence before concurrency is accepted.

The isolated `ModelService.loadModel()` probe resolves transport with `{status: 'error'}`. The unchanged service still assigns `activeModel` and emits `model-loaded`; both assertions pass. This proves the service lacks semantic rejection handling for such a body, not that the current unmounted load handler returns that response. Require explicit accepted-result validation across the UI/API boundary, with model revision and usable inference attested separately.

CP-01/03/06/07/08/09 closeout should first select and mount the actual server modules with request/response tests; establish recording/dataset/teacher identity; choose truthful algorithm and metric contracts; implement job lifecycle and verified model/profile installation; then demonstrate recording → training → export → reload → changed inference in the browser. UI, sensing-server and training maintainers are proposed accountable roles. Historical effort estimates are unratified planning values, and compute/latency, data-volume, adaptation and model-quality claims remain unmeasured in this review.

## macOS helper and runtime contract

[ADR-025](../../../RuView/docs/adr/ADR-025-macos-corewlan-wifi-sensing.md) remains Partially Implemented. The Rust adapter exists and is macOS-gated, but its named helper and server integration do not establish the proposed multi-AP scan path. The [receipt](evidence/ruview-macos-snapshot.json) records eight source hashes and the absent planned helper directory. This was source-only: no Swift build, macOS scan, adapter test, hardware run or external platform/research validation occurred.

| Design principle | Current source | Closeout disposition |
|---|---|---|
| Independent Swift subprocess | Rust defaults to `mac_wifi --scan-once`; the named source is `v1/src/sensing/mac_wifi.swift`, not the absent `tools/macos-wifi-scan/main.swift`/`build.sh` pair | Select and version one executable, CLI and output protocol; test discovery, exit status, timeout and packaging |
| Shared observation type | Adapter constructs `BssidObservation` with RSSI, percentage, channel, inferred band/radio type, SSID and receipt-time `Instant` | Preserve measurement time, source identity and available noise/SNR deliberately; the current type has no noise/SNR field |
| Synthetic identity under redaction | Implementation hashes SSID bytes plus channel using FNV-1a-style arithmetic and sets local/unicast bits, rather than the documented SHA-256 string derivation | Ratify an identity algorithm and migration; test collisions, redaction forms, hidden SSIDs and same-SSID/channel AP collapse |
| macOS compile gating | `adapter/mod.rs` gates module/re-export with `target_os = "macos"` | Compile on the selected macOS target and verify Windows/Linux regressions; Linux builds do not exercise the gated parser |
| Graceful automatic fallback | The inspected Rust `main.rs` probes ESP32 then Windows, otherwise simulation; explicit `wifi` dispatch starts the Windows task | Implement or revise the macOS probe/task/dispatch contract and verify both explicit and automatic modes |

### Helper and parser disagree

The existing Swift helper obtains the connected interface and prints timestamp/RSSI/noise/transmit-rate objects every 100 ms in an unconditional loop. It does not inspect arguments or implement the proposed full-scan, probe or connected modes. It emits no SSID, BSSID or channel fields. The Rust adapter uses blocking `Command::output()` and waits for process termination, with no timeout in this method. If built from that source and invoked successfully, the continuous helper does not complete as a one-shot scan; even an isolated output line lacks fields the parser requires. This is a source-level incompatibility, not a measured hang on a Mac.

The parser expects one JSON object per line, rather than the ADR's JSON array. It uses manual string/number extraction instead of `serde_json`, requires BSSID to be a quoted string, and silently skips unparseable observations. The ADR's `bssid: null` case therefore does not reach synthetic identity generation. Zeroed, empty or invalid *string* BSSIDs can reach that fallback. String extraction retains escape sequences rather than JSON-decoding them, and channel is cast from a floating number to `u8` without range/integrality validation. Require fixtures for the selected wire format and explicit malformed/empty-scan diagnostics.

Noise, band and PHY fields are not read by the adapter. Band/radio type are inferred from channel; the mapping table's SNR and direct PHY promises remain incomplete. Synthetic address bits do not establish uniqueness against all other locally administered addresses. Identity remains a data-quality key, not authentication.

### Separate Python consumer

There is a distinct Python path: `ws_server.py` selects `MacosWifiCollector` on Darwin and labels it `macos_wifi`; `rssi_collector.py` compiles the adjacent Swift source if the binary is absent and runs a streaming collector. Its stop path terminates/kills the process and joins the thread. This credits connected-network RSSI source code, not the proposed Rust multi-AP integration. Selection constructs the collector before actual start/compile, so successful selection alone is not verified hardware readiness. The claimed approximately 10 Hz is a configured polling interval, not a measured delivery rate.

### Verification and future scope

All original verification groups remain open at their stated scope:

| Original plan | Evidence needed |
|---|---|
| Five Swift checks: build, probe, scan, connected, WiFi disabled | Build the selected helper and record finite CLI results, actual schema, permissions and failure behaviour for each mode |
| Four adapter checks: JSON, synthetic identity, missing helper, real scan | Parser/identity fixtures must match the real helper; missing executable and hung helper need distinct outcomes; run on macOS |
| Seven end-to-end steps: release build, workspace tests, explicit WiFi launch, latest endpoint, motion response, UI, auto mode | Demonstrate the same consumed binary/helper/data source through every step; a source label or green unit suite does not establish physical motion sensitivity |
| Three cross-platform checks: macOS, Windows, Linux | Build all selected targets and exercise platform dispatch/fallback, including the gated adapter |

All eight proposed pipeline stages depend first on a valid observation stream with accurate timestamps and measured cadence. Reusing `WindowsWifiPipeline` types does not validate gating, attention, correlation, motion, breathing, quality, fingerprint matching or orchestration for slower macOS sampling. The ADR's presence/motion effectiveness and research percentages remain unevaluated; no physiological accuracy follows from a polling loop.

The five future items also need explicit dispositions. Connected-AP polling already exists in the separate Swift/Python path, while integration/performance remains open. A Linux `iw` adapter source exists, so it is no longer a wholly absent-code proposal. The `WindowsWifiPipeline` rename remains a compatibility decision. Future Apple CSI exposure remains an external dependency. The claimed pre-bundled helper and proposed macOS container image require an actual supported deployment artifact and host-framework access evidence; neither is demonstrated by these files.

CP-01/06/08/09 should sequence helper/protocol selection, parser and identity reconciliation, selected-runtime dispatch, platform builds and measured scan/cadence/UI scenarios. Platform adapter, sensing-server and packaging maintainers are proposed accountable roles. Keep connected-AP RSSI, multi-AP scanning and raw CSI distinct throughout acceptance and estate consumption.

## Python proof replay and mock boundaries

[ADR-011](../../../RuView/docs/adr/ADR-011-python-proof-of-reality-mock-elimination.md) remains Partially Implemented. Its historical inventory is no longer a reliable description of every listed path: mock generation has moved behind settings in several inspected modules, unsupported parsers raise errors, and uptime is calculated. Conversely, deterministic processing does not establish physical capture provenance or pose accuracy. The [receipt](evidence/ruview-proof-reality-snapshot.json) records seventeen source hashes and a failed replay attempt: `python3 v1/data/proof/verify.py` exits 1 at import because this interpreter has no NumPy. No dependencies were installed and no expected hash was regenerated.

### Five decisions and concrete file dispositions

| Original decision | Current evidence | Closeout requirement |
|---|---|---|
| 1: hard-fail silent mock fallbacks | Selected CSI parsers and router methods now reject unsupported/incomplete data; Doppler uses cached phase differences and FFT | Verify each actual entrypoint on valid, incomplete and unsupported input; source absence of random calls does not prove accurate processing |
| 2: isolate mock infrastructure | Testing modules exist and selected pose/router branches import them only under explicit settings; both default mock flags are false; dedicated test settings enable them | Audit every runtime entrypoint and configuration source, with explicit output provenance. Module existence alone is not evidence of a default mock path |
| 3: proof bundle | Committed metadata explicitly states `is_synthetic: true`, `is_real_capture: false`; generator uses seed 42. Replay preprocesses/features the first 100 frames and hashes outputs | Describe this as deterministic synthetic replay; supply provenance for a separate captured-data fixture if physical acquisition is claimed |
| 4: pinned build environment | Four numerical/settings dependencies are pinned. `docker/Dockerfile.python` adds unpinned websockets/uvicorn/fastapi, does not run proof verification, and launches the sensing WebSocket server | Select a reproducible build target and run verification during its actual build; the proposed root `docker build .` target is absent |
| 5: CI verification | Workflow executes replay twice, but the random-code scan only warns and excludes named RNG constructors and comment patterns | Decide and implement enforceable scope; cover all relevant source/config changes and retain a failing-gate fixture |

| Original concrete file-change row | Current disposition |
|---|---|
| `core/csi_processor.py` Doppler | Cached phase differences and FFT replace the cited random placeholder; numerical correctness and hardware sensitivity require separate tests |
| `hardware/csi_extractor.py` incomplete ESP32 fallback | Raises explicit extraction errors for incomplete payloads and invalid numeric data |
| `hardware/csi_extractor.py` Atheros parser | Explicitly unsupported and raises `CSIExtractionError`; no Atheros parsing acceptance |
| `hardware/router_interface.py` fallback | `_parse_csi_response()` specifies an explicit unsupported-parser error rather than random output |
| `services/pose_service.py` mock poses | Delegates to the testing module after checking `mock_pose_data`; direct disabled use raises |
| `services/pose_service.py` mock CSI | Bounded search of the five original processor/extractor/router/pose files finds no `np.random` or `random.` calls; real inference still needs configured input and weights |
| `services/pose_service.py` statistics/history | Mock statistics are delegated under the flag; production branches return stored/empty information instead. Verify API semantics and real persistence independently |
| `core/router_interface.py` mock generator | Imports testing generator through the explicit mock branch and displays its banner; real collection remains an explicit unsupported path |
| `api/dependencies.py` mock auth | Inspected credential-validation branch rejects unconfigured JWT credentials rather than returning a mock user. Request-state/middleware identity and optional authentication require their own acceptance |
| `data/proof/` | Synthetic reference JSON, metadata, hash, generator and verifier exist; the proposed captured binary does not |
| `requirements-lock.txt` | Four exact versions exist; this is a minimal numerical replay lock, not the complete deployed application dependency closure |
| `verify-pipeline.yml` | Exists with narrower path triggers and warning-only random scan; no hosted workflow run is claimed here |

The original low-severity uptime examples have also changed: health calculates elapsed time from application start, and the pose service uses its recorded start time. The planned `v1/docs/hardware-setup.md` path is absent in this checkout; parser errors referencing hardware setup need resolvable instructions for the actual supported capture path.

### Meaning and limits of the gate

The workflow regenerates the reference and then checks only metadata flags (`is_synthetic`, seed 42) in its purported committed-data comparison step. It does not compare regenerated data bytes to the committed bytes there. The subsequent feature hash is a separate check over a subset of frames. Passing either gate would not validate unprocessed frames, device provenance, all features/entrypoints or an independent physical result.

The verifier exits 2 when the expected hash file is absent, which is a failure to complete verification rather than a successful skipped check. Its optional codebase audit is not enabled by the workflow's replay commands. Workflow path filters cover core, hardware, proof and the workflow itself; changes confined to services, API, settings or the requirements lock do not trigger it via those filters. The final random scan explicitly leaves findings as warnings. Seeded or deterministic synthetic output can reproduce a hash, so a matching digest cannot prove that no randomness or simulation was used.

The UI's mock-server enable and auto-detect settings are both false in the inspected config. This credits its default mock-server gate, while the independent [sensing-service simulation path](#sensing-ui-source-and-freshness-contract) remains. The four acceptance steps—clone, one command, matching hash and a no-random CI badge—plus five clean-machine runs remain unverified as a combined claim. The local missing-dependency failure is not proof that the pinned environment fails; it simply supplies no successful replay evidence.

CP-01/03/06/08/09 should first ratify distinct claims for deterministic replay, physical acquisition and validated inference; complete runtime/config mock-boundary review; wire an actual locked build and rejecting CI gate; then record five clean-environment replays and a provenance-bound captured-data journey. Python sensing, validation and delivery maintainers are proposed accountable roles. The dated claim that Rust dependencies are merely commented/unpublished should be read against the [current dependency scope](upstream-decision-scope.md), not retained as a blanket current blocker. No external hardware/platform or accuracy claims were validated in this pass.

## Roadmap planning design and evidence state

[ADR-038](../../../RuView/docs/adr/ADR-038-sublinear-goal-oriented-action-planning.md) remains Deferred. This is a proposed development coordinator, not a sensing-runtime planner. All seven named `.claude-flow/goap/` modules are absent at their declared location. The [design receipt](evidence/ruview-goap-design-snapshot.json) preserves the source hash, proposed state/action/goal tables, six performance budgets and two mathematical counterexamples. No planner implementation, CLI integration, persistent plan or agent dispatch was executed.

### State and action catalogue

The proposal defines 25 feature flags, five hardware properties and six quality metrics, plus twenty example actions and seven goal templates. These are a useful list of intended planning inputs, but several observations overstate their evidence: a file existing, crate building or helper suite passing does not establish a capability in the selected deployment. USB enumeration does not establish authenticated, calibrated, communicating multistatic nodes. A GPU utility response does not establish model/provider readiness. Each property needs an evidence level, source revision, target/configuration, observation time and explicit unknown/stale state. A cached boolean cannot retain those distinctions.

The action table also needs semantic reconciliation. `adr025_wasm` assigns browser deployment to the current macOS CoreWLAN ADR; `adr011_mat` assigns survivor detection to the Python proof-replay ADR. Route these to the actual runtime/edge and MAT/tracking decisions before execution. Action effects that assign PCK 0.6, SNR 10 or jitter 30 are predictions to verify, not measurements created by completing a coding task. An increment such as `max_persons_tracked += 2` needs a non-repeatable implementation/admission condition; repeating the same action must not manufacture acceptance evidence. Preserve all twenty examples for reconciliation rather than treating them as an executable catalogue of approximately eighty actions.

The seven goal templates require different proofs: multi-person tracking needs count and identity continuity; vital monitoring needs measured signal validity; production accuracy needs a defined dataset and jitter benchmark; browser deployment needs target execution; MAT needs authoritative survivor/alert semantics; multistatic mesh needs synchronised physical nodes and fusion; cross-environment robustness needs held-out environments. Dependencies and helper availability alone do not satisfy any of these.

### Search and prioritisation contracts

| Proposed technique | Design finding | Acceptance requirement |
|---|---|---|
| Backward relevance pruning | Pseudocode repeatedly scans all actions and has no visited-condition/action expansion guard. Cyclic prerequisites can keep returning to the frontier. Numeric increments also need state-dependent reasoning | Terminate on cycles; cover delete effects, repeated increments and alternative prerequisite chains; compare reachable plans with an exhaustive small oracle |
| Tier decomposition | The prose calls this three tiers but lists four. ADR-037 person-count and NMF actions are both Tier 2 although NMF requires person count; channel/multiband/mesh dependencies similarly do not justify unrestricted within-tier parallelism | Derive a DAG and resource/conflict constraints; validate each parallel batch against actual preconditions, shared files and exclusive hardware |
| Incremental replanning | Removing an invalidated action can invalidate later consumers; the proposed patch/merge lacks a transitive validity contract | Revalidate the resulting complete plan from the new observed state, preserving in-flight action ownership and cancellation semantics |
| A* heuristic | Summing minimum cost per condition double-counts an action satisfying multiple conditions. With one idle agent, one cost-5 action satisfying two missing goals gives h=10 although the optimal cost is 5 | Define whether cost means total work or elapsed time; prove a lower bound for the chosen objective before claiming admissibility or optimality |
| PageRank priority | Edges point from prerequisite to dependent. With conventional incoming-link PageRank, rank flows toward dependents, so foundations do not necessarily rank highest | Specify orientation, dangling nodes, weights and cost semantics; validate the desired priority policy on small dependency graphs |

The receipt's three-node foundation → middle → leaf calculation, with damping 0.85 and uniform dangling redistribution, gives ranks approximately 0.184, 0.341 and 0.474. It contradicts the blanket assertion that this orientation naturally surfaces foundations. This is a counterexample to a proposed property, not a failed test of repository planner code. Likewise, the heuristic calculation proves an overestimate in the stated formula without running A*.

“Sublinear” needs a defined input size and measured workload. The proposal's O(G×A) relevance scan is linear in action count for fixed G; pruning an exponential state space does not prove sublinear action-count complexity or a universal four-step search depth. Keep the six budgets—cached observation <100 ms, fresh observation <30 s, planning <5 ms, PageRank <2 ms, replanning <1 ms and DOT <1 ms—as unmeasured targets, including catalogue size, cache hit/miss and process-start overhead in future receipts.

### Execution, persistence and remaining increments

| Proposed module/integration | Required closeout evidence |
|---|---|
| `state.ts`, hardware/test/git observation and task hooks | Versioned observations with exit status and target; no shell pipeline may promote matching output text over a failed command; cache invalidation must include consumed dependencies/configuration |
| `actions.ts` and `goals.ts` | Reconcile all action references, preconditions, numerical outcomes and seven goal templates; distinguish unavailable, unknown and temporarily reserved resources |
| `planner.ts` and `pagerank.ts` | Correctness oracle, terminating search, objective/heuristic contract, resource-aware priority and all measured budgets |
| `executor.ts` and swarm coordinator | Authorised bounded dispatch, per-action IDs, verification of effects, failure/cancellation and recovery; illustrative agent roles do not authorise this review to launch teams |
| Memory state/plan/history | Revision-bound persistence, concurrent update policy and restart tests; stored predicted effects must not become observed facts |
| `visualize.ts` and graph CLI | Graph generated from the same catalogue/state as the executable plan; colours must reflect evidence, not the example's static completion labels |
| Five CLI operations: plan, observe, prioritize, execute, graph | A pinned supported CLI and real command results; the unversioned example commands are not verified interfaces |

The six declared external integration points—memory, swarm, hooks, cargo tests, USB enumeration and Git—therefore remain contracts to implement. The initial status arithmetic (14+4+19+1=38) also disagrees with “37 ADRs”; use the [current local inventory](closeout/sensing-decisions.md) instead of hard-coded status totals. Alternatives remain design options, but a goal-filtered dependency scheduler should be evaluated against the actual requirements before accepting the claim that topological planning necessarily executes everything.

CP-01/06/07/08/09 should first ratify observed-versus-predicted state and authoritative catalogue links; correct search/priority semantics; implement and test the seven modules and five commands; then demonstrate execution/persistence/recovery on bounded development tasks. Planning, developer-tooling and evidence maintainers are proposed accountable roles. The [estate execution sequence](closeout/execution-sequence.md) remains the review's explicit roadmap; this deferred automation is not a prerequisite for manually resolving its established gaps.

## Rust primary-backend migration

[ADR-020](../../../RuView/docs/adr/ADR-020-rust-ruvector-ai-model-migration.md) remains Accepted as a direction: prefer Rust for production signal/model services while retaining Python for prototyping. It does not establish completed migration. The [seven-source receipt](evidence/ruview-migration-snapshot.json) records manifests and selected entrypoint/router code. No Cargo resolution/build, model conversion, linkage inspection, size/startup benchmark or runtime comparison ran.

The workspace now lists fifteen members, rather than the ADR's twelve. `wifi-densepose-nn` defaults to ONNX and declares optional tch/Candle features; as the [foundational backend assessment](#neural-backend-selection) explains, dependency declarations do not establish three working backend implementations or actual GPU provider selection. The workspace declares `ort` 2.0.0-rc.11. A native Rust executable is not automatically a static, self-contained ONNX deployment; that needs inspection of the built artifact and runtime dependencies.

### Three phases and twelve replacement mappings

| Phase-1 component | Current evidence and acceptance boundary |
|---|---|
| CSI processing | Rust processing exists; [codec and input-path findings](#firmware-and-server-parser-disagree) prevent treating language migration as input parity |
| Motion detection | Feature/rule implementations exist; validate comparable inputs, thresholds and false-positive/negative behaviour against the selected Python path |
| BVP extraction | [Signal integration](#signal-and-mat-integration-helper-availability-versus-execution) distinguishes helper implementation from selected weighted/fallback execution |
| Fresnel geometry | Solver helpers exist; acceptance needs consumed geometry, node positions and localisation error evidence |
| Subcarrier selection | Graph/min-cut helpers exist; establish selection behaviour and actual consumer cadence rather than inheriting the algorithm label |
| Spectrogram | Signal helpers exist; compare scaling, windows, time/frequency axes and selected attention path on defined inputs |
| Pose inference | ONNX session implementation exists; require a compatible trained model, exact input/output contract and reported provider execution |
| DensePose mapping | DensePose head/module exists; helper availability does not establish complete trained inference through the server |
| REST API | MAT exposes an Axum router for MAT-specific operations; it is not a FastAPI parity manifest |
| WebSocket stream | MAT provides `/ws/mat/stream`; the separate sensing server provides sensing/pose streams. Consumer and source contracts differ |
| Survivor detection | MAT components exist; [tracker-to-response integration](#survivor-tracking-and-operational-integration) remains incomplete |
| Vital signs | Multiple extraction/proxy paths exist; presence of MAT/NN modules does not establish physical or clinical validation |

Phase 2's five RuVector declarations and role mappings require the earlier [signal/MAT review](#signal-and-mat-integration-helper-availability-versus-execution): min-cut, attention-gated processing, attention, solver and temporal storage each need actual caller and numerical evidence. The selected signal manifest declares four of the five RuVector names plus Midstreamer dependencies; temporal storage appears elsewhere, so the five-crate table should not imply all five are active in every signal execution. “Subpolynomial” remains an algorithm-specific claim to establish, not a property inherited by all callers.

Phase 3 has a partial consolidation. The sensing-server binary serves HTTP at default port 8080 and a dedicated sensing WebSocket listener at 8765; it also exposes `/ws/sensing` on the HTTP router. This supports same-origin sensing access, but it is not the proposed single listener at 8000. The inspected binary does not reference `wifi_densepose_mat` or mount `/ws/mat/stream`; MAT's `create_router()` is a separate library surface. Require one selected deployment topology and an endpoint/auth/error/stream parity matrix before deprecating Python.

### Build and migration gates

| Named command or migration step | Manifest/source finding | Required disposition |
|---|---|---|
| Lightweight MAT build with `std,api,onnx` | MAT declares `std` and `api`, but no `onnx` feature. Its NN dependency does not disable dependency defaults | Correct package/feature selection; `--no-default-features` on MAT alone is not a declaration of every transitive feature |
| Full build with `all-backends` | The feature is declared on NN in a virtual workspace | Select the package/features explicitly and prove each claimed backend; building dependencies does not establish runtime selection |
| MAT build with `std,api` and inspect `target/release/wifi-densepose-mat` | MAT has no explicit binary entry and no implicit `src/main.rs` or `src/bin` directory | Name the actual deployable binary and router composition; building the library cannot establish the documented executable |
| `cargo check --workspace` | Typechecking scope differs from a selected ONNX-only deployment and executable acceptance | Pin target/features/dependencies and retain full exit status/output; run actual release/link checks separately |
| `cargo test --workspace` | Workspace tests cover selected host members; the WASM edge crate is excluded | Run relevant target/runtime tests separately and state their coverage |
| One-time ONNX conversion | Example omits a concrete model, dummy input and preprocessing/output contract | Export a revision-bound compatible model and compare Python/Rust outputs on shared fixtures, including unsupported operations and dynamic shapes |
| Retain Python fallback | Python sensing code and deployment remain | Define supported fallback, source labelling, failure handling and retirement conditions explicitly |
| UI backend adaptation | Original `sensingOnlyMode` gating no longer matches inspected handlers | Use the [current UI contract](#sensing-ui-source-and-freshness-contract), then verify both selected backends |
| Deprecate Python after parity | No complete parity receipt is established | Ratify the parity matrix and migration/rollback acceptance before declaring production replacement |

The root workspace includes an OpenBLAS-configured dependency declaration, but the inspected signal manifest does not itself consume `ndarray-linalg`; neither “OpenBLAS required everywhere” nor “no OpenBLAS anywhere” follows from that declaration alone. Resolve the selected dependency graph and inspect the output. The claimed approximately 50 MB binary, 20 MB memory, sub-100 ms startup, static delivery and cross-target portability remain unmeasured. Browser WASM, Raspberry Pi and ESP32-S3 have distinct runtime requirements; a generic `cargo build --target` promise is insufficient.

CP-01/03/06/08/09 should sequence deployable target and feature selection, exact API/input/model contracts, Python/Rust numerical and error parity, measured release/runtime properties, target-specific tests, and deprecation/rollback. Rust backend, model and delivery maintainers are proposed accountable roles. Accepted architecture and testable Rust helpers advance the vision, but the production replacement claim requires the complete selected path.

## CRV stage facade and identity evidence

[ADR-033](../../../RuView/docs/adr/ADR-033-crv-signal-line-sensing-integration.md) says Accepted — Implemented. Preserve that historical status while qualifying its scope: a feature-gated `crv/mod.rs` facade exists, but this does not establish the complete proposed six-stage sensing journey. The [receipt](evidence/ruview-crv-snapshot.json) records four selected source hashes, all 37 original acceptance rows and a non-ignored symbol search across local Rust crates. That search finds `WifiCrvPipeline` and the proposed bridge names only in the facade file; it establishes no named external runtime caller or implemented bridge trait. Aliased or external consumers were not exhaustively traced.

The CRV analogy is a naming/design proposal. Structural similarity between stage names does not validate remote viewing, physical sensing accuracy or human identity. Assess the algorithms against measured sensing outcomes independently of that analogy. No upstream dependency build, unit suite, benchmark, hardware test or identity evaluation ran here.

### Four implementation phases

| Phase | Current disposition |
|---|---|
| 1: module/config/session/output | Consolidated in `crv/mod.rs`, with `WifiCrvPipeline` and `CsiCrvResult` instead of the planned separate `WifiCrvSession`/`WifiCrvOutput` files. Default dimensions are 32 and convergence threshold 0.6, rather than the proposed 128 and cited conservative 0.75 |
| 2: six encoders | Stage I classifier and Stage II descriptor extraction are local; facade methods delegate stage data to `CrvSessionManager`. Separate named encoder files are absent from the module directory. Validate each selected upstream implementation and wire contract |
| 3: convergence | `find_cross_room_convergence(room_id, threshold)` delegates by a single target-room key. It does not accept the proposed list of room/session identities or apply a temporal transition constraint in this wrapper |
| 4: existing-module integration | `pub mod crv` is gated by the optional `crv` feature, with empty defaults. Named source/consumer bridges are not found outside the facade by the bounded search. Existing standalone signal/training algorithms do not establish that wiring |

### Stages and acceptance groups

| Stage / original IDs | Source evidence and required acceptance |
|---|---|
| I: S1-1–S1-5 | Classifier scores six gestalt categories from amplitude/phase statistics and converts the envelope to a synthetic stroke for the upstream encoder. Prove configured dimension/norm and classification fixtures; movement labels from one frame's subcarrier statistics do not establish temporal Doppler discrimination |
| II: S2-1–S2-4 | Converts roughness, centroid, energy, periodicity and phase coherence into text descriptors. Temperature is derived from amplitude energy, contrary to the phase-drift criterion. Prove actual modality mappings, embedding dimensions and upstream attention normalisation |
| III: S3-1–S3-4 | Maps each AP into a circle and each supplied link into a relationship whose strength is copied from `signal_strength`. Empty nodes error. Validate link identity/range, intended SNR transform and positional sensitivity of the selected encoder |
| IV: S4-1–S4-5 | Accept emits no AOL entry; PredictOnly and Recalibrate use clamped supplied score, Reject uses 1.0. All generated AOL timestamps are zero. The local enum has no ForcedAccept; Recalibrate's entry is not flagged. Prove real gate mapping, burst/refractory behaviour and accepted-frame policy |
| V: S5-1–S5-4 | Probes stages 1–4 with a query vector and fixed k=min(3, entry count). Empty query or history returns an error, rather than the specified empty result. Define entry-versus-frame identity, attention/candidate shape and ordering before asserting 50 frames imply 50 weights |
| VI: S6-1–S6-5 | Delegates partitioning of accumulated session data, without a person-count hint in the wrapper. Prove two-cluster/single-cluster behaviour, centroid shape, separation meaning and exhaustive disjoint membership against labelled persons |
| Convergence: C-1–C-5 | Same-target session agreement is not a demonstrated cross-room identity transition. Establish which room/person/session keys are compared, threshold semantics, stage agreement, consensus shape and false matches with time constraints |
| Complete pipeline: E-1–E-5 | `process_csi_frame()` executes only Stages I and II and returns their embeddings. Mesh, coherence, query and partition methods require separate calls. It has no optional pose-hypothesis parameter. Establish a real orchestrator with six outputs, missing inputs, accumulating history and p95 latency evidence |

The 37 criteria remain open at their specified scope. Existing unit tests are source evidence of intended local cases; they were not executed in this pass. The Criterion function called `pipeline_full_session` constructs upstream `CrvSessionManager` directly with synthetic stage data and 64 dimensions. Even a result from that benchmark would need qualification before supporting the local facade's proposed <5 ms six-stage per-frame contract.

Stages I and II are added sequentially to the manager. No rollback surrounds the two additions in `process_csi_frame()`, so error handling and partial-session mutation need an explicit acceptance case. Query, partition and convergence results must carry source frames/session revision if callers are to use them as auditable evidence. The current facade's naming cannot establish that a partition represents a person rather than environmental variation.

The seven-stage/dependency table and five additional RuVector-benefit mappings are not a dependency activation receipt. The crate's optional CRV/GNN declarations and existing min-cut/attention/solver/temporal dependencies need the [consumed-revision review](upstream-decision-scope.md); the local vendored copy is not automatically the selected package. Likewise, the proposed `PersonSeparator` interchangeability with training metrics and cross-room tracker integration remain unestablished by the bounded bridge search.

CP-01/03/06/07/09 should ratify configuration/dimensions, stage data/time semantics and session identity; wire selected signal/training consumers; validate the 37 criteria against actual upstream versions; then demonstrate measured six-stage operation and labelled cross-room transitions. CRV adapter, signal and identity maintainers are proposed accountable roles. Keep Accepted — Implemented scoped to the shipped facade until that larger evidence exists.

## Cognitive container contract and durable lifecycle

[ADR-003](../../../RuView/docs/adr/ADR-003-rvf-cognitive-containers-csi.md) remains Deferred for its proposed contracts. The [seven-source receipt](evidence/ruview-container-snapshot.json) records the current server writer/reader, pipeline, recording module, main caller and edge format, plus the ADR itself. The five proposed type/trait names have no exact matches in local crate Rust files; this bounded search does not establish absence in upstream dependencies or under different names.

| Proposed contract | Current evidence and closeout requirement |
|---|---|
| Fingerprint `.rvf.csi` | No dedicated proposed type/adapter found. Require fixed dimensions, finite values, feature/normalisation version, annotations and durable index reconstruction; the [fingerprint helper review](#fingerprint-retrieval-and-hnsw-acceptance) does not establish persistent HNSW or COW. |
| Model `.rvf.model` | Server `RvfBuilder` writes weights as little-endian f32 VEC bytes, plus manifest, metadata, quantisation, vital profile, embedding, LoRA and witness helpers. Main pretraining caller exports weights and embedding. These establish packaging source, not the proposed ONNX blob, branch history, A/B metrics or complete adaptation provenance. |
| Session `.rvf.session` | Separate recording module writes `.csi.jsonl`; no proposed session type found. Require aligned raw CSI, detections and poses, clock/schema identity, temporal queries, bounded retention and deterministic replay against a pinned model. Recording route integration remains subject to the [training UI review](#training-ui-and-model-operation-contracts). |
| Vector adapter | Proposed `RvfVectorizable` round trip is absent by exact name. Specify information lost during feature extraction, supported antenna/subcarrier counts and how reconstruction differs from original raw CSI; do not promise lossless raw reconstruction from summary statistics. |

The server format uses version-1, 64-byte segment headers and alignment. `from_file` reads the whole file; `build` materialises output bytes. It is not evidence of memory-mapped opening. Reader checks magic, version, payload extent and CRC-derived content hash and rejects CRC mismatches in the inspected source. Its warning concerns missing signature verification and capability enforcement. A version string in manifest metadata is not schema migration or model lineage. `add_raw_segment` can package bytes without supplying an index, COW, WASM, kernel or eBPF consumer; all 25 advertised segment capabilities need explicit adopt/defer/retire decisions.

The edge source instead defines a packed 32-byte header with manifest/WASM/signature/test-vector lengths. A shared RVF name does not prove server/edge interoperability. Require a documented format identity, version negotiation, unknown-segment policy and fixtures through each selected writer and consumer. The server parser loop also allows fewer than 64 trailing bytes to remain outside parsed segments; define and test exact-length, empty, duplicate and trailing-data policies before claiming complete-file validation.

`write_to_file` directly creates/truncates the destination, writes bytes and flushes. This does not implement the ADR's transactional-write promise: no temporary-file replacement or durable-sync protocol is present in that method. Establish crash/interrupted-write recovery, concurrent readers/writers, retained last-good state and error propagation before treating export as deployment. This is source review, not an injected failure experiment.

The lifecycle covers create, ingest, query, branch, compare, merge, export and deploy. Builder creation/export have concrete source; remaining operations need actual consumer evidence and acceptance, including index rebuild, branch isolation/conflicts and deployment rollback. The six positive consequences—single-file transfer, versioned models, replay, atomicity, portability and space efficiency—remain separate acceptance claims. Four negative consequences require format migration/exit, measured serialisation overhead, maintainer documentation and bounded high-rate retention. No latency or capacity benchmark ran.

Storage estimates need correction or an explicit compression assumption. At 329 f32 values a vector occupies 1,316 bytes before metadata/index overhead. One hour at 100 Hz is 360,000 vectors and 473,760,000 bytes, already above the proposed 10–200 MB session range. The four illustrated branch deltas total 8,300 vectors and 10,922,800 raw bytes, rather than about 250 KB. These arithmetic checks do not measure a compressed format. Validate the fingerprint 5–50 MB/10K–100K and model 50–500 MB ranges against actual payloads as well.

CP-01/04/06/07/08/09 require a selected format and accountable consumer, followed by semantic round trips, malformed/truncated-file handling, provenance and authority enforcement, durable recovery, branch/replay acceptance and cross-target deployment. Measure all six original targets: open <10 ms, insert <0.1 ms, 100K-vector query <1 ms, branch <1 ms, merge <100 ms and export about 1 ms/MB. Record hardware, sizes, compression, cold/warm state and tail latency. These are proposed gates, not achieved results. No Cargo suite, deployed consumer, physical device or performance acceptance ran in this pass.

## Commodity sensing capabilities and observation readiness

[ADR-013](../../../RuView/docs/adr/ADR-013-feature-level-sensing-commodity-gear.md) has concrete Python components, but its Accepted—Implemented status covers more than the current evidence establishes. The [probe](evidence/ruview-commodity-probe.py) executes unchanged AST-selected dataclasses, classifier and two extractor methods in isolation, avoiding numerical dependencies. Its [receipt](evidence/ruview-commodity-probe.json) records eight source hashes, four passed assertions, 36 declared test methods and the absent `v1/data/proof/commodity` directory. No complete test suite, FFT execution, installed CLI or physical capture ran. The ADR's external research percentages, prices and setup times remain unevaluated historical claims.

| Contract group | Inspected implementation and closeout boundary |
|---|---|
| Eight input metrics | Linux collector reads RSSI, noise and link quality from `/proc/net/wireless`, plus TX/RX bytes and retries from `iw station dump`. MCS/PHY rate, beacon timing and channel utilisation are not collected by this path. Noise uses the proc value, not the proposed survey method. Driver availability, actual units and permission requirements need target evidence. |
| Collection fallback | RSSI requires proc wireless support; there is no proposed `iw link` RSSI fallback in this collector. Interface validation/parsing uses substring matching. Station command has a two-second timeout; unavailable/timed-out stats become zero, and return status is not checked. Distinguish missing statistics from observed zero and bind the exact interface. |
| Five extraction groups | Time statistics, Hann-windowed FFT and CUSUM exist. The inspected extractor does not implement PELT, cross-receiver signal correlation or packet timing jitter. Noise/SNR and maximum-step features from the proposed specification are absent from `RssiFeatures`; collected link statistics are not used by this RSSI extractor. Ratify these omissions or implement and validate them. |
| Numerical semantics | Actual variance uses `ddof=1`; FFT uses Hann window and power divided by N; CUSUM drift scales with sample standard deviation. These differ from the illustrative population-variance/unwindowed code. Pin expected features for the implemented semantics. Sample rate is inferred from mean timestamp differences, with 10 Hz fallback; this does not resample irregular acquisition. |
| Classification | Default presence threshold is 0.5 with inclusive comparison, and active motion requires band energy at least 0.1. The ADR instead illustrates variance thresholds 2/5 and a different confidence formula. Actual confidence weights base/spectral/agreement 60/20/20; single receiver agreement defaults to 1.0. `_max_receivers` is stored but not consulted in classification. Define and calibrate the accepted rule. |
| Output and multi-receiver contract | Actual `SensingResult` includes presence, level, confidence, variance, band energies and change-point count, without the proposed timestamp/receivers-agreeing fields. Classifier accepts other results, but `CommodityBackend.get_result` passes only its own features. Capability, receiver identity, observation time and source provenance need an explicit consumer envelope. |

The probe confirms that an empty extractor input returns default features, which the classifier labels absent with confidence 1.0. It also confirms that variance 0.5 and motion power 0.1 yield active/confidence 1.0 with one receiver. These are deterministic rule outcomes, not calibrated probabilities. `CommodityBackend.get_result` has no readiness guard. The separate WebSocket tick loop does check at least four buffered samples before extraction, which narrows the empty-start finding; it does not establish freshness after collection stops. Trimming is relative to the last sample timestamp, not wall time: the actual method retains timestamps 1–4 within a thirty-second window regardless of current age. Collection errors are logged while old buffer contents remain. Require unknown/insufficient/stale states, monotonic timing checks, minimum observation duration and explicit recovery before accepting presence outputs.

The four named modules exist. The seven test groups still contain the stated 4/5/8/4/7/6/2 methods, totalling 36; their declaration is not a fresh passing run or a live Linux driver test. The backend accurately restricts its declared capabilities to PRESENCE and MOTION, even though the constructor can receive a simulated collector. The separate WebSocket server exposes source labels and includes a seeded simulation fallback. Therefore “no mock data” must be scoped to a selected, verified collector. Linux selection constructs the collector before `start` validates the interface, so construction alone does not prove acquisition readiness or successful fallback. Windows and macOS collectors now exist as separate paths; the original Windows-only-simulation note is historical.

All eight capability rows need explicit disposition. Binary presence and coarse motion have rule implementations but no accuracy receipt here. Room-level location, person count and walk/sit/stand classification lack an implementation/evaluation mapping in this commodity backend. Respiration is represented by a spectral band feature, which does not establish reliable respiration detection. Heartbeat and body pose remain outside its declared capabilities. Do not promote the ADR's percentage ranges into release acceptance results. Validate receiver count, environment, interference, labelled scenarios, false positives/negatives and uncertainty for any adopted capability.

The nine-row ESP32/commodity comparison is a planning comparison, not a measured deployment trial. Preserve capability and data-source distinctions; remeasure cost, setup time, technical barrier and reproducibility on selected targets, and treat credibility judgments as rationale. All seven promised commodity proof artifacts are absent at the named directory: thirty-second three-receiver capture, metadata, scenario, expected features, expected classification, feature hash and verifier. Require captured provenance and reproducible expected outputs, retaining simulation fixtures separately.

`v1/setup.py` declares `wdp=src.cli:cli`, but the inspected CLI defines no `sense` command. The advertised `pip install wifi-densepose && wdp sense --interface wlan0` journey therefore needs a selected package/revision and executable installation smoke test or a corrected supported command. No package-publication or installation claim was tested. The six positive consequences remain bounded by actual acquisition, capability reporting, setup and deterministic captured replay; the five negative consequences should inform documented scope, calibration and interference tests rather than be treated as resolved by test counts.

CP-01/03/06/08/09 sequence closeout as: pin collector and output contracts; settle the specification/implementation differences; handle readiness, stale/missing input and source transitions; deliver and verify the captured proof bundle; test the supported CLI and target permissions; then evaluate advertised capabilities. Keep the broader [sensing UI freshness](#sensing-ui-source-and-freshness-contract), [macOS bridge](#macos-helper-and-runtime-contract) and [Python proof](#python-proof-replay-and-mock-boundaries) gates attached to their actual consumers.

## Signal algorithm semantics and runtime adoption

[ADR-014](../../../RuView/docs/adr/ADR-014-sota-signal-processing.md) names six modules that all exist and are exported by the signal library. The [seven-source receipt](evidence/ruview-signal-algorithms-probe.json) records them and `lib.rs`. Exact searches for ten principal API names across local crate Rust files find each only in its defining module, including its tests. None of these six files mentions `CsiData`. This establishes standalone array/slice helpers, not the ADR's promised integration with that type or a selected runtime pipeline. Aliases, generated or external callers are outside this bounded search.

| Algorithm | Current semantics and required acceptance |
|---|---|
| Conjugate multiplication | Computes the documented complex product; checks equal, non-empty streams and at least two antennas for the matrix path. Existing tests exercise common-offset cancellation. Require synchronised antenna identity, shared-offset assumptions and real phase-error evidence; the product alone cannot establish that every remaining change represents human motion. |
| Hampel filter | Uses the original input window, median/MAD and a special zero-MAD branch that replaces deviations above 1e-15. Rejects empty input/zero half-window. Pin edge windows, finite values and threshold policy, and show that selected filtering removes interference without removing useful motion. |
| Fresnel estimator | Geometry provides the radius formula, but rate estimation uses autocorrelation over a lag range derived from 1.5–10 seconds and amplitude-based confidence. Its confidence path depends on wavelength/displacement, without using the stored TX/body or body/RX distances. Require a specified geometry contribution, amplitude calibration and error/uncertainty evaluation before claiming weak-signal geometric detection. |
| Spectrogram | Computes per-subcarrier positive-frequency STFT with selectable windows and magnitude/power output. Input is used directly without automatic DC removal in this helper. Shape is frequency × time. Pin sample rate, units, scaling, history length, padding, window and the actual CNN consumer; a matrix is not an accepted model input until that contract is tested. |
| Subcarrier selection | Implements motion/static variance ratio with epsilon, threshold and top-K; extraction separately checks selected indices. The alternative online path ranks absolute variance and has different semantics. Require labelled calibration periods, dimensional/finite checks, stable subcarrier identity and measured SNR benefit; do not import the ADR's 6–10 dB claim as a result. |
| Body velocity profile | Averages per-subcarrier STFT magnitudes after mean removal, then maps speed to FFT bins using the absolute Doppler frequency. Equal positive/negative speed magnitudes therefore select the same bin; signed direction is not established. Require validated rate/carrier/velocity configuration, a stated directional contract and held-out environment evidence for robustness. |

The [isolated probe](evidence/ruview-signal-algorithms-probe.py) compiles the unchanged Fresnel geometry, estimator and amplitude helper, removing only the `thiserror` derive/attributes and excluding the later external solver helper. Three assertions pass: a deterministic 0.25 Hz input yields the same 15 BPM and approximately 0.739 confidence for distances 1/1 m and 20/30 m at the same carrier; a NaN distance is accepted by the geometry constructor; constant input returns `NoSignal`. These are algorithm-boundary observations, not live breathing measurements or accuracy evidence. The later geometry solver and RuVector integrations retain their [separate review](#signal-and-mat-integration-helper-availability-versus-execution).

Configuration admission needs its own tests. For example, BVP and spectrogram allow window size one through their positive-size guards, while the Hann expression divides by `window_size - 1`. Positive/finite sampling and carrier frequencies and velocity-bin counts also need a consistent contract. This is source-derived risk, not an executed FFT failure. The Fresnel NaN constructor finding is directly reproduced.

The four per-module promises have different evidence: declarations comprise 5/7/10/8/8/8 tests (46 total); source uses constructed deterministic examples without a `rand`, `mock` or `seed` token in the six inspected files; module documentation carries paper references; direct `CsiData` integration is not shown by the selected APIs. Synthetic mathematical fixtures can be useful without being captured observations. Neither test declarations nor absence of random calls prove research-grade behaviour. External literature and the ADR's state-of-the-art comparisons were not evaluated in this local-source pass.

All five claimed benefits remain acceptance work: publication-level accuracy needs a specified benchmark and reproduced comparison; physical formulas need calibrated inputs; cross-environment robustness needs held-out environments; CNN readiness needs consumed shape/scaling fixtures; and SNR improvement needs measured baseline comparisons. The three costs require runtime CPU/memory/latency budgets, acquisition and uncertainty for geometry, and adequate timestamped history at the selected sample rate. The estimator's actual lag/history guards, rather than a generic “more than one second” statement, must govern readiness.

CP-01/03/06/09 should first select algorithms per device/sensing tier and wire explicit adapters/order, then validate parameter and timing boundaries, then prove model/consumer compatibility and evaluate captured data against baselines. Keep unadopted algorithms explicitly deferred. No full crate build or suite, physical capture, publication comparison, cross-environment accuracy or performance acceptance ran in this pass.

## Contrastive training objective and consolidation boundary

This is a **partial assessment of ADR-024**, limited to the projection, contrastive objective and pretraining-to-fine-tuning boundary. The [ADR](../../../RuView/docs/adr/ADR-024-contrastive-csi-embedding-model.md) now has a [cumulative review](#aether-cumulative-closeout-contract); this section retains its narrower evidence scope. Its augmentation, indexing, complete seven-phase rollout, cross-modal alignment, quantisation/edge targets, performance, named acceptance tests and future increments are not closed by this section.

The [four-source receipt](evidence/ruview-contrastive-objective-probe.json) covers the ADR and sensing-server `embedding.rs`, `trainer.rs` and `main.rs`. The actual pretraining method builds two deterministic augmented views, reconstructs transformer and projection weights inside the loss closure, mean-pools body features, computes embeddings, estimates gradients over combined parameters and updates both transformer and projection. The main CLI calls this method and exports weights/embedding configuration. This is stronger evidence than an empty training stub, but no complete training run or captured-data evaluation occurred here.

| Stated contract | Current implementation and closeout requirement |
|---|---|
| Symmetric NT-Xent over 2N views | `info_nce_loss` averages A-to-B cross-entropy over N B candidates. It excludes same-view negatives and the reverse B-to-A contribution. Ratify that alternative objective or implement the specified symmetric loss; tests must verify candidate sets and symmetry. |
| VICReg variance/covariance terms | Pretraining returns only `info_nce_loss`; the inspected training path does not add the specified weighted regularisers. No exact `variance_loss`, `covariance_loss`, `vicreg` or `aether_loss` symbols appear in the three selected source files. Trace equivalent implementations if claimed; otherwise retain collapse prevention and monitoring as unfinished work. |
| BatchNorm projection and learned temperature | Current forward path is linear, optional LoRA, ReLU, linear, optional LoRA, optional L2 normalisation. It has no proposed BatchNorm stage. Temperature is an input scalar, with CLI 0.07 and helper floor 1e-6; this path does not optimise the ADR's learned temperature with floor 0.01. |
| Joint supervised contrastive training | Loss structures contain a contrastive field, but the inspected supervised epoch accumulates six supervised terms and does not calculate the contrastive term. A weight/config field alone does not establish the proposed joint objective. Require a traced loss and gradient contribution with data-backed regression evidence. |
| EWC consolidation and application | `consolidate_pretrained` computes Fisher using squared parameter magnitude, explicitly described in source as a proxy, rather than the contrastive data loss. Penalty and gradient accessors exist. Searches in the three files find calls in tests, not the main pretraining handoff or supervised update. Require the actual data objective, consolidation handoff, penalty/gradient application and retention evaluation. |

The [reproducible probe](evidence/ruview-contrastive-objective-probe.py) compiles the unchanged cosine and InfoNCE helpers. Three assertions pass: a singleton pair returns zero loss even for orthogonal embeddings; four identical embeddings return `ln(4)`; an asymmetric fixture gives A-to-B about 1.127 and B-to-A about 0.693. These establish objective semantics, not observed training collapse. A singleton has no negative candidate in this objective, so the trainer's `batch_size.max(1)` and short final batches need explicit handling and acceptance. Empty input also returns zero in the inspected helper/pretraining method; a zero result must not be presented as convergence without sample and batch evidence.

CP-01/03/07/09 should settle the accepted objective before comparing training metrics across revisions. Pin candidate construction, batch policy, projection/temperature semantics and regulariser weights; verify nonzero expected gradients and optimiser state; wire any adopted consolidation/joint objective into real callers; then measure held-out representation quality and downstream retention against a baseline. Existing [dataset separation](#dataset-separation-and-evaluation-meaning), [SONA profile](#sona-feedback-admission-and-profile-lifecycle) and [model-operation](#training-ui-and-model-operation-contracts) requirements remain attached. This pass does not establish the ADR's convergence, label-efficiency, identity, device or performance claims.

## Embedding augmentation and projection state

This continues the **partial ADR-024 review**, covering all seven augmentation proposals and selected projection/LoRA persistence paths. It does not complete the remaining deployment, cross-modal, performance, phase and acceptance-test assessment. The [probe](evidence/ruview-embedding-state-probe.py) compiles four unchanged local modules—embedding, graph transformer, SONA and sparse inference—and runs six focused assertions. The [receipt](evidence/ruview-embedding-state-probe.json) hashes those modules plus trainer, main and RVF container. No full suite, full training or RVF/device round trip ran.

| Proposed augmentation | Current `CsiAugmenter` contract |
|---|---|
| Temporal jitter | Default ±2 rows, versus the ADR's ±3 frames. Indices clamp at edges, duplicating boundary rows; there is no timestamp-aware shift. Define row/time identity and padding semantics at the data/model boundary. |
| Subcarrier masking | Fixed probability 0.15 independently per element, applied to view A. This differs from sampling a probability in 0.05–0.20 and needs an explicit decision on whether a mask persists across a temporal window. |
| Gaussian noise | Fixed sigma 0.05 for both views, rather than sampling sigma 0.01–0.05. Validate units, normalisation order and acceptable perturbation strength. |
| Phase rotation | Chooses one offset in ±π/4 and multiplies every amplitude in the entire window by its cosine. This does not implement a per-frame complex phase rotation sampled from 0–2π. Preserve that distinction in model metadata and any physical-invariance claim. |
| Amplitude scaling | One factor in 0.8–1.2 scales the full B window. This is present; accepted units and preservation of the intended label still require data evidence. |
| Adjacent subcarrier permutation | No corresponding implementation in the inspected augmenter. Ratify omission or define a hardware-specific mapping and validate it. |
| Temporal crop/interpolation | No corresponding implementation in the inspected augmenter. Ratify omission or define timestamp-aware packet-loss handling and validate it. |

The augmenter uses fixed compositions: A receives jitter/noise/masking; B receives jitter/scaling/phase-labelled scaling/noise. It does not randomly select two to four augmentation types per view. In the isolated probe, disabling all other effects leaves A unchanged and B uniformly scaled by approximately 0.9627645. The same factor applies to every tested row and column. This confirms the numerical operation, not whether it preserves room, person or activity identity. Those three semantic invariances need separate labelled tests and an explicit statement of which embedding task each augmentation serves.

Projection-state findings use nonzero rank-one adapters on two zero-base linear layers with normalisation disabled. `forward([1,1])` returns `[4,4]`; after `merge_lora`, it returns `[16,16]`. The merge adds adapter deltas into base weights while leaving adapters present, and `forward` continues to add their outputs. The matching unmerge restores `[4,4]` in this fixture. Require an explicit merged-state invariant, idempotence and output-equivalence checks across merge/unmerge and environment switching; do not rely only on a merge-then-unmerge round trip.

The same probe verifies that ordinary `flatten_into`/`unflatten_from` loses the nonzero adapter contribution: flatten records only base linear parameters and restore sets both adapters to `None`. Separate `flatten_lora`/`unflatten_lora` helpers exist, so this is an incomplete ordinary round-trip contract, not absence of all adapter serialisation. The pretraining loop reconstructs the projection using the ordinary round trip on every loss evaluation and after each update; passing an adapted projection into that path therefore needs an explicit adapter policy. `freeze_base_train_lora` computes only the adapter path, without the frozen base contribution, so it also needs a defined training objective rather than being treated as equivalent to freezing optimiser updates on the base.

The server RVF helper stores JSON configuration length, JSON and f32 projection weights together in `SEG_EMBED`; `embedding()` checks prefix/config bounds and a four-byte-aligned weight payload. This differs from the ADR's proposal to put projection weights alongside backbone weights in VEC. The main pretraining export writes base projection weights through this helper. Separate LoRA segment helpers do not prove that this caller exports/restores active adapters or that loading applies them. Require versioned dimensions, exact parameter counts, finite values, adapter identity/rank and observed output equality through the actual writer and inference consumer. The [container lifecycle](#cognitive-container-contract-and-durable-lifecycle) requirements remain applicable.

The main pretraining command attempts the selected dataset and explicitly logs a synthetic-data fallback for errors or empty results, then proceeds with constructed windows. This is visible fallback, but the inspected export configuration does not carry the full data-source/fallback identity. Bind export provenance to the actual loaded corpus, augmentation policy, seed, objective and revision, and make a failed requested dataset distinguishable from successful captured-data training. No dataset failure or exported artifact was executed in this pass.

CP-01/03/07/08/09 must settle augmentation semantics and label preservation, separate base and adapter state, enforce merge/output invariants, and verify actual model export/load equality before accepting environment-adaptive embeddings. The prior [objective and EWC review](#contrastive-training-objective-and-consolidation-boundary) remains a prerequisite. Remaining requirements are dispositioned in the [cumulative review](#aether-cumulative-closeout-contract).

## Embedding quantisation and deployment evidence

This is a further **partial ADR-024 assessment**, covering quantisation, parameter budgets, the eleven performance targets and the two deployment proposals. The [seven-source receipt](evidence/ruview-embedding-quant-probe.json) and [probe](evidence/ruview-embedding-quant-probe.py) compile four unchanged local modules and run six focused assertions. No full training, quantised-model inference, index benchmark, RVF round trip or physical-device execution occurred.

`validate_quantized_embeddings` receives already-produced FP32 vectors. It symmetrically quantises and dequantises each vector and the query, then compares cosine-distance ranks. It does not quantise the backbone or projection and run them on CSI. The passed quantiser reference is unused; the function calls static symmetric helpers. Therefore this receipt addresses storage precision of existing embeddings, not the ADR's mandatory preservation of neighbours through INT8 model inference or its proposed mixed INT8-backbone/FP16-projection fallback. Require the actual selected kernels, weights, input calibration and output vectors through both inference paths.

The validator ranks all supplied candidates, without a k=10 selection. Its rank helper assigns average ranks for ties, but the final expression uses the simplified squared-rank-difference formula without tie correction. The compiled fixture uses query `[1,0]` and candidates `[1,0.001]`, `[1,0.002]`, `[0,1]`. Quantisation ties the first two candidates; the validator reports 0.875, whereas correlation of the resulting average-rank arrays is approximately 0.8660254. Two further assertions show empty and singleton candidate sets return 1.0. These values must not be treated as useful validation success without a minimum candidate/query policy. Define tied-rank correlation, invalid-value rejection, query sampling and neighbour coverage explicitly before using the 0.95 gate. This fixture establishes a metric discrepancy, not a demonstrated production threshold crossing.

| Parameter budget | Executed count and implication |
|---|---|
| Backbone: 56 inputs, 17 keypoints, width 64, four heads, two GNN layers | 29,956 parameters, matching the detailed backbone subtotal. |
| Default projection: 64→128→128 | 24,832 parameters; the implemented projection lacks the ADR's additional 256 BatchNorm parameters. |
| Optional pose encoder: 51→128→128 | 23,168 parameters, rather than the ADR's 7,040. The printed two-layer formula itself sums to 23,168. |
| Backbone plus projection | 54,788 parameters, or 219,152 bytes if every parameter is stored as f32. |
| Including pose encoder | 77,956 parameters, or 311,824 f32 bytes. One byte per parameter alone is 77,956 bytes, exceeding the proposed <65 KB model target for this full configuration. |

Parameter count is not a measured deployed file size or peak SRAM requirement. Quantisation scales, biases, container metadata, buffers, stack, allocator, index, runtime and firmware must be included. Optional pose encoding may remain a training/server capability, but that deployment choice needs explicit disposition. Do not infer ESP32 memory margin from the ADR's 53–60 KB headline.

All eleven performance targets remain unmeasured here: x86 extraction <1 ms; ESP32 INT8 extraction <2 ms; 10K-vector HNSW search <0.5 ms; convergence <200 epochs; five-room identification >95%; six-activity classification >85%; five-subject re-identification >80%; anomaly F1 >0.90; INT8 rank correlation >0.95; INT8 model <65 KB; and training peak RSS <50 MB. Resolve the re-identification row's mAP label versus Rank-1 measurement before defining its gate. Bind each result to a dataset split, selected model, corpus size, target and reproducible measurement. The [fingerprint assessment](#fingerprint-retrieval-and-hnsw-acceptance) still identifies a linear local index, so HNSW latency cannot be assigned to that helper.

The ESP32 proposal needs a real cross-compiled model path, precision-specific kernels, measured activation/runtime memory, an actual mapping for the claimed 256-core/8K-reference arrangement, and capture→embedding→search timing at the selected sampling rate. The 27.1 ms arithmetic budget is a proposal, not a measured latency distribution. The WASM proposal requires a deployed FP32/FP16 consumer, worker lifecycle and capacity evidence for the selected index. Exact searches across local crate Rust files found no `/embedding/extract`, `/embedding/search`, `/embedding/stream` or `/embedding/drift` route strings; generated or external routing is outside that search. Public helpers do not prove these REST/WebSocket endpoints or non-blocking browser execution.

CP-01/03/06/08/09 require corrected counts and target scope, a valid neighbour-preservation metric, a comparison through actual quantised inference, then packaging/memory and target latency evidence. Retain Phase 5 mixed precision and full packaging, and Phase 6 end-to-end/convergence/benchmark acceptance as open. The [cumulative review](#aether-cumulative-closeout-contract) now reconciles the remaining ADR sections; this section alone remains a partial evidence scope.

## AETHER cumulative closeout contract

This completes the **dedicated documentation assessment** of [ADR-024](../../../RuView/docs/adr/ADR-024-contrastive-csi-embedding-model.md), retaining Partially Implemented and open runtime acceptance. The previous four focused sections establish [objective/EWC](#contrastive-training-objective-and-consolidation-boundary), [augmentation/state](#embedding-augmentation-and-projection-state), [quantisation/deployment](#embedding-quantisation-and-deployment-evidence) and the earlier [fingerprint path](#fingerprint-retrieval-and-hnsw-acceptance). The [requirements receipt](evidence/ruview-aether-closeout.json) preserves Phase 1–6 bullet requirements, all 24 unit and nine integration test names, five Phase 7 subphases and five future dispositions. It records six additional source hashes. No complete training, suite, target benchmark or model-quality acceptance is asserted.

The five original embedding gaps—persistence, similarity, unlabelled pretraining, transfer and meaningful index inputs—have partial helpers and CLI paths. They remain system requirements, rather than consequences of merely exposing intermediate features. Reusing a backbone supports the design rationale; literature comparisons, architecture naming, estimated effort and speculative superiority over generative alternatives do not establish accuracy or runtime efficiency. The proposed 8–12 days and per-phase day/line estimates require replanning after scope and ownership are ratified.

| Phase | Cumulative disposition and acceptance |
|---|---|
| 1: embedding module | Exported module, projection, extractor and base-weight round trip exist. Proposed BatchNorm/training forward, full augmentation/config/loss structures and monitoring APIs differ or are missing. `EmbeddingExtractor.extract` exists; the promised `forward_dual` API does not appear in the inspected files. Require the specified shared pose/embedding execution or revise that promise. |
| 2: pretraining | Real finite-difference parameter updates and a CLI loop exist, but the stated symmetric/VICReg objective, result/history API, alignment/uniformity monitoring and joint objective remain unestablished. Settle those contracts and prove useful non-collapsed representations on pinned data. |
| 3: four indices and API | The four `IndexType` variants annotate one linear in-memory index implementation. Its string metadata is narrower than the proposed typed environment/person/activity/confidence/profile structure. No proposed prune/stats lifecycle or four independently configured HNSW deployments is shown. Establish the specified collection triggers, storage and consumers before accepting the REST/index journey. |
| 4: optional cross-modal alignment | `PoseEncoder` and `cross_modal_loss` helpers exist; selected main/trainer files show no paired-data training consumer or pose-retrieval evaluation. Explicitly adopt or defer Phase C. If adopted, bind paired timestamps/labels and measure cross-modal retrieval, not just helper shape or aligned-vector loss. |
| 5: quantisation and packaging | Vector-rounding validation, embedding/LoRA segment helpers and export paths exist. The metric, actual quantised model, mixed-precision fallback, complete state restoration and ESP32 latency remain open. |
| 6: integration and benchmarks | Require CSI→embedding→selected HNSW insert/search correctness, MM-Fi convergence, quantisation preservation, target timing and all eleven performance gates. An isolated helper probe cannot satisfy these journey-level claims. |
| 7: required adaptation integration | Keep all five subphases required until superseded explicitly. Nonzero LoRA merge/restore issues and unapplied proxy EWC remain open. Drift detection and mining are helpers with different semantics from the complete workflow; profile storage must be proven through actual activation. |

For the four index uses, preserve the proposed ten-second environment aggregation and drift-triggered update; activity-transition segmentation; first-sixty-second calibration and recalibration baseline; and confirmed person-trajectory identity. Each needs a consumer and evidence, not just an enum. Current `is_anomaly` tests one query against all entries and treats an empty index as anomalous; it does not implement the specified five-consecutive-frame rule. Distances do not themselves establish intrusion, fall or person re-identification labels. Comparing every stored profile is linear in profile count, even if each 128-dimensional comparison has fixed cost; clarify the ADR's O(1) wording.

`EmbeddingExtractor::with_drift_detection` updates the existing mean/variance detector and exposes drift accessors. This does not replace statistical drift with semantic profile matching. `sona.rs` has no `env_embedding` field in the inspected source, and the optional extractor detector does not automatically insert into an index, pause insertion, switch profiles or launch adaptation. Require a single observable orchestration path with admitted data, profile identity and recovery.

The [mining probe](evidence/ruview-embedding-mining-probe.py) and [seven-source receipt](evidence/ruview-embedding-mining-probe.json) add three assertions against four unchanged modules: warmup returns all twelve negatives for four anchors; a 0.25 ratio then selects three pairs; all three can belong to the same anchor. The helper ranks negatives globally, while the ADR promises the hardest fraction per anchor. Its mined-loss caller gives anchors without negatives only a positive term. Main/pretraining uses the ordinary loss, so these observations do not establish that mining currently affects training. Decide global versus per-anchor policy and prove actual loss/gradient consumption.

The 33 named tests need semantic correction as well as execution. For unit-normalised vectors, the sum of population variances across dimensions is at most one. Thus a test requiring variance >0.5 in every one of 128 dimensions is impossible; likewise an alert on any dimension below 0.1 will inevitably fire on such vectors. Even sample variance with N≥2 has total at most N/(N−1)≤2. Apply variance regularisation/monitoring to a specified representation with attainable scale, or change the criterion. Identical embeddings across multiple pairs also do not imply zero InfoNCE loss: the prior probe shows collapsed N=4 yields ln(4). Clarify positive-pair identity versus whole-batch collapse. Projection, augmentation, loss, metric and extractor tests must assert the chosen semantics; all nine integration tests still need real consumer/data/target scope.

The nine benefits require separate evidence: unlabelled pretraining and reduced labels; infrastructure reuse; consumed HNSW inputs; a genuine shared dual-output pass; validated anomaly tasks; useful compact environment descriptors; held-out transfer; complete edge operation; and measured information-disclosure properties. A 128-dimensional bottleneck is not proof of non-invertibility or privacy, especially when identity and room recognition are intended tasks. CP-04 requires explicit access, retention, identity and disclosure boundaries before cross-site sharing. The 512-byte descriptor comparison does not replace the full adaptation profile or prove greater discrimination.

The five costs remain accountable: backbone coupling, augmentation sensitivity, added training, quantisation distortion and the proposed BatchNorm train/inference distinction. Record BatchNorm as a design difference until adopted. The six risk mitigations require working collapse monitoring/regularisation (or explicitly adopted BYOL), measured quantised recall/mixed precision, a tested supervised fallback, explicit optional cross-modal scope, a verified freezing/stop-gradient policy, and labelled person-retrieval evidence. A comparable published architecture does not validate this model.

The five future directions—masked reconstruction, hyperbolic embeddings, temporal CPC, federated training and advanced attention—are proposed deferrals pending explicit adoption, not silently delivered or mandatory blockers of the selected baseline. If federated training is adopted, gradient transfer needs its own authority and disclosure contract; absence of raw CSI transfer is not a privacy proof.

CP-01/03/04/06/07/08/09 sequence work through ratified objective/data/identity contracts; corrected state and metric semantics; wired index/adaptation/API consumers; reproducible complete artifacts; then held-out quality and target acceptance. Accountable roles are proposed, not assigned. The documentation assessment is complete for ADR-024; all implementation acceptance remains evidence-dependent.
