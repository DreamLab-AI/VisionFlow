# Solid, SideStr and knowledge repairs — 2026-11-03

Source and native-test acceptance only unless a publication is explicitly stated. No live balances, chain activation, server deployment or Git export against production data was performed. Foreign primary-checkout changes and foreign worktrees were preserved.

## Revisions

| Repository | Local main repair | Version / delivery |
| --- | --- | --- |
| solid-pod-rs | Product `c27cdfed1d96fc12dc2c7db5df866e9bed75524a`; later test-only `7718095`; main clean | Workspace alpha.19 publication handled by root from frozen product revision |
| sidestr-rs | `7f04644411220ba0d3b9c7bdba58bbf56be790fa` | Public core0.4.3 published by root; private EVM remains0.1.0; no running chain activation |
| ontomark | `6c3f693b36301ee95c8b684a000b5aeec3ac9bbd` | Root published certificate0.1.1, compile0.6.4 and CLI0.7.3 |
| vowl-wasm | product `756f49f5ab7bac9a97b3701157bb923da34abe22`, metadata `2aff199e2de06f4bba512301acf7023477c492ac` | Official pinned-toolchain0.2.3 GitHub bundle uploaded; npm permission/package E404 remains open |
| knowledgeGraph | `4809ea9772e61766058cf20d38e7527082e78fed` | Retained explorer transfer repair; no deployment claim |
| visionGraph | `fa7604d65e717bbae297003099d192c9dfc66d04`, root adoption `f7186a7d2` | Promoted certificate path; root pins released CLI0.7.3 and official VOWL0.2.3 tarball |

## External concern dispositions

| Concern | Disposition and acceptance boundary |
| --- | --- |
| SP-04.10 malformed Turtle conditions | Missing type, missing/invalid payment cost, unbalanced or trailing conditions return AclParse and settle Invalid. Valid typed conditions still parse. Follow-up preserves restrictions across same-subject statements and normalises HTTPS ACL predicates consistently. `core/src/wac/parser.rs` |
| Turtle lexical edge cases | Quoted hashes, escaped quotes and semicolons inside IRIs are preserved. Subject grouping is indexed, avoiding a quadratic scan. This remains the declared Turtle subset. Parser regressions pass. |
| SP-07.6 provenance leak | Resolve caller then resource Read with the shared WAC path; audience cache policy retained. Current sidecar included only for the canonical single matching activity. Arbitrary authored Turtle is not certified as historical provenance. `server/src/handlers/prov.rs:387,390,409,435` |
| Provenance anchor double replay verification | Verify once, compare captured owner in memory: authenticated non-owner403; missing/invalid authentication401. Existing owner-only route tests pass. `prov.rs:472` |
| SP-14.1 double pod prefix | GitHttpService receives data_root and extracts slug once; owner clone fixture returns200 with real git-http-backend. Earlier claim that the routing bug was already fixed was incorrect. |
| SP-14.2 whole-history child/history disclosure | Current root Read plus authenticated principal equal to the cryptographically named pod key. Delegated/public root Read cannot export private child/history objects. Alias pods without cryptographic owner binding denied. Push remains refused. Owner custody/recovery semantics recorded in ADR2010. |
| SP-07.15 stale payment fallback | Only NotFound falls back for state/replay/legacy ledger; other read/parse failures propagate. Explicit failure-injection test and legacy migration controls pass. |
| WAC balance/debit mirror divergence | Both consume canonical PaymentStore. Charge owns shared-lock read/check/debit/commit. Repeated-grant regression explicitly creates canonical state before reading/charging and verifies actual ledger depletion. |
| MRC20 proof presenter credited shared-address deposit | Require DID-tweaked deposit target matching authenticated beneficiary; configured network restriction. Same proof presented by another key is refused before rightful credit; existing generic-address funds need explicit migration/recovery, not bearer credit. `server/src/handlers/pay.rs:610` |
| Payment arithmetic | Fee>=10000, token price overflow and swap numerator overflow refused. AMM reversed pair maps amounts into stored currency order. Real transaction-fixture buy/withdraw tests retain integer division remainder as spendable sats. |
| Payment intent signing key duplication | Commit clears secret copies; recovery loads matching current issuer custody. Legacy copies scrub on next commit. No backup erasure, key revocation or rotation claim; issuer trail remains Control-gated. |
| Unbounded sell offers | Funded listing reservation check,32 per principal/1024 global caps and authenticated seller-only cancel route. Non-escrow settlement retains balance recheck. Legacy oversized state and separating hot balance reads from order storage remain operational/performance concerns. |
| RSA advisory | Existing deny.toml explicit RUSTSEC-2023-0071 ignore retained with corrected signing/verification rationale. No RSA decryption/key-unwrapping oracle; dependency still present, upstream issue remains. No invented remediation or dependency removal claim. |
| SP-06.3 capability overclaim | Owning ADR narrows confinement claim to FsBackend cap-std operations; subprocesses and other storage backends are separate boundaries. |
| SP-08.9 typestate strength | Retained source boundary, including the explicitly named unchecked escape hatch; no new claim of universal cryptographic acceptance. |
| SP consumer pins/shared replay and policy law | Producer alpha19 includes nonexhaustive ReplayError::Unavailable for durable store uncertainty. Forum integration is identity-lane work; root owns publication and consumer pin adoption. Producer declaration alone is not an edge deployment receipt. |
| SR-03.1 duplicate withdrawal output reuse | Consume each exact script/value output at most once. Duplicate underpayment regression observed red then green; separate legitimate output control passes. |
| SR-03.1 snapshot bytecode tampering | Validate Keccak bytecode map commitments and non-empty account-code presence in every retained world before restore mutation. Tampered-code refusal and valid round-trip pass. Snapshot metadata and Hitch findings are not closed by this fix. |
| SR-02.4 / SR-03.3 header eviction | Shared core Error::is_header_rejection handles Bitcoin/Knots header prefixes; core and private EVM preserve queued transactions for header failures. Genuine transaction-rule eviction retained and tested. |
| OM single producer | Existing sole-publisher contract retained; no new competing compiler introduced. Source claim does not prove every consumer has adopted the published release. |
| OM certificate promotion | Certificate files and checksums now part of the same Staged generation and promotion as ontology. External/traversing/symlink certificate destinations refused early; help/docs updated. Failed promotion preserves matched old bundle/certificate. |
| OM formal proof/parity overclaim | VersionIRI excluded from semantic consequence delta; proof parity remains the named declared fragment, not proof of corpus factual truth or every classifier feature. |
| OM-07 resource identity | Explicit compiled amendment rejects old/new resource IRI change. VC producer/signer paths were assigned to VC lane and must not be inferred complete from this compiler fix. |
| OM-07 grouped proposals | Atomic multi-page application was not implemented in this lane. Parent owns Agentbox refusal/consumer integration; VC signer lane was stopped, so remaining signer work stays explicit. |
| VW-03.3 coincident nodes | Legacy quadtree max depth32; coincident51-point regression and finite force result pass. CSR controls reject NaN/infinity and invalid reheat. |
| VW-04.1 graph reload | Successful load replaces PinManager and ForceSimulation. Reload regression passes; stale graph interaction state not retained. |
| VW-05.3 npm retry | Workflow reuses immutable GitHub archive, verifies sha256 and package name/version, then independently attempts absent npm publication. Fixture accepted matching archive/rejected corruption. Version0.2.3 avoids modifying a0.2.2 release. |
| VOWL release evidence | Local WASM built Rust1.99.1 with matching wasm-bindgen0.2.128 and passed5Node tests; kept separate from exact1.99.0 CI acceptance. Root reports exact-toolchain build/inventory/tests succeeded and bundle uploaded; registry failure remains. |
| KG-04.2 retained explorer | Transfer pool counts in-flight buffers instead of wrapper object identity; generation guard retained. Actual structuredClone transfer regression plus existing tests pass. |
| VG certificate consumer | Check, review artefact and published site consume site-data/certificate from the promoted generation. Full certificate now included, subject to existing size gate. Root additionally adopted released CLI/hash and official VOWL tarball. |
| VG-04.3 source_sha pull check | VisionClaw ownership; not repaired or rerouted by this lane after the stopped VC boundary. |

## Validation

Solid: final follow-up core537/server121 pass,1ignored; Git5, MRC20 11, pay8, provenance9, phase4 7 pass. Git fixture requires `GIT_HTTP_BACKEND_PATH=$(git --exec-path)/git-http-backend` in this Nix environment; an invocation without it correctly failed500 for missing distro-default binary before the configured run passed. Canonical-charge test1 passes. Fmt, Clippy all-targets and rustdoc warnings denied pass. Full-diff Deepsec initially BLOCKed2HIGH and prompted the follow-up. High-effort revalidation run `20261103063807-cc085d7f039f86d9` returned9/9 dispositions:8Fixed/1FP/0TP/0Uncertain, covering both HIGH findings and the AMM bug. Parser-only run `20261103064638-0c46c644dd9384d5` returned5/5Fixed/0TP/0Uncertain. Combined13Fixed+1FP. The remaining two historical report entries are the documented RSA retained risk and the anchor status bug fixed in code with explicit403 native regression in test-only commit7718095. Full resolved export `/tmp/solid-revalidation-all.json`; per-run JSON, logs, original BLOCK receipt and native acceptance preserved under `workspace/.tmp/external-fixes-artifacts/solid-deepsec-final/` and its parent. Do not relabel the historical BLOCK receipt as PASS.

SideStr core/EVM native+doc suites and focused eviction/chain fixtures pass; Clippy/rustdoc warnings denied pass. Ontomark512compile+12certificate tests pass, including publication failure/identity cases; Clippy/rustdoc pass. VOWL156native pass/1ignored,5WASMNode pass; Clippy/rustdoc pass. KnowledgeGraph11transfer tests pass. Publisher17tests passed before root's later consumer acceptance checks.

Owning ADRs updated. Six focused VisionFlow successor topics contain16panels with strict declared-revision citations and Pele renders;12historical topics have dated successor pointers without broad re-stamping. Diagram commit `94b03f2` merged VisionFlow main, own diagram worktree/branch removed after preserving renders/contact sheet under `workspace/.tmp/external-fixes-artifacts/`. Every lane-owned product worktree/branch has also been removed after inspecting/preserving ignored evidence. Central indexes/freshness ledger remain root-owned. Five temporary citation symlinks `wt/{sidestr-rs,ontomark,vowl-wasm,knowledgeGraph,visionGraph}` were handed to root for its final citation run and cleanup; pre-existing Solid/forum aliases were preserved.
