| ID | Repository | Severity | Status | Verified by | Summary | Queue |
|---|---|---|---|---|---|---|
| AW1-01 | solid-pod-rs | high | open | lead | The mirror is built with DefaultSsrfCheck, so it refuses the operator's loopback relay that ADR-2010 says it admits | `remediation/audit-wave1-2026-11-02.json` |
| AW1-02 | agentbox | high | open | lead | agentbox pins solid-pod-rs 76ab7aab, which predates the SP-GIT-02 publication repair | `remediation/audit-wave1-2026-11-02.json` |
| AW1-03 | solid-pod-rs | medium | open | lead | LwsCidVerifier never rejects an old or future iat | `remediation/audit-wave1-2026-11-02.json` |
| AW1-04 | solid-pod-rs | low | open | lead | Relay::store() exposes an EventStore whose put() stores events without verification | `remediation/audit-wave1-2026-11-02.json` |
| AW1-05 | solid-pod-rs | medium | open | lead | The OIDC DPoP replay cache is a plain LRU and can evict a still-valid proof | `remediation/audit-wave1-2026-11-02.json` |
| AW1-06 | solid-pod-rs | low | open | lead | IdP discovery advertises grant types and endpoints the server does not serve | `remediation/audit-wave1-2026-11-02.json` |
| AW1-07 | solid-pod-rs | low | open | lead | The 5xx error logger is the innermost middleware, not the outermost its comment claims | `remediation/audit-wave1-2026-11-02.json` |
| AW1-08 | solid-pod-rs | low | open | lead | ADR-2011's three follow-ons are all still open | `remediation/audit-wave1-2026-11-02.json` |
| AW1-09 | VisionClaw | high | open | lead | The /ontology/derived write door checks power_user only; ADR-135 D7 requires a signed distiller envelope and an allow-list | `remediation/audit-wave1-2026-11-02.json` |
| AW1-10 | VisionClaw | medium | open | lead | ADR-135 D2.3 retires the CLEAR+INSERT corpus reload, which VisionClaw ADR-2017 re-affirms and still runs | `remediation/audit-wave1-2026-11-02.json` |
| AW1-11 | loom | medium | open | lead | The Rust reload port carries a bundle's declared content digest but never compares it, contrary to ADR-141 decision 6 | `remediation/audit-wave1-2026-11-02.json` |
| W2-01 | - | high (data loss on a live path, availability of the agent-activity feature) | open | worker | agentbox forwards bare agent events; VisionClaw ingest drops every frame as Malformed | `remediation/audit-wave2-2026-11-02.json` |
| W2-02 | - | medium (a consumer pinning main would believe the change shipped) | open | worker | forum ADR-2027 describes an inbox-key relay change that is not on main | `remediation/audit-wave2-2026-11-02.json` |
| W2-03 | - | low (record drift; the configuration is stronger than the record, not weaker) | open | worker | website ADR-2007 says one zone is encrypted; configuration encrypts three | `remediation/audit-wave2-2026-11-02.json` |
| W2-04 | - | housekeeping | closed | worker | four citation-allowlist entries were stale after the VF-02 corrections | `remediation/audit-wave2-2026-11-02.json` |
| W3-01 | agentbox | high | open | lead | management-api's RelayConsumer admits every signed author when its allowlist is empty, and the shipped flake never sets AGENTBOX_RELAY_ALLOWED_PUBKEYS, so by default it dispatches governance, intent and payment events fr | `remediation/audit-wave3-2026-11-02.json` |
| W3-02 | VisionClaw | medium | open | lead | Saved API keys can never be read back. POST /api/auth/nostr/api-keys writes through NostrService; GET reads ProtectedSettingsActor, which starts empty and is never sent a write message by anything in src. A user saving P | `remediation/audit-wave3-2026-11-02.json` |
| W3-03 | agentbox | medium | open | lead | Panel AB-28.9 drew agentbox's services/dream-engine and crates/headroom-napi under MIT/Apache; both declare AGPL-3.0-only. The panel is corrected. The separate dream-engine repository's crates are MIT OR Apache-2.0, so t | `remediation/audit-wave3-2026-11-02.json` |
| W3-04 | agentbox | low | open | lead | The licensing document says scripts/ci/check-crate-licensing.sh enforces the rule that sidestr crates come from crates.io only. The script checks only each crate's own package fields and licence files and never inspects  | `remediation/audit-wave3-2026-11-02.json` |
| W3-05 | agentbox | medium | open | worker | The NIP-98 door is not fully fail-closed: if the boot step writing the proxy config fails (it ends in `// true`) the allowlist is empty and any validly signed key is accepted; if the verifier cannot load, a valid session | `remediation/audit-wave3-2026-11-02.json` |
| W3-06 | agentbox | medium | open | worker | Boot now creates a devuser-owned /run/secrets/ruvector-pg/ inside the role-only secrets mount in both modes, outside the role plan; ADR-2122 records that the activation rehearsal was not re-run with it present. | `remediation/audit-wave3-2026-11-02.json` |
| W3-07 | agentbox | low | open | worker | k_spend is minted as an independent key, not derived from k_id; INGRESS-identity.md and ADR-2101 still describe the derivation. | `remediation/audit-wave3-2026-11-02.json` |
| W3-08 | agentbox | low | open | worker | ADR-2007's review trigger has fired: router.sh is a third harness wrapper with no profile pin, absent from verified_paths, and a missing wrapper silently launches plain `claude` instead of failing. | `remediation/audit-wave3-2026-11-02.json` |
| W3-09 | VisionClaw | low | open | worker | A layout reset is undone by the next physics change, because the reset updates SQLite and the GPU but not the settings actor; and when the GPU refuses a physics update the actor already holds the new value while SQLite d | `remediation/audit-wave3-2026-11-02.json` |
| W3-10 | VisionClaw | low | open | worker | The ADR-2049 production Dockerfile stage pipes the build into `tail` without pipefail, so a dependency that stops compiling cannot fail the stage; NVCC_CCBIN and CUDAHOSTCXX are missing from the build-input environment l | `remediation/audit-wave3-2026-11-02.json` |
| W3-11 | VisionClaw | low | open | worker | docs/BASELINE-architecture.md still routes positions through PhysicsOrchestratorActor; a types.rs comment calls the visibility filter off by default when it is on; an exported event-coordination function has no callers. | `remediation/audit-wave3-2026-11-02.json` |
| SP-ACL-REVIEW-01 | solid-pod-rs | HIGH | open-pre-existing | worker | [HIGH] Git clone exposes private child resources with only pod-root Read | `remediation/solid-acl-control-2026-11-01.json` |
| SP-ACL-REVIEW-02 | solid-pod-rs | HIGH_BUG | open-pre-existing | worker | [HIGH_BUG] Web-Ledger fee debited before the request is validated or served | `remediation/solid-acl-control-2026-11-01.json` |
| SP-ACL-REVIEW-03 | solid-pod-rs | MEDIUM | open-pre-existing | worker | [MEDIUM] Proxy forwards X-Upstream-Authorization across cross-origin and HTTPS->HTTP redirects | `remediation/solid-acl-control-2026-11-01.json` |
| SP-ACL-REVIEW-04 | solid-pod-rs | MEDIUM | open-pre-existing | worker | [MEDIUM] /.well-known/apps publishes per-pod manifest.json bypassing WAC | `remediation/solid-acl-control-2026-11-01.json` |
| SP-ACL-REVIEW-05 | solid-pod-rs | BUG | open-pre-existing | worker | SPARQL DELETE/INSERT ignores WHERE and skips variable templates | `remediation/solid-acl-control-2026-11-01.json` |
| SP-GIT-01 | solid-pod-rs | - | runtime-reproduced-open | worker | Pod Git applies the pod component twice and authorises only the root | `remediation/solid-git-export-2026-11-01.json` |
| SP-GIT-02 | solid-pod-rs | - | publication-isolation-repaired-local-main | worker | Git CGI permits known hidden-tip requests despite mirror ref hiding | `remediation/solid-git-export-2026-11-01.json` |
| SP-GIT-03 | solid-pod-rs | MEDIUM | open-pre-existing | worker | [MEDIUM] git-http-backend stderr and absolute server paths leak to remote client on backend errors | `remediation/solid-git-export-2026-11-01.json` |
| SP-GIT-04 | solid-pod-rs | BUG | open-pre-existing | worker | [BUG] GitResponse::error builds JSON by unescaped string interpolation | `remediation/solid-git-export-2026-11-01.json` |
| SP-GIT-05 | solid-pod-rs | BUG | open-pre-existing | worker | [BUG] Substring-based is_write misclassifies reads of repos named like a service, and can auto-init on GET | `remediation/solid-git-export-2026-11-01.json` |
| SP-GIT-06 | solid-pod-rs | BUG | fixed-documentation | worker | [BUG] with_auth doc claims anonymous push without a provider, but the code fails closed | `remediation/solid-git-export-2026-11-01.json` |
| SP-ID-01 | solid-pod-rs | - | producer-fixed-adoption-open | worker |  | `remediation/solid-identity-2026-11-01.json` |
| SP-ID-02 | solid-pod-rs | - | open | worker |  | `remediation/solid-identity-2026-11-01.json` |
| SP-ID-03 | solid-pod-rs | - | open | worker |  | `remediation/solid-identity-2026-11-01.json` |
| SP-ID-04 | solid-pod-rs | - | open | worker |  | `remediation/solid-identity-2026-11-01.json` |
| SP-OIDC-01 | solid-pod-rs | - | producer-fixed-adoption-open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-02 | solid-pod-rs | - | open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-03 | solid-pod-rs | - | open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-04 | solid-pod-rs | - | open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-05 | solid-pod-rs | - | open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-06 | solid-pod-rs | - | open | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| SP-OIDC-07 | solid-pod-rs | - | documentation-corrected | worker |  | `remediation/solid-oidc-2026-11-01.json` |
| ? | solid-pod-rs | MEDIUM | open-pre-existing | worker | ssrf | `remediation/solid-oidc-2026-11-01.json` |
| ? | solid-pod-rs | MEDIUM | open-pre-existing | worker | expensive-api-abuse | `remediation/solid-oidc-2026-11-01.json` |
| ? | solid-pod-rs | MEDIUM | open-pre-existing | worker | other-cache-exhaustion | `remediation/solid-oidc-2026-11-01.json` |
| ? | solid-pod-rs | MEDIUM | open-pre-existing | worker | other-info-disclosure | `remediation/solid-oidc-2026-11-01.json` |
| ? | solid-pod-rs | BUG | open-pre-existing | worker | other-logic-bug | `remediation/solid-oidc-2026-11-01.json` |
| SP-PAR-265 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-266 | solid-pod-rs | - | producer-repair-partial: malformed and target/condition guard regressions pass; HTTP status parity and parser completeness remain open | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-267 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-268 | solid-pod-rs | - | producer-repair-partial: nine production NIP-98 route tests added; full upstream fixture parity remains open | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-269 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-270 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-271 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-272 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-273 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-274 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-275 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-276 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-277 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-278 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-279 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-280 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-281 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-282 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-283 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-284 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-285 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-286 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-287 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-288 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-289 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-290 | solid-pod-rs | - | producer-repair-partial: discovery now shares enforced sibling mapping; existing inner ACL migration and provisioning remain open | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-291 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-292 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-293 | solid-pod-rs | - | open-triage | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-294 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-295 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-296 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-297 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-298 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| SP-PAR-299 | solid-pod-rs | - | retain-contract-classification; not automatically a defect | worker |  | `remediation/solid-parity-intake-2026-11-01.json` |
| COMPARE-F1 | - | medium | open | worker | Two accepted ADRs name different sole producers of the sovereign corpus, and both producers run at the pins. ADR-2013 Decision 3 makes VisionClaw crates/vault the only parser, reasoner and build. ADR-2014's estate alignm | `audit-wave4/w4-compare.json` |
| COMPARE-F2 | - | low | open | worker | parallel-remediation.md:34 asks for VCS-10 to be revalidated and /pay to be routed through the established verifier. At the VisionClaw pin it is: extract_caller_pubkey calls validate_nip98_token_bound with body binding ( | `audit-wave4/w4-compare.json` |
| COMPARE-F3 | - | medium | open | worker | Almost no external comparison in the corpus was executed. The reports read alternatives from documentation and state that nothing was installed or benchmarked. The exceptions are three. The Open Ontologies oo-cert v2.0.1 | `audit-wave4/w4-compare.json` |
| COMPARE-F4 | - | low | open | worker | prose-sanitiser and diagram-ir have no primary-source comparison of any kind. dream-machine has only its own ADR Alternatives sections. Loom, codescape runtime and the agentbox platform have partial coverage. | `audit-wave4/w4-compare.json` |
| COMPARE-F5 | - | low | open | worker | Most external comparators are cited by a moving URL (latest, main or master) with a consultation date. A minority pin a commit or version. Commit pins: NIPs 400d157 (forum-spaces-and-replication.md:112), Teller 7c00cea ( | `audit-wave4/w4-compare.json` |
| COMPARE-F6 | - | medium | open | worker | This re-confirms DE-F2 at the pin; it is not a new defect. enable_automerge receives no evaluated SHA. It runs `gh pr ready` and then `gh pr merge --auto --squash` without `--match-head-commit`, the option dream-forge-an | `audit-wave4/w4-compare.json` |
| CONTRACTS-F1 | - | high | open | worker | The live agent-event forwarder never authenticates, so a release VisionClaw refuses the socket before W2-01 is reached | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F2 | - | high | open | worker | W2-01 still holds at the pins, the rejection is silent, and agentbox's own test asserts the defective shape | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F3 | - | high | open | worker | GET /v1/tasks strips three fields VisionClaw requires, so list_tasks always fails | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F4 | - | medium | open | worker | GET /v1/tasks/:taskId strips taskDir and logFile, which TaskStatus requires | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F6 | - | medium | open | worker | Kind 31402 priority is a label on every producer and an integer on the relay | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F7 | - | medium | open | worker | The agentbox governance-bridge writes 31402 events without the tags the relay projects | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F9 | - | medium | open | worker | visionclaw-wire says every producer encoder is tested against it, but the VisionClaw server does not depend on it | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F11 | - | medium | open | worker | Cross-repository path dependencies build against whatever sibling checkout is present | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F12 | - | medium | open | worker | Loom reads ontology scaffold bundles with ontomark-compile 0.1.1 while the corpus is built by ontomark 0.7.0 | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F20 | - | medium | open | worker | Only one contract has a fixture shared by both sides; the rest are tested against hand-written copies | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F5 | - | low | open | worker | GET /v1/status strips pid, which ApiStatus requires | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F8 | - | low | open | worker | agentbox signs the NIP-98 u tag without the query string; the solid-pod-rs server verifies against the full URI | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F10 | - | low | open | worker | Agent-action envelope: failure_mode and authority_class are dropped by VisionClaw, and intent is never emitted | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F13 | - | low | open | worker | Estate crate consumers lag their producers | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F14 | - | low | open | worker | The website builds two nostr-bbs-core versions, and its kit pin is 22 commits behind the forum | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F15 | - | low | open | worker | The standalone WasmVOWL repository still pins the vowl-wasm v0.1.1 bundle | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F16 | - | low | open | worker | VisionClaw's submodules lag the audit pins: agentbox by 51 commits, codescape by 6 | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F17 | - | low | open | worker | The agentbox protocol registry still describes the pre-ADR-2061 kind-map asymmetry | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F18 | - | low | open | worker | The legacy :9500 MCP-TCP poll has no producer in any pinned repository | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F19 | - | low | open | worker | The strict-nip98 auto-elevation reads an env name the image never sets | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F21 | - | low | open | worker | The rationale gate's outcome set differs between the relay and the VisionClaw host | `audit-wave4/w4-contracts.json` |
| CONTRACTS-F22 | - | low | open | worker | Authority-denial agent events carry string ids where the canonical envelope needs integers | `audit-wave4/w4-contracts.json` |
| COV-A-F1 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F2 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F3 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F4 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F5 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F6 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F7 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F8 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F9 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F10 | - | low | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-A-F11 | - | medium | open | worker |  | `audit-wave4/w4-cov-a.json` |
| COV-B-F1 | - | medium | open | worker | the SPARQL write-verb clamp refuses ordinary label searches | `audit-wave4/w4-cov-b.json` |
| COV-B-F2 | - | medium | open | worker | a non-ASCII Xinference error body can abort the façade | `audit-wave4/w4-cov-b.json` |
| COV-B-F3 | - | medium | open | worker | the model engine listens on every host interface with no key | `audit-wave4/w4-cov-b.json` |
| COV-B-F4 | - | low | open | worker | loom-client can report a stale transient error instead of `Truncated` | `audit-wave4/w4-cov-b.json` |
| COV-B-F5 | - | low | open | worker | the attestation chain detects corruption, not a capable tamperer | `audit-wave4/w4-cov-b.json` |
| COV-B-F6 | - | low | open | worker | a crash between append and head update reads as tamper | `audit-wave4/w4-cov-b.json` |
| COV-B-F7 | - | low | open | worker | proofgate still calls itself build/CI-time only | `audit-wave4/w4-cov-b.json` |
| COV-B-F8 | - | low | open | worker | the corpus and the model Dockerfile still name Qwen3.8-27B | `audit-wave4/w4-cov-b.json` |
| COV-B-F9 | - | low | open | worker | the token floor is implemented twice | `audit-wave4/w4-cov-b.json` |
| COV-B-F10 | - | low | open | worker | stale default paths | `audit-wave4/w4-cov-b.json` |
| COV-B-F11 | - | medium | open | worker | loom has no topic set; the adapter ring and model engine are undrawn | `audit-wave4/w4-cov-b.json` |
| W4COVC-F1 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F2 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F3 | - | medium | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F4 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F5 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F6 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F7 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| W4COVC-F8 | - | low | open | worker |  | `audit-wave4/w4-cov-c.json` |
| CD-F1 | - | medium | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F2 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F3 | - | medium | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F4 | - | medium | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F5 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F6 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F7 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F8 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F9 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F10 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F11 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| CD-F12 | - | low | open | worker |  | `audit-wave4/w4-cov-d.json` |
| COV-E-F1 | - | low | open | worker | The module header says nights are stored as vectors in an RVF container for HNSW recall when RuVector is installed. In fact `openMemory` always builds `FlatMemory`, and with `backend: 'ruvector-rvf'` it overwrites the la | `audit-wave4/w4-cov-e.json` |
| COV-E-F2 | - | low | open | worker | `load` turns any read or JSON parse failure into an empty list (:81-83). The next `remember` then persists that list over the file (:99-100 → :91). | `audit-wave4/w4-cov-e.json` |
| COV-E-F3 | - | low | open | worker | `visionGraph/transcripts/weekly_ingest.py` presents itself as the weekly cron job, but the live jobs run agentbox's Rust port. The two have diverged: Python keeps confidence ≥ 0.7 flat (:486), while Rust uses tier thresh | `audit-wave4/w4-cov-e.json` |
| COV-E-F4 | - | low | open | worker | Both ingest paths name their output 'verified' (Python docstring: 'verifies them via Perplexity'). Phase 3 only filters by the model's self-reported confidence, and no external check runs in cron mode. | `audit-wave4/w4-cov-e.json` |
| COV-E-F5 | - | low | open | worker | The legacy-toolchain migration guard is not run by CI, the release workflow or the nightly evaluator. It scans sibling checkouts on disk, and its `legacy-crate` condition (`project/crates/vault-core` present, :146-147) h | `audit-wave4/w4-cov-e.json` |
| COV-E-F6 | - | low | open | worker | After a grant is sent, a failed ledger save is only logged (:194-196), and the save neither fsyncs nor checks the rename beyond `?` (:70-74). The in-memory ledger keeps enforcing limits until the process restarts. | `audit-wave4/w4-cov-e.json` |
| COV-E-F7 | - | low | open | worker | site-facts claims to be the one place pages take figures from. Its guard test checks only the specialist count and retired figures, not prices or accommodation. '£8M+' is typed by hand in three places, and the print flye | `audit-wave4/w4-cov-e.json` |
| COV-E-F8 | - | low | open | worker | `openNote` awaits `api.index` without a try/catch, and both callers discard the promise with `void` (:57, :64). | `audit-wave4/w4-cov-e.json` |
| COV-E-F9 | - | low | open | worker | The header says outputs land in `src/generated/`, but the default output directory is `target/plugin-engine` (`ONTOMARK_ENGINE_OUT`, :11). | `audit-wave4/w4-cov-e.json` |
| SIDESTR-F1 | - | high | open | worker |  | `audit-wave4/w4-sidestr.json` |
| SIDESTR-F2 | - | medium | open | worker |  | `audit-wave4/w4-sidestr.json` |
| SIDESTR-F3 | - | medium | open | worker |  | `audit-wave4/w4-sidestr.json` |
| SIDESTR-F4 | - | low | open | worker |  | `audit-wave4/w4-sidestr.json` |
| SIDESTR-F5 | - | low | open | worker |  | `audit-wave4/w4-sidestr.json` |
| SIDESTR-F6 | - | low | open | worker |  | `audit-wave4/w4-sidestr.json` |

- `parallel-implementations-2026-11-01.json`: 134 findings in a static source report; not recounted here.

Total indexed findings: 173
