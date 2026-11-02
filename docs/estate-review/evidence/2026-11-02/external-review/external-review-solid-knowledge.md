# Solid, SideStr and knowledge lane — 2026-11-02

Source checks, with no new product tests or deployment. Owning ADR changes are dated scoped notes; original decisions, statuses and verification stamps are retained.

| Owner | Reviewed source | Local main after docs | ADRs updated |
|---|---|---|---|
| solid-pod-rs | 9b6143c | b24a8e1 | 2004, 2005, 2008, 2010 |
| sidestr-rs | 814b9e54 | 9ba0764f | 0002 |
| ontomark | 4b9d15d | a525902 | OM-005, OM-009 |
| visionGraph | 556a0156b | 91452aa66 | VG-002 |
| knowledgeGraph | 4ed9ac159da | a5a74c068fa | 2002 |
| vowl-wasm | 28af436 | ca1deae | 001 |

## Concern dispositions

| External concern | Disposition and current evidence |
|---|---|
| Solid library-first, LDP/WAC, crypto feature composition | Supported as architecture. Exact eight-crate census is historical; do not present a package count as a security assurance. |
| cap-std makes all symlink escapes impossible | Overstated. FsBackend stores Arc<Dir> and opens its capability root at storage/fs.rs:25-28,53-57; this guarantee does not extend automatically to Git/provenance subprocess paths or other backends. ADR-2005 qualified. |
| UncheckedEvent -> VerifiedEvent | Supported within that API (solid-pod-rs-nostr/src/typestate.rs:118 onward); explicit into_inner_unchecked escape remains. This is not proof that every ingestion caller uses typestate. |
| Only absent WAC inherits | Supported typed-resolver design; actual variants are Missing, Found, Invalid, Unavailable. Parser omissions can precede this boundary. |
| Turtle malformed/missing-type conditions disappear | Confirmed source: solid-pod-rs/src/wac/parser.rs:317,418 and condition bucket:220-222. ADR-2005 records the gap; no new exploit run. |
| Git double pod-prefix 404 | Still source-confirmed at 9b6143c: server lib.rs:4740-4755 forwards original path into service.rs:296-301 slug join. Earlier hidden-object repair is separate. ADR-2010 records both. |
| Child/history Git WAC export | Still open: server lib.rs:4785-4801 gates pod-root read only. |
| Mirror publication still unfixed | Historical if asserted about current Solid main: A/B1/B2 merged9b6143c; separate consumer adoption and B3 availability follow-ups remain. ADR-2010 qualifies shipping versus source repair. |
| Private provenance sidecar disclosure and historical mismatch | Confirmed source: server handlers/prov.rs:351-405 has no underlying-resource WAC check, reads current sidecar without requested SHA match. ADR-2004 updated. |
| Canonical payment read failure selects legacy | Confirmed source handlers/pay.rs:179-195. Any storage get error enters legacy fallback. ADR-2008 updated. |
| OIDC DPoP not wired into extract_pubkey | Prior Nov1/Nov2 audit remains the evidence; this lane did not rerun OIDC probes or claim repair. |
| NIP98 empty-body payload requirement | Prior merged repair remains; not newly executed here and consumer versions still matter. |
| SideStr optional crate isolation | Supported. Does not establish activation or consensus safety. |
| Duplicate withdrawals reuse one output | Confirmed sidestr-evm/src/state.rs:511-514 .any output matching; sum allowance at :71-78. ADR-0002 updated. |
| Future-header errors evict valid transactions | Source path persists: sidestr-core/src/state.rs:874-891 and sidestr-evm/src/rule.rs:364-373 classify candidate verdict failure as transaction eviction. ADR-0002 updated. |
| Snapshot code bytes not validated | Confirmed restore compares world root to block record only (evm/state.rs:305-336); world root commits stored code_hash, execution uses mapped bytes (world.rs:112-124,184-201). ADR-0002 updated. |
| Reserve/private origin-neutral attestations | Existing ADR-0003 boundary unchanged; no new live reserve or chain check. |
| Sole Ontomark producer, explicit machine/public projection, Staged output | Supported publication ownership/source design at compile/build/mod.rs:575,609,965-1026. No remote generation verified. |
| Lean certificate proves all OWL truth | Overstatement. Named-closure parity is conditional on certificate_out (build/mod.rs:700-711); it certifies the defined fragment, not factual truth. Historical 351421/731141/9.5s receipt is not a fresh benchmark. OM-009 qualified. |
| Certificate outside promotion transaction | Confirmed cert.write(dir) at build/mod.rs:724-727 before Staged::promote:1026. OM-009 updated. |
| versionIRI contaminates consequence delta | Confirmed comparable_closure filters only blank nodes (rdfs-certificate/lib.rs:611-622); no versionIRI exclusion. OM-009 updated. |
| Resource IRI mutation across amendment | Current apply reparses and checks introduced errors without old/new resource equality (ontomark-compile/write/apply.rs:280-286); vault propose's touched-IRI union does not enforce immutability (VisionClaw crates/vault/src/propose.rs:568 onward). OM-005 updated. |
| Grouped proposals versus single-page consumer | Remains cross-consumer contract gap; OM-005 records acceptance. Agentbox digest/create-path checks owned by parent's lane. |
| visionGraph publisher pins compiler + checker and deploys distributions | Supported pipeline source: publish.yml pins compiler v0.7.0+sha at :48-51, source/build/check/deploy steps remain. Compiler checkout is newer; this does not prove a remote publish. VG-002 updated. |
| ontology_pull ignores source_sha for up-to-date | Confirmed consumer src/services/ontology_pull.rs:312-318 build_sha+atomic only; VG-002 records equal-build/different-source acceptance case. |
| VOWL shared NGG1/WASM worker views | Supported architecture, not proof of all legacy paths. NGG1 contract unchanged. |
| Coincident-node QuadTree subdivision | Confirmed legacy src/layout/quadtree.rs:166-238 has no stop; distinct from bounded CSR tree. No new 51-node WASM test run. VOWL ADR-001 updated. |
| Non-finite CSR parameters | Common dispatch guard absent src/layout/csr_sim.rs:692-705. Per-setter behaviour needs distinct tests, not blanket infinity exploit claims. |
| Ontology reload retains pin/alpha state | load_ontology replaces metadata/graph only (bindings/mod.rs:56-76). VOWL ADR-001 updated. |
| npm retry blocked by existing GitHub asset | Confirmed release.yml:86-90 fails before npm step:165-173. VOWL ADR-001 updated. |
| vowl-wasm0.2.2 npm publication missing | Read-only npm view @dreamlab-ai/vowl-wasm@0.2.2 version dist.integrity --json returned E404 on 2026-11-02. No publication performed. |
| KG retained explorer v0.1.1 and DoubleBuffer | Confirmed package.json:20, src/workers/positionTransport.ts:76-82 identity-based ownership; visionGraph publishing-tools/WasmVOWL copy uses inFlight count. KG ADR-2002 updated. |
| Ontomark Obsidian offline host portability | No specific new defect alleged; prior shared-host contract evidence stands. Do not infer installed-plugin runtime from renderer/source success. |

Validation: all six worktree diffs passed git diff --check; only dated ADR prose changed. Clean/ignored checks passed before all six task worktrees/branches were removed. No push, release or deployment. Existing visionGraph journal files and knowledgeGraph dirty README/LICENSING/baseline files preserved. Hash-based estate claims now need scoped freshness handling because these ADR bytes changed; these notes do not claim a new whole-ADR audit.
