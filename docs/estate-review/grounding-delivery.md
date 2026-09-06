---
title: Loom grounding and generation delivery
status: source-inspected-runtime-unverified
date: 2026-09-04
type: explanation
---

# Loom grounding and generation delivery

Loom gives the knowledge layer a useful application boundary: assemble a bounded context from a known corpus, then either serve that context directly or pass it to a replaceable model backend. Source inspection shows a deliberate separation between retrieval, injection policy, graph access, model transport and generation reporting. It also shows that availability and generation reporting have weaker guarantees than the phrase “every answer is grounded” suggests.

This is the first source pass over Loom, at local base `8cdef36bb571`. [Source hashes](evidence/knowledge-boundaries.json) record the inspected files. The initial pass did not run Rust tests. The later closeout pass below adds default-feature library and generation tests; research benchmarks and runtime deployment still need review. No new recall or performance claim is verified here.

## Composition and request path

The [workspace manifest](../../../loom/Cargo.toml) contains eight crates. Domain ports separate the scaffold logic from Oxigraph, RuVector, Xinference, the model backend, attestation and the HTTP façade. The [composition root](../../../loom/crates/loom-facade/src/lib.rs) instantiates the lexical retriever, an in-memory graph, HNSW index, embedder, backend, generation reader and injection policy.

[Retrieval fusion](../../../loom/crates/loom-facade/src/fusion.rs) first scores lexical matches. A sufficient lexical match returns through the scaffold assembler without calling the embedder. Only a below-threshold result can reach semantic fallback, and then only when enabled and ready, with matching semantic/lexical generations and a configured semantic injection threshold. HNSW results become candidates for the same assembler rather than a second answer format.

This is a strong design choice: expensive or less predictable retrieval is conditional, and the output still passes through one assembly policy. Semantic fallback defaults off in the [configuration](../../../loom/crates/loom-facade/src/config.rs), however. The existence of vector adapters should not be described as proof that every deployment is using semantic retrieval.

The [chat route](../../../loom/crates/loom-facade/src/routes/mod.rs) grounds the last user message and merges the assembled block into the model conversation. With the relevant options enabled, a non-streaming delivery-shaped request above the verbatim threshold is answered directly from the scaffold. Otherwise the backend receives the conversation. The [Profile A configuration](../../../loom/deploy/compose.profile-a.yml) enables verbatim mode and backend no-thinking behaviour, whereas the library defaults leave those switches off. A benchmark result must identify which serving path and configuration produced it.

## Availability can outlive grounding

Failure to load the lexical index produces an empty retriever rather than preventing the service from starting. Graph and semantic-adapter failures also degrade without necessarily stopping HTTP serving. On the chat path, scaffold errors are logged and the unmodified request is still delegated to the backend.

The [grounding response code](../../../loom/crates/loom-facade/src/routes/grounding.rs) exposes an explicit object describing engagement, score scale, threshold and seeds, including a no-match result. That is useful evidence for a consumer that reads it. It does not force an OpenAI-compatible client to interpret the extra fields or refuse an ungrounded answer.

Likewise, [health reporting](../../../loom/crates/loom-facade/src/routes/health.rs) sets `ok: true` while exposing index size, graph availability, semantic readiness and backend reachability separately. This is a liveness-and-diagnostics response, not an unconditional guarantee of grounding readiness.

**Assessment:** graceful degradation is defensible for general assistance, but a private-knowledge product also needs an explicit consumer policy for requests that require grounding. Model availability and evidence availability must be distinguishable at the user-facing boundary. Confirm that agent callers inspect grounding status before relying on an answer as corpus-backed.

## Generation identity is split between memory and disk

Lexical and graph adapters load their data during state construction. The [lexical retriever](../../../loom/crates/loom-scaffold/src/lib.rs) stores its parsed index and generation, while the [graph adapter](../../../loom/crates/loom-graph-oxigraph/src/lib.rs) bulk-loads `ontology.ttl` and `ontology-inferred.ttl` into memory. The inspected router has no reload endpoint.

By contrast, [MirrorStore](../../../loom/crates/loom-facade/src/mirror.rs) reads the filesystem whenever its current generation is requested. It prefers `build-manifest.json`, then `.generation.json`, then the scaffold timestamp. Manifest presence sets `verified_single_generation` true; hash verification is a separate `verify_atomicity` method. Inspection of the façade source found its definition but no invocation from state construction or request handling. That method also returns success when no manifest or no artefact hashes are available.

This permits a source-level mismatch: after files are replaced, the reported top-level disk generation can advance while already-loaded content remains unchanged. The semantic fusion path checks its loaded retriever/index generations, but that does not establish agreement with the disk-based identity reported elsewhere. A process-reload test with two distinguishable generations is still needed to quantify which response fields and routes expose the mismatch.

**Assessment:** generation identity should belong to the same immutable loaded bundle as the content. A successful disk promotion is not evidence that a serving process has switched to that bundle. Generation reporting should distinguish downloaded, validated, activated and served state.

## Mirroring is staged but not a transaction over the whole set

The [mirror script](../../../loom/app/mirror.sh) downloads to staging and checks that embedded timestamps cluster within a tolerance, defaulting to 300 seconds. It requires at least two usable stamps. `ontology.ttl` has no stamp in this scheme and is carried with the other artefacts. A fetch failure with an existing local file can be treated like an unchanged file; that prior file remains a candidate.

Promotion uses `os.replace` on each artefact in sequence, then replaces `.generation.json` last. Each replacement is atomic at the file level; readers or a crash between replacements can still encounter a partially updated set. A commit-marker protocol could make that safe if all consumers verified the marker and loaded from an immutable generation, but the inspected serving path does not establish that protocol.

The header describes a future upstream manifest preference, but the executable body inspected here contains timestamp verification and local hash recording, not a download-and-check path for an upstream generation manifest. Timestamp proximity is weaker evidence than a publisher-signed or content-addressed set, particularly for the unstamped Turtle file.

There is also a concrete directory boundary. The script writes `app/data`, while Profile A mounts the repository's `data` directory. The compose file already warns about this and requires synchronising the artefacts before startup. This is documented operational debt, not a newly discovered secret failure.

## RDF loading does not repair upstream semantics

The graph adapter restricts loading to two named Turtle files and exposes read-only queries. It does not itself classify the ontology, normalise URNs to HTTP IRIs, or reconstruct missing publication filtering. Therefore the producer issues in [knowledge production](knowledge-production.md), including different asserted/inferred identity forms, matter to this consumer. Restricting filenames is a useful boundary against accidentally loading arbitrary working-graph files; it cannot prove the allowed files contain only intended public data.

The lexical scaffold is a different projection again. Its successful delivery cannot be taken as proof that SPARQL joins work across the Turtle files. Test each consumer path against the same generation and a small set of known relations.

## Remaining evaluation

Trace the benchmark from corpus and question selection through the copy/exposure controls, model settings, serving mode and scoring before adopting the reported recall gains. Exercise a missing-index request, a mixed-generation promotion and a source-to-query IRI case through the real router. Then inspect the agent tools that consume Loom: their behaviour determines whether the façade's diagnostic honesty reaches the human or disappears behind a fluent answer.

## ADR reconciliation and build identity — 2026-09-04

All four in-repository Loom ADR candidates (135–138) now carry scoped closeout extensions, alongside RUST-ARCHITECTURE. Their historical status is retained. The workspace directly consumes `../ruvector/crates/ruvector-core` with hnsw/storage/simd/parallel features, so a release needs both checkout identities. Generation tests explicitly exercise the standalone verifier; their existence does not wire it into serving. The proposed grounding interface is now represented in successful route source, while backend non-200 and error paths require separately specified contracts.

Current validation: `cargo test --workspace --lib` passes, and all seven `exp009_generation` tests pass. The latter verify helper-level generation matching, metadata preference, tamper rejection when the verifier is called, and its no-manifest success case. They do not test atomic serving activation. [Receipt](evidence/loom-closeout-snapshot.json).

## Consumed vector configuration

The [local vector review](consumed-vector-storage.md) traces the actual RuVector dependency and reproduces acceptance of incompatible stored configurations. Semantic readiness currently proves an opened nonempty database, not cosine/model compatibility. This adds an artefact-validation prerequisite to the generation and confidence contracts.
