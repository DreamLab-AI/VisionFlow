---
id: LM-01
title: Workspace composition, the hexagonal ring and the retired Python reference
area: loom
governing:
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/docs/design/ADR-137-loom-rust-replatform.md
adrs: [loom:ADR-136, loom:ADR-137, loom:ADR-140, loom:ADR-141]
sources:
  - ../loom/Cargo.toml
  - ../loom/deny.toml
  - ../loom/crates/loom-domain/Cargo.toml
  - ../loom/crates/loom-domain/src/ports.rs
  - ../loom/crates/loom-scaffold/Cargo.toml
  - ../loom/crates/loom-scaffold/src/lib.rs
  - ../loom/crates/loom-graph-oxigraph/Cargo.toml
  - ../loom/crates/loom-graph-oxigraph/src/lib.rs
  - ../loom/crates/loom-vector-ruvector/Cargo.toml
  - ../loom/crates/loom-vector-ruvector/src/hnsw.rs
  - ../loom/crates/loom-embed-xinference/Cargo.toml
  - ../loom/crates/loom-embed-xinference/src/lib.rs
  - ../loom/crates/loom-backend-openai/Cargo.toml
  - ../loom/crates/loom-backend-openai/src/lib.rs
  - ../loom/crates/loom-attest-proofgate/Cargo.toml
  - ../loom/crates/loom-attest-proofgate/src/lib.rs
  - ../loom/crates/loom-mcp/Cargo.toml
  - ../loom/crates/loom-mcp/src/lib.rs
  - ../loom/crates/loom-client/Cargo.toml
  - ../loom/crates/loom-bench/Cargo.toml
  - ../loom/crates/loom-facade/Cargo.toml
  - ../loom/crates/loom-facade/src/lib.rs
  - ../loom/crates/loom-facade/src/bundle.rs
  - ../loom/crates/loom-facade/src/mirror.rs
  - ../loom/crates/loom-facade/src/mcp_host.rs
  - ../loom/crates/loom-facade/src/routes/mod.rs
  - ../loom/crates/loom-vector-ruvector/tests/recall_gate.rs
  - ../loom/docs/design/RUST-ARCHITECTURE.md
  - ../loom/tests/golden-python/meta.json
  - ../loom/crates/loom-scaffold/src/tests.rs
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-01.1 Eleven workspace members and their internal dependency edges

```mermaid
flowchart TB
    WS["workspace members, resolver 2, MSRV 1.89, unsafe_code deny<br/>../loom/Cargo.toml:3,24,30"]
    subgraph CORE["pure core"]
        DOM["loom-domain: types and ports, depends on ontomark-parse only<br/>loom-domain/Cargo.toml:12"]
    end
    subgraph ADAPT["adapters, each depends on loom-domain"]
        SCAF["loom-scaffold + ontomark-compile<br/>loom-scaffold/Cargo.toml:12,13"]
        OXI["loom-graph-oxigraph + oxigraph<br/>loom-graph-oxigraph/Cargo.toml:12,13"]
        VEC["loom-vector-ruvector + ruvector-core<br/>loom-vector-ruvector/Cargo.toml:12,14"]
        EMB["loom-embed-xinference<br/>loom-embed-xinference/Cargo.toml:12"]
        BACK["loom-backend-openai<br/>loom-backend-openai/Cargo.toml:12"]
        ATT["loom-attest-proofgate<br/>loom-attest-proofgate/Cargo.toml:12"]
        MCP["loom-mcp: JSON-RPC framing, no retrieval logic<br/>loom-mcp/Cargo.toml:12"]
    end
    subgraph ROOT["composition root"]
        FAC["loom-facade: lib + binary + six tool binaries<br/>loom-facade/Cargo.toml:11,15,61,63,64,65,66,67,68,73"]
    end
    subgraph LOOSE["no internal dependency"]
        CLI["loom-client: reqwest, serde, thiserror, tokio only<br/>loom-client/Cargo.toml:14,15"]
        BEN["loom-bench: publish false, talks HTTP to a running node<br/>loom-bench/Cargo.toml:10,12"]
    end
    WS --> CORE --> ADAPT --> ROOT --> LOOSE
    EDGE["optional adapter-to-adapter edge: pg-write pulls loom-embed-xinference<br/>loom-vector-ruvector/Cargo.toml:29,40"]
    VEC --- EDGE
    DRIFT1["DOC-DRIFT: the architecture says eight crates and that no adapter<br/>depends on another adapter<br/>../loom/docs/design/RUST-ARCHITECTURE.md:19,57"]
    EDGE --- DRIFT1
```

**What it shows.** The ring holds for the default build; the one adapter-to-adapter edge exists only under `pg-write`. `loom-client` and `loom-bench` sit outside the ring and reach a node over HTTP.

## LM-01.2 Ports and the one concrete adapter bound to each

```mermaid
flowchart TB
    subgraph PORTS["loom-domain ports, loom-domain/src/ports.rs:1,3"]
        P1["LexicalIndex<br/>loom-domain/src/ports.rs:15"]
        P8["ManifestSource<br/>loom-domain/src/ports.rs:137"]
        P2["VectorIndex<br/>loom-domain/src/ports.rs:59"]
        P3["GraphStore<br/>loom-domain/src/ports.rs:78"]
        P4["EmbeddingProvider<br/>loom-domain/src/ports.rs:86"]
        P5["ModelBackend<br/>loom-domain/src/ports.rs:100"]
        P6["GenerationStore<br/>loom-domain/src/ports.rs:117"]
        P7["AttestationLedger<br/>loom-domain/src/ports.rs:125"]
    end
    subgraph IMPLS["adapters"]
        A1["LexicalRetriever<br/>loom-scaffold/src/lib.rs:384,526"]
        A2["HnswIndex<br/>loom-vector-ruvector/src/hnsw.rs:255"]
        A3["OxigraphStore<br/>loom-graph-oxigraph/src/lib.rs:419"]
        A4["XinferenceEmbedder<br/>loom-embed-xinference/src/lib.rs:146"]
        A5["OpenAiBackend<br/>loom-backend-openai/src/lib.rs:173"]
        A6["LoadedBundle and MirrorStore<br/>loom-facade/src/bundle.rs:278<br/>loom-facade/src/mirror.rs:236"]
        A7["ChainedLedger<br/>loom-attest-proofgate/src/lib.rs:305"]
    end
    PORTS --> IMPLS
    HOST["AppState implements loom-mcp ToolHost over the same ports<br/>loom-facade/src/mcp_host.rs:93<br/>loom-mcp/src/lib.rs:66"]
    IMPLS --> HOST
    INV["INVARIANT: one retriever Arc is both the lexical index and the manifest source<br/>loom-facade/src/lib.rs:268,269,270,273,274"]
    A1 --- INV
    INV2["INVARIANT I-P1: retrieval ports return Iri, ConceptMatch, CanonicalUnit or Scaffold,<br/>never a raw engine artefact<br/>loom-domain/src/ports.rs:3,4"]
    P1 --- INV2
```

**What it shows.** Every external system is behind a trait so the router can be exercised in memory; `GenerationStore` has two implementations because the loaded identity and the disk view are kept apart (see LM-02.4).

## LM-01.3 External pins and the feature planes

```mermaid
flowchart TB
    RUV["ruvector-core by PATH from the sibling checkout, explicit feature set<br/>../loom/Cargo.toml:67"]
    PIN["fork commit pinned by script and CI, not by the lockfile<br/>../loom/Cargo.toml:58,59,60"]
    ONT["published ontomark-compile 0.9.0 and ontomark-parse 0.1.4<br/>../loom/Cargo.toml:56,57"]
    DENY["cargo-deny: yanked denied, unknown-git allowed for the path dep<br/>../loom/deny.toml:6,61"]
    RUV --- PIN
    ONT --- DENY
    subgraph FEATURES["feature planes"]
        F1["loom-facade default = attest, mounts the ledger routes<br/>loom-facade/Cargo.toml:57,58"]
        F2["proofgate attest = ruvector-core, deliberately NOT enabled by the facade<br/>loom-attest-proofgate/Cargo.toml:25<br/>loom-facade/Cargo.toml:70,71,72"]
        F3["pg-write: postgres staging bins, compiled out of serving<br/>loom-vector-ruvector/Cargo.toml:40,49,56"]
        F4["semantic-fallback: gates only the ignored recall test<br/>loom-vector-ruvector/Cargo.toml:42<br/>../loom/crates/loom-vector-ruvector/tests/recall_gate.rs:26"]
    end
    RUV --> FEATURES
    PG["ProofGate and MutationLedger are not in ruvector-core, the attest feature<br/>re-exports the sha2 ChainedLedger instead<br/>loom-attest-proofgate/src/lib.rs:26,27,45,46"]
    F2 --- PG
    TENS["TENSION: the proofgate crate calls itself build and CI time only,<br/>never on the serving hot path, yet the default facade serves its ledger<br/>loom-attest-proofgate/src/lib.rs:1,2<br/>loom-facade/src/routes/mod.rs:71,72,73"]
    F1 --- TENS
    DRIFT["DOC-DRIFT: the layout names a flake.nix Nix build, absent at this commit<br/>../loom/docs/design/RUST-ARCHITECTURE.md:27,833"]
    PIN --- DRIFT
    SEE["see LM-06.5 for how the build receipt records the path revision"]
    RUV --- SEE
```

**What it shows.** Two upstream inputs decide what the binary contains: a sibling RuVector checkout identified only by script, and two published Ontomark crates. Feature flags separate the serving binary from build-time and evaluation code.

## LM-01.4 The retired Python reference that the Rust port still names

```mermaid
flowchart TB
    MAP["deprecation map: loom_facade.py, loom_graph.py and ontology_scaffold.py<br/>ported, the proxy and pipeline dropped<br/>../loom/docs/design/RUST-ARCHITECTURE.md:763,764,765,767,768"]
    GOLD["byte-parity anchors generated by app/ontology_scaffold.py<br/>../loom/tests/golden-python/meta.json:18,19"]
    TEST["scaffold tests compare served blocks with the golden files<br/>loom-scaffold/src/tests.rs:5,229,239,249,257"]
    MAP --> GOLD --> TEST
    D1["DOC-DRIFT: the scaffold crate is documented as a 1:1 port of a file that no longer exists<br/>loom-scaffold/src/lib.rs:1,2"]
    D2["DOC-DRIFT: the graph adapter ports app/loom_graph.py, also gone<br/>loom-graph-oxigraph/src/lib.rs:3"]
    D3["DOC-DRIFT: the backend adapter cites app/loom_facade.py line numbers<br/>loom-backend-openai/src/lib.rs:4,98"]
    TEST --- D1
    TEST --- D2
    TEST --- D3
    INV["INVARIANT: Python byte-parity is the correctness bar for the serialised block<br/>loom-scaffold/src/lib.rs:3,4"]
    GOLD --- INV
    OPEN["OPEN: with the reference deleted, the golden files are the only oracle left,<br/>so a parity claim cannot be re-derived from this tree"]
    INV --- OPEN
```

**What it shows.** The parity comments are history, not live cross-references: `app/` now holds only `mirror.sh`. The frozen golden fixtures are what keep the port honest.
