---
id: OM-06
title: Machine scope, the records emitter and the ingest plan/accept contract
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-003, ADR-OM-005, ADR-OM-008, ADR-2136, ADR-2137]
sources:
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
  - ../ontomark/crates/ontomark-compile/src/build/records.rs
  - ../ontomark/crates/ontomark-compile/src/build/redact.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/build/graph_tiers.rs
  - ../ontomark/crates/ontomark-compile/src/build/turtle.rs
  - ../ontomark/crates/ontomark-compile/tests/machine_scope.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/embed.rs
  - ../ontomark/crates/ontomark-cli/tests/machine_scope_cli.rs
  - ../ontomark/crates/ontomark-wasm/src/engine.rs
  - ../ontomark/crates/ontomark-wasm/src/lib.rs
  - ../ontomark/crates/ontomark-ingest/src/lib.rs
  - ../ontomark/crates/ontomark-ingest/src/contract.rs
  - ../ontomark/crates/ontomark-ingest/src/accept.rs
  - ../ontomark/crates/ontomark-ingest/src/judge.rs
  - ../ontomark/crates/ontomark-ingest/src/error.rs
  - ../ontomark/crates/ontomark-ngg1/src/lib.rs
  - ../ontomark/docs/DEPRECATION.md
  - ../ontomark/docs/design/machine-bundle-brief.md
  - ../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md
  - ../loom/crates/loom-facade/src/mirror.rs
  - ../loom/crates/loom-vector-ruvector/src/artifact.rs
  - ../project/agentbox/docs/adr/ADR-2136-pin-the-shared-ontology-toolchain.md
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: 13af42340448022d4d43ac349a8bb5f3ae6d1d47, loom: 5ad860863b588c0cdb2290e31c69e39d82755aeb, agentbox: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc, visionflow: ff15faf83d82fef7f23d96da381c8ab78d0206b0}
---

## OM-06.1 build --scope public|machine — the surface, the refusals and where run_with forks

```mermaid
flowchart TD
    ARG["CLI --scope public or machine, default public; ScopeArg is the clap twin of build::Scope<br/>ontomark-cli/src/main.rs:195-210, ontomark-cli/src/main.rs:390-391"]
    OPT["build::Options.scope reaches run_with; Scope is the enum Public (default) and Machine<br/>ontomark-compile/src/build/mod.rs:88, ontomark-compile/src/build/generation.rs:58"]
    GUARD["ensure_machine_scope names EVERY conflict, then ScopeConflict exits 2: --publish-out, --vault working or all, redaction off, a non-knowledge root, a layout with no working vault or no working folder<br/>ontomark-compile/src/build/mod.rs:476-518, ontomark-compile/src/build/mod.rs:282-297"]
    CENSUS["knowledge AND working run the census; working is loaded once and a public flag that is not a bool refuses the build<br/>ontomark-compile/src/build/mod.rs:596-603"]
    CRED["credential gate scans working's public pages with the knowledge ones; the staged Markdown is then discarded, publish/ is never written<br/>ontomark-compile/src/build/mod.rs:633, ontomark-compile/src/build/mod.rs:766-768"]
    VALW["working validated on its own under its own role; an error refuses the build<br/>ontomark-compile/src/build/mod.rs:645-656"]
    FORK["projection forks: project_machine returns knowledge + plain records, else the public projection<br/>ontomark-compile/src/build/mod.rs:658-676"]
    KNOW["corpus = the knowledge projection exactly as a public build makes it: Turtle, reasoner, scaffold, prose index, WebVOWL, NGG1, OKF read only this<br/>ontomark-compile/src/build/mod.rs:678-689"]
    LIST["listed = knowledge records followed by plain working records; only the page API and the search index read it, and page_count counts it<br/>ontomark-compile/src/build/mod.rs:756-761, ontomark-compile/src/build/mod.rs:791"]
    BACK["scaffold backlinks stay knowledge-only (D4), page-document backlinks may include working sources<br/>ontomark-compile/src/build/mod.rs:679, ontomark-compile/src/build/mod.rs:795-800"]
    MARK["gen.scope = options.scope; the field is omitted from a public marker so public marker bytes do not change<br/>ontomark-compile/src/build/mod.rs:1002, ontomark-compile/src/build/generation.rs:118-121"]
    REP["Report.machine carries working_records and yielded; published is empty in machine scope<br/>ontomark-compile/src/build/mod.rs:190-196, ontomark-compile/src/build/mod.rs:1036-1040, ontomark-compile/src/build/mod.rs:1062"]
    ARG --> OPT --> GUARD --> CENSUS --> CRED --> VALW --> FORK
    FORK --> KNOW --> LIST --> BACK --> MARK --> REP
    SAME["INVARIANT: data/ (bar the markers) and okf/ are byte-identical across scopes, in both directions, and ontology_digest and content_digest are equal, because every ontology artefact reads KNOW and the scaffold backlinks stay knowledge-only<br/>test machine_and_public_builds_share_every_ontology_artefact ontomark-compile/tests/machine_scope.rs:143"]
    KNOW -.- SAME
    BACK -.- SAME
    DIFF["INVARIANT: what differs is the markers (scope), api/pages/working/, the search index, page_count, and the absence of publish/<br/>test machine_scope_writes_no_publish_tree_and_stamps_its_scope ontomark-compile/tests/machine_scope.rs:244"]
    MARK -.- DIFF
    CLI["CLI contract: machine stamps scope, public writes no scope field, refusals exit 2<br/>ontomark-cli/tests/machine_scope_cli.rs:52, ontomark-cli/tests/machine_scope_cli.rs:70, ontomark-cli/tests/machine_scope_cli.rs:85"]
    ARG -.- CLI
```
- DIVERGENCE: ADR-OM-005 decision 4 named the private mode `--vault all`, which is the public publish's value; the mode shipped as its own flag, so `--vault` keeps its meaning (`ontomark-cli/src/main.rs:192-193`, `DEPRECATION.md:468`).
- EXTERNAL: the design record is `docs/design/machine-bundle-brief.md:9-13`; the decisions D1-D4 are in the ADR amendment (`ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:79-88`).

## OM-06.2 project_machine — the union split, the directional twin rule and the plain records

```mermaid
flowchart TD
    PM["project_machine knowledge, working, vocab, held identities, PrivateLinks<br/>ir/projection.rs:740-747"]
    KP["knowledge side first: project_vault_with_links, untouched by working, so every ontology artefact keeps its bytes<br/>ir/projection.rs:750-751"]
    UNION["union_extra = held identities of both vaults<br/>ir/projection.rs:754-755"]
    PUBK["public_knowledge = every identity a PUBLIC knowledge record holds, as authored<br/>ir/projection.rs:763-768"]
    STRIP["strip_shared_public_identity blanks, on a PRIVATE record, each identity a public knowledge page already holds; public records pass through<br/>ir/projection.rs:253-268, ir/projection.rs:769-774"]
    CHECK1["Split::checked over knowledge + hold_back(working): a public identity also private in either vault is PublicationBlocked::Collision<br/>ir/projection.rs:534-553, ir/projection.rs:776"]
    TWIN["DIRECTIONAL TWIN RULE: public working beside PRIVATE page, either vault: refused; public working beside PUBLIC knowledge: yields to knowledge, counted; private working beside PUBLIC knowledge: harmless, yields silently<br/>ir/projection.rs:725-731"]
    CLEAN["link-clean each vault (clean_vault), rebuild the corpora; working's held pages forced private by hold_back<br/>ir/projection.rs:613-649, ir/projection.rs:654-662, ir/projection.rs:778-780"]
    YIELD["yields = public working records whose identity a cleaned public knowledge record holds<br/>ir/projection.rs:785-806"]
    LINKS["each plain record's links are pointed at the target's published slug through PrivateLinks::resolve_public: knowledge slug after disambiguation, working slug, else slug-verbatim<br/>ir/projection.rs:813-835, ontomark-compile/src/build/redact.rs:239-248"]
    UREDACT["private markers over the UNION: a private working twin keeps its own title and label as markers, so public working prose naming it is masked<br/>ir/projection.rs:837-850, ir/projection.rs:558-583"]
    SLUGC["two public working pages may not claim one working slug: Collision before staging<br/>ir/projection.rs:855-867"]
    PLAIN["plain override: page_id working/id, slug = working_slug, page_iri = page namespace + slug, has_ontology false, iri cleared, entity_type Class<br/>ir/projection.rs:869-883"]
    OUT["MachineProjection knowledge, redactor, plain, yielded; merged = knowledge then plain<br/>ir/projection.rs:691-717, ir/projection.rs:887-892"]
    PM --> KP --> UNION --> PUBK --> STRIP --> CHECK1 --> CLEAN --> YIELD --> LINKS --> UREDACT --> SLUGC --> PLAIN --> OUT
    TWIN -.- CHECK1
    TWIN -.- YIELD
    WS["working_slug = working/ + each path segment slugified, subdirectories kept; a segment slugify empties (CJK, emoji, punctuation) keeps x- + the first 4 bytes of its SHA-256, so two folders never share a file<br/>ir/projection.rs:668-671, ir/projection.rs:678-686"]
    PLAIN -.- WS
    T1["tests: a_public_working_twin_of_a_private_knowledge_page_is_refused :386, a_public_working_twin_of_a_public_knowledge_page_yields :428, a_private_working_twin_of_a_public_knowledge_page_yields_silently :666, a_non_ascii_working_folder_keeps_its_own_slug_segment :711<br/>ontomark-compile/tests/machine_scope.rs:386, ontomark-compile/tests/machine_scope.rs:428, ontomark-compile/tests/machine_scope.rs:666, ontomark-compile/tests/machine_scope.rs:711"]
    TWIN -.- T1
    WS -.- T1
    NOTE["INVARIANT: no urn page IRI under working/ reaches a public bundle; machine page IRIs carry no spaces and equal the slug path<br/>ontomark-compile/tests/machine_scope.rs:293, ontomark-compile/tests/machine_scope.rs:696"]
    PLAIN -.- NOTE
```
- INVARIANT: edges from a working record to knowledge are resolved; edges from knowledge to working are not, because knowledge records are never re-resolved (D3), so working links are one-directional in v1.
- DIVERGENCE: the twin rule was amended after the first real-corpus build found a private working twin of a public knowledge page (`Web3`); the amended text is `docs/design/machine-bundle-brief.md:44-49`.

## OM-06.3 The records emitter and Loom's promote_vault_build

```mermaid
sequenceDiagram
    autonumber
    participant CLI as ontomark-cli<br/>ontomark-cli/src/main.rs:416
    participant RUN as build::run_with<br/>ontomark-compile/src/build/mod.rs:545
    participant REC as build::records<br/>ontomark-compile/src/build/records.rs:103
    participant XIN as Xinference embedder<br/>ontomark-cli/src/embed.rs:14
    participant LOOM as promote_vault_build<br/>loom-vector-ruvector/src/artifact.rs:165
    CLI->>CLI: --with-rvdb builds Some(Xinference::new endpoint) from --embed-endpoint, the<br/>only network the build touches (ontomark-cli/src/main.rs:819-824)
    CLI->>RUN: run_with options, profile, vocab, classifier, embedder<br/>(ontomark-compile/src/build/mod.rs:545-551)
    RUN->>REC: concept_records scaffold, prose, generation_id<br/>(ontomark-compile/src/build/mod.rs:949-950)
    REC-->>RUN: ConceptRecord key urn:ngm:class:slug, id loom:ontology-corpus:key, text header,<br/>metadata.generation (ontomark-compile/src/build/records.rs:120-123)
    RUN->>REC: embed_all in batches of 96 (ontomark-compile/src/build/records.rs:40,<br/>ontomark-compile/src/build/records.rs:173)
    REC->>XIN: embed texts: one vector per text (ontomark-compile/src/build/records.rs:61-67)
    XIN-->>REC: vectors - each is width-checked to 384 and normalised, any other width is an<br/>error (ontomark-compile/src/build/records.rs:74-81,<br/>ontomark-compile/src/build/records.rs:86,<br/>ontomark-compile/src/build/records.rs:173)
    REC-->>RUN: to_jsonl, one record per line (ontomark-compile/src/build/records.rs:195)
    RUN->>RUN: staged.add data/ontology-corpus.records.jsonl, so the marker lists it with its<br/>sha256 and size (ontomark-compile/src/build/mod.rs:953)
    Note over RUN,REC: INVARIANT: the compile crate never opens a socket - the Embedder trait is the<br/>one seam, and a fake embedder tests the emitter<br/>(ontomark-compile/src/build/records.rs:14-20,<br/>ontomark-compile/src/build/records.rs:61-67)
    RUN-->>LOOM: bundle on disk - the host runs Loom's promotion as its own process<br/>(loom-vector-ruvector/src/artifact.rs:20-25)
    LOOM->>LOOM: read marker, require id and the records entry, verify the file against the<br/>listed sha256 and bytes (loom-vector-ruvector/src/artifact.rs:166-195)
    LOOM->>LOOM: every record must name the bundle generation and carry a 384-wide vector<br/>(loom-vector-ruvector/src/artifact.rs:268-290)
    LOOM->>LOOM: build ontology-corpus.rvdb + sidecar, rename into place, add both to the marker<br/>with a derived note, under .promotion-in-flight<br/>(loom-vector-ruvector/src/artifact.rs:198-202,<br/>loom-vector-ruvector/src/artifact.rs:205-260)
    Note over LOOM: INVARIANT: the marker content_digest stays the source-corpus digest, only the<br/>builder of a file describes it, so ontomark lists records.jsonl and Loom lists<br/>the rvdb pair (loom-vector-ruvector/src/artifact.rs:154-157)
```
- INVARIANT: the embedder lock is `bge-small-en-v1.5` at 384 dimensions (`ontomark-compile/src/build/records.rs:34-36`); the HNSW rebuild after a bulk load is a separate non-concurrent step and is never folded into the build.
- EXTERNAL: C3 lists the file, its `--with-rvdb` condition and the `derived` note in the marker (`sovereign-corpus-contracts.md:115-122`).

## OM-06.4 Assisted ingest — the host-driven plan/accept sequence over the wasm engine

```mermaid
sequenceDiagram
    autonumber
    participant HOST as host<br/>ontomark-wasm/src/lib.rs:73
    participant ENG as Engine<br/>ontomark-wasm/src/engine.rs:217
    participant ING as Ingest plan and accept<br/>ontomark-ingest/src/lib.rs:170
    participant MODEL as host-controlled model
    HOST->>ENG: load generation g with the vault files (ontomark-wasm/src/engine.rs:289-302)
    HOST->>ENG: ingest_plan generation g, path, options (ontomark-wasm/src/engine.rs:159-166)
    ENG->>ENG: read guard: a generation other than the engine's is answered stale and nothing runs (ontomark-wasm/src/engine.rs:445-458)
    ENG->>ING: ensure_model, then plan the note against the profile and the typed model (ontomark-wasm/src/engine.rs:733-751)
    ING->>ING: the note must declare an entity and the profile must grant assertion keys (ontomark-ingest/src/contract.rs:176-195)
    ING-->>ENG: ModelRequest contract ontomark-ingest/facts/1, note<br/>digest, KeySpec list, chunks, response schema<br/>(ontomark-ingest/src/contract.rs:196-212,<br/>ontomark-ingest/src/contract.rs:13)
    ENG-->>HOST: path, digest, request (ontomark-wasm/src/engine.rs:750)
    HOST->>MODEL: request.instructions, chunks and schema - the engine never calls a model
    MODEL-->>HOST: one ModelResponse per chunk: key, a name or literal, a<br/>verbatim evidence quote<br/>(ontomark-ingest/src/contract.rs:82-98)
    HOST->>ENG: ingest_accept generation g, path, the request UNCHANGED, responses, options (ontomark-wasm/src/engine.rs:170-181)
    ENG->>ING: accept doc, request, responses (ontomark-wasm/src/engine.rs:753-773)
    ING->>ING: refuse a foreign contract, a stale note digest, a request whose note, keys or chunks no longer equal what plan produces, an unknown chunk (ontomark-ingest/src/accept.rs:216-258)
    ING->>ING: per assertion: key in request, quote located in its chunk, name grounded and not the note itself, kind fits the predicate, quote mentions the value, not already present, not conflicting (ontomark-ingest/src/accept.rs:293-300, ontomark-ingest/src/accept.rs:330, ontomark-ingest/src/accept.rs:345, ontomark-ingest/src/accept.rs:410-418)
    ING->>ING: build ONE patch bound to the source digest, keys in profile order, and prove it applies before returning it (ontomark-ingest/src/accept.rs:467-536)
    ING-->>ENG: Candidates patch, accepted with evidence spans, withheld<br/>with a reason, diagnostics<br/>(ontomark-ingest/src/accept.rs:170-182,<br/>ontomark-ingest/src/accept.rs:537-545)
    ENG->>ENG: apply the patch once more to give the preview shape of propose_patch: base_text, new_text, new_digest, diff, edits (ontomark-wasm/src/engine.rs:774-790)
    ENG-->>HOST: path, candidates, preview, or preview null when nothing<br/>was accepted (ontomark-wasm/src/engine.rs:777-778,<br/>ontomark-wasm/src/engine.rs:793)
    HOST->>HOST: the user reviews the diff - only the host writes the file, and only if the file still hashes to the planned digest
    HOST->>ENG: upsert generation g+1 with the written text - the model and every cache are rebuilt (ontomark-wasm/src/engine.rs:304-314, ontomark-wasm/src/engine.rs:410-416)
    Note over ENG: INVARIANT: model faults are data, not errors: only a bad<br/>request is an IngestError with a stable wire code<br/>(ontomark-ingest/src/error.rs:9-58,<br/>ontomark-wasm/src/engine.rs:799-810)
    Note over ING: INVARIANT: the opt-in Jev hooks (FactJudge,<br/>CandidateSelector, FieldSuggester) are host-supplied and<br/>monotone, a judge can only withhold what accept already<br/>allowed, and NoJudge makes accept_judged equal accept<br/>(ontomark-ingest/src/judge.rs:13-26,<br/>ontomark-ingest/src/judge.rs:341)
    Note over ING: INVARIANT: starting provenance is fixed, confidence 0.5<br/>and method inferred - ingest never raises it, a reviewer<br/>does (ontomark-ingest/src/accept.rs:18-23)
```
- INVARIANT: the engine is pure; `ingest_plan` and `ingest_accept` never write and never call a model, and a mutation carries a generation strictly greater than the engine's (`ontomark-wasm/src/engine.rs:1-15`, `ontomark-wasm/src/engine.rs:429-443`).
- EXTERNAL: the instructions are versioned prompt data, `facts-v1.md`, carried inside the request (`ontomark-ingest/src/contract.rs:16`).

## OM-06.5 Open items carried by the machine scope and the ingest contract

```mermaid
flowchart TB
    subgraph NGG["NGG1 per-predicate collapse — KEEP"]
        N1["twelve relation keys all become EDGE_RELATION, and the dedup key merges two relations between one pair<br/>ontomark-compile/src/build/graph_tiers.rs:283, ontomark-compile/src/build/graph_tiers.rs:306-316"]
        N2["EDGE_RELATION is a published wire byte; per-relation edge types need an NGG2 format bump, a per-edge property IRI left undecided in ADR-OM-005<br/>ontomark-ngg1/src/lib.rs:142-144"]
        N3["the TTL keeps every relation, so only the explorer tiers lose the distinction<br/>ontomark-compile/src/build/turtle.rs:337-341"]
        N1 --> N2
        N1 -.- N3
    end
    subgraph LOOMR["Loom marker reader tolerance of scope — VERIFIED field by field"]
        L1["vault_build_generation reads named fields by key with Value::get: generated_at, commit, promoted_at, artifacts, class_count, content_digest, page_count, vocabulary_version, stale_after; scope is never read, so it is ignored<br/>loom-facade/src/mirror.rs:266-285"]
        L2["parse_marker_strict gates on id, dirty, commit, content_digest, class_count, page_count, vocabulary_version and artifacts; an unknown key is not a refusal<br/>loom-facade/src/mirror.rs:394-437"]
        L3["promote_vault_build reads id and artifacts and rewrites the marker as a Value, so scope survives the rewrite<br/>loom-vector-ruvector/src/artifact.rs:168-175, loom-vector-ruvector/src/artifact.rs:255-260"]
        L4["INVARIANT: C3 records that Loom tolerates the field, verified live 2026-10-12, and that a consumer cache keyed on ontology_digest must also key on scope<br/>sovereign-corpus-contracts.md:148-152"]
        L1 --> L2 --> L3 --> L4
    end
    subgraph ABP6["AB-P6 — DECIDED: the --scope machine flag on build; ADR-2136 now says so"]
        A1["ontomark names the producer as build --scope machine, with --vault all the public publish's value<br/>DEPRECATION.md:468"]
        A2["agentbox ADR-2136 names AB-P6 as the same released-CLI build --scope machine, no separate verb<br/>ADR-2136-pin-the-shared-ontology-toolchain.md:84"]
        A3["the only verb split ADR-2136 states is elsewhere: the released ontomark CLI versus a thin AGPL adapter for propose, apply and panel publish<br/>ADR-2136-pin-the-shared-ontology-toolchain.md:80"]
        A4["DECIDED 2026-10-12 (owner, via ADR-OM-005 decision 4): AB-P6 is `build --scope machine`, no separate verb<br/>ontomark-cli/src/main.rs:192-193"]
        A5["DRIFT: the DEPRECATION input-set bullet still names `ontomark build --vault all --private-machine-bundle` (name provisional), a flag that does not exist<br/>DEPRECATION.md:470"]
        A1 --> A4
        A2 --> A4
        A3 -.-> A4
        A5 -.->|DOC-DRIFT| A4
    end
```
- DIVERGENCE: the phrase "AB-P6 verb split" appears in no document at these revisions; the entry records what the sources do say: the scope flag (`DEPRECATION.md:468`), ADR-2136's matching `build --scope machine` wording (`ADR-2136-pin-the-shared-ontology-toolchain.md:84`) and the CLI-versus-adapter verb division (`ADR-2136-pin-the-shared-ontology-toolchain.md:80`).
- INVARIANT: of the three rows, AB-P6 is decided (the scope flag) and the Loom tolerance is verified (C3 corrected at VisionFlow HEAD); only the NGG1 collapse stays open, pending a format bump.
