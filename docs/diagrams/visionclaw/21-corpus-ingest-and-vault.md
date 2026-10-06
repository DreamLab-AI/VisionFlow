---
id: VC-21
title: Corpus ingest (CorpusSource port, ADR-2114) and the frontmatter inclusion gate
area: visionclaw
governing:
  - ../project/docs/VAULT-corpus-format.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2014, ADR-2040, ADR-2070, ADR-2112, ADR-2113, ADR-2114]
sources:
  - ../project/crates/visionclaw-ontology/src/services/jsonld_validator/mod.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/services/github/content_enhanced.rs
  - ../project/src/services/corpus_source/mod.rs
  - ../project/src/services/corpus_source/github.rs
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/page_parser.rs
  - ../project/src/services/parsers/knowledge_graph_parser.rs
  - ../project/src/handlers/admin_sync_handler.rs
  - ../project/src/app_state.rs
  - ../project/src/bin/sync_corpus.rs
  - ../project/src/bin/validate_md.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/crates/visionclaw-domain/src/vault/mod.rs
  - ../project/crates/visionclaw-domain/src/vault/link.rs
  - ../project/crates/visionclaw-domain/src/models/node.rs
  - ../project/scripts/pre-commit-validate.sh
verified_commit: {visionclaw: af3dff3f25300cf12bceda5650688ec223270eca}
---

## VC-21.1 Corpus sync entry points and the CorpusSource ingest path

<!-- 2026-09-29: rewritten for ADR-2114 - GitHubSyncService now reads through the CorpusSource port (GitHubSource / LocalDirectorySource) instead of calling the GitHub API directly, and the sync_github/sync_local binaries and local_file_sync_service.rs are gone, replaced by one sync_corpus.rs binary plus a boot task and an HTTP admin endpoint that all call the same sync_graphs(_with) -->
```mermaid
sequenceDiagram
    autonumber
    participant Admin as admin_sync_handler::trigger_sync<br/>src/handlers/admin_sync_handler.rs:66
    participant Boot as boot sync task<br/>src/app_state.rs:614
    participant Bin as sync_corpus.rs::main<br/>src/bin/sync_corpus.rs:25
    participant Svc as GitHubSyncService::sync_graphs_with<br/>src/services/github_sync_service.rs:254
    participant Src as CorpusSource::list_pages<br/>src/services/corpus_source/mod.rs:106
    participant SHA as filter_changed_files<br/>src/services/github_sync_service.rs:1928
    participant Batch as process_batch_incremental<br/>src/services/github_sync_service.rs:1454
    participant File as process_fetched_file<br/>src/services/github_sync_service.rs:1636
    participant KG as kg_repo (Oxigraph)<br/>src/services/github_sync_service.rs:1568

    Admin->>Svc: sync_graphs_with(force_full) - POST /api/admin/sync
    Boot->>Svc: sync_graphs() - spawned at server boot
    Bin->>Svc: sync_graphs() - sync_corpus binary
    Note over Bin: CORPUS_SOURCE=local or github selects LocalDirectorySource / GitHubSource - sync_corpus.rs:56-64
    Svc->>Src: list_pages()
    Note over Src: GitHubSource delegates to the Trees API with a Contents API fallback (corpus_source/github.rs:60)<br/>LocalDirectorySource walks base_paths on disk (corpus_source/local.rs:207)
    Note over Src: listing excludes /bak/ /.recycle/ /journals/ /.obsidian/ /.trash/ (ADR-2040 D6) - content_enhanced.rs:127-131 (files) :245-249 (dirs)<br/>identity decode folds legacy Namespace___Title and %2F/%2f percent-encoding to / - vault::page_name_from_path vault/mod.rs:361-368<br/>Path authority: VAULT_ROOT is the single root (agentbox vault manifest, see AB-05)
    Svc->>Svc: VaultIndex::from_identities(files) - full listing, github_sync_service.rs:297
    Note over Svc: INVARIANT: index built before the change-marker filter, or an incremental sync mints phantom stubs for unchanged pages
    alt FORCE_FULL_SYNC=1/true env, or ?force_full=true query
        Note over Svc: bypasses the SHA1 filter entirely
        Svc->>Svc: files_to_process = files.clone()
    else incremental
        Svc->>SHA: filter_changed_files(files)
        SHA-->>Svc: files whose change_marker differs from stored metadata
    end
    opt force_full_sync
        Svc->>KG: clear_graph()
    end
    loop chunks(BATCH_SIZE=50) - github_sync_service.rs:60
        Svc->>Batch: process_batch_incremental(batch, deferred_edges, scope)
        loop PARALLEL_FETCHES=8 concurrent - :1470
            Batch->>Src: fetch_page(file)
        end
        Batch->>File: process_fetched_file(file, content, scope)
        File->>File: page_parser::parse_page(content, path) - src/services/page_parser.rs:66
        alt Ok(Some(parsed)) - under knowledge/ AND frontmatter type is Class or Property or Individual
            File->>File: page_is_kg_included(content) - the same gate as the plain path, :2108
            Note over File: DRIFT: ADR-2014's "formal data ingests unconditionally, bypasses the publish gate" no longer holds post-ADR-2114 - both paths share one gate
            File->>File: build_node_from_entity + enrich_node_from_frontmatter - :134,2155
        else Ok(None) - plain vault page
            File->>File: process_plain_vault_file(file, content, scope) - :1832
            Note over File: see VC-21.2 for the inclusion gate applied here
        else Err(invalid frontmatter YAML)
            File->>File: skip (warn log, fail-closed)
        end
        Batch->>KG: batch_add_nodes(real_nodes) / batch_add_nodes_if_absent(stub_nodes) - :1568,1579
        Batch->>KG: batch_add_edges(immediate_edges) - :1602 same-batch endpoints only
    end
    Svc->>Svc: deferred_edges partition: resolvable vs dangling - :455
    Svc->>KG: batch_add_edges(resolvable)
    Note over Svc,KG: DIVERGENCE: dangling wikilinks mint no stub - they fold into wikilink_count weight plus co-citation springs (FANOUT_NODE_THRESHOLD, default 3)
    Svc->>Svc: update_file_metadata(all_files_to_process) - :1965
    Svc-->>Admin: SyncStatistics{total_files,kg_files_processed,total_nodes,total_edges,errors}
```

## VC-21.2 Frontmatter inclusion gate — one gate, both ingest paths

<!-- RESOLVED 2026-09-29: the ADR-2040 D3 legacy Logseq `key:: value` tolerance is deleted - the vault/mod.rs module doc now states such lines are body text and never the publish gate -->
```mermaid
sequenceDiagram
    autonumber
    participant Caller as process_fetched_file / process_plain_vault_file<br/>src/services/github_sync_service.rs:1636,1832
    participant Gate as page_is_kg_included<br/>src/services/github_sync_service.rs:2108
    participant Vault as vault::parse<br/>crates/visionclaw-domain/src/vault/mod.rs:278
    participant PM as PageMeta::is_kg_included<br/>crates/visionclaw-domain/src/vault/mod.rs:162

    Caller->>Gate: page_is_kg_included(content)
    Gate->>Vault: vault::parse(content) -> PageMeta{public,owl_class,...}
    Vault->>PM: is_kg_included()
    alt frontmatter public: true (real YAML boolean, not the string "true")
        PM-->>Gate: true
    else frontmatter owl-class satisfies is_class_marker (see VC-21.12)
        PM-->>Gate: true
        Note over Vault: ADR-2040 D2: owl-class in frontmatter satisfies the gate directly
    else public:: true or owl:class:: written anywhere in the body (Logseq-style)
        PM-->>Gate: false
        Note over Vault: RESOLVED - the ADR-2040 D3 bounded legacy tolerance is deleted<br/>key:: value lines are body text everywhere now (vault/mod.rs:22-25)
    else absent both public and owl-class
        PM-->>Gate: false
        Note over PM: INVARIANT: fail-closed - no frontmatter, or public absent/false and no owl-class, means private (vault/mod.rs:160-163)
    end
    alt Gate returns false
        Caller-->>Caller: skip page (debug/warn log, no node or edge emitted)
    else Gate returns true
        Caller-->>Caller: node ingested - page, linked_page or ontology-typed per VC-21.3
    end
    Note over Gate: RESOLVED ADR-2070: the VAULT readers table now cites github_sync_service.rs:2108 for this fn
    Note over Caller: DRIFT: ADR-2014's ontology bypass is superseded post-ADR-2114 - an ontology-declaring page (VC-21.1) is gated here too, not exempted
```

## VC-21.3 Node-type derivation: file to page / linked_page / ontology-typed

<!-- 2026-09-29: the JSON-LD Page/Class fence entry is gone (ADR-2112/2113); ontology classification now reads frontmatter `type: Class|Property|Individual` under knowledge/ via page_parser::parse_page. The downstream classify/population-type path (knowledge_graph_parser.rs, graph_state_actor.rs, node.rs, vault/link.rs) is unchanged and its citations still land on the same lines -->
```mermaid
flowchart TD
    F["markdown file (post source fetch)"] --> PP{"page_parser::parse_page<br/>src/services/page_parser.rs:66"}
    PP -->|Ok Some - knowledge/ path AND type is Class/Property/Individual| GATE1{"page_is_kg_included?<br/>github_sync_service.rs:2108"}
    GATE1 -->|true| PN["build_node_from_entity<br/>github_sync_service.rs:134<br/>metadata ontology_type=Class/Property/Individual"]
    GATE1 -->|false| SKIP1["page skipped - see VC-21.2"]
    PP -->|Ok None - plain vault page| KP["KnowledgeGraphParser::create_page_node<br/>knowledge_graph_parser.rs:145"]
    KP --> OC{"meta.owl_class.is_some()?<br/>vault::parse(content).owl_class"}
    OC -->|yes| ON["node_type = ontology_node<br/>owl_class_iri set<br/>knowledge_graph_parser.rs:198"]
    OC -->|no| PG["node_type = page<br/>metadata public=true"]
    KP --> WL["outbound [[wikilink]] targets"]
    WL --> VR{"VaultContext::resolve target<br/>crates/visionclaw-domain/src/vault/link.rs:215"}
    VR -->|exact identity match, target has slash| RES1["resolves to real page identity"]
    VR -->|bare target, exactly one basename match| RES2["resolves to that page's full identity"]
    VR -->|bare target, several matches| RES3["same-folder wins, else first in sorted path order"]
    VR -->|no match anywhere| STUB["linked_page stub id"]
    STUB -.->|DIVERGENCE| DROP["dropped at ingest - stub nodes never materialise, fold into wikilink_count plus co-citation weight instead"]
    PN --> CLS["GraphStateActor::classify_node<br/>src/actors/graph_state_actor.rs:242"]
    ON --> CLS
    PG --> CLS
    CLS --> POP["Node::population_type()<br/>crates/visionclaw-domain/src/models/node.rs:271"]
    POP -->|page or linked_page| KGSET["knowledge_node_ids"]
    POP -->|owl_class or ontology_node| OCSET["ontology_class_ids"]
    POP -->|owl_individual| OISET["ontology_individual_ids"]
    POP -->|owl_property| OPSET["ontology_property_ids"]
    POP -->|agent or bot| AGSET["agent_node_ids"]
    POP -->|unknown, owl_class_iri set| OCSET
    POP -->|unknown, no owl_class_iri| KGSET
```

## VC-21.6 validate_md binary (JSON-LD validator CLI)
```mermaid
sequenceDiagram
    autonumber
    participant Main as validate_md.rs::main<br/>src/bin/validate_md.rs:27
    participant V as Validator::new<br/>crates/visionclaw-ontology/src/services/jsonld_validator/mod.rs:239
    participant VF as Validator::validate_markdown_file<br/>crates/visionclaw-ontology/src/services/jsonld_validator/mod.rs:249

    Main->>Main: args = env::args().skip(1)
    alt args empty
        Main-->>Main: eprintln usage, ExitCode::from(2)
    end
    Main->>V: Validator::new() - CanonicalContext::v1 + OwlProfile::default
    alt validator init fails
        Main-->>Main: eprintln error, ExitCode::from(2)
    end
    loop for each path argument
        Main->>VF: validate_markdown_file(path)
        VF->>VF: extract_jsonld_blocks(markdown)
        alt blocks empty and contains_unfenced_jsonld(markdown)
            VF-->>Main: ValidationIssue::error(MissingCodeFenceMarker)
        end
        loop for each fenced json-ld block
            VF->>VF: serde_json::from_str + validate_jsonld_block
        end
        VF-->>Main: Vec~ValidationIssue~
        alt issues empty
            Main->>Main: print path OK
        else has issues
            loop each issue
                alt severity Error
                    Main->>Main: total_errors+=1, print_issue
                else severity Warning
                    Main->>Main: total_warnings+=1, print_issue
                end
            end
        end
    end
    Main->>Main: print_summary(total_files, total_errors, total_warnings, failed_files)
    alt total_errors > 0
        Main-->>Main: ExitCode::from(1)
    else
        Main-->>Main: ExitCode::from(0)
    end
    Note over Main: INVARIANT: exit 1 on any Error-severity issue is the contract pre-commit-validate.sh depends on (validate_md.rs:10-12)
```

## VC-21.12 `owl-class` class-marker grammar — the typed inclusion half
```mermaid
sequenceDiagram
    autonumber
    participant Y as frontmatter scalar<br/>crates/visionclaw-domain/src/vault/mod.rs:413
    participant CM as is_class_marker<br/>crates/visionclaw-domain/src/vault/mod.rs:469
    participant PM as PageMeta<br/>crates/visionclaw-domain/src/vault/mod.rs:49
    participant G as is_kg_included<br/>crates/visionclaw-domain/src/vault/mod.rs:162
    Y->>CM: candidate value for owl-class or owl:class (vault/mod.rs:413-417)
    CM->>CM: trim, reject any whitespace or control character (vault/mod.rs:469)
    alt absolute IRI beginning http:// https:// or urn:
        CM-->>PM: accepted, meta.owl_class = Some(v) (vault/mod.rs:415,501)
    else CURIE prefix:local - prefix starts with a letter then letters digits _ - . and local non-empty
        CM-->>PM: accepted, meta.owl_class = Some(v) (vault/mod.rs:415,501)
    else bare word with no colon, or a coerced YAML boolean or number
        CM-->>PM: rejected, meta.owl_class_rejected = Some(v) retained verbatim (vault/mod.rs:416,501)
    end
    PM->>G: is_kg_included()
    alt public true or owl_class is Some
        G-->>PM: true - page becomes a KG node (vault/mod.rs:162-163)
    else public false or absent and owl_class None
        G-->>PM: false - fail-closed, a rejected marker does NOT open the gate (vault/mod.rs:160-163)
    end
    Note over CM,G: INVARIANT the class marker is a policy, not any scalar that renders to a non-empty string -<br/>owl_class_rejected keeps the offending value so an author can be told why (vault/mod.rs:57-64)
    Note over CM: RESOLVED ADR-2070 the VAULT Inclusion closeout qualification records this as closed -<br/>is_class_marker (vault/mod.rs:469) enforces an absolute IRI or prefix:local CURIE grammar and rejects into<br/>owl_class_rejected (vault/mod.rs:57-64), so owl-class true and owl-class 42 shut the gate
    Note over G: DIVERGENCE that same closeout also records that an explicit public false plus a class marker<br/>remains included by policy - is_kg_included is an OR, so a valid owl-class overrides public false (vault/mod.rs:162-163)
```
