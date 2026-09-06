---
title: VisionClaw graph ingestion, persistence and recovery
status: source-and-backup-probe-verified
date: 2026-09-04
type: explanation
---

# VisionClaw graph ingestion, persistence and recovery

VisionClaw is where authored knowledge becomes a queryable and rendered runtime graph. Its embedded storage design removes a networked database dependency and makes ownership of state easier to locate. That simplification is real, but sharing a store does not make the ingestion, reasoning, decision and backup lifecycle one transaction.

This chapter follows persistence setup, full-sync reconstruction, derived writeback and provenance, plus an isolated execution of the backup script. It does not yet establish complete reasoner correctness, GPU/rendering behaviour, all ingestion formats, the proposal transaction or deployed disaster recovery. [Source receipts](evidence/visionclaw-data-snapshot.json) pin the inspected files. No production database was opened or changed.

## Storage ownership is explicit

[AppState](../../../project/src/app_state.rs) opens an Oxigraph ontology repository below `DATA_DIR`, obtains its store handle and constructs the graph repository from that same handle. Settings, enrichment, liveness and KPI repositories use separate SQLite files. This supports the operative persistence decision: the two graph repositories share the embedded store rather than independently opening its RocksDB directory.

The arrangement avoids a second networked graph database, but each SQLite file and each graph operation has its own commit boundary. A consistent cross-store restore requires more than copying those files individually. The implementation also has actor-held graph state and post-sync reload signalling; served state must be checked against persisted state rather than assumed identical.

The adapter's manifest pins the external Whelk repository to a specific revision. Its dependency must be included in the build/evidence manifest even though no sibling `whelk` checkout was found at the workspace root. Reasoner correctness remains an explicit investigation item, not something inferred from the presence of its dependency.

## Full sync is destructive reconstruction, not generation activation

[GitHubSyncService](../../../project/src/services/github_sync_service.rs) lists files and builds a vault index. Incremental operation filters changed files; full sync processes the complete listed set. Before processing full-sync batches, it calls `kg_repo.clear_graph`. The current [graph repository](../../../project/src/adapters/oxigraph_graph_repository.rs) drops both the knowledge and agent named graphs in separate updates, then performs best-effort orphan bridge cleanup.

Batch fetching and parsing can fail after that clear. Failures are recorded in sync statistics while other files continue. Later, full sync rebuilds the asserted ontology from the graph then available in storage. Thus a later single atomic asserted-graph update does not make the whole fetch/clear/process/reason/reload sequence atomic. Source inspection identifies a possible partial reconstructed corpus and temporary mismatch between graph families; this pass did not induce failure in a live sync.

The asserted rebuild is skipped during incremental sync, so its current content is not automatically synonymous with the newest changed-page projection. Full rebuild uses one `CLEAR + INSERT DATA` update for the asserted graph. Its source comments explicitly acknowledge that this removes runtime classes/axioms added through governed write paths unless those changes have reached the authoritative corpus. Provenance and inferred graphs are separate; preserving them does not preserve the removed asserted semantics.

**Closeout:** define authoritative ownership for each graph population, protect agent data from unintended corpus resets, stage/validate a replacement generation before activation, and bind serving/reasoning to the activated generation. Test fetch failure, invalid pages, interrupted batches, governed runtime additions and restart during full sync. Explicitly choose when old or new inferred results may be served.

## Derived writes have a useful repository fence

[append_derived_quads](../../../project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs) rejects asserted and inferred graph targets, rejects other non-allowlisted graphs, checks subject/predicate/object IRI safety, and accumulates permitted summary/observed statements into one update. Because validation precedes the update, a later forbidden quad does not cause the earlier validated subset to be submitted by this method.

That is a substantive boundary and belongs in the roadmap as a capability to preserve. Its scope is this repository method, not every writer holding the shared store. The stronger ADR wording that asserted content is written only by sync needs qualification: the sync source itself identifies runtime governed writers to that graph.

The next proof should exercise mixed allowed/forbidden batches, unsafe IRIs, handler bypass through direct repository calls, and rollback on update failure. This chapter inspected the implementation; it did not execute those Rust adapter tests.

## Insert-only provenance can still be incomplete

The [provenance emitter](../../../project/crates/visionclaw-adapters/src/provenance_emitter.rs) makes individual `store.insert` calls. It inserts an activity type before validating the agent IRI, then adds association, agent type, time, action and optional entity/source links. A later invalid IRI or storage failure can return an error after earlier inserts. No encompassing transaction is used in this function.

This is a source-established partial-write possibility, not a reproduced production event. It contradicts an unconditional assertion that the complete Entity/Activity/Agent structure always exists. Insert-only programming also does not by itself provide a cryptographic proof against a different privileged writer changing the store. Keep separate claims for append-only emitter behaviour, atomic record completeness, retention, access control and tamper detection.

**Closeout:** validate all terms before mutation and write each provenance record atomically, or expose and repair incomplete records. Bind decision/mutation receipts to the completed record. Retention and erasure need an explicit policy that also covers replicas, external memory and backups; this review makes no legal-compliance claim.

## The backup primitive works; coverage and destination need enforcement

The [actual backup script](../../../project/scripts/backup-sqlite.sh) uses SQLite's online backup API and checks the resulting database. A [temporary-data probe](evidence/visionclaw-backup-probe.py) held an open WAL connection, committed a row and invoked host-mode backup. The restored copy contained that row and passed `integrity_check`. This verifies a useful local backup primitive.

The same [receipt](evidence/visionclaw-backup-probe.json) shows two limits: requesting `settings.sqlite3 missing.sqlite3` copied one database and exited zero; with default configuration, the destination was `./data/backups`, inside the source data directory. The script records the actual count and logs missing files, but success does not require the complete requested set. Its off-volume property is therefore configuration-dependent, especially in host mode. A path outside a container also does not by itself establish a separate failure domain.

SQLite files are backed up sequentially. Their individual consistency does not create a common checkpoint with each other or Oxigraph. Recovery of the asserted graph from upstream also leaves questions about runtime assertions, derived graphs and provenance. These are already reasons for partial recovery status; the ADR should not describe complete off-volume coverage as an unconditional implemented fact.

**Closeout:** define required versus optional databases, reject missing required members, enforce or attest the destination failure domain, record revision/checkpoint identities, and restore into a separate environment. Verify completeness, ownership and application-level behaviour after restore, not just database structural integrity. No production backup or restore was run here.
