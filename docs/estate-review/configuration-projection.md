---
title: Configuration projection and runtime truth
status: source-and-isolated-probe-verified
date: 2026-09-04
type: explanation
---

# Configuration projection and runtime truth

The manifest describes desired configuration, the Nix build determines available software, boot projection writes harness configuration, and running processes use the state they loaded. These are distinct observations. The system view's live/boot/rebuild labels help explain the transition, but a manifest-enabled flag alone cannot establish that a process is using the new configuration.

The [projector probe](evidence/mcp-projection-probes.py) executes the real script against temporary files with a fresh environment. [Results and source hashes](evidence/mcp-projection-probes.json) establish the cases below; no actual MCP configuration or server was changed.

## Managed reconciliation has a bounded ownership set

The [projector](../../../project/agentbox/scripts/project-mcp-servers.mjs) loops over the current registry and handles only entries marked projector-managed. It checks the gate and requirements, strips its annotations and expands environment references into the target. Bespoke/reference entries are left untouched.

| Fixture | Exit | Target result |
|---|---|---|
| Current managed entry, gate on, empty requirements array | 0 | Updated to the projected command |
| Current managed entry, gate off | 0 | Removed |
| Previously managed entry absent from new registry | 0 | Old target entry retained |
| Current managed entry without requirements array | 0 | Removed |
| Malformed source registry, gate environment off | 0 | Old target entry retained |

All five fixtures preserve a bespoke target entry. This supports the current-name ownership boundary, but not complete removal of stale managed entries. Since projected entries lose their management annotations and the loop only visits current source names, deleting a source definition leaves no record that the projector owns the old target entry. Removal requires retained ownership history or an explicit migration/tombstone, not deletion of arbitrary user-owned entries.

The missing-requirements case has a separate type mismatch: requiresMet returns boolean true for a non-array, while the caller reads req.ok. The current registry's nine managed entries all have arrays, so this is a schema-edge reproduction rather than an observed current server omission. Malformed registry or target input is a no-op by design.

## Boot success is not a configuration receipt

The source catches target write errors and still exits zero. It writes the target directly rather than staging and renaming it; interruption/concurrent writers require separate testing. Counters describe planned changes and do not independently verify persisted output. These are source findings; no disk-error or interruption probe ran.

The [entrypoint](../../../project/agentbox/config/entrypoint-unified.sh) runs the projector after bespoke setup, conditional on its inputs and executable being present. Missing inputs can therefore skip projection without preventing boot. Its consultant-model projection gives a nonempty environment override priority over the manifest. That is an intentional configuration precedence, not evidence of the model argument held by an already-running session.

[system-manifest.js](../../../project/agentbox/management-api/lib/system-manifest.js) builds a view from the supplied parsed manifest and adapter registry. The catalogue declares apply classes. Reading desired configuration is useful, but cannot make the built image, projected target or loaded process change retroactively. Reproducibility also requires pinned sources/dependencies, build inputs and feature selections in addition to TOML.

## Closeout requirements

CP-01/04/08 require receipts connecting desired manifest digest, built image identity, projected target digest and loaded process configuration. Report skipped, failed, persisted and active states distinctly. Define ownership across registry deletion/rename and test those transitions while preserving bespoke entries. Validate registry schema before mutation and make requirements semantics consistent.

Exercise missing/malformed source and target, write failure, interruption, idempotent rerun and boot/rebuild precedence. Verify atomic replacement and reader reload without exposing expanded secret values in receipts. ADR-2008 becomes partial for the broader stale-entry guarantee. ADR-2003 retains the composition decision with stronger evidence requirements; ADR-2031 retains its staged model-projection status and historical tariff claims are not re-certified by this source review.

## Service package and release metadata

Agentbox ADR-2030's declared licence split is reflected in the eight current local service manifests: seven MIT OR Apache-2.0, one AGPL-3.0-only. Its promised adjacent licence texts and per-crate README statements are absent from all eight package directories. The [collector](evidence/service-package-inventory.py) and [receipt](evidence/service-package-inventory.json) inventory tracked/nonignored manifests and actual adjacent files; they do not infer package contents from manifest labels.

The ecosystem documentation separately records prose-sanitiser and diagram-ir extraction and Nix consumption. Local source directories, extracted repositories, package archives and deployed derivations are separate release identities. The older ten-crate account therefore needs reconciliation rather than being repeated as a current count. This pass does not look up registries or certify the extracted releases.

CP-01/08 requires an accountable package owner, source/dependency revisions, declared notices/texts, inspected archive contents and release digest. Review distribution-specific licensing through the designated maintainer process; this is a local metadata consistency finding, not a legal compatibility assessment. ADR-2030 becomes partial for its concrete packaging commitments. Its broad services-wide verification baseline remains stale, and no baseline or generated index is advanced to hide that fact.

## Knowledge settings migration

VisionClaw ADR-2041 renames the graph vocabulary while retaining bounded inbound compatibility. Three current Rust alias tests and eight client migration tests pass; the [receipt](evidence/graph-settings-migration.json) records their exact scope. GraphsSettings deserialises the old name as an alias of knowledge and emits the canonical field. The client persistence merge hook calls migrateGraphSettingsKey, which removes logseq and preserves knowledge when both keys exist. It creates a new object for the migration rather than mutating the original.

The binary settings registry normalises embedded .graphs.logseq. path segments before registration and lookup. Its IDs remain registration-order counters, so alias normalisation prevents a second slot for that path but does not itself prove consistent registration order across independently running versions. The domain graph-type helper passes unknown values through for callers to reject; normalisation is not request validation.

These are several distinct compatibility mechanisms: typed deserialisation, object migration, dynamic JSON lookup, dotted path lookup and transport graph values. A passing test of one does not certify every patch or persistence path. CP-01/02/06/08 requires a common matrix for legacy-only, canonical-only, both keys, null/wrong types, repeated migration, old client/new server and rollback. Demonstrate the actual saved settings and binary IDs after restart, and verify rejected values reach a clear client error. Record the release in which aliases are accepted and prove consumers have migrated before removal; a relative phrase such as one release is not an operational retirement receipt. ADR-2041 keeps its scoped complete/staged declaration; live migration remains open.

## Development restart and build-input coverage

The [development source receipt](evidence/dev-docs-closeout.json) reaches ADR-2008's prebuilt-default review trigger. The image copies supervisord.dev.conf; dev-entrypoint selects supervisord when that file exists. Its backend command invokes rust-backend-wrapper.sh. That wrapper uses visionclaw-server, skips Cargo when its timestamp heuristic reports fresh output, and otherwise builds with gpu,ontology,dev-auth. It retries a failed build after cargo clean and exits if that retry fails. The older individual-service entrypoint branch uses webxr and a build pipeline through tee without pipefail. These are distinct paths; the cited old branch cannot establish normal-image behaviour.

The [extracted timestamp probe](evidence/dev-build-input-probe.py) uses temporary files and the actual decision block. [Three cases](evidence/dev-build-input-probe.json) show a newer crate Rust file triggers a build, but a newer crate CUDA file or crate Cargo.toml does not. The wrapper scans Rust under src and crates, CUDA only under src, and only the root manifest/lock/build script. Timestamp freshness also lacks content, feature, environment and toolchain identity. No stale production binary or actual missed live rebuild was observed.

ADR-2008 is partial against unconditional restart compilation and complete source-change coverage. CP-01/06/08 requires one explicit development startup contract, complete build inputs or Cargo-owned invalidation, and source/feature/toolchain/module identity for the selected executable. Test edits to every input class, preserved timestamps, changed environment/features, build failure, retry, skip mode and process startup through the actual image. The fixture never invokes Cargo, cleanup, Docker or a server. Release safety requirements still apply separately to the compiled dev-auth feature.
