---
title: Codescape extraction, command policy and retained GPU artefacts
status: in-progress
date: 2026-10-31
type: explanation
---

# Codescape extraction, command policy and retained GPU artefacts

Codescape's extraction and dependency isolation are supported by current source and tests. Three operational claims need qualification: an explicit Jev endpoint bypasses the local/cloud permission check, a confidence of exactly `1.0` selects a command at the default threshold, and a startup path announcing CPU fallback leaves a previously valid PTX pair available to the loader.

This pass reads the standalone Codescape decision pack at `58108c0357bd0dbf6588f8b6adf6e044f4c04556`, compares the consumed checkout at `e3b76f63e801c6a8460110b93e88a0b420e9b0b4`, and checks host configuration at VisionClaw `75b72416bac176ca8fde715cac3bd7c77657b3e7`. [Source receipt](evidence/2026-10-31/codescape-sources.json). Both Codescape checkouts were clean. VisionClaw's existing unrelated changes were preserved. No service was restarted, no model provider contacted and no GPU kernel loaded.

## Reproduced boundaries

The [Rust probe](evidence/2026-10-31/codescape-command-probe.rs) links the freshly built production library. [Results](evidence/2026-10-31/codescape-command-probe.json) contain eleven cases, including passing controls. Scripted replies test policy; a separate loopback HTTP exchange exercises the real transport. Selection of a typed view command is tested, not its subsequent execution.

| Case | Observed result | Interpretation |
|---|---|---|
| Explicit `https://remote.example.invalid` with Jev enabled, cloud/local flags all false | Resolves and can enter the derived egress allowlist; the address is never contacted | Endpoint configuration is trusted before the local/cloud gate. This is a configuration-policy gap, not arbitrary URL input from an unauthenticated caller. |
| Implicit proxy endpoint with no local/cloud permission | Refused | The gate exists on the fallback resolution branches. |
| No enabling flag | Refused | Jev remains off by default in the library. The host Compose default differs. |
| Default threshold, valid offered command with confidence `0.99` | Preserves grammar `NoMatch` | Below-threshold fallback works. |
| Same response with confidence exactly `1.0` | Selects `ZoomIn`, including over real loopback HTTP | A threshold of one is not a disabled-acceptance state. |
| Confidence above one, `none`, or unknown command | Preserves `NoMatch` | Range, closed-choice and abstention guards work. |
| Grammar recognises `help`, tested against all five replies | Returns `Help` without another transport call | Grammar hits retain precedence. |
| Valid cached PTX pair, missing GPU source directory, verbatim production `build_ptx` function | Exit 0, diagnostic promises CPU fallback, cached pair unchanged and still accepted by `PtxSource::from_dir` | A failure to refresh the artefact does not invalidate the previous one or force CPU selection. |

### Endpoint locality is conditional

`crates/codescape-server/src/voice/jev.rs:171-227` computes the local/cloud permission flags, but returns an explicit `CODESCAPE_JEV_ENDPOINT` at lines 190–198 before checking them at line 200. `egress_url` uses this same resolver (`:231`), and `config.rs:282` adds the result to the egress list. The allowlist therefore cannot independently enforce the resolver's stated local-only default.

Egress checks configured origins and redirect targets (`egress.rs:77,98`). It does not determine where an allowed proxy ultimately sends a request. The current host configuration explicitly opts into the shared proxy and cloud route, so this counterexample does not show that the current deployment violates its operator's selection. It shows that setting an explicit remote endpoint without the documented permission flag is enough to resolve it. Acceptance should distinguish trusted endpoint configuration, assertions that a backend is local, and independently enforced network locality.

### Default confidence is not advisory-only

Resolution defaults to `1.0` (`voice/jev.rs:182-186`); selection rejects only `confidence < min_confidence` (`:409`). A valid offered response with confidence one reaches `Command::validate` and returns a command. `after_grammar` replaces a `NoMatch` with that command (`:435-446`), and the real route treats the result as an ordinary command (`routes/mod.rs:299-302`). The probe does not execute that final route or demonstrate any model's real confidence distribution.

Host Compose enables Jev by default and leaves the threshold empty (`docker-compose.unified.yml:504-508`), while its comment calls the model advisory. That comment and VC-39's previous “accepts nothing” wording overstate the gate. The offered set is bounded to argument-free view commands; this finding is not evidence of repository mutation, arbitrary code execution or an automatic ontology write. If calibration is intended to precede all model-selected actions, acceptance needs a separate disabled state, with an exact-one response in its test matrix.

### Missing GPU sources can leave a usable prior artefact

`build_ptx` (`deploy/codescape-entrypoint.sh:53-96`) returns success when the source manifest is absent. It does not remove the existing output pair. The probe executes that function in isolation with disposable paths, then validates the retained pair through the real loader. It uses the repository's ABI fixture plus a generated matching manifest, so this is neither real compiler output nor a GPU conformance measurement.

`CudaEngine::new` reads the configured PTX directory (`crates/codescape-layout/src/cuda.rs:467-472`), and `select_engine` falls back when construction fails (`src/lib.rs:375-388`). Missing build sources alone do not establish that failure. The implementation can deliberately reuse a prior artefact, but the startup message must identify that behaviour and its provenance. Actual device acceptance and rendering remain untested here. The baseline already discloses that a hand-written manifest passes its non-cryptographic consistency check; this pass does not recast that known limit as signature verification.

## Standalone and consumed revisions

VisionClaw's committed gitlink still names `e3b76f63e801`. The standalone diff contains ten changed paths: nine documentation/ignore/evidence paths and formatting in `voice/jev.rs`. Formatting each revision's Jev file through the same installed rustfmt yields identical bytes, recorded in the source receipt. The entrypoint, resolver's neighbouring configuration, egress implementation and command route are byte-identical between those checkouts. The two command-policy findings also exist at VC-39's historical Codescape pin `13e056d9aa90d6f1b4aaf3b72d69a8cb29446ad1`, whose Jev file matches the consumed file exactly.

These newer phase receipts add evidence about the standalone checkout. They do not show that the host consumes a different feature implementation. Its GPU conformance receipt also explicitly names a different PTX artefact from the recorded running service. Preserve those identities when using a phase result to discuss deployment readiness.

## Local decision pack, section by section

Of the four ADR-like census files, the pack contains one decision (`ADR-0001`), a routing index, a preamble and a template. They are not four implemented decisions. The local preamble makes the living baseline authoritative; pre-extraction VisionClaw ADR-2143 supplies history. This pass does not claim a complete re-audit of every ADR-2143 amendment or every inherited invariant.

| Record or section | Assessment |
|---|---|
| ADR-0001 metadata and opening status | Accepted/complete/live records the owner's extraction/activation judgement. `verified_commit` is descriptive prose, contrary to the preamble's required full SHA. The opening says activation has not happened; the later dated close-out supersedes it. Both should be legible without reconciling contradictory present-tense statements. |
| Context | The earlier path couplings and extraction origin are historical rationale. Current manifests and isolation tests support their removal. The historical repository-size estimate was not remeasured. |
| Decision 1: separate repository | Separate Git owners, workspace and lockfile are present. The host consumes the separate repository through a gitlink. |
| Decision 2: licence split | Manifest declarations and transitive normal/build dependency checks distinguish the two permissive crates from the AGPL crates. This is a source/metadata observation, not a new legal assessment of relicensing. |
| Decision 3: shared wire | Registry dependency on `visionclaw-wire`, the prefix exception and no external path packages are checked. Model tests pass against the consumed dependency. No new publication or downstream XR acceptance is claimed. |
| Decision 4: runtime PTX | Runtime file/manifest loading, ABI checks and CPU fallback on loader failure exist. Missing source during startup retains prior files, as reproduced above. Build-source availability and loader acceptance are different conditions. |
| Decision 5: graphicle revision | Workspace dependency remains pinned to `003f93605d895ed69f0cf06690dd296474385b99`; isolation tests resolve it. No upstream graphicle audit is implied. |
| Decision 6: host deployment | Host Compose names the submodule entrypoint and the gitlink exists. Current service/process activation was not inspected in this pass. |
| Invariants | Eight current isolation tests pass, including the injected forbidden-package control, default/cuda dependency resolutions, boot refusal, provenance of dependencies and licence filtering. This does not prove a network firewall or absence of corpus mounts. The baseline itself keeps those outside enforced invariants. |
| Consequences | Independent workspace checks support decoupled source builds. Runtime PTX and the host gitlink remain explicit external dependencies. The permissive publish dry-run is historical evidence, not rerun here. |
| Alternatives rejected | Historical tradeoffs remain attributable to the extraction decision. Repository size and fetching-cost estimates were not independently validated. |
| Verification | Historical import receipts are labelled. The undated “follow-on, not done” line contradicts later completion notes unless read as history. Current tests below add a narrower fresh receipt. |
| Close-out 30 October | Records a particular running container and CUDA artefact. It is not a current health observation or evidence that later standalone commits are deployed. |
| Close-out 31 October | Raw P0/P1/P2 files exist, retain the formatting failure and distinguish GPU conformance from outstanding latency, isolation and mount checks. No GPU result was rerun here. The raw workspace log records seven ignored tests, not six CUDA device tests: one ingest dogfood case, five layout/artefact cases and one ASR/TTS network round-trip. Several are CPU benchmarks. The summary miscounts and misclassifies this coverage. |
| README routing and ledger | Routes to baseline, deployment and ADR-0001; status agrees with the amended metadata. |
| PREAMBLE | Defines the baseline as normative and requires a full-SHA verification stamp. ADR-0001 does not meet that stamp rule. |
| TEMPLATE | Describes verification-path staleness checks, but the inspected `.github/workflows/ci.yml` has Rust, CUDA compilation and client jobs, with no ADR metadata/staleness invocation. This workflow does not establish enforcement of the template's claim. |

## Verification and remaining work

Fresh tests passed **189 cases**: [server library](evidence/2026-10-31/codescape-server-suite.txt), 99 passed and one network round-trip ignored; [isolation](evidence/2026-10-31/codescape-isolation-suite.txt), eight passed; [model/collaboration](evidence/2026-10-31/codescape-pure-suite.txt), 65 unit/integration and 17 documentation tests passed. Eleven additional probe cases passed their assertions, including the three counterexamples. These results do not certify the full desktop/XR, voice, GPU or deployed path.

Reproduce with `cargo build --offline --locked -p codescape-server --lib --message-format=json`, then compile the probe with `rustc --edition=2021`, `-L dependency=<codescape>/target/debug/deps` and the build's selected `codescape_server`, `codescape_collab`, `codescape_layout`, `tokio` and `serde_json` rlibs. Cargo emits more than one serde_json artefact: use the runtime instance with `arbitrary_precision`, not the build-dependency instance. Pass the Codescape root to the probe. The [reproduction runner](evidence/2026-10-31/codescape-probe-run.cjs) performs that selection and removes its executable. It writes only audit evidence and disposable fixtures.

[VC-42](../diagrams/visionclaw/42-codescape-command-policy-and-runtime-artefacts.md) adds the inspected boundaries. VC-39's two policy claims and ES-09's generic no-acceptance claim are corrected without advancing unrelated historical source pins. Product code and owner ADR statuses are unchanged. Remaining Codescape work includes the full inherited ADR-2143 review, client/XR behaviour, presence semantics, authentication and proxy boundaries, mounted-data isolation, real model calibration and current GPU/deployment acceptance. The wider estate review remains open.

The citation tool needed one mapping correction: it recognised `../project/codescape/` but not the standalone `../codescape/`. Before that correction, all 51 new citations fell back to working-tree bytes; afterwards they resolve at the declared revision. The three changed topics rendered all 35 diagrams with zero citation warnings. The [full citation gate](evidence/2026-10-31/codescape-citations.txt) passed across 163 topics and 1,542 diagrams: 23,926 revision-bound references, no fallbacks or unresolved references, and nine explicitly allowlisted generated-file references. Verification is fresh for all 39 declared revisions. These are structural and citation checks, not a claim that every diagram has received semantic review.

Follow-on: the [presence and concurrent-XR review](2026-10-31-codescape-presence-and-xr.md) now supplies the inherited ADR-2143 and ADR-2159 section assessments and current admission, proxy and presence probes. Its remaining operational acceptance limits still apply.
