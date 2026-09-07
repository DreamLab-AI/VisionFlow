# Agentbox and Loom closeout execution — 2026-09-07

This executes the assigned master-register items after the source audit. Existing
unrelated changes were preserved, including the contract workflow dependency fix
and semantic-rule tests. The master register is edited by the root coordinator.
Evidence below distinguishes source implementation, observed activation and
retained-disabled profile decisions. No publisher key was rotated, credential
content read, local Agentbox container rebuilt, or paid/model task dispatched.

## Changes and verification

- **Boot identity:** failed mint, malformed configured DID, inconsistent public
  components and extra executable output now abort before consumers. Mint output
  is parsed as three exact exports and never evaluated. Six isolated shell tests
  pass, including failed mint with valid-looking output and an injection probe.
- **Encrypted custody packaging:** `lib/secret-backup.nix` builds the locked crate
  with checks enabled; `flake.nix::secretBackupPkg` includes it in runtime packages.
  Eight synthetic backup/restore/permissions/wrong-key and ancestor-pruning tests pass locally and in the HP Nix derivation. This packages
  a tool; it does not perform G-5's actual key migration.
- **Loom client identity:** GET generation precedes cache lookup. The client binds
  loaded digest, graph/semantic generation and bge-small-en-v1.5 / 384 / cosine;
  mismatches, missing proof or unavailability return empty labelled degradation.
  Each read response must have matching generation/digest/atomicity headers.
  Loom preserves the existing response body shapes; Agentbox now accepts Rust
  label-search arrays as well as the historical hits envelope. Twenty-seven JS
  tests pass, including cached-answer refusal and mixed expansion responses.
- **Loom server/source closure:** new read-response identity headers and generation
  model qualification are covered by a router integration test. The full locked,
  offline workspace test run passes (exact totals in verification-summary.json).
  The new Rust workflow checks out the selected fork at a full SHA and runs a
  source guard before Cargo. Local guard passes; hosted Rust contracts run 34121825633 passes at pushed 397b86c.
- **Documentation:** AoE activation now has a process and negative-request receipt;
  the Loom confidence interface is accepted separately from deployment; the LAN
  threat model and retained-disabled profile are explicit; model alias and documented
  weight variant are distinguished; the routing observation window is specified.

## Per-item disposition

| ID | Disposition | Concrete evidence / remaining dependency |
|---|---|---|
| G-4 | Source gap closed; explicit recipient rollout staged | Mandatory recipient enumeration implemented before turn text is read/composed, including dry-run. 25 isolated tests pass: allowed/denied/unset/empty/invalid and no body leakage. Activation needs an explicit reviewed public-recipient set; external retention/custody are separate. |
| G-5 | Blocked on concrete custody migration | Tracked operator publisher remains shared. No reviewed replacement public identity, encrypted recovery custody and corresponding bilateral allowlist migration has been supplied or executed. G-17 packaging does not rotate it. |
| G-6 | Source fixed and tested | `config/entrypoint-unified.sh` strict mint guard; identity-entrypoint.log: six pass. Running local image is not replaced. |
| G-10 | Documentation decision complete | `agentbox/docs/LAN-door-threat-model.md` covers AoE, unauthenticated VNC, CDP, voice/GUI, relay and Loom doors, distinguishing host publication, Docker siblings and same-UID custody. No universal authentication claim. |
| G-17 | Source and standalone Nix build verified; image activation separate | Locked Nix derivation and runtime package inclusion. Actual HP build found absolute-ancestor pruning; `collect` now prunes relative to the requested root. Eight tests pass locally and under Nix; installed package confirmed by hp-custody-nix-build-fixed.log. Source e7bfc158a, ADR be0fc078a pushed. HP M-3 baseline does not activate this package. |
| G-19 | Closed: source closure and hosted CI verified | `.github/workflows/rust-contracts.yml` checks out Loom plus fork `677b2475409c50cb964be8a3b848da2952390535`; source guard rejects wrong revisions or modified core/workspace inputs. Local guard and workspace tests pass. Relative sibling layout remains deliberate and documented; plain Cargo alone does not run the source guard. Hosted Rust contracts run 34121825633 passes at pushed 397b86c, including all workspace tests and the reload guard. |
| G-25 | Closed: local and hosted contract checks pass | Preserved root `npm ci --ignore-scripts` dependency fix. Exact contract selection passes 29 suites, 580 tests, 34 explicit TODOs locally. Hosted Contract tests run 34120897048 succeeds at pushed 7bf2382c0 (agentbox-hosted-after-push.json). Separate Security Invariants run failed on 7 documentation-only stale revision anchors, corrected in be0fc078a with 74 ADRs validating locally and hosted Security Invariants run 34121401119 succeeding; do not infer all workflows green. |
| G-29 / DOC-5 | Naming reconciliation complete; weight attestation separate | Manifest keeps wire alias `qwen3.8-27B`; comments, workspace CLAUDE.md and Loom guide name the documented Heretic abliterated Qwen3.8-27B Q8_0 variant. Historical August deployment is labelled. Alias does not attest GGUF/template digest or base-model behaviour. |
| DOC-4 | Observed activation complete | Running AoE 1.13.2 uses token auth on loopback 9095; unauthenticated `/api/sessions` returns 401. ADR-2002 activation set live with scoped receipt and relation to ADR-2009. No token read or sent. |
| EA-05 | Source implemented; live activation blocked | Strict handshake/cache/response checks pass locally. Live `/loom/generation` has no loaded identity/model fields, while health reports graph generation 2026-08-22 and semantic generation 2026-08-17. Rebuild a matching bundle/server before activating the client; current deployment would deliberately be refused. HTTP service assertions are not cryptographic remote attestation. |
| M-2 / T-1 | Closed for retained-disabled supported profile | Relay expose=false/mobile=false are retained. No rail publication selected while G-5 custody remains unresolved. A future rail profile needs its own reviewed admission/deployment evidence. |
| M-3 | HP image built; loading/activation in progress | SSH/Nix/Docker available. Deployed HP source 647aabb10 has a dirty operator manifest. Separate worktree at published b5bfc03db uses an opaque copy of that manifest; old source/config/image retained. `hp-runtime-build.log` and locked-retry log record two actual fixed-output failures (Mermaid and AQE floating transitive resolution). The next build uses original lockfiles recovered from the working local packages via npm ci, reviewing changed output hashes against installed package manifests (only already-locked optional musl packages added, no existing package version or manifest changed). All nine materialise and pass an explicit forced `--rebuild` replay. Source b0c963c66 is pushed. Complete runtime image built at /nix/store/2m758brvq5b7c0b8q4dixxhnln6hsyxs-image-agentbox.json; Docker loading is in progress. Provenance in npm-lock-provenance.json. No container replacement yet. |
| M-4 | Stale wiring premise corrected; end-to-end fan-out not observed | Current flake already projects AGENTBOX_RELAY_FANOUT from relayCfg.external_fanout. RelayConsumer uses it plus NOSTR_RELAYS to construct peer URLs. No fabricated federation-table requirement or automatic exposure change; live multi-peer delivery still needs receipts. |
| M-5 | Stale source premise corrected; live nightly acceptance separate | Existing commit 2c521c5bb added engine::clone_repo_and_siblings and annexe_subpath, preserving real workspace depth for annexeInclude siblings. The targeted symlink/depth/fallback regression passes. dispatch::clone_to_hp archives each repository HEAD, excluding dirty changes. This corrects the original standalone-clone escape path; no completed nightly sovereign-mesh evaluator receipt was observed. The HP baseline b5bfc03db includes this engine fix; runtime nightly execution remains separate from packaging. |
| T-4 | Retained-disabled decision complete | All eight held surface families reaffirmed in Agentbox backlog: relay/mobile, multi-user, public git/gateway, payments, OIDC, pod MCP and pip. Selected tracked booleans captured in selected-profile.json; no held surface enabled. |
| D-1 | Window defined; actual observation still required | Seven complete UTC days of deployed retrieval/corpus identity and daily passing recall receipts; restart on failure/model/index change. Existing sample/independence floors remain. Historical one-day recall does not satisfy this window; feed_routing=false unchanged. No prohibited SQL/CLI bypass used to invent measurements. |
| LM-1 | Interface acceptance complete | ADR-138 accepted under authorised closeout after rerunning eleven exp014 cases. Live health exposes confidence counters; full source contract acceptance remains distinct from matching loaded binary identity. |
| E-4 | Optional capability remains excluded; no false repair claim | Nagual is pinned in its existing Nix derivation and toolchains.nagual_qe=false. The specific upstream sqlx compile defect was not reproduced with a new dependency build; no speculative override/hash was inserted. Enabling Nagual still requires its build/probe evidence. |
| E-5 | Concrete runtime blocker | `http://visionclaw-server:4000/health` fails DNS from this container (curl 6); direct_axiom_load=false remains. Endpoint and authenticated publication path must be restored before enabling writes; no alias was invented. |
| L-2 | Live-session evidence missing | No second-model-family traffic canary executed. Source tests are not model-diversity measurements; no paid model dispatch was triggered merely to fill a row. |
| L-1 / L-6 | Physical/session blocker | No connected phone session/Amethyst+Amber interaction and envelope canary receipts were supplied or produced; retained mobile/relay exposure profile also excludes activation. Unit fixtures cannot substitute for on-phone evidence. |
| V-1 P2 | Reload trigger source implemented; data/server activation blocked | A once-per-tick verifier/reloader and opt-in user timer are implemented, with 8 tests for complete/hash-verified bundle, no-op, restart/probe success, failure and concurrent publication. It requires all 6 artefacts and matching semantic generation/model, records pending before action and served only after proof. Live bundle and legacy API fail prerequisites, so timer remains uninstalled; matching publication and server rollout are the concrete blockers. |

## Evidence and limits

[Execution artefacts](execution-2026-09-07/) include exact tool/test outputs,
read-only live probes, hosted failure and local contract results, source-pin
checks, the selected manifest flags and HP preflight/build output. Logs describe
working source above the heads in verification-summary.json; no source hash was
silently changed to imply acceptance. The AB ADR index validates all 74 records.
The updated behaviour diagrams pass rendering and worktree citation checks:
AB-08 14/14, AB-11 16/16, AB-13 17/17, AB-16 12/12, AB-24 9/9 and
ES-06 10/10, each with zero citation warnings. These cover recipient admission
before body access, strict mint parsing, packaged custody and generation identity/reload.
Subsequent source/owner/port corrections reduced the whole Agentbox lane to zero
worktree citation warnings across 29 topics / 353 diagrams (agentbox-citations-current.log).
ES-01/07/09 additionally render 6/9/21 panels with zero warnings. This is a bounded
snapshot during concurrent editing, not global G-18 acceptance; the earlier residual
log remains retained as history.

The HP build is intentionally of the published baseline, which includes the
previous M-3 source fixes. Local EA-05/G-6/G-17 changes require a later coordinated
rollout; upgrading the client alone would suppress currently unverified Loom
results. No current local session is restarted by this work.

## Additional verified execution receipts

The standalone custody package initially failed five tests inside Nix because
absolute `/build` ancestors matched directory exclusions. Root-relative pruning
fix e7bfc158a passes all eight tests in Nix and locally; this is package acceptance,
not deployment or real-key recovery. `hp-custody-nix-build-fixed.log` ends
`CUSTODY_NIX_BUILD_PASS`. The nine npm fixed-output closures each passed an explicit
forced rebuild (`hp-npm-reproducibility-rebuild.log`), not only cache lookup.
Agentbox source and scoped ADR/workflow documentation are pushed through be0fc078a.

Agentbox CI run 34121400849 and Secret Scan run 34121401072 also succeed
at be0fc078a. Loom docs/workflow commit 48b2659 is pushed; Rust contracts are
running in hosted CI (run 34121700475), with deployment unchanged.

The first hosted Loom run (34121700475) correctly failed: `cargo fmt --all`
recursively checked unrelated pinned upstream path dependencies. The workflow
now uses `cargo fmt --check`, whose verbose local receipt enumerates all eight
Loom workspace members without changing upstream files. Commit 397b86c is pushed;
retry 34121825633 has passed checkout, fork guard and format, and is testing.

Hosted Loom retry 34121825633 **passes** at 397b86c: exact dependency source
guard, all Loom workspace formatting, locked workspace Rust tests and reload
guard tests. See loom-hosted-success.json and loom-hosted-success.log. G-19
source/CI acceptance is complete; EA-05 data/server activation remains separate.

## Final current-source diagram reconciliation

AB-01/03/08/13/16 are qualified to published Agentbox be0fc078a. All 71 panels
render successfully with zero worktree citation warnings, and the five file hashes
remain unchanged after rendering (agentbox-final-five-topic-hashes.sha256).
Source reads confirmed mandatory recipient enumeration before stdin/body access,
including dry-run. Named flake symbols were reanchored to their current locations.
The custody diagram now correctly orders encryption selection before collection,
shows root-relative exclusions, and describes the manifest as names and metadata
without file hashes. The cuda-runtime diagram describes its explicit local-CUDA
dispatch rather than incorrectly making that output conditional on the manifest.
No root Git staging was performed; the coordinator owns the final snapshot.

## Other hosted workflow dispositions

Ontology Federation run 33862998777 failed to check out the private `jjohare/logseq`
source using the repository GITHUB_TOKEN. Its workflow, ontology-publish.yml, was
deleted by 7b2811272 and is absent from current main; this is an inactive historical
failure, not an active publication gate. It was not recreated or supplied credentials.

The active flake updater had two concrete faults: an unsupported boolean flag and
an installer defaulting to Nix 2.22.1, which rejects the current relative-path lock
entries. Commits 8a6bf696f and c11059cba remove the flag and select Nix 2.35.2 in all
three Nix workflows, preserving action SHA pins and the project lock. An isolated
HP fixture confirms the corrected command writes a lock without a Git commit.
Hosted retry 34126738773 passes update and evaluation after the development-shell
CLI repair, then fails because repository policy denies Actions-created pull
requests. Review branch `deps/nix-flake-update` exists at 7e17343; no PR, merge or
deployment occurred. The Actions permission setting was not changed. Historical failure logs
and nix-flake-update-fixture.log retain the exact evidence.

The development-shell executable selection repair is 8fcc7b79b, with six governed
ADR anchors qualified in 771d96ed5. Main is pushed through that commit. The
updater’s unresolved failure is now the concrete repository PR-permission policy,
not Nix command parsing or evaluation. HP image construction has completed at
`/nix/store/2m758brvq5b7c0b8q4dixxhnln6hsyxs-image-agentbox.json`; Docker loading
is in progress, so M-3 runtime acceptance is not yet asserted.

A further metadata-only pass pins AB-14/23/30 to Agentbox 771d96ed5 and
ES-04/09 to that Agentbox commit plus VisionClaw dd82a07b0. All 60 panels pass
declared-revision citation checks with zero warnings. Diagram blocks were
unchanged, so existing render receipts remain applicable. See
agentbox-estate-final-declared-pins.log.

## HP rollout result and rollback

The built candidate `agentbox:closeout-built-20260907-e3fbe18f` was loaded and tested with the existing deployment configuration and mounts. It failed the bounded health/readiness checks; the rollout script recreated the original image from the retained rollback tag. M-3 remains blocked by startup failure. No successful runtime upgrade is claimed.

The independent follow-up [health receipt](execution-2026-09-07/hp-rollback-health.log) confirms original image `sha256:c64e5acaf902cbda96cced5665f9bda7dd18f1a067545c3ea0dc41105a79bc55` is running, Docker health is healthy and `http://127.0.0.1:9090/ready` returns HTTP200. The [rollout log](execution-2026-09-07/hp-runtime-rollout.log) preserves the failed checks and rollback. The candidate and build receipts remain available for startup diagnosis; there is no attribution of the failure to an unverified cause.
