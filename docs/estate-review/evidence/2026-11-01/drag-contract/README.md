# Drag-contract evidence

Evidence for the [contract review](../../../2026-11-01-drag-holds-and-acceptance.md).

`sources.json` pins VisionClaw source at `531df5ef74453a2ed320bff968c203a9b91faed7`.
Full files retain their original bytes. `graph-scene-excerpts.txt` preserves numbered
lines and records the full original file hash. `visual-budget-delta.patch` compares
all seven ADR-2167 governed paths with the prior audit's `75b72416b` baseline.

## Native registry tests

From this directory:

```sh
CARGO_TARGET_DIR="$HOME/workspace/.tmp/vf-drag-audit-target" cargo test --locked --offline --manifest-path probe/Cargo.toml
CARGO_TARGET_DIR="$HOME/workspace/.tmp/vf-drag-audit-target" cargo clippy --locked --offline --manifest-path probe/Cargo.toml --all-targets -- -D warnings
```

The crate compiles the exported production `hold_registry.rs` directly: 14 native
unit tests pass. Three additional probes pass, recording late-hold readmission,
socket-only capacity enforcement and the ownership divergence. The last probe
invokes the real registry but **models** the node-only orchestrator HashSet; it
must not be presented as an actual actor test. No GPU is loaded. The locked glam
version is 0.30.5, compatible with the production manifest's declared version.
This private probe is audit machinery, not a published estate crate.

## Full-script negative control

Supply a local installation of the Node `ws` package:

```sh
node acceptance-negative-control.cjs source/scripts/drag-physics-acceptance.mjs /absolute/path/to/ws
```

The fixture binds only `127.0.0.1` on an ephemeral port and launches the unchanged
acceptance script against it. It deliberately keeps the held node pinned and
emits no post-release position frames. During the hold, the dragger receives
neighbour updates while its held-node sample remains stale; the observer sees
both the held node and neighbours. All five script checks pass and the script
exits 0. The fixture itself exits 0 only when this false acceptance is reproduced.
That exit code means **counterexample reproduced**, not correct backend behaviour.

The synthetic peer accepts V5 acknowledgements but does not implement server
pacing, permissions, graph physics or release semantics. It establishes what the
script can falsely accept, not whether that exact trace occurs in deployment.
No production endpoint or private graph was used.

## Documentation and visuals

VC-52's five panels were rendered and inspected. The ImageMagick default preview
uses a serif fallback for the SVG's inherited CSS font. A second preview applies
explicit 16 px DejaVu Sans presentation attributes to text, matching the declared
sans-serif style for inspection; both previews and untouched generated SVGs are
retained in the scratch archive. This is diagram legibility evidence, not a browser
or product screenshot. No historical VC-16 panel gains source-review credit from
its added revision notice.

The full corpus gate resolved 25,480 citations at their declared revisions, with
zero citation warnings, zero working-tree fallbacks and nine existing allowlisted
citations. The generated index and verification receipt checks pass. JEV's advisory
sample flagged AB-01.3 and AB-01.5 for manual review; those older panels remain in
the audit queue and acquire no credit from this run. The sample is not a semantic
review of the whole corpus.

D3 force simulation and Yjs awareness documentation were opened on 2026-11-01.
The report's comparisons distinguish temporary holds, explicit fixed coordinates
and presence from live physics performance; they do not recommend replacing the
estate's shared owner or infer guarantees from another project's implementation.
