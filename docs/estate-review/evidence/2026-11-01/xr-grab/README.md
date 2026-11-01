# XR grab evidence

Source pin: VisionClaw `531df5ef74453a2ed320bff968c203a9b91faed7`.
See the [assessment](../../../2026-11-01-xr-grab-boundaries.md).

`sources.json` records 22 unmodified source exports. `extraction.json` identifies
14 complete production function bodies copied verbatim into `fixture/graph_methods.gd`.
Its base class is audit scaffolding: connection sends, node positions, UI, haptics,
source invalidation and arrangement state are explicit spies. The fixture does
not instantiate GraphScene, the gdextension, controllers, a backend or a renderer.
The double-click constant matches production at 0.45 s.

Run from this directory on a machine with Godot:

```sh
godot --headless --path fixture --script res://probe.gd
CARGO_TARGET_DIR="$HOME/workspace/.tmp/vf-xr-grab-audit-target" cargo test --locked --offline --manifest-path rust-probe/Cargo.toml --lib
CARGO_TARGET_DIR="$HOME/workspace/.tmp/vf-xr-grab-audit-target" cargo clippy --locked --offline --manifest-path rust-probe/Cargo.toml --tests -- -D warnings
```

Godot 4.6.1 on the authorised HP host ran the fixture with a private XDG data
directory: 25 checks, zero failures. A successful exit means the controls and
recorded counterexamples matched expectations, **not** that product requirements
passed. `probe-final.log`, `results.json` and the remote file hashes preserve the run.
The generation test models id reuse; the double-click case replays callbacks;
the box case seeds an existing grip. None simulates the full controller loop.

The Rust probe compiles the complete, unmodified production interaction module
under its existing test configuration: 13 original tests and one new capture
boundary test pass. It confirms capture at 0.1, 1, 6, 10 and 30 m and refusal at
30.1 m. The separate Godot fixture measures hold-depth conversion. These are
complementary unit observations, not an integrated Rust/Godot scene test.
Strict Clippy passes. This private probe is not an estate library release.

`portal-source-delta.json` covers all eight ADR-2166 governed paths and preserves
the delta against the prior audit. Old portal and performance results are linked
from the report and remain historical. No prior headset or deployment limitation
is removed by a source hash matching.

The two official Unity documentation pages were opened on 2026-11-01. They inform
the input and attachment comparisons only; no competing engine was benchmarked.

VC-47's three panels and VC-53's five panels pass the strict citation and Pele
render gates and were visually inspected. Inspection previews explicitly apply
16 px DejaVu Sans text attributes to avoid ImageMagick's serif fallback; generated
SVGs remain unchanged. This is diagram legibility evidence, not browser rendering.

The full index contains 205 topics and 1,730 panels. Its citation gate verifies
25,543 citations at declared revisions, with zero warnings or working-tree
fallbacks and nine existing allowlisted exceptions. JEV's 120-citation advisory
sample flags older AB-01.3/AB-01.5 citations for manual review; those panels gain
no semantic review credit here. The index/verification freshness check passes.
