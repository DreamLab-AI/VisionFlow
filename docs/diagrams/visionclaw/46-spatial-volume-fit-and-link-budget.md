---
id: VC-46
title: Spatial volume fit cadence, link highlights and native fallback
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2158-xr-content-is-presented-in-spatial-volumes-emulated-until-the-runtime-provides-them.md
  - ../project/docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md
adrs: [ADR-2158, ADR-2164]
sources:
  - ../project/xr-client/scripts/spatial_volume.gd
  - ../project/xr-client/scripts/link_layer.gd
  - ../project/xr-client/rust/src/render_store_pack.rs
  - ../project/xr-client/scripts/native_container_shim.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The [headless audit](../../estate-review/2026-10-31-xr-volumes-and-code-links.md) runs unchanged production scripts on Godot 4.6.1 with isolated fixtures. It tests geometry/state behaviour without a headset, live streams or a native spatial container.

The subsequent [portal review](47-portal-entry-and-scaled-walkthrough.md) covers entered-room transitions, scale-dependent walk-through re-arming and membrane accounting; entered rooms inherit the fit boundary shown here.

## For the business

The multi-volume interface is implemented, but its budget and containment promises need narrower wording or repairs. Highlight markers now share the link cap rather than exceeding it, but links they displace vanish without a count, and a newly expanded graph extent can remain outside its fitted box for several ticks.

## VC-46.1 Links and highlight markers now share one 512-instance allowance

```mermaid
flowchart TD
    SORT["Sort all link records and count overflow<br/>link_layer.gd:343,344"] --> CAP["Keep at most 512 records<br/>link_layer.gd:38,345"]
    CAP --> PACK["Rust pack_links walks the capped specs<br/>link_layer.gd:498; render_store_pack.rs:698"]
    PACK --> DRAWN["Both ends must be drawn in this volume's slot<br/>render_store_pack.rs:725"]
    DRAWN --> ALLOW{"links + markers + 1 + new markers<br/>over 512?<br/>render_store_pack.rs:738,746"}
    ALLOW -->|yes| SKIP["Link skipped, no counter raised<br/>render_store_pack.rs:747"]
    ALLOW -->|no| MARK["Highlighted links collect one marker per distinct end<br/>render_store_pack.rs:754,757"]
    ALLOW -->|no| SEG["Pack the link segment<br/>render_store_pack.rs:761"]
    SEG --> APPEND["Append marker instances after the links<br/>render_store_pack.rs:772,778,781"]
    MARK --> APPEND
    APPEND --> MM["Set MultiMesh instance count to links plus markers<br/>link_layer.gd:501,502,503,504"]
    MM --> COST["Charge all instances at 12 triangles each<br/>link_layer.gd:179,180"]
```

**What it shows.** Since `6b811f7a1` the Rust pack reserves room for a lit link's not-yet-seen end markers before it packs the link, so links plus markers never exceed 512 instances (6,144 triangles); the 2026-10-31 fixture (512 links plus 513 markers) predates this and was not re-run. **Why it is this way.** ADR-2164 D8/L5 bound instances and triangles, and the earlier record-only cap let markers ride past it; the allowance closes that gap in the packer rather than in the GDScript record cap.
**Open (risk):** a link dropped by the instance allowance is skipped without touching `unresolved` or any overflow count (`xr-client/rust/src/render_store_pack.rs:746-747`), while the only reported overflow is the record cap (`xr-client/scripts/link_layer.gd:344`), so highlighting can silently remove links from view.
**Debt (parallel):** the packer hard-codes `512` (`xr-client/rust/src/render_store_pack.rs:746`) beside the GDScript `MAX_LINKS := 512` it must agree with (`xr-client/scripts/link_layer.gd:38`); nothing ties the two.
**Drift:** ADR-2164's 2026-11-02 review note still says Rust adds deduplicated markers after the record cap (`docs/adr/ADR-2164-code-graph-entities-link-to-knowledge-and-ontology-nodes-resolved-at-runtime.md:28-29`); the next day's packer counts them inside the 512 allowance (`xr-client/rust/src/render_store_pack.rs:746`).

## VC-46.2 A cached extent can lag moving content despite a safe fit calculation

```mermaid
flowchart TD
    TICK["Visible, open, unlatched volume ticks<br/>spatial_volume.gd:184"] --> DUE{"Refit requested or sixth tick?<br/>spatial_volume.gd:189"}
    DUE -->|yes| READ["Read current source AABB<br/>spatial_volume.gd:190"]
    READ --> FIT["Uniform minimum-axis scale, ignoring tiny extents<br/>spatial_volume.gd:105"]
    DUE -->|no| CACHE["Retain cached target and AABB<br/>spatial_volume.gd:189,205"]
    FIT --> CHECK["Ease, then snap if cached AABB would escape<br/>spatial_volume.gd:205"]
    CACHE --> CHECK
    BOUNDS["Explicit bounds change re-reads source immediately<br/>spatial_volume.gd:290,296"] --> FIT
    FITBTN["HUD Fit button: refit sets the request flag<br/>graph_scene.gd:2033,2043; spatial_volume.gd:156,157"] --> DUE
```

**What it shows.** After a fitted source expands, the headless fixture observes its new corner outside the box for four ticks before the sixth-tick refresh. Explicit resizing refits immediately and the anisotropic/zero-extent controls pass. **Why it is this way.** The containment check uses the cached extent. This is a transform-level counterexample; no headset image or real node-motion frequency was measured.

## VC-46.3 Native support is a guarded wrapper around the emulated arrangement

```mermaid
flowchart TD
    BOOT["Scene constructs shim and evaluates once<br/>graph_scene.gd:1694"] --> GATE["Require app setting, Android, engine setting and 4.8+<br/>native_container_shim.gd:140"]
    GATE --> EXT["Require extension class, instance and enabled state<br/>native_container_shim.gd:154,157,160"]
    EXT --> START["Start around an existing arrangement<br/>native_container_shim.gd:197"]
    START --> WATCH["Poll activity and activation timeout<br/>native_container_shim.gd:220"]
    WATCH --> FIT["Fit whole arrangement, with explicit enum translation<br/>native_container_shim.gd:252,336"]
    WATCH -->|timeout or loss| FALL["One-shot fallback restores scale and default arrangement<br/>native_container_shim.gd:411"]
```

**What it shows.** Existing fake-extension tests cover fallback and arrangement fitting; real native-container support is not established by the headless run. **Why it is this way.** Every graph volume stays emulated. The optional native arm wraps the application arrangement rather than turning each graph into a separate native container.
