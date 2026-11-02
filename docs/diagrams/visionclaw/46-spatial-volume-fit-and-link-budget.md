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
  - ../project/xr-client/scripts/xr_prefs.gd
verified_commit: {visionclaw: 86cdd7635582}
---

## For developers

The [headless audit](../../estate-review/2026-10-31-xr-volumes-and-code-links.md) runs unchanged production scripts on Godot 4.6.1 with isolated fixtures. It tests geometry/state behaviour without a headset, live streams or a native spatial container.

The subsequent [portal review](47-portal-entry-and-scaled-walkthrough.md) covers entered-room transitions, scale-dependent walk-through re-arming and membrane accounting; entered rooms inherit the fit boundary shown here.

## For the business

The multi-volume interface is implemented, but its budget and containment promises need narrower wording or repairs. Highlight markers add geometry beyond the link cap, and a newly expanded graph extent can remain outside its fitted box for several ticks.

## VC-46.1 Highlight markers are added after the 512-link cap

```mermaid
flowchart TD
    SORT["Sort all link records and count overflow<br/>link_layer.gd:343,344"] --> CAP["Keep at most 512 records<br/>link_layer.gd:38,345"]
    CAP --> PACK["Rust pack_links walks the capped specs<br/>link_layer.gd:498; render_store_pack.rs:667"]
    PACK --> SEG["Pack each drawable link segment<br/>render_store_pack.rs:729"]
    PACK --> MARK["Highlighted links collect one marker per distinct end<br/>render_store_pack.rs:722,725"]
    SEG --> APPEND["Append marker instances after the links<br/>render_store_pack.rs:740,746,749"]
    MARK --> APPEND
    APPEND --> MM["Set MultiMesh instance count to links plus markers<br/>link_layer.gd:501,502,503,504"]
    MM --> COST["Charge all instances at 12 triangles each<br/>link_layer.gd:179,180"]
```

**What it shows.** The audit fixture started with 512 links and 6,144 triangles. Highlighting their common source added 513 markers: 1,025 instances and 12,300 triangles, still one draw call and zero record overflow. That fixture ran when packing was GDScript. Since then packing has moved to the Rust `pack_links`, with the same order: links first, then markers. The source trace at the pin shows the same arithmetic, but the fixture was not re-run. **Why it is this way.** The cap applies to link records, and the markers are appended later. Triangle accounting includes the extra geometry, so this is not an uncounted rendering cost.

**Tension:** ADR-2164 D8/L5 caps instances and total triangles, while the current highlight path exceeds both (`render_store_pack.rs:740`; `link_layer.gd:504`).

## VC-46.2 A cached extent can lag moving content despite a safe fit calculation

```mermaid
flowchart TD
    TICK["Visible, open, unlatched volume ticks<br/>spatial_volume.gd:179"] --> DUE{"Refit requested or sixth tick?<br/>spatial_volume.gd:184"}
    DUE -->|yes| READ["Read current source AABB<br/>spatial_volume.gd:185"]
    READ --> FIT["Uniform minimum-axis scale, ignoring tiny extents<br/>spatial_volume.gd:105"]
    DUE -->|no| CACHE["Retain cached target and AABB<br/>spatial_volume.gd:184,200"]
    FIT --> CHECK["Ease, then snap if cached AABB would escape<br/>spatial_volume.gd:200"]
    CACHE --> CHECK
    BOUNDS["Explicit bounds change re-reads source immediately<br/>spatial_volume.gd:285,291"] --> FIT
```

**What it shows.** After a fitted source expands, the headless fixture observes its new corner outside the box for four ticks before the sixth-tick refresh. Explicit resizing refits immediately and the anisotropic/zero-extent controls pass. **Why it is this way.** The containment check uses the cached extent. This is a transform-level counterexample; no headset image or real node-motion frequency was measured.

## VC-46.3 Native support is a guarded wrapper around the emulated arrangement

```mermaid
flowchart TD
    BOOT["Scene constructs shim and evaluates once<br/>graph_scene.gd:1528"] --> GATE["Require app setting, Android, engine setting and 4.8+<br/>native_container_shim.gd:140"]
    GATE --> EXT["Require extension class, instance and enabled state<br/>native_container_shim.gd:154,157,160"]
    EXT --> START["Start around an existing arrangement<br/>native_container_shim.gd:197"]
    START --> WATCH["Poll activity and activation timeout<br/>native_container_shim.gd:220"]
    WATCH --> FIT["Fit whole arrangement, with explicit enum translation<br/>native_container_shim.gd:252,336"]
    WATCH -->|timeout or loss| FALL["One-shot fallback restores scale and default arrangement<br/>native_container_shim.gd:411"]
```

**What it shows.** Existing fake-extension tests cover fallback and arrangement fitting; real native-container support is not established by the headless run. **Why it is this way.** Every graph volume stays emulated. The optional native arm wraps the application arrangement rather than turning each graph into a separate native container.
