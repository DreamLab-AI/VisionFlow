---
id: VC-53
title: XR grab lifetimes, input arbitration and catch depth
area: visionclaw
governing:
  - ../project/docs/XR-client.md
adrs: [ADR-2172, ADR-2166, ADR-2169]
sources:
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/scripts/backend_connection.gd
  - ../project/xr-client/scripts/connection_volume_source.gd
  - ../project/xr-client/rust/src/interaction.rs
  - ../project/docs/XR-client.md
  - ../project/xr-client/tests/unit/test_grab_client.gd
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

The [assessment](../../estate-review/2026-11-01-xr-grab-boundaries.md) traces ADR-2172 and refreshes the portal amendment. Its method probes use exact production function bodies on explicit spies; they are not a full scene, controller or backend run.

## For the business

Two hands can address separate nodes, but reliable manipulation also needs a clear end to each gesture. A data refresh now ends every hold cleanly; opening a document and catching a distant node still expose gaps that ordinary single-step demonstrations miss.

## VC-53.1 Per-hand holds route through their volume and share one id domain

```mermaid
flowchart TD
    VOL["Knowledge and ontology share VisionClaw<br/>graph_scene.gd:95,96"] --> GATE["Capability and entered-space checks,<br/>the merged-space guest also grabbable<br/>graph_scene.gd:301,324,325,3735"]
    GATE --> PACK["Merge each volume's own drawn candidates<br/>graph_scene.gd:3764,3774,3784"]
    PACK --> HOLD["Record hand id, volume and exclusion set<br/>graph_scene.gd:3838,3839,3840"]
    HOLD --> DRIVE["Invert Content transform and send on volume connection<br/>graph_scene.gd:3877,3879,3881"]
    HOLD --> RELEASE["End one hand and update remaining exclusions<br/>graph_scene.gd:3888,3898,3905"]
```

**What it shows.** Per-hand state and per-volume transforms coexist with a shared draggable connection. **Why it is this way.** The class-filtered views share one graph and server id domain; this is not a general multi-server identity scheme.

## VC-53.2 A topology change now releases the holds of that connection

```mermaid
flowchart TD
    GEN["Topology bumps connection generation, then signals<br/>backend_connection.gd:391,392,393"] --> RELEASE["Release every hold on a volume of that connection first<br/>graph_scene.gd:4662,4663,4669,4670,4671"]
    RELEASE --> END["_end_grab per hand sends drag end and clears the hand<br/>graph_scene.gd:166,168,169,3895,3898"]
    END --> SOURCE["Invalidate volume sources<br/>graph_scene.gd:4672,4674"]
    SOURCE --> CLEAR["Clear selection and stale links<br/>graph_scene.gd:4678,4681,4682"]
    DROP["VisionClaw disconnect releases every hold<br/>graph_scene.gd:4636,4639,4640"] --> END
    TEST["GUT: both hands released before ids are reused<br/>test_grab_client.gd:305,310,311,312"] --> RELEASE
```

**What it shows.** Since `6b811f7a1` the topology callback ends both hands' holds before invalidating sources and selection, and a VisionClaw disconnect does the same. **Why it is this way.** Hand state carries an id and a volume but no generation (`xr-client/scripts/graph_scene.gd:849-850`), so the release has to happen at the boundary rather than at each drive.
**Invariant:** no hand holds a wire id across a topology generation of its connection (`xr-client/scripts/graph_scene.gd:4669-4671`), pinned by `xr-client/tests/unit/test_grab_client.gd:305-312`.

## VC-53.3 Double-click consumes one callback rather than the held press

```mermaid
flowchart TD
    FRAME["Evaluate an unheld hand each active frame<br/>graph_scene.gd:3673,3684,3701"] --> EMIT["Active trigger emits node_grabbed<br/>interaction.rs:172,180,181"]
    EMIT --> DOUBLE["Second same-id callback opens document<br/>graph_scene.gd:4952,4954"]
    DOUBLE --> RESET["Clear last-click id and return<br/>graph_scene.gd:4953,4955"]
    RESET --> NEXT["Next active callback can start a grab<br/>graph_scene.gd:4956,4961"]
```

**What it shows.** Replaying one more callback after document opening starts a hold in the fixture. **Why it is this way.** The callback has per-hand click history but no consumed-press latch; a real HUD may intercept some subsequent rays, which the fixture does not model.

## VC-53.4 Capture reach exceeds the retained hold depth

```mermaid
flowchart TD
    RAY["Capture accepts forward hits through 30 m<br/>interaction.rs:14,60"] --> START["Catch stores depth clamped to 0.2 through 6 m<br/>graph_scene.gd:3852,3858"]
    START --> DRIVE["First drive uses the clamped depth<br/>graph_scene.gd:3874,3877"]
    DRIVE --> TARGET["Target and outbound update use that new point<br/>graph_scene.gd:3878,3881"]
```

**What it shows.** Unit observations accept a 10 m capture and convert its first hold update to 6 m, with no hand movement. **Why it is this way.** Capture reach and hand attachment have separate limits. The 1 m and 6 m controls retain their depth.

## VC-53.5 Hold admission and containment have distinct exceptions

```mermaid
flowchart TD
    HELD["Held node blocks a new box grip<br/>graph_scene.gd:647,648"] --> REVERSE["Node-start lock checks transition and ease only<br/>graph_scene.gd:3725,3728,3833"]
    REVERSE --> LOCAL["Local hold written before send result<br/>graph_scene.gd:3838,3859"]
    LOCAL --> SEND["Connection can return false<br/>backend_connection.gd:282-283,285"]
    E7["E7 states no node centre outside box<br/>XR-client.md:1168"] --> EXCLUDE["Fit excludes held ids to prevent feedback runaway<br/>connection_volume_source.gd:47,50,206"]
```

**What it shows.** Reverse acquisition and send refusal are separate admission boundaries; fit containment is also qualified by intentional exclusions. **Why it is this way.** These guards were added at different entry points, while excluding held outliers avoids shrinking the content repeatedly beneath the hand.

**Tension:** E7's universal wording needs to account for the intentionally excluded held nodes (`XR-client.md:1168`; `connection_volume_source.gd:47,206`).

## VC-53.6 Two headset pin sets survive with readers but no writer

```mermaid
flowchart TD
    END["Grab end sends drag end and records no pin<br/>graph_scene.gd:3885,3888,3895"] --> CONN["Connection drag end records nothing<br/>backend_connection.gd:304,305"]
    SCENE["Scene _pinned_ids: count, HUD list, CSV, Unpin All<br/>graph_scene.gd:966,2514,2527,2575,2805"] --> ACK["Removed only on unpin ack or capability loss<br/>graph_scene.gd:4578,2798"]
    CONNSET["Connection _pinned: read, unpinned, cleared on close<br/>backend_connection.gd:52,324,331,333"] --> ACK2["Erased by send_unpin<br/>backend_connection.gd:319"]
    END -.-> D1["DEBT: no path adds to either set<br/>backend_connection.gd:49,50,51"]
    SCENE -.-> D2["DRIFT: comment still says drag-end adds<br/>graph_scene.gd:963,964"]
```

**What it shows.** Unpin All, the HUD count and connection teardown still iterate sets that nothing fills, so after ADR-2172 F3 they are always empty. **Why it is this way.** F3 removed the writer, and the client-side Pin sender that would fill the sets is a recorded follow-up (ADR-2172 F5).

**Drift:** the `_pinned_ids` comment says drag-end adds (`graph_scene.gd:963`), but `_end_grab` records nothing (`graph_scene.gd:3885`).
