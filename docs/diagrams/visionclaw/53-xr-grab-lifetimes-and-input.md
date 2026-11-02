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
verified_commit: {visionclaw: 86cdd7635582}
---

## For developers

The [assessment](../../estate-review/2026-11-01-xr-grab-boundaries.md) traces ADR-2172 and refreshes the portal amendment. Its method probes use exact production function bodies on explicit spies; they are not a full scene, controller or backend run.

## For the business

Two hands can address separate nodes, but reliable manipulation also needs a clear end to each gesture. Refreshing data, opening a document and catching a distant node expose gaps that ordinary single-step demonstrations miss.

## VC-53.1 Per-hand holds route through their volume and share one id domain

```mermaid
flowchart TD
    VOL["Knowledge and ontology share VisionClaw<br/>graph_scene.gd:94,95"] --> GATE["Capability and entered-space checks<br/>graph_scene.gd:299,322,3306"]
    GATE --> PACK["Merge each volume's own drawn candidates<br/>graph_scene.gd:3335,3345,3355"]
    PACK --> HOLD["Record hand id, volume and exclusion set<br/>graph_scene.gd:3409,3410,3411"]
    HOLD --> DRIVE["Invert Content transform and send on volume connection<br/>graph_scene.gd:3448,3450,3452"]
    HOLD --> RELEASE["End one hand and update remaining exclusions<br/>graph_scene.gd:3459,3469,3476"]
```

**What it shows.** Per-hand state and per-volume transforms coexist with a shared draggable connection. **Why it is this way.** The class-filtered views share one graph and server id domain; this is not a general multi-server identity scheme.

## VC-53.2 Topology invalidates selections but leaves held ids alive

```mermaid
flowchart TD
    GEN["Topology bumps connection generation<br/>backend_connection.gd:359,360,361"] --> SOURCE["Invalidate volume sources<br/>graph_scene.gd:4224,4227"]
    SOURCE --> CLEAR["Clear selection and stale links<br/>graph_scene.gd:4231,4234,4235"]
    HOLD["Hand state has id and volume, no generation<br/>graph_scene.gd:845,846"] --> UPDATE["Drive uses retained id without generation check<br/>graph_scene.gd:3441,3450,3452"]
    CLEAR --> UPDATE
```

**What it shows.** The extracted callback clears selection while a subsequent drive still sends the held id. **Why it is this way.** Selection lifetime is explicitly reset; grab lifetime is not part of that callback.

**Debt:** Held ids can survive the generation boundary even though the connection documents generation-scoped identity (`backend_connection.gd:18,19`; `graph_scene.gd:4224`).

## VC-53.3 Double-click consumes one callback rather than the held press

```mermaid
flowchart TD
    FRAME["Evaluate an unheld hand each active frame<br/>graph_scene.gd:3244,3255,3272"] --> EMIT["Active trigger emits node_grabbed<br/>interaction.rs:172,180,181"]
    EMIT --> DOUBLE["Second same-id callback opens document<br/>graph_scene.gd:4504,4506"]
    DOUBLE --> RESET["Clear last-click id and return<br/>graph_scene.gd:4505,4507"]
    RESET --> NEXT["Next active callback can start a grab<br/>graph_scene.gd:4508,4513"]
```

**What it shows.** Replaying one more callback after document opening starts a hold in the fixture. **Why it is this way.** The callback has per-hand click history but no consumed-press latch; a real HUD may intercept some subsequent rays, which the fixture does not model.

## VC-53.4 Capture reach exceeds the retained hold depth

```mermaid
flowchart TD
    RAY["Capture accepts forward hits through 30 m<br/>interaction.rs:14,60"] --> START["Catch stores depth clamped to 0.2 through 6 m<br/>graph_scene.gd:3423,3429"]
    START --> DRIVE["First drive uses the clamped depth<br/>graph_scene.gd:3445,3448"]
    DRIVE --> TARGET["Target and outbound update use that new point<br/>graph_scene.gd:3449,3452"]
```

**What it shows.** Unit observations accept a 10 m capture and convert its first hold update to 6 m, with no hand movement. **Why it is this way.** Capture reach and hand attachment have separate limits. The 1 m and 6 m controls retain their depth.

## VC-53.5 Hold admission and containment have distinct exceptions

```mermaid
flowchart TD
    HELD["Held node blocks a new box grip<br/>graph_scene.gd:643,644"] --> REVERSE["Node-start lock checks transition and ease only<br/>graph_scene.gd:3296,3299,3404"]
    REVERSE --> LOCAL["Local hold written before send result<br/>graph_scene.gd:3409,3430"]
    LOCAL --> SEND["Connection can return false<br/>backend_connection.gd:250,253"]
    E7["E7 states no node centre outside box<br/>XR-client.md:949"] --> EXCLUDE["Fit excludes held ids to prevent feedback runaway<br/>connection_volume_source.gd:47,50,186"]
```

**What it shows.** Reverse acquisition and send refusal are separate admission boundaries; fit containment is also qualified by intentional exclusions. **Why it is this way.** These guards were added at different entry points, while excluding held outliers avoids shrinking the content repeatedly beneath the hand.

**Tension:** E7's universal wording needs to account for the intentionally excluded held nodes (`XR-client.md:949`; `connection_volume_source.gd:47,186`).

## VC-53.6 Two headset pin sets survive with readers but no writer

```mermaid
flowchart TD
    END["Grab end sends drag end and records no pin<br/>graph_scene.gd:3456,3459,3466"] --> CONN["Connection drag end records nothing<br/>backend_connection.gd:272,273"]
    SCENE["Scene _pinned_ids: count, HUD list, CSV, Unpin All<br/>graph_scene.gd:958,2092,2105,2153,2383"] --> ACK["Removed only on unpin ack or capability loss<br/>graph_scene.gd:4145,2376"]
    CONNSET["Connection _pinned: read, unpinned, cleared on close<br/>backend_connection.gd:52,292,299,301"] --> ACK2["Erased by send_unpin<br/>backend_connection.gd:287"]
    END -.-> D1["DEBT: no path adds to either set<br/>backend_connection.gd:49,50,51"]
    SCENE -.-> D2["DRIFT: comment still says drag-end adds<br/>graph_scene.gd:955,956"]
```

**What it shows.** Unpin All, the HUD count and connection teardown still iterate sets that nothing fills, so after ADR-2172 F3 they are always empty. **Why it is this way.** F3 removed the writer, and the client-side Pin sender that would fill the sets is a recorded follow-up (ADR-2172 F5).

**Drift:** the `_pinned_ids` comment says drag-end adds (`graph_scene.gd:955`), but `_end_grab` records nothing (`graph_scene.gd:3456`).
