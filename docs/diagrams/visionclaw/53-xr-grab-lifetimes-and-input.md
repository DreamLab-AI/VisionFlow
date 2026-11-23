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
verified_commit: {visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e}
---

## For developers

The [assessment](../../estate-review/2026-11-01-xr-grab-boundaries.md) traces ADR-2172 and refreshes the portal amendment. Its method probes use exact production function bodies on explicit spies; they are not a full scene, controller or backend run.

## For the business

Two hands can address separate nodes, but reliable manipulation also needs a clear end to each gesture. A data refresh now ends every hold cleanly; opening a document and catching a distant node still expose gaps that ordinary single-step demonstrations miss.

## VC-53.1 Per-hand holds route through their volume and share one id domain

```mermaid
flowchart TD
    VOL["Knowledge and ontology share VisionClaw<br/>graph_scene.gd:91,92"] --> GATE["Capability and entered-space checks,<br/>the merged-space guest also grabbable<br/>graph_scene.gd:297,320,321,3740"]
    GATE --> PACK["Merge each volume's own drawn candidates<br/>graph_scene.gd:3769,3779,3789"]
    PACK --> HOLD["Record hand id, volume and exclusion set<br/>graph_scene.gd:3843,3844,3845"]
    HOLD --> DRIVE["Invert Content transform and send on volume connection<br/>graph_scene.gd:3882,3884,3886"]
    HOLD --> RELEASE["End one hand and update remaining exclusions<br/>graph_scene.gd:3893,3903,3910"]
```

**What it shows.** Per-hand state and per-volume transforms coexist with a shared draggable connection. **Why it is this way.** The class-filtered views share one graph and server id domain; this is not a general multi-server identity scheme.

## VC-53.2 A topology change now releases the holds of that connection

```mermaid
flowchart TD
    GEN["Topology bumps connection generation, then signals<br/>backend_connection.gd:450,451,452"] --> RELEASE["Release every hold on a volume of that connection first<br/>graph_scene.gd:4667,4668,4674,4675,4676"]
    RELEASE --> END["_end_grab per hand sends drag end and clears the hand<br/>graph_scene.gd:162,164,165,3900,3903"]
    END --> SOURCE["Invalidate volume sources<br/>graph_scene.gd:4677,4679"]
    SOURCE --> CLEAR["Clear selection and stale links<br/>graph_scene.gd:4683,4686,4687"]
    SWAP["Manual reconnect releases holds before closing the old socket<br/>graph_scene.gd:3020-3028"] --> END
    DROP["VisionClaw disconnect releases every hold<br/>graph_scene.gd:4641,4644,4645"] --> END
    TEST["GUT: both hands released before ids are reused<br/>test_grab_client.gd:311,316,317,318"] --> RELEASE
```

**What it shows.** Since `6b811f7a1` the topology callback ends both hands' holds before invalidating sources and selection, and a VisionClaw disconnect does the same. The current reconnect path also releases both hands before replacing the socket; the source regression asserts that each end precedes close (`xr-client/tests/unit/test_grab_client.gd:743-755`). This review did not rerun GUT. **Why it is this way.** Hand state carries an id and a volume but no generation (`xr-client/scripts/graph_scene.gd:835-836`), so the release has to happen at the boundary rather than at each drive.
**Invariant:** no hand holds a wire id across a topology generation of its connection (`xr-client/scripts/graph_scene.gd:4674-4676`), pinned by `xr-client/tests/unit/test_grab_client.gd:311-318`.

## VC-53.3 Double-click consumes one callback rather than the held press

```mermaid
flowchart TD
    FRAME["Evaluate an unheld hand each active frame<br/>graph_scene.gd:3678,3689,3706"] --> EMIT["Active trigger emits node_grabbed<br/>interaction.rs:172,180,181"]
    EMIT --> DOUBLE["Second same-id callback opens document<br/>graph_scene.gd:4841,4843"]
    DOUBLE --> RESET["Clear last-click id and return<br/>graph_scene.gd:4842,4844"]
    RESET --> NEXT["Next active callback can start a grab<br/>graph_scene.gd:4845,4850"]
```

**What it shows.** Replaying one more callback after document opening starts a hold in the fixture. **Why it is this way.** The callback has per-hand click history but no consumed-press latch; a real HUD may intercept some subsequent rays, which the fixture does not model.

## VC-53.4 Capture reach exceeds the retained hold depth

```mermaid
flowchart TD
    RAY["Capture accepts forward hits through 30 m<br/>interaction.rs:14,60"] --> START["Catch stores depth clamped to 0.2 through 6 m<br/>graph_scene.gd:3857,3863"]
    START --> DRIVE["First drive uses the clamped depth<br/>graph_scene.gd:3879,3882"]
    DRIVE --> TARGET["Target and outbound update use that new point<br/>graph_scene.gd:3883,3886"]
```

**What it shows.** Unit observations accept a 10 m capture and convert its first hold update to 6 m, with no hand movement. **Why it is this way.** Capture reach and hand attachment have separate limits. The 1 m and 6 m controls retain their depth.

## VC-53.5 Hold admission and containment have distinct exceptions

```mermaid
flowchart TD
    HELD["Held node blocks a new box grip<br/>graph_scene.gd:643,644"] --> REVERSE["Node-start lock checks transition and ease only<br/>graph_scene.gd:3730,3733,3838"]
    REVERSE --> LOCAL["Local hold written before send result<br/>graph_scene.gd:3843,3864"]
    LOCAL --> SEND["Connection can return false<br/>backend_connection.gd:331-332,334"]
    E7["E7 states no node centre outside box<br/>XR-client.md:1188"] --> EXCLUDE["Fit excludes held ids to prevent feedback runaway<br/>connection_volume_source.gd:47,50,206"]
```

**What it shows.** Reverse acquisition and send refusal are separate admission boundaries; fit containment is also qualified by intentional exclusions. **Why it is this way.** These guards were added at different entry points, while excluding held outliers avoids shrinking the content repeatedly beneath the hand.

**Tension:** E7's universal wording needs to account for the intentionally excluded held nodes (`XR-client.md:1188`; `connection_volume_source.gd:47,206`).

## VC-53.6 Two headset pin sets survive with readers but no writer

```mermaid
flowchart TD
    END["Grab end sends drag end and records no pin<br/>graph_scene.gd:3890,3893,3900"] --> CONN["Connection drag end records nothing<br/>backend_connection.gd:353,354"]
    SCENE["Scene _pinned_ids: count, HUD list, CSV, Unpin All<br/>graph_scene.gd:952,2525,2538,2586,2816"] --> ACK["Removed only on unpin ack or capability loss<br/>graph_scene.gd:4583,2809"]
    CONNSET["Connection _pinned: read, unpinned, cleared on close<br/>backend_connection.gd:61,373,380,382"] --> ACK2["Erased by send_unpin<br/>backend_connection.gd:368"]
    END -.-> D1["DEBT: no path adds to either set<br/>backend_connection.gd:58,59,60"]
    SCENE -.-> D2["DRIFT: comment still says drag-end adds<br/>graph_scene.gd:949,950"]
```

**What it shows.** Unpin All, the HUD count and connection teardown still iterate sets that nothing fills, so after ADR-2172 F3 they are always empty. **Why it is this way.** F3 removed the writer, and the client-side Pin sender that would fill the sets is a recorded follow-up (ADR-2172 F5).

**Drift:** the `_pinned_ids` comment says drag-end adds (`graph_scene.gd:949`), but `_end_grab` records nothing (`graph_scene.gd:3890`).
