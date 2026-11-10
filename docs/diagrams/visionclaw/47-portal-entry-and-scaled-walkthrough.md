---
id: VC-47
title: Portal entry, comfort transitions and scaled walk-through
area: visionclaw
governing:
  - ../project/docs/adr/ADR-2166-spatial-volumes-can-be-entered-the-others-become-portals.md
adrs: [ADR-2158, ADR-2166]
sources:
  - ../project/xr-client/scripts/space_portals.gd
  - ../project/xr-client/scripts/volume_arrangement.gd
  - ../project/xr-client/scripts/spatial_volume.gd
  - ../project/xr-client/scripts/graph_scene.gd
  - ../project/xr-client/tests/unit/test_space_portals.gd
  - ../project/docs/adr/ADR-2166-spatial-volumes-can-be-entered-the-others-become-portals.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9}
---

## For developers

The [portal audit](../../estate-review/2026-10-31-portals-and-federation.md) records 41 existing logic tests and seven adversarial/control cases on isolated Godot 4.6.1. Those runs remain historical. The [grab and portal refresh](../../estate-review/2026-11-01-xr-grab-boundaries.md) checks the current source delta; seven native scene tests and headset acceptance remain outside the portal fixture.

## For the business

Entering a volume changes the local viewing arrangement. It grants no remote-data permission. State restoration works in the tested cases; the walk-through hysteresis now scales with the space, and the scope of comfort promises still needs correction.

## VC-47.1 Entry snapshots the arrangement and chooses a transition

```mermaid
flowchart TD
    ENTER["Refuse native, unavailable target or transition<br/>space_portals.gd:375"] --> SNAP["First entry snapshots arrangement and locks layout<br/>space_portals.gd:399,411"]
    SNAP --> CHOICE["Reduced motion, swap or transformed arrangement chooses dim<br/>space_portals.gd:413"]
    CHOICE --> ROOM["Apply bounded room and lay out real portal boxes<br/>space_portals.gd:582,597,600"]
    ROOM --> LEAVE["Leave goes through dim then restores snapshot<br/>space_portals.gd:429,669"]
    LEAVE --> RING["Unlock layout and snap visible content fit<br/>space_portals.gd:675,685"]
```

**What it shows.** Enter/swap/leave and exact restoration have passing logic tests. **Why it is this way.** The room is a bounded-volume layout state, with a saved ring arrangement; it is not a native immersive mode or a remote room admission.

## VC-47.2 Re-arm distance is measured in metres against an arrangement-scaled margin

```mermaid
flowchart TD
    HAND["Two-hand gesture scales the whole arrangement<br/>graph_scene.gd:3491,3526"] --> MARGIN["Margin = REARM_DISTANCE 0.3 times the largest arrangement axis scale<br/>space_portals.gd:60,829,830"]
    INV["distance_to_box: inverse transform gives the local overshoot<br/>space_portals.gd:805,807,808"] --> WORLD["Per-axis world scale restored, result in metres<br/>space_portals.gd:811"]
    WORLD --> LIMIT["Head within the scaled margin of any portal keeps it disarmed<br/>space_portals.gd:831,832"]
    MARGIN --> LIMIT
    LIMIT --> ARM["Re-arm when clear of every portal<br/>space_portals.gd:834,835"]
    ARM --> DWELL["Inside test then 0.25 second dwell enters target<br/>space_portals.gd:841,851,852,856"]
    TEST["GUT: margin tracks arrangement scale, 0.03 m at scale 0.1<br/>test_space_portals.gd:523,530,536"] --> MARGIN
```

**What it shows.** Since `6b811f7a1` the distance is in world metres and the margin scales with the arrangement, so at scale 0.1 the head re-arms beyond 0.03 m; the 2026-10-31 half- and double-scale fixture predates the change. **Why it is this way.** ADR-2166's 2026-11-05 update closes the scale-relative finding by making the hysteresis proportional to how large the user has made the space (`docs/adr/ADR-2166-spatial-volumes-can-be-entered-the-others-become-portals.md:32`).
**Invariant:** the walk-through re-arm margin is `REARM_DISTANCE` times the arrangement's largest axis scale (`xr-client/scripts/space_portals.gd:830`), pinned by `xr-client/tests/unit/test_space_portals.gd:523-536`.

## VC-47.3 Resize detection and membrane capacity have explicit boundaries

```mermaid
flowchart TD
    API["Runtime bounds apply immediately<br/>spatial_volume.gd:269,272"] --> SIG["Portal tick detects changed bounds signature<br/>space_portals.gd:692,761,762"]
    SIG --> DIM["Reduced motion starts a dim for ring repositioning<br/>space_portals.gd:763,764"]
    ADD["Arrangement accepts unique volume ids<br/>volume_arrangement.gd:75,79,80"] --> IDS["List every visible open non-room volume<br/>space_portals.gd:562"]
    IDS --> MM["Draw at most eight membrane instances<br/>space_portals.gd:73,776,777"]
    MM --> GAP["Scene budget collector has no portal-layer count<br/>graph_scene.gd:3578,3613,3616"]
```

**What it shows.** The API probe changes bounds before the dim begins; nine admitted portal boxes get eight membranes. **Why it is this way.** Detection follows mutation, and the documented eight-portal capacity is enforced only for membrane instances. The present scene has three volumes; the ninth-box case tests an extension boundary.

**Debt:** Portal membranes and fade meshes are absent from the scene's explicit draw-call sum; primitive totals are measured separately (`graph_scene.gd:3613,3614`).
