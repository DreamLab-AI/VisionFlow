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
  - ../project/xr-client/scripts/frame_budget_pass.gd
verified_commit: {visionclaw: 531df5ef74453a2ed320bff968c203a9b91faed7}
---

## For developers

The [portal audit](../../estate-review/2026-10-31-portals-and-federation.md) records 41 existing logic tests and seven adversarial/control cases on isolated Godot 4.6.1. Those runs remain historical. The [grab and portal refresh](../../estate-review/2026-11-01-xr-grab-boundaries.md) checks the current source delta; seven native scene tests and headset acceptance remain outside the portal fixture.

## For the business

Entering a volume changes the local viewing arrangement. It grants no remote-data permission. State restoration works in the tested cases; physical-distance hysteresis and the scope of comfort promises need correction.

## VC-47.1 Entry snapshots the arrangement and chooses a transition

```mermaid
flowchart TD
    ENTER["Refuse native, unavailable target or transition<br/>space_portals.gd:284"] --> SNAP["First entry snapshots arrangement and locks layout<br/>space_portals.gd:308,320"]
    SNAP --> CHOICE["Reduced motion, swap or transformed arrangement chooses dim<br/>space_portals.gd:322"]
    CHOICE --> ROOM["Apply bounded room and lay out real portal boxes<br/>space_portals.gd:415,430,433"]
    ROOM --> LEAVE["Leave goes through dim then restores snapshot<br/>space_portals.gd:338,471"]
    LEAVE --> RING["Unlock layout and snap visible content fit<br/>space_portals.gd:477,480"]
```

**What it shows.** Enter/swap/leave and exact restoration have passing logic tests. **Why it is this way.** The room is a bounded-volume layout state, with a saved ring arrangement; it is not a native immersive mode or a remote room admission.

## VC-47.2 World points become local distances before a metre-labelled threshold

```mermaid
flowchart TD
    HAND["Two-hand gesture scales the whole arrangement<br/>graph_scene.gd:3062,3097"] --> INV["Inverse global transform maps head into portal coordinates<br/>space_portals.gd:596,597"]
    INV --> DIST["Return local excess-vector length<br/>space_portals.gd:598,599"]
    DIST --> LIMIT["Compare directly with REARM_DISTANCE 0.3<br/>space_portals.gd:51,617"]
    LIMIT --> ARM["Re-arm when clear of every portal<br/>space_portals.gd:620,621"]
    ARM --> DWELL["Inside test then 0.25 second dwell enters target<br/>space_portals.gd:627,637,638,642"]
```

**What it shows.** Half scale re-arms at a physical 0.2 m gap; double scale remains disarmed at 0.4 m. Unit-scale controls pass. **Why it is this way.** Distance is measured in local units after inverse scaling, making the physical threshold 0.3 times the arrangement scale.

**Tension:** The documented re-arm distance is in metres, while the implementation compares a local-coordinate length (`space_portals.gd:599,617`).

## VC-47.3 Resize detection and membrane capacity have explicit boundaries

```mermaid
flowchart TD
    API["Runtime bounds apply immediately<br/>spatial_volume.gd:248,251"] --> SIG["Portal tick detects changed bounds signature<br/>space_portals.gd:487,551,552"]
    SIG --> DIM["Reduced motion starts a dim for ring repositioning<br/>space_portals.gd:553,554"]
    ADD["Arrangement accepts unique volume ids<br/>volume_arrangement.gd:75,79,80"] --> IDS["List every visible open non-room volume<br/>space_portals.gd:395"]
    IDS --> MM["Draw at most eight membrane instances<br/>space_portals.gd:64,566,567"]
    MM --> GAP["Scene budget collector has no portal-layer count<br/>graph_scene.gd:3149,3180,3183"]
```

**What it shows.** The API probe changes bounds before the dim begins; nine admitted portal boxes get eight membranes. **Why it is this way.** Detection follows mutation, and the documented eight-portal capacity is enforced only for membrane instances. The present scene has three volumes; the ninth-box case tests an extension boundary.

**Debt:** Portal membranes and fade meshes are absent from the scene's explicit draw-call sum; primitive totals are measured separately (`graph_scene.gd:3180,3181`).
