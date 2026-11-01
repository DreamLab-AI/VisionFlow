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
verified_commit: {visionclaw: 531df5ef74453a2ed320bff968c203a9b91faed7}
---

## For developers

The [assessment](../../estate-review/2026-11-01-xr-grab-boundaries.md) traces ADR-2172 and refreshes the portal amendment. Its method probes use exact production function bodies on explicit spies; they are not a full scene, controller or backend run.

## For the business

Two hands can address separate nodes, but reliable manipulation also needs a clear end to each gesture. Refreshing data, opening a document and catching a distant node expose gaps that ordinary single-step demonstrations miss.

## VC-53.1 Per-hand holds route through their volume and share one id domain

```mermaid
flowchart TD
    VOL["Knowledge and ontology share VisionClaw<br/>graph_scene.gd:93,94"] --> GATE["Capability and entered-space checks<br/>graph_scene.gd:298,321,3302"]
    GATE --> PACK["Merge each volume's own drawn candidates<br/>graph_scene.gd:3331,3341,3351"]
    PACK --> HOLD["Record hand id, volume and exclusion set<br/>graph_scene.gd:3405,3406,3407"]
    HOLD --> DRIVE["Invert Content transform and send on volume connection<br/>graph_scene.gd:3444,3446,3448"]
    HOLD --> RELEASE["End one hand and update remaining exclusions<br/>graph_scene.gd:3455,3465,3472"]
```

**What it shows.** Per-hand state and per-volume transforms coexist with a shared draggable connection. **Why it is this way.** The class-filtered views share one graph and server id domain; this is not a general multi-server identity scheme.

## VC-53.2 Topology invalidates selections but leaves held ids alive

```mermaid
flowchart TD
    GEN["Topology bumps connection generation<br/>backend_connection.gd:359,360,361"] --> SOURCE["Invalidate volume sources<br/>graph_scene.gd:4205,4208"]
    SOURCE --> CLEAR["Clear selection and stale links<br/>graph_scene.gd:4212,4215,4216"]
    HOLD["Hand state has id and volume, no generation<br/>graph_scene.gd:841,842"] --> UPDATE["Drive uses retained id without generation check<br/>graph_scene.gd:3437,3446,3448"]
    CLEAR --> UPDATE
```

**What it shows.** The extracted callback clears selection while a subsequent drive still sends the held id. **Why it is this way.** Selection lifetime is explicitly reset; grab lifetime is not part of that callback.

**Debt:** Held ids can survive the generation boundary even though the connection documents generation-scoped identity (`backend_connection.gd:18,19`; `graph_scene.gd:4205`).

## VC-53.3 Double-click consumes one callback rather than the held press

```mermaid
flowchart TD
    FRAME["Evaluate an unheld hand each active frame<br/>graph_scene.gd:3240,3251,3268"] --> EMIT["Active trigger emits node_grabbed<br/>interaction.rs:172,180,181"]
    EMIT --> DOUBLE["Second same-id callback opens document<br/>graph_scene.gd:4485,4487"]
    DOUBLE --> RESET["Clear last-click id and return<br/>graph_scene.gd:4486,4488"]
    RESET --> NEXT["Next active callback can start a grab<br/>graph_scene.gd:4489,4494"]
```

**What it shows.** Replaying one more callback after document opening starts a hold in the fixture. **Why it is this way.** The callback has per-hand click history but no consumed-press latch; a real HUD may intercept some subsequent rays, which the fixture does not model.

## VC-53.4 Capture reach exceeds the retained hold depth

```mermaid
flowchart TD
    RAY["Capture accepts forward hits through 30 m<br/>interaction.rs:14,60"] --> START["Catch stores depth clamped to 0.2 through 6 m<br/>graph_scene.gd:3419,3425"]
    START --> DRIVE["First drive uses the clamped depth<br/>graph_scene.gd:3441,3444"]
    DRIVE --> TARGET["Target and outbound update use that new point<br/>graph_scene.gd:3445,3448"]
```

**What it shows.** Unit observations accept a 10 m capture and convert its first hold update to 6 m, with no hand movement. **Why it is this way.** Capture reach and hand attachment have separate limits. The 1 m and 6 m controls retain their depth.

## VC-53.5 Hold admission and containment have distinct exceptions

```mermaid
flowchart TD
    HELD["Held node blocks a new box grip<br/>graph_scene.gd:642,643"] --> REVERSE["Node-start lock checks transition and ease only<br/>graph_scene.gd:3292,3295,3400"]
    REVERSE --> LOCAL["Local hold written before send result<br/>graph_scene.gd:3405,3426"]
    LOCAL --> SEND["Connection can return false<br/>backend_connection.gd:250,253"]
    E7["E7 states no node centre outside box<br/>XR-client.md:940"] --> EXCLUDE["Fit excludes held ids to prevent feedback runaway<br/>connection_volume_source.gd:47,50,186"]
```

**What it shows.** Reverse acquisition and send refusal are separate admission boundaries; fit containment is also qualified by intentional exclusions. **Why it is this way.** These guards were added at different entry points, while excluding held outliers avoids shrinking the content repeatedly beneath the hand.

**Tension:** E7's universal wording needs to account for the intentionally excluded held nodes (`XR-client.md:940`; `connection_volume_source.gd:47,186`).
