---
id: VT-05
title: Effects and force models
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
adrs: [visiontouch-rs:ADR-0002]
sources:
  - ../visiontouch-rs/crates/effects/src/lib.rs
  - ../visiontouch-rs/apps/calibration/src/protocol.rs
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/crates/device/src/lib.rs
  - ../visiontouch-rs/docs/adr/ADR-0004-reset-pose-registration.md
verified_commit: a9e52b6154b44fb7afc158230c4ec26d4213c015
---

## For developers

Source review and qualifications: [1 November audit](../../estate-review/2026-11-01-visiontouch-software-boundaries.md). The source pins are retained; this review changes claims, not device behaviour.

`visiontouch-effects` is a hardware-free library of seven force models in model metres, seconds and newtons. A caller owns an `Effect`, feeds it the pen position, velocity and an interval, and gets back an unclamped `Frame` (force, contact proxy, spring-demo bodies, balloon state). Stiffness and damping come from one sanitised `Tuning`, with 150 N/m as the reference for several gains. Cube, plane and balloon also contain a damping term independent of stiffness, so their total force is not generally proportional to stiffness. The crate never limits force: the guard in VT-03 does. The Studio constructs a fresh effect for each effect attempt (apps/calibration/src/protocol.rs:1359-1361).

## For the business

These are the test objects the owner uses to judge whether the arm feels right: a hard cube, a flat plane, a water ball, a balloon that pops, three bobbing springs, a thick treacle field and a lattice of soft clicks. They are simulated entirely in software so they can be tuned and replayed without touching the arm. The numbers are commanded strengths from a model, not a promise of what the arm delivers, and the library cannot cause any push by itself.

## VT-05.1 What an effect call returns, and what it refuses

```mermaid
sequenceDiagram
    autonumber
    participant CL as caller
    participant NW as Effect::new<br/>crates/effects/src/lib.rs:207
    participant ST as Effect::step<br/>crates/effects/src/lib.rs:231
    participant TU as Tuning::sanitised<br/>crates/effects/src/lib.rs:80
    participant GD as ForceGuard<br/>crates/device/src/lib.rs:348
    CL->>NW: kind, radius, tuning
    NW->>TU: clamp stiffness 20 to 1500 and damping 0 to 100, NaN falls back<br/>(crates/effects/src/lib.rs:80-95)
    NW-->>CL: radius clamped to 0.2 mm to 100 mm, 20 mm default if not finite<br/>(crates/effects/src/lib.rs:207-212)
    CL->>ST: position, velocity, dt
    ST->>ST: non-finite input or dt outside 0 to 25 ms gives an all-zero frame and no<br/>state change<br/>(crates/effects/src/lib.rs:232-234)
    ST-->>CL: Frame with unclamped force and a contact proxy<br/>(crates/effects/src/lib.rs:169-181)
    CL->>GD: force becomes a Demand, limited there<br/>(crates/device/src/lib.rs:420)
    Note over ST: INVARIANT: pure simulation, never opens hardware or arms motors<br/>crates/effects/src/lib.rs:3-5
    Note over ST: DEBT: finite but extreme velocity can overflow the Water force<br/>calculation<br/>crates/effects/src/lib.rs:232 and crates/effects/src/lib.rs:263
    Note over ST,GD: INVARIANT: Frame force is unclamped and the hardware layer must limit it<br/>crates/effects/src/lib.rs:167
    Note over NW: DOC-DRIFT: positions are said to be relative to the calibrated centre<br/>crates/effects/src/lib.rs:6 but the model frame is the registered reset<br/>pose, see docs/adr/ADR-0004-reset-pose-registration.md
```

An invalid step returns a zero frame the caller should treat as a fault, not a quiet success.

## VT-05.2 Seven models and how stiffness reaches each

```mermaid
flowchart TB
    TU["Tuning k and b, default 150 N/m and 4 N·s/m<br/>crates/effects/src/lib.rs:62-77"]
    SC["s = k / 150 scales fixed constants<br/>crates/effects/src/lib.rs:240"]
    CU["Cube: spring on the nearest face plus inward damping<br/>crates/effects/src/lib.rs:245-254"]
    PL["Plane: spring below Y = 0 plus damping<br/>crates/effects/src/lib.rs:255-260"]
    BA["Balloon: spring plus quadratic term, uses k only through s<br/>crates/effects/src/lib.rs:266-281"]
    WA["Water: drag and small buoyancy by immersion<br/>crates/effects/src/lib.rs:261-265"]
    VI["Viscous: minus 8 s times velocity<br/>crates/effects/src/lib.rs:282"]
    DE["Detents: sine wells, spacing 0.6 r, plus damping<br/>crates/effects/src/lib.rs:283-288"]
    SP["Springs: three bodies, contact reaction scaled by s<br/>crates/effects/src/lib.rs:289-346"]
    OUT["Frame force, unclamped<br/>crates/effects/src/lib.rs:347"]
    TU --> SC
    TU --> CU
    TU --> PL
    SC --> BA
    SC --> WA
    SC --> VI
    SC --> DE
    SC --> SP
    CU --> OUT
    PL --> OUT
    BA --> OUT
    WA --> OUT
    VI --> OUT
    DE --> OUT
    SP --> OUT
    N1["DRIFT: total cube, plane and balloon force includes damping independent of stiffness<br/>crates/effects/src/lib.rs:252 and crates/effects/src/lib.rs:278"]
    N2["DOC-DRIFT: Tuning docs say the balloon uses k directly<br/>crates/effects/src/lib.rs:64-65 while the code scales it through s at crates/effects/src/lib.rs:277"]
    SC --- N1
    BA --- N2
```

Cube and plane use stiffness and damping directly. Water, viscosity, detents and the spring-demo pen reaction scale by the ratio. Balloon scales its spring term but keeps a separate damping term. The balloon and the doc comment disagree on which group it belongs to, though the result is the same at the default.

## VT-05.3 Contact proxy for the cube and the plane

```mermaid
flowchart TB
    IN["pen position p inside cube half-width r?<br/>crates/effects/src/lib.rs:246"]
    FACE["nearest face is the axis with least remaining depth<br/>crates/effects/src/lib.rs:247-249"]
    SNAP["proxy snaps onto that face<br/>crates/effects/src/lib.rs:251"]
    FC["force along the face normal, k times depth plus b times inward speed<br/>crates/effects/src/lib.rs:252"]
    PL["plane: p.y below 0, proxy y set to 0, force minus k p.y plus damping<br/>crates/effects/src/lib.rs:256-259"]
    FR["Frame proxy differs from the measured pen<br/>crates/effects/src/lib.rs:172-174"]
    IN --> FACE --> SNAP --> FC
    PL --> FR
    SNAP --> FR
    N1["INVARIANT: the proxy is the nearest surface point, so the marker stays on the face while the pen shows real penetration<br/>crates/effects/src/lib.rs:6-7 and crates/effects/src/lib.rs:143"]
    FR --- N1
```

Damping acts only while the pen moves deeper, so it never pulls the pen back in on exit.

## VT-05.4 The balloon latch

```mermaid
stateDiagram-v2
    [*] --> Intact: new effect (crates/effects/src/lib.rs:207)
    Intact --> Intact: indentation below 35 percent gives spring force (crates/effects/src/lib.rs:272-280)
    Intact --> Popped: deformation reaches 0.35 (crates/effects/src/lib.rs:269-271)
    Popped --> Popped: zero force for any later input (crates/effects/src/lib.rs:272)
    Popped --> [*]: recreate the Effect (crates/effects/src/lib.rs:179-180)
    note right of Popped
        INVARIANT: reinflation requires a new Effect, created for each Studio effect attempt
        protocol.rs:1359-1361 and crates/effects/src/lib.rs:391-405
    end note
```

A new Studio effect attempt recreates the effect. Releasing ends the current held attempt; a fresh attempt starts with an intact balloon.

## VT-05.5 Springs demo: sub-stepped three-body integration

```mermaid
sequenceDiagram
    autonumber
    participant ST as Effect::step Springs<br/>crates/effects/src/lib.rs:289
    participant BD as three bodies<br/>crates/effects/src/lib.rs:293
    participant PN as pen contact<br/>crates/effects/src/lib.rs:321
    ST->>ST: sub-steps of at most 1 ms<br/>(crates/effects/src/lib.rs:290-292)
    ST->>BD: anchor attraction with softened core and light drag<br/>(crates/effects/src/lib.rs:294-300)
    ST->>BD: pair springs toward an equilateral rest length<br/>(crates/effects/src/lib.rs:302-318)
    ST->>PN: contact within 0.28 r, spring 100 and damping 1.5<br/>(crates/effects/src/lib.rs:324-335)
    PN-->>BD: contact removed from the body, added to the pen reaction<br/>(crates/effects/src/lib.rs:336-337)
    ST->>BD: velocity limited to 0.2 m per s then positions advance<br/>(crates/effects/src/lib.rs:339-341)
    ST-->>ST: reaction summed over sub-steps, scaled by s<br/>(crates/effects/src/lib.rs:343)
    Note over ST: INVARIANT: no-contact test checks body positions and symmetry under 2 ms<br/>versus two 1 ms steps<br/>crates/effects/src/lib.rs:407-415
    Note over BD: DEBT: body mass is an inline 0.02 and every gain is a literal, no named<br/>constants<br/>crates/effects/src/lib.rs:340
```

Contact is subtracted from the body, then the pen reaction is averaged over substeps and multiplied by stiffness / 150. Changing stiffness scales the returned pen force without changing body motion; equal returned and body-contact magnitudes therefore hold only at the reference scale, before device limiting.
