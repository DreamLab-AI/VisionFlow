---
id: VT-04
title: Mass Lab servo-rate coupling
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/spec/11-relative-pointer.md
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/docs/adr/ADR-0002-layered-software-force-envelope.md
adrs: [visiontouch-rs:ADR-0002]
sources:
  - ../visiontouch-rs/apps/mass-lab/src/pointer.rs
  - ../visiontouch-rs/apps/mass-lab/src/physics.rs
  - ../visiontouch-rs/apps/mass-lab/src/main.rs
  - ../visiontouch-rs/crates/device/src/lib.rs
  - ../visiontouch-rs/docs/spec/11-relative-pointer.md
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/docs/adr/ADR-0002-layered-software-force-envelope.md
verified_commit: 7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d
---

## For developers

Mass Lab is a rigid-body sandbox where the pen drags blocks. The force path is deliberately split: the UI thread only publishes a coupling descriptor (proxy position, proxy velocity, stiffness, damping, issue time) and a worker thread recomputes the force from every fresh sample at servo rate, so output no longer depends on the frame rate. A descriptor older than 100 ms, an unregistered arm, a paused or non-real-time simulation, or feedback off all produce an explicit zero with a named reason. Mass Lab always inherits `ForceLimits::default` and never raises limits. The Calibration Studio, which builds higher limits, is not covered here.

## For the business

Mass Lab is the lab's proof that a person can feel a simulated object. It is also the first thing that could later drive a graph tool, so it is built to stop pushing the moment anything looks wrong: if the screen freezes for a tenth of a second the arm goes slack rather than tugging toward a stale picture, and the screen explains in words why force is off. The strength it uses is the small, already-proven default. What it feels like is a command, not a measured force, and the on-screen label says so.

## VT-04.1 One frame and one servo tick

```mermaid
sequenceDiagram
    autonumber
    participant UI as App send_coupling<br/>apps/mass-lab/src/main.rs:104
    participant PT as Pointer send_coupling<br/>apps/mass-lab/src/pointer.rs:590
    participant WK as worker loop<br/>apps/mass-lab/src/pointer.rs:256
    participant SV as service<br/>apps/mass-lab/src/pointer.rs:180
    participant DV as Device guard<br/>crates/device/src/lib.rs:512
    UI->>PT: proxy position and velocity only when running at 1.00x (apps/mass-lab/src/main.rs:105-110)
    PT->>WK: Command Coupling in registered metres with an issue time (apps/mass-lab/src/pointer.rs:594-608)
    WK->>WK: drain applies every queued coupling first so UI floods cannot starve sampling (apps/mass-lab/src/pointer.rs:225-229)
    WK->>SV: one service per loop then sleep 2 ms (apps/mass-lab/src/pointer.rs:342 and apps/mass-lab/src/pointer.rs:381)
    SV->>DV: sample first, then position (apps/mass-lab/src/pointer.rs:181-183)
    SV->>SV: servo tick computes the demand and mute reason (apps/mass-lab/src/pointer.rs:184)
    SV->>DV: request stamped with the evaluation time, so never older than one loop (apps/mass-lab/src/pointer.rs:185-187)
    Note over SV,DV: INVARIANT: one sample then at most one fresh request per iteration<br/>apps/mass-lab/src/pointer.rs:178-179
    Note over UI,WK: INVARIANT: the worker reads and services the device independently of rendering<br/>docs/spec/11-relative-pointer.md:41
    Note over WK,DV: DEBT: the device path is hard-coded to /dev/parport0<br/>apps/mass-lab/src/pointer.rs:324
```

The UI frame rate only affects how fresh the descriptor is. The force itself is recomputed from the newest position every loop.

## VT-04.2 Servo tick: mute ladder and the spring law

```mermaid
flowchart TB
    S0["Servo tick with an optional registered position<br/>apps/mass-lab/src/pointer.rs:99"]
    S1["position missing: Unregistered if feedback else FeedbackOff, velocity reset<br/>apps/mass-lab/src/pointer.rs:105-113"]
    S2["velocity low-pass, tau 10 ms hides single-count steps<br/>apps/mass-lab/src/pointer.rs:114-122"]
    S3["feedback off gives FeedbackOff<br/>apps/mass-lab/src/pointer.rs:124-126"]
    S4["coupling issued within 100 ms? else SceneStalled<br/>apps/mass-lab/src/pointer.rs:127-133"]
    S5["Demand Force of K (proxy - p) - B (v - v_proxy) with the pen clamped into the scene box<br/>apps/mass-lab/src/pointer.rs:74-86"]
    Z["Demand Force zero plus a MuteReason<br/>apps/mass-lab/src/pointer.rs:135-140"]
    S0 --> S1
    S0 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S1 --> Z
    S3 --> Z
    S4 --> Z
    N1["INVARIANT: a stalled UI cannot tether the pen to a frozen proxy, age limit is about six frames<br/>apps/mass-lab/src/pointer.rs:11-13"]
    N2["INVARIANT: a pen outside the scene box feels no spurious force, clamped axes take the proxy velocity<br/>apps/mass-lab/src/pointer.rs:77-82"]
    N3["DEBT: the FeedbackOff reason is not counted but every other mute increments muted_ticks, so a paused sim counts as stalled<br/>apps/mass-lab/src/pointer.rs:136-138"]
    S4 --- N1
    S5 --- N2
    Z --- N3
```

Defaults are 150 N/m stiffness and 1 N·s/m damping, with a slider range of 20 to 1500 N/m.

## VT-04.3 Two places decide why the pen is muted

```mermaid
flowchart TB
    subgraph UIPRED["UI predicate, first match wins<br/>apps/mass-lab/src/main.rs:88"]
        U1["FeedbackOff<br/>apps/mass-lab/src/main.rs:90"]
        U2["Unregistered<br/>apps/mass-lab/src/main.rs:92"]
        U3["NotRunning<br/>apps/mass-lab/src/main.rs:94"]
        U4["NotRealTime, within half a percent of 1.00x<br/>apps/mass-lab/src/main.rs:96"]
        U5["SceneStalled, no sample for 500 ms<br/>apps/mass-lab/src/main.rs:98"]
        U6["else the reason the worker last reported<br/>apps/mass-lab/src/main.rs:100"]
        U1 --> U2 --> U3 --> U4 --> U5 --> U6
    end
    subgraph WKR["Worker emits only three reasons<br/>apps/mass-lab/src/pointer.rs:99"]
        W1["Unregistered or FeedbackOff<br/>apps/mass-lab/src/pointer.rs:108"]
        W2["FeedbackOff<br/>apps/mass-lab/src/pointer.rs:125"]
        W3["SceneStalled<br/>apps/mass-lab/src/pointer.rs:132"]
    end
    SEND["paused or not 1.00x sends coupling None, so the worker sees a stalled scene<br/>apps/mass-lab/src/main.rs:105"]
    LBL["labelled in the sidebar beside Pen force<br/>apps/mass-lab/src/main.rs:343"]
    SEND --> W3
    U6 --> LBL
    W3 -. "worker_muted" .-> U6
    N1["DIVERGENCE: NotRunning and NotRealTime exist only in the UI predicate, the same pause is counted by the worker as SceneStalled<br/>apps/mass-lab/src/main.rs:94-96 vs apps/mass-lab/src/pointer.rs:129-132"]
    LBL --- N1
```

The operator sees one reason; the counter beside it, `muted servo ticks`, counts by the worker's coarser view.

## VT-04.4 Scene proxy and the device coupling are two springs

```mermaid
flowchart TB
    TGT["pen target in the scene from RelativeMotion<br/>apps/mass-lab/src/pointer.rs:424"]
    PX["proxy body chases the target: 600 N/m, 60 N·s/m, 250 N cap, target speed 8 m/s<br/>apps/mass-lab/src/physics.rs:342-350"]
    POS["pointer_position and pointer_velocity<br/>apps/mass-lab/src/physics.rs:300-306"]
    DESC["Coupling in registered model metres, scene to device map x10000 times sensitivity<br/>apps/mass-lab/src/pointer.rs:415-456"]
    SPR["device spring K 150 N/m B 1 N·s/m at servo rate<br/>apps/mass-lab/src/pointer.rs:15-19"]
    GUARD["ForceGuard 0.5 N, 96 counts<br/>apps/mass-lab/src/pointer.rs:277"]
    REAC["reaction_force = minus the proxy force<br/>apps/mass-lab/src/physics.rs:353"]
    TGT --> PX
    PX --> POS
    POS --> DESC
    DESC --> SPR
    SPR --> GUARD
    PX --> REAC
    N1["DOC-DRIFT: spec says the proxy reaction is reported to the device SDK<br/>docs/spec/12-force-feedback.md:129 but reaction_force is read only by tests, see apps/mass-lab/src/physics.rs:529"]
    N2["INVARIANT: spec and code agree that 150 N/m is a default, with a 20 to 1500 slider bounded in practice by the 0.5 N, 96-count limits<br/>docs/spec/12-force-feedback.md:112-113 and apps/mass-lab/src/main.rs:353"]
    N3["INVARIANT: Mass Lab sets ForceLimits::default and cannot raise limits, only the Studio ladder does<br/>apps/mass-lab/src/pointer.rs:272-278"]
    REAC --- N1
    SPR --- N2
    GUARD --- N3
```

The device sees a tether to the proxy, and the proxy in turn is a light body dragged by the measured pen. Contact loads reach the pen only through proxy lag.

## VT-04.5 Register reset and the feedback switch

```mermaid
sequenceDiagram
    autonumber
    participant UI as Pointer buttons<br/>apps/mass-lab/src/pointer.rs:571
    participant WK as worker commands<br/>apps/mass-lab/src/pointer.rs:260
    participant DV as Device<br/>crates/device/src/lib.rs:449
    UI->>WK: RegisterReset, UI clears feedback and trail (apps/mass-lab/src/pointer.rs:579-588)
    WK->>DV: register_reset switches feedback off then records the pose (apps/mass-lab/src/pointer.rs:294)
    WK->>WK: servo state reset so any held coupling is dropped (apps/mass-lab/src/pointer.rs:296)
    WK-->>UI: Feedback false then Registered (apps/mass-lab/src/pointer.rs:297-298)
    UI->>WK: Feedback true, only if registered (apps/mass-lab/src/pointer.rs:571-577)
    WK->>WK: refuse and report Feedback false when no registration (apps/mass-lab/src/pointer.rs:268-270)
    WK->>DV: set default limits then enable_feedback (apps/mass-lab/src/pointer.rs:275-279)
    DV-->>WK: any error drops the device and reports Error (apps/mass-lab/src/pointer.rs:283-285)
    Note over UI,DV: INVARIANT: feedback cannot be enabled before registration on this connection<br/>apps/mass-lab/src/pointer.rs:268 and crates/device/src/lib.rs:515-516
    Note over WK,DV: INVARIANT: SIGINT or SIGTERM only set a flag and the worker does the cleanup<br/>apps/mass-lab/src/pointer.rs:257-259 and docs/spec/12-force-feedback.md:145-147
```

Registering always turns force off, so the operator has to enable it again deliberately after zeroing.
