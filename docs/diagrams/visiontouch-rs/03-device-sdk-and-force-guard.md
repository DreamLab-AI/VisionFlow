---
id: VT-03
title: Device SDK and force guard
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/adr/ADR-0002-layered-software-force-envelope.md
  - ../visiontouch-rs/docs/adr/ADR-0004-reset-pose-registration.md
adrs: [visiontouch-rs:ADR-0002, visiontouch-rs:ADR-0004]
sources:
  - ../visiontouch-rs/crates/device/src/lib.rs
  - ../visiontouch-rs/crates/hal/src/device.rs
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/docs/adr/ADR-0002-layered-software-force-envelope.md
  - ../visiontouch-rs/docs/adr/ADR-0004-reset-pose-registration.md
verified_commit: a9e52b6154b44fb7afc158230c4ec26d4213c015
---

## For developers

Source review and qualifications: [1 November audit](../../estate-review/2026-11-01-visiontouch-software-boundaries.md). The source pins are retained; this review changes claims, not device behaviour.

`visiontouch-device` turns raw packets into a registered pose and turns an application force request into three bounded DAC offsets. Ceilings derive from one model-independent bound, the transport clamp of 300 counts, and applications may only lower them. Every tick runs the same ordered guard: expiry, cap, slew, uniform DAC scaling, rounding. Cartesian force is refused until the arm has been registered at the reset pose on this connection, and a failed shutdown outranks every other error. All newtons here are model commands, not measured forces. The Calibration Studio's use of the SDK (its verification ladder and higher limits) is outside this topic.

## For the business

This layer bounds model commands between an application and the transport. Cartesian commands default to a 0.5 N cap and require registration and explicit arming; raw channel diagnostics have a separate 96-count cap and can run unregistered. A disarmed caller can raise defaults up to the 2 N and 300-count SDK ceilings. When serviced, the guard zeros stale requests. The continuous-force policy and dwell budget belong to the application, and none of these command figures measures delivered force.

## VT-03.1 Where each limit comes from

```mermaid
flowchart TB
    HAL["transport clamp MAX_MOTOR_OFFSET = 300 counts<br/>device.rs:19"]
    DACMAX["MAX_DAC_OFFSET equals the clamp<br/>crates/device/src/lib.rs:72"]
    FMAX["MAX_FORCE_N 2.0 model N<br/>crates/device/src/lib.rs:64"]
    SLEW["MAX_SLEW_N_S 100<br/>crates/device/src/lib.rs:79"]
    TO["COMMAND_TIMEOUT 25 ms<br/>crates/device/src/lib.rs:62"]
    DEF["ForceLimits::default 0.5 N, 96 counts, 100 N/s<br/>crates/device/src/lib.rs:108-115"]
    CONT["CONTINUOUS_FORCE_N 1.0 application policy, not a guard timer<br/>crates/device/src/lib.rs:69"]
    CH["MAX_CHANNEL_TEST_OFFSET 96<br/>crates/device/src/lib.rs:76"]
    VAL["ForceLimits::valid rejects anything above a ceiling, NaN or non-positive<br/>crates/device/src/lib.rs:118-127"]
    SET["set_limits refuses while armed<br/>crates/device/src/lib.rs:360-369"]
    DER["UNIT_DERATING 1.0 applied once in tick<br/>crates/device/src/lib.rs:95"]
    HAL --> DACMAX
    DACMAX --> VAL
    FMAX --> VAL
    SLEW --> VAL
    VAL --> SET
    DEF --> SET
    DACMAX -. "const assert" .-> CH
    SET --> DER
    N1["INVARIANT: ceilings are compile-time ordered, DAC ceiling cannot exceed the transport clamp<br/>crates/device/src/lib.rs:88-90"]
    N2["INVARIANT: the 134 counts per newton gain gives 268 counts at 2 N so 300 is the only bound independent of the uncertain gains<br/>12-force-feedback.md:71-73"]
    N3["DEBT: CONTINUOUS_FORCE_N is documented as the cap for content and ADR-0002 relies on it<br/>ADR-0002-layered-software-force-envelope.md:38 but no code reads the constant, only crates/device/src/lib.rs:50 mentions it"]
    N4["INVARIANT: ADR-0002 and ADR-0004 are accepted, complete, live, and the code and tests exist<br/>ADR-0002-layered-software-force-envelope.md:5-7 and crates/device/src/lib.rs:776"]
    CONT --- N3
    DACMAX --- N1
    HAL --- N2
    DEF --- N4
```

Applications can raise the defaults while disarmed, up to the immutable SDK ceilings. The SDK does not verify a Studio unlock or a fixed dwell. The default is exactly the envelope that was armed live on the unit. A characterisation session armed through enable_characterisation uses its own raised ceiling and slew for Demand::Characterise only (crates/device/src/lib.rs:80-87, crates/device/src/lib.rs:616-643); the ceilings above govern every other demand.

## VT-03.2 Registration, kinematics and the Jacobian

```mermaid
sequenceDiagram
    autonumber
    participant DV as Device<br/>crates/device/src/lib.rs:558
    participant ES as EncoderState<br/>crates/device/src/lib.rs:216
    participant PS as position_from_counts<br/>crates/device/src/lib.rs:293
    participant JC as jacobian_from_counts<br/>crates/device/src/lib.rs:306
    DV->>ES: update with each sample, unwrap counter deltas<br/>(crates/device/src/lib.rs:228-239)
    ES->>ES: continuous count adds wrapping_sub of old<br/>(crates/device/src/lib.rs:232)
    DV->>ES: register_reset needs one sample, feedback is switched off first<br/>(crates/device/src/lib.rs:583-590)
    ES->>ES: keep all six continuous counts as the reset pose<br/>(crates/device/src/lib.rs:247-254)
    ES->>PS: position_m uses counts minus reset counts<br/>(crates/device/src/lib.rs:264-274)
    PS-->>ES: metres from reset pose, origin at counts zero<br/>(crates/device/src/lib.rs:293-301)
    ES->>JC: guard evaluates J at the same registered counts<br/>(crates/device/src/lib.rs:306-315)
    Note over ES: INVARIANT: a reconnect builds a new EncoderState so registration never<br/>survives it<br/>crates/device/src/lib.rs:569-575 and<br/>ADR-0004-reset-pose-registration.md:36-37
    Note over ES,PS: EXTERNAL: angle gains and the 0.1397 m link length are a black-box fit,<br/>physical accuracy is unestablished<br/>12-force-feedback.md:40-46 and ADR-0004-reset-pose-registration.md:55
    Note over ES: DEBT: unwrapping assumes less than half a counter revolution between<br/>samples, nothing checks it<br/>crates/device/src/lib.rs:226-227
    Note over ES,PS: INVARIANT: unregistered position exists for display only and is never a<br/>demand input<br/>crates/device/src/lib.rs:275-279
```

Registration records where the arm was when the operator said it was at the reset pose, then everything is measured from there. Expected error at the tip is centimetres, which is why the Studio adds ruler checks.

## VT-03.3 ForceGuard::tick order

```mermaid
flowchart TB
    A["armed and a request held? else zero, telemetry cleared<br/>crates/device/src/lib.rs:456-459"]
    B["request age and servo gap within COMMAND_TIMEOUT? else zero and expired<br/>crates/device/src/lib.rs:460-464"]
    C["Channels demand: clamp to min of 96 and the DAC limit, skip slew and model<br/>crates/device/src/lib.rs:472-476"]
    D["Force demand while unregistered gives zero<br/>crates/device/src/lib.rs:516-517"]
    E["cap magnitude at force_n times derating, direction kept<br/>crates/device/src/lib.rs:519-521"]
    F["desired exactly zero gives immediate zero<br/>crates/device/src/lib.rs:522-523"]
    G["slew: step from last applied force by slew times dt<br/>crates/device/src/lib.rs:525-531"]
    H["largest channel demand above the DAC limit scales the whole vector<br/>crates/device/src/lib.rs:533-538"]
    I["round, clamp, return offsets<br/>crates/device/src/lib.rs:542-543"]
    J["transport clamp 300 on every packet<br/>device.rs:365"]
    A --> B
    B --> C
    B --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    C --> I
    I --> J
    N1["INVARIANT: uniform scaling keeps the force direction so a wall never gains tangential force<br/>crates/device/src/lib.rs:50-51 and crates/device/src/lib.rs:537"]
    N2["INVARIANT: zero is never slewed and stale demand is zero, no stale force is held<br/>crates/device/src/lib.rs:522 and crates/device/src/lib.rs:464"]
    N3["INVARIANT: slew memory is telemetry.applied_force, an expired tick resets it so output restarts from zero<br/>crates/device/src/lib.rs:528 and crates/device/src/lib.rs:442-450"]
    G --- N3
    H --- N1
    F --- N2
```

The first force tick after arming applies nothing, because there is no previous tick and the slew step is zero. The documented order in the type comment matches the code.

## VT-03.4 Request, sample and the shutdown-precedes-rejection rule

```mermaid
sequenceDiagram
    autonumber
    participant AP as application worker
    participant RQ as Device::request<br/>crates/device/src/lib.rs:655
    participant GD as ForceGuard<br/>crates/device/src/lib.rs:348
    participant SM as Device::sample<br/>crates/device/src/lib.rs:680
    participant HL as DeviceReader<br/>device.rs:205
    AP->>RQ: Demand and creation time
    RQ->>RQ: not armed gives reason not armed, Force or Characterise while unregistered gives<br/>unregistered<br/>(crates/device/src/lib.rs:656-661)
    RQ->>GD: request, a non-finite force disarms the guard<br/>(crates/device/src/lib.rs:420-434)
    RQ->>RQ: any rejection calls disable_feedback<br/>(crates/device/src/lib.rs:671-674)
    RQ-->>AP: Rejected, or ShutdownFailed which outranks it<br/>(crates/device/src/lib.rs:712-717)
    AP->>SM: sample every servo loop even with no new demand
    SM->>HL: read packet, update state<br/>(crates/device/src/lib.rs:682-683)
    SM->>GD: tick with registered counts, then write offsets<br/>(crates/device/src/lib.rs:684-688)
    SM->>HL: any error leads to disable_feedback, shutdown failure is folded into the<br/>error text<br/>(crates/device/src/lib.rs:694-700)
    Note over RQ,SM: INVARIANT: no recoverable force fault, every sampling error ends the<br/>session and needs a reconnect<br/>crates/device/src/lib.rs:677-679 and<br/>ADR-0002-layered-software-force-envelope.md:42-44
    Note over RQ: INVARIANT: arming is explicit, enable_feedback is the only path to the<br/>handshake<br/>crates/device/src/lib.rs:602-608
    Note over SM,HL: DEBT: expiry is only evaluated when the worker runs, there is no<br/>hardware watchdog<br/>crates/device/src/lib.rs:60-62 and 12-force-feedback.md:121-122
```

Rejecting a bad request also shuts output down, so a buggy caller cannot leave the last good force running. The price is that a rejected request leaves the session disarmed and the caller must enable again.

## VT-03.5 Device lifecycle: feedback state, with registration as a separate flag

```mermaid
stateDiagram-v2
    [*] --> Disarmed: connect with feedback off (crates/device/src/lib.rs:569-575)
    Disarmed --> Disarmed: register_reset records the reset pose (crates/device/src/lib.rs:583-590)
    Disarmed --> Armed: enable_feedback runs the zero handshake then arms the guard (crates/device/src/lib.rs:602-608)
    Armed --> Disarmed: disable_feedback or any rejected request (crates/device/src/lib.rs:646-650)
    Armed --> Faulted: sample or write error, shutdown attempted (crates/device/src/lib.rs:694-700)
    Disarmed --> Faulted: shutdown I/O failure (crates/device/src/lib.rs:706)
    Faulted --> [*]: close and reconnect, registration is lost
    note right of Disarmed
        INVARIANT: register_reset switches feedback off first, so output never runs
        while the reset pose is being recorded
        crates/device/src/lib.rs:587
    end note
    note right of Armed
        Registered or not, a channel liveness test may run here. Cartesian force also needs the registration flag
        crates/device/src/lib.rs:658-661
    end note
```

Registered and armed are independent: registration survives a disarm, and only a reconnect clears it.
