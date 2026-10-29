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
verified_commit: 7b95fb20d50a9d06c64fb6d6cc393b16edbbc35d
---

## For developers

`visiontouch-device` turns raw packets into a registered pose and turns an application force request into three bounded DAC offsets. Ceilings derive from one model-independent bound, the transport clamp of 300 counts, and applications may only lower them. Every tick runs the same ordered guard: expiry, cap, slew, uniform DAC scaling, rounding. Cartesian force is refused until the arm has been registered at the reset pose on this connection, and a failed shutdown outranks every other error. All newtons here are model commands, not measured forces. The Calibration Studio's use of the SDK (its verification ladder and higher limits) is outside this topic.

## For the business

This layer is the safety envelope between any application and the arm. Whatever a game or a graph tool asks for, the arm receives at most half a newton by default, never more than two, and only after the operator has placed it in a known reset pose and explicitly enabled force. A request that is more than a fortieth of a second old is dropped to zero, a fault ends the session rather than carrying on, and the limits can only be tightened by software, never loosened. None of these figures is a measured force on the real arm, and no measurement has yet been made.

## VT-03.1 Where each limit comes from

```mermaid
flowchart TB
    HAL["transport clamp MAX_MOTOR_OFFSET = 300 counts<br/>crates/hal/src/device.rs:19"]
    DACMAX["MAX_DAC_OFFSET equals the clamp<br/>crates/device/src/lib.rs:71"]
    FMAX["MAX_FORCE_N 2.0 model N<br/>crates/device/src/lib.rs:63"]
    SLEW["MAX_SLEW_N_S 100<br/>crates/device/src/lib.rs:78"]
    TO["COMMAND_TIMEOUT 25 ms<br/>crates/device/src/lib.rs:61"]
    DEF["ForceLimits::default 0.5 N, 96 counts, 100 N/s<br/>crates/device/src/lib.rs:99-106"]
    CONT["CONTINUOUS_FORCE_N 1.0 for content with no fixed dwell<br/>crates/device/src/lib.rs:68"]
    CH["MAX_CHANNEL_TEST_OFFSET 96<br/>crates/device/src/lib.rs:75"]
    VAL["ForceLimits::valid rejects anything above a ceiling, NaN or non-positive<br/>crates/device/src/lib.rs:109-118"]
    SET["set_limits refuses while armed<br/>crates/device/src/lib.rs:327-336"]
    DER["UNIT_DERATING 1.0 applied once in tick<br/>crates/device/src/lib.rs:86"]
    HAL --> DACMAX
    DACMAX --> VAL
    FMAX --> VAL
    SLEW --> VAL
    VAL --> SET
    DEF --> SET
    DACMAX -. "const assert" .-> CH
    SET --> DER
    N1["INVARIANT: ceilings are compile-time ordered, DAC ceiling cannot exceed the transport clamp<br/>crates/device/src/lib.rs:79-81"]
    N2["INVARIANT: the 134 counts per newton gain gives 268 counts at 2 N so 300 is the only bound independent of the uncertain gains<br/>docs/spec/12-force-feedback.md:64-66"]
    N3["DEBT: CONTINUOUS_FORCE_N is documented as the cap for content and ADR-0002 relies on it<br/>docs/adr/ADR-0002-layered-software-force-envelope.md:38 but no code reads the constant, only crates/device/src/lib.rs:50 mentions it"]
    N4["INVARIANT: ADR-0002 and ADR-0004 are accepted, complete, live, and the code and tests exist<br/>docs/adr/ADR-0002-layered-software-force-envelope.md:5-7 and crates/device/src/lib.rs:625"]
    CONT --- N3
    DACMAX --- N1
    HAL --- N2
    DEF --- N4
```

Application code can lower every figure here and raise none. The default is exactly the envelope that was armed live on the unit.

## VT-03.2 Registration, kinematics and the Jacobian

```mermaid
sequenceDiagram
    autonumber
    participant DV as Device<br/>crates/device/src/lib.rs:449
    participant ES as EncoderState<br/>crates/device/src/lib.rs:186
    participant PS as position_from_counts<br/>crates/device/src/lib.rs:263
    participant JC as jacobian_from_counts<br/>crates/device/src/lib.rs:276
    DV->>ES: update with each sample, unwrap counter deltas (crates/device/src/lib.rs:198-209)
    ES->>ES: continuous count adds wrapping_sub of old (crates/device/src/lib.rs:202)
    DV->>ES: register_reset needs one sample, feedback is switched off first (crates/device/src/lib.rs:474-481)
    ES->>ES: keep all six continuous counts as the reset pose (crates/device/src/lib.rs:217-224)
    ES->>PS: position_m uses counts minus reset counts (crates/device/src/lib.rs:234-244)
    PS-->>ES: metres from reset pose, origin at counts zero (crates/device/src/lib.rs:263-271)
    ES->>JC: guard evaluates J at the same registered counts (crates/device/src/lib.rs:276-285)
    Note over ES: INVARIANT: a reconnect builds a new EncoderState so registration never survives it<br/>crates/device/src/lib.rs:460-466 and docs/adr/ADR-0004-reset-pose-registration.md:36-37
    Note over ES,PS: EXTERNAL: angle gains and the 0.1397 m link length are a black-box fit, physical accuracy is unestablished<br/>docs/spec/12-force-feedback.md:35-41 and docs/adr/ADR-0004-reset-pose-registration.md:55
    Note over ES: DEBT: unwrapping assumes less than half a counter revolution between samples, nothing checks it<br/>crates/device/src/lib.rs:196-197
    Note over ES,PS: INVARIANT: unregistered position exists for display only and is never a demand input<br/>crates/device/src/lib.rs:245-249
```

Registration records where the arm was when the operator said it was at the reset pose, then everything is measured from there. Expected error at the tip is centimetres, which is why the Studio adds ruler checks.

## VT-03.3 ForceGuard::tick order

```mermaid
flowchart TB
    A["armed and a request held? else zero, telemetry cleared<br/>crates/device/src/lib.rs:386-389"]
    B["request age and servo gap within COMMAND_TIMEOUT? else zero and expired<br/>crates/device/src/lib.rs:390-394"]
    C["Channels demand: clamp to min of 96 and the DAC limit, skip slew and model<br/>crates/device/src/lib.rs:402-406"]
    D["Force demand while unregistered gives zero<br/>crates/device/src/lib.rs:408"]
    E["cap magnitude at force_n times derating, direction kept<br/>crates/device/src/lib.rs:410-412"]
    F["desired exactly zero gives immediate zero<br/>crates/device/src/lib.rs:413-414"]
    G["slew: step from last applied force by slew times dt<br/>crates/device/src/lib.rs:416-422"]
    H["largest channel demand above the DAC limit scales the whole vector<br/>crates/device/src/lib.rs:424-429"]
    I["round, clamp, return offsets<br/>crates/device/src/lib.rs:433-434"]
    J["transport clamp 300 on every packet<br/>crates/hal/src/device.rs:300"]
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
    N1["INVARIANT: uniform scaling keeps the force direction so a wall never gains tangential force<br/>crates/device/src/lib.rs:50-51 and crates/device/src/lib.rs:428"]
    N2["INVARIANT: zero is never slewed and stale demand is zero, no stale force is held<br/>crates/device/src/lib.rs:413 and crates/device/src/lib.rs:394"]
    N3["INVARIANT: slew memory is telemetry.applied_force, an expired tick resets it so output restarts from zero<br/>crates/device/src/lib.rs:419 and crates/device/src/lib.rs:373-380"]
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
    participant RQ as Device::request<br/>crates/device/src/lib.rs:512
    participant GD as ForceGuard<br/>crates/device/src/lib.rs:318
    participant SM as Device::sample<br/>crates/device/src/lib.rs:529
    participant HL as DeviceReader<br/>crates/hal/src/device.rs:156
    AP->>RQ: Demand and creation time
    RQ->>RQ: not armed gives reason not armed, Force while unregistered gives unregistered (crates/device/src/lib.rs:513-516)
    RQ->>GD: request, a non-finite force disarms the guard (crates/device/src/lib.rs:359-365)
    RQ->>RQ: any rejection calls disable_feedback (crates/device/src/lib.rs:520-523)
    RQ-->>AP: Rejected, or ShutdownFailed which outranks it (crates/device/src/lib.rs:561-566)
    AP->>SM: sample every servo loop even with no new demand
    SM->>HL: read packet, update state (crates/device/src/lib.rs:531-532)
    SM->>GD: tick with registered counts, then write offsets (crates/device/src/lib.rs:533-537)
    SM->>HL: any error leads to disable_feedback, shutdown failure is folded into the error text (crates/device/src/lib.rs:543-549)
    Note over RQ,SM: INVARIANT: no recoverable force fault, every sampling error ends the session and needs a reconnect<br/>crates/device/src/lib.rs:526-528 and docs/adr/ADR-0002-layered-software-force-envelope.md:42-44
    Note over RQ: INVARIANT: arming is explicit, enable_feedback is the only path to the handshake<br/>crates/device/src/lib.rs:493-499
    Note over SM,HL: DEBT: expiry is only evaluated when the worker runs, there is no hardware watchdog<br/>crates/device/src/lib.rs:59-60 and docs/spec/12-force-feedback.md:91-92
```

Rejecting a bad request also shuts output down, so a buggy caller cannot leave the last good force running. The price is that a rejected request leaves the session disarmed and the caller must enable again.

## VT-03.5 Device lifecycle: feedback state, with registration as a separate flag

```mermaid
stateDiagram-v2
    [*] --> Disarmed: connect with feedback off (crates/device/src/lib.rs:460-466)
    Disarmed --> Disarmed: register_reset records the reset pose (crates/device/src/lib.rs:474-481)
    Disarmed --> Armed: enable_feedback runs the zero handshake then arms the guard (crates/device/src/lib.rs:493-499)
    Armed --> Disarmed: disable_feedback or any rejected request (crates/device/src/lib.rs:503-507)
    Armed --> Faulted: sample or write error, shutdown attempted (crates/device/src/lib.rs:543-549)
    Disarmed --> Faulted: shutdown I/O failure (crates/device/src/lib.rs:555)
    Faulted --> [*]: close and reconnect, registration is lost
    note right of Disarmed
        INVARIANT: register_reset switches feedback off first, so output never runs
        while the reset pose is being recorded
        crates/device/src/lib.rs:478
    end note
    note right of Armed
        Registered or not, a channel liveness test may run here. Cartesian force also needs the registration flag
        crates/device/src/lib.rs:515-516
    end note
```

Registered and armed are independent: registration survives a disarm, and only a reconnect clears it.
