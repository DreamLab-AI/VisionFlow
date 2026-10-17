---
id: VT-08
title: Calibration Studio and the verification ladder
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/adr/ADR-0003-staged-force-verification-ladder.md
  - ../visiontouch-rs/docs/adr/ADR-0004-reset-pose-registration.md
adrs: [visiontouch-rs:ADR-0003, visiontouch-rs:ADR-0004]
sources:
  - ../visiontouch-rs/apps/calibration/src/main.rs
  - ../visiontouch-rs/apps/calibration/src/ui.rs
  - ../visiontouch-rs/apps/calibration/src/worker.rs
  - ../visiontouch-rs/apps/calibration/src/controller.rs
  - ../visiontouch-rs/apps/calibration/src/protocol.rs
  - ../visiontouch-rs/apps/calibration/src/registration.rs
  - ../visiontouch-rs/apps/calibration/src/evidence.rs
  - ../visiontouch-rs/apps/calibration/src/arm.rs
  - ../visiontouch-rs/apps/calibration/src/scene.rs
  - ../visiontouch-rs/apps/calibration/src/coverage.rs
  - ../visiontouch-rs/apps/calibration/src/sim.rs
  - ../visiontouch-rs/apps/calibration/src/smoke.rs
  - ../visiontouch-rs/apps/calibration/README.md
  - ../visiontouch-rs/crates/hal/src/device.rs
  - ../visiontouch-rs/docs/spec/14-force-calibration.md
  - ../visiontouch-rs/docs/adr/ADR-0003-staged-force-verification-ladder.md
  - ../visiontouch-rs/docs/adr/ADR-0004-reset-pose-registration.md
verified_commit: 77898bc42b29d5928b4eae00040ef0c26c3cd0a3
---

## For developers

The Calibration Studio is the one consumer of `visiontouch-device` that builds force limits above the SDK defaults, and it does so only inside fixed-dwell steps. It is two threads. The interface thread draws and sends intent, the servo thread owns `Device` and every motor request. Three pure modules decide what may happen: `protocol` (the per-connection unlock `Session`), `controller` (the per-tick interlocks) and `registration` (stillness, return, gravity, ruler). Only `worker` touches the device, so `sim` and `smoke` can drive the same `Engine` with no hardware and the whole ladder runs in `cargo test`. Every attempt ends the same way: demand zeroed on the stopping tick, output disarmed, and only then evidence written, with an `OutputOff` token making the order a type rule. ADR-0003 is accepted and complete but its activation is staged until an owner session exists, so everything below is verified against a simulated plant only. All newtons are model commands, not measured forces.

## For the business

This is the program the lab owner will run, attended, to find out whether the arm pushes back at all and by how much. It cannot be rushed: the operator must register the arm at its reset pose, prove each of the three motors responds, prove the directions are right, then climb a ladder of test forces from a tenth of a newton to two, with a gauge reading typed in at each step. A button press, a released key, a too-fast hand or a frozen screen stops that one attempt at once. A gauge reading that comes out much higher than the command locks every further force step for the rest of the connection, and registering the arm again does not lift it. A reading of almost nothing (under 0.05 N) is recorded as nothing delivered: it neither unlocks the next rung nor locks anything. Disconnecting forgets everything, so each sitting starts from the bottom. The result is a folder per sitting holding every attempt as a table of numbers and a signed-off summary, which becomes the evidence for the first real force figures. Until that sitting happens, nothing in this program has been measured on the real arm.

## VT-08.1 UI thread and servo thread

```mermaid
sequenceDiagram
    autonumber
    participant UI as App::hold<br/>apps/calibration/src/main.rs:187
    participant WK as run<br/>apps/calibration/src/worker.rs:1266
    participant EN as Engine::step<br/>apps/calibration/src/worker.rs:764
    participant CT as Controller::tick<br/>apps/calibration/src/controller.rs:443
    participant EV as Engine::finish<br/>apps/calibration/src/worker.rs:863
    UI->>WK: every frame bump pulse and store held (apps/calibration/src/main.rs:200-201)
    UI->>WK: commands on an mpsc channel, Start only after a 120 ms fresh hold (apps/calibration/src/main.rs:204)
    WK->>WK: drain commands then derive heartbeat age from pulse (apps/calibration/src/worker.rs:1288-1308)
    WK->>EN: step with held and ui age every 2.4 ms (apps/calibration/src/worker.rs:1275 and apps/calibration/src/worker.rs:1306-1320)
    EN->>EN: sample the device on every tick even with no demand (apps/calibration/src/worker.rs:769)
    EN->>CT: TickInput, then interlocks, then planned demand (apps/calibration/src/worker.rs:835)
    CT-->>EN: TickOutput, demand forced to zero when any stop fires (apps/calibration/src/controller.rs:548)
    EN->>EN: hw.request with the demand, a rejection ends the attempt (apps/calibration/src/worker.rs:843-848)
    EN->>EV: finish(reason) on the same tick as the stop (apps/calibration/src/worker.rs:856-857)
    EV->>EV: disable_feedback first, always (apps/calibration/src/worker.rs:864-866)
    EV->>EV: score, then OutputOff token, then write CSV and JSON (apps/calibration/src/worker.rs:986-990)
    WK-->>UI: latest Snapshot every 16 ms, dropped if the UI is behind (apps/calibration/src/worker.rs:1248 and apps/calibration/src/worker.rs:1313-1315)
    UI->>UI: poll drains snapshots and keeps the newest (apps/calibration/src/main.rs:127)
    Note over UI,WK: INVARIANT: rendering never issues a motor command, the UI sends intent and reads copies<br/>apps/calibration/src/main.rs:3
    Note over UI,WK: DEBT: on Wayland a held Space can stay latched after focus loss because no key-up or focus event arrives, only the 250 ms heartbeat and Stop remain<br/>apps/calibration/src/main.rs:295-302
    Note over WK,EN: OPEN: the servo period is a fixed 2.4 ms sleep, a slow real I/O round trip stretches the interval toward the 25 ms gap guard<br/>apps/calibration/src/worker.rs:1275 and apps/calibration/src/controller.rs:36
    Note over WK,EN: INVARIANT: the loop never catches up after an overrun, the real interval is measured and the 25 ms gap interlock judges it<br/>apps/calibration/src/worker.rs:1318-1320
```

**What it shows.** The interface can freeze or lose focus and the arm still stops, because hold and heartbeat are shared atomics the servo thread reads every tick, and snapshots flow back through a channel of depth one.
**Why it is this way.** ADR-0003 requires held attempts with operator interlocks, and the focus-loss comment records a rejected alternative: a recent-key-down fallback needs autorepeat, whose delay exceeds any safe timeout.

## VT-08.2 The per-connection unlock ladder

```mermaid
flowchart TB
    CON["connect: Session::connected, every verdict forgotten, Over included<br/>apps/calibration/src/protocol.rs:501 and apps/calibration/src/worker.rs:413"]
    BASE["Backdrive and liveness need only a connection<br/>apps/calibration/src/protocol.rs:610-622"]
    LIVE["liveness pass on channels 0, 1 and 2<br/>apps/calibration/src/protocol.rs:568-570"]
    REG["registered, plus return, gravity and ruler checks complete<br/>apps/calibration/src/protocol.rs:565-571"]
    SIGN["sign pulses 0.25 N on X, Y and Z, limits 0.5 N 96 counts<br/>apps/calibration/src/protocol.rs:24 and apps/calibration/src/protocol.rs:623-629"]
    R1["rungs 0.1, 0.25 and 0.5 N, effects at 0.5 N, lift to 0.5 N<br/>apps/calibration/src/protocol.rs:630-660"]
    R2["1.0 N rung needs 0.5 N cleared on all axes, limits 300 counts<br/>apps/calibration/src/protocol.rs:633 and apps/calibration/src/protocol.rs:427-442"]
    R3["2.0 N rung needs 0.5 and 1.0 N cleared<br/>apps/calibration/src/protocol.rs:634-637"]
    LIFT["lift cap 0.5, then 1.0, then 1.5 N<br/>apps/calibration/src/protocol.rs:582-592"]
    WALL["wall ramp and thermal need the 1.0 N rung, capped at 1.0 N<br/>apps/calibration/src/protocol.rs:661-669"]
    FX["effects cap 0.5 N, then 1.0 N, never above<br/>apps/calibration/src/protocol.rs:594-602"]
    OVER["Over on any rung, axis or sign sets Session::over, which locks every force step for the whole connection and lives outside rungs<br/>apps/calibration/src/protocol.rs:488 and apps/calibration/src/protocol.rs:551-559 and apps/calibration/src/protocol.rs:760-762"]
    CLR["re-registration clears checks, signs and rungs but keeps liveness and Over<br/>apps/calibration/src/protocol.rs:735-746"]
    RECON["only a reconnect clears Over: Session::reconnect forgets everything<br/>apps/calibration/src/protocol.rs:508-510"]
    NONE["None: a completed dwell under 0.05 N is recorded, clears no rung and locks nothing<br/>apps/calibration/src/protocol.rs:357 and apps/calibration/src/protocol.rs:365-366"]
    SCORE["rung score: Over above 1.2c plus 0.05, Pass from 0.8c minus 0.05, None under 0.05 N, else Under<br/>apps/calibration/src/protocol.rs:360-370"]
    CON --> BASE
    BASE --> LIVE
    LIVE --> REG
    REG --> SIGN
    SIGN --> R1
    R1 --> R2
    R2 --> R3
    R1 --> LIFT
    R2 --> LIFT
    R2 --> WALL
    R1 --> FX
    R2 --> FX
    R1 --> SCORE
    SCORE --> OVER
    SCORE --> NONE
    OVER --> BASE
    CLR --> REG
    RECON --> CON
    N1["INVARIANT: an Over verdict is sticky for the connection, a retest or later passing reading cannot overwrite it, and a None cannot erase an earlier real verdict<br/>apps/calibration/src/protocol.rs:760-767 and apps/calibration/src/worker.rs:1075-1076"]
    N2["INVARIANT: unlocked kinds are derived by probing check, never stored, so table and gate cannot disagree<br/>apps/calibration/src/protocol.rs:673-712"]
    N3["INVARIANT: lift never exceeds 1.5 N, and README, code and ADR-0003 now agree<br/>apps/calibration/README.md:46 and apps/calibration/src/protocol.rs:30 and docs/adr/ADR-0003-staged-force-verification-ladder.md:39-41"]
    N4["OPEN: liveness is kept across re-registration while signs and rungs are cleared, a new pose is accepted without re-proving the force channels<br/>apps/calibration/src/protocol.rs:739-742"]
    N5["INVARIANT: the Over lock lasts the whole connection in ADR, spec and code alike, re-registration does not lift it<br/>docs/adr/ADR-0003-staged-force-verification-ladder.md:47-49 and docs/spec/14-force-calibration.md:200 and apps/calibration/src/protocol.rs:551-552"]
    OVER --- N1
    RECON --- N5
    BASE --- N2
    LIFT --- N3
    CLR --- N4
```

**What it shows.** Each Start is judged by `Session::check`, which names the first missing prerequisite, and every attempt sets explicit limits (disarm, set, arm) so nothing is inherited. Under-delivery is recorded with its slope but never blocks, only Over does, and Over is the one verdict held outside the per-rung table so that re-registration cannot clear it.
**Why it is this way.** Three powered sessions on 2026-10-16 produced no perceptible force, so ADR-0003 (accepted 2026-10-17) replaced hold-and-feel with discriminating steps, and only a completed dwell is scored (apps/calibration/src/worker.rs:921-926).

## VT-08.3 Interlocks per attempt class

```mermaid
flowchart TB
    IN["TickInput from the sample: position, button, interval, heartbeat age, held<br/>apps/calibration/src/worker.rs:791-804"]
    NF["position not finite gives non-finite-position<br/>apps/calibration/src/controller.rs:476-479"]
    UR["Cartesian step but no registration gives unregistered<br/>apps/calibration/src/controller.rs:480-483"]
    BT["pen button gives pen-button<br/>apps/calibration/src/controller.rs:484"]
    GAP["sample interval over 25 ms gives sample-gap<br/>apps/calibration/src/controller.rs:36 and apps/calibration/src/controller.rs:485"]
    HB["heartbeat older than 250 ms gives ui-heartbeat-stale<br/>apps/calibration/src/controller.rs:38 and apps/calibration/src/controller.rs:486"]
    HOLD["key not held gives hold-released, thermal uses Confirm instead<br/>apps/calibration/src/controller.rs:487-491"]
    SPD["speed over class limit, measured as displacement over a 9.6 ms window, gives speed<br/>apps/calibration/src/controller.rs:18 and apps/calibration/src/controller.rs:447-467 and apps/calibration/src/controller.rs:492"]
    EXC["excursion from start pose gives excursion<br/>apps/calibration/src/controller.rs:493-496"]
    OSC["ladder or wall oscillation gives oscillation<br/>apps/calibration/src/controller.rs:497-505"]
    LFT["lift only: pen rises over 2 mm gives lift-off, a normal end<br/>apps/calibration/src/controller.rs:506-509"]
    BUD["applied force above 1.0 N accrues, over 10 s gives over-continuous-cap<br/>apps/calibration/src/controller.rs:14 and apps/calibration/src/controller.rs:510-518"]
    END["plan duration reached gives completed, class bound gives time-bound<br/>apps/calibration/src/controller.rs:519-521"]
    ZERO["any stop: demand replaced by zero on the same tick<br/>apps/calibration/src/controller.rs:548"]
    LIM["class table: speed, excursion, time bound, hold<br/>apps/calibration/src/controller.rs:53-70"]
    REJ["device refuses a demand gives request-rejected via reject<br/>apps/calibration/src/controller.rs:400-404"]
    IN --> NF
    NF --> UR
    UR --> BT
    BT --> GAP
    GAP --> HB
    HB --> HOLD
    HOLD --> SPD
    SPD --> EXC
    EXC --> OSC
    OSC --> LFT
    LFT --> BUD
    BUD --> END
    END --> ZERO
    LIM --> SPD
    REJ --> ZERO
    N1["INVARIANT: checks run in a fixed order and the first reason wins, later ones cannot overwrite it<br/>apps/calibration/src/controller.rs:469-476"]
    N2["INVARIANT: the interlock velocity spans a fixed 9.6 ms of time, not a count of samples, so cadence cannot change what the speed interlock sees (an unpaced 0.28 ms loop once turned encoder quantisation into a phantom 50 mm/s, P-0065)<br/>apps/calibration/src/controller.rs:18 and apps/calibration/src/controller.rs:447-467"]
    N5["INVARIANT: README time bounds 4, 1.5, 4, 60, 180, 660 and 60 s now match the class table<br/>apps/calibration/README.md:58 and apps/calibration/src/controller.rs:55-62"]
    N3["DEBT: a thermal attempt has no hold, only a Confirm every 30 s with a 10 s grace, so release is no longer a stop and the arm relies on Confirm and the 660 s bound<br/>apps/calibration/src/controller.rs:26-28 and apps/calibration/src/controller.rs:61"]
    N4["OPEN: oscillation is judged on the ladder axis, or Y otherwise, but only wall and ladder act on it, so a ringing liveness or sign attempt is not stopped by it<br/>apps/calibration/src/controller.rs:497-505"]
    ZERO --- N1
    SPD --- N2
    LIM --- N5
    HOLD --- N3
    OSC --- N4
```

**What it shows.** Every interlock reduces to a named `StopReason` written into the evidence, and the 1.0 N budget counts cumulative time above one newton across the attempt, which is what lets the 2.0 N rung exist.
**Why it is this way.** Limits come from design section 3.2 and are restated in docs/spec/14-force-calibration.md:128-138, where the table now agrees with both the code and the README.

## VT-08.4 Registration at the reset pose

```mermaid
flowchart TB
    STILL["still pen: 250 ms within 0.5 mm<br/>apps/calibration/src/registration.rs:12-14 and apps/calibration/src/registration.rs:40"]
    REGC["RegisterReset: refused while a step runs or the pen moves<br/>apps/calibration/src/worker.rs:586-591"]
    SDK["hw.register_reset records the six continuous counts, output off<br/>apps/calibration/src/worker.rs:595"]
    RESET["session.record_registered false then true, checks cleared<br/>apps/calibration/src/worker.rs:598-599"]
    RET["return to reset: capture within 2 mm, spread of captures within 2 mm<br/>apps/calibration/src/registration.rs:16 and apps/calibration/src/registration.rs:166-171 and apps/calibration/src/registration.rs:187-190"]
    ARMG["ArmGravityDrop: output off, no step, one drop per arming<br/>apps/calibration/src/worker.rs:617-632"]
    GRAV["drop 10 mm within 0.5 s, mostly vertical, down is a pass<br/>apps/calibration/src/registration.rs:22-24 and apps/calibration/src/registration.rs:233-262"]
    STICKY["a failed gravity drop is sticky, a later pass never overwrites it<br/>apps/calibration/src/registration.rs:152-156"]
    RULER["ruler slide per axis: scale, sign and off-axis from two still marks<br/>apps/calibration/src/registration.rs:91-113"]
    RULEOK["sign and off-axis within 15 percent stop the line, scale 0.8 to 1.25 only warns<br/>apps/calibration/src/registration.rs:18-20 and apps/calibration/src/registration.rs:84-87"]
    DONE["Checks::complete unlocks sign pulses<br/>apps/calibration/src/registration.rs:196-198"]
    POSE["arm::pose from registered counts, tip equals the SDK position<br/>apps/calibration/src/arm.rs:27-51"]
    WIRE["arm::segments: base block, shoulder post, two links, gimbal stub<br/>apps/calibration/src/arm.rs:53-64"]
    LIVE["live pose uses continuous minus reset counts<br/>apps/calibration/src/scene.rs:159-162"]
    GHOST["ghost arm at zero counts, white 0.5 box and a line to the pen<br/>apps/calibration/src/scene.rs:163-170"]
    STILL --> REGC
    REGC --> SDK
    SDK --> RESET
    RESET --> RET
    RET --> ARMG
    ARMG --> GRAV
    GRAV --> STICKY
    STICKY --> RULER
    RULER --> RULEOK
    RULEOK --> DONE
    SDK --> POSE
    POSE --> WIRE
    WIRE --> LIVE
    LIVE --> GHOST
    N1["INVARIANT: a mark is accepted only from a still pen, so a moving reading never becomes a reference<br/>apps/calibration/src/registration.rs:158-161"]
    N2["DEBT: arm.rs duplicates the link length and per-count angle gains from the SDK and is kept equal only by a test<br/>apps/calibration/src/arm.rs:9-10 and apps/calibration/src/arm.rs:107"]
    N3["TENSION: the box is a visibility aid larger than the 2 mm return tolerance, so its size could be read as an acceptance region<br/>apps/calibration/src/scene.rs:165-167 and apps/calibration/src/registration.rs:16"]
    N4["OPEN: the return check measures the pen against the registered origin only after the operator reproduces the physical pose, steering to the marker proves nothing about physical repeatability<br/>apps/calibration/README.md:22-27"]
    STILL --- N1
    WIRE --- N2
    GHOST --- N3
    RET --- N4
```

**What it shows.** Registration records where the arm physically was at the reset pose, then four independent checks follow: repeatable return, gravity toward -Y, ruler sign on three axes, and a straight slide.
**Why it is this way.** ADR-0004 (accepted, live) replaced bottom-reference workflows with a published reset pose, and a reconnect clears it because it is a new device (docs/adr/ADR-0004-reset-pose-registration.md:36-37).

## VT-08.5 Evidence schema 2

```mermaid
classDiagram
    class SessionDir {
        captures/calibration/utc-sessionid
        session.json
        attempt json and csv pairs
        coverage.json optional
    }
    class SessionRecord {
        schema 2
        kind haptics-verification-session
        session_id
        connected_utc
        registration
        checks
        unlock_history
        attempts as stems
        lift with F_0, F_m per mass and gain
        k_marginal_n_mm
        claim
    }
    class AttemptRecord {
        schema 2
        kind haptics-verification-attempt
        session_id and attempt_id
        step and params
        started_utc and duration_s
        software and limits and registration
        stop and verdict
        telemetry_summary and auto
        operator
        claim
    }
    class CsvRef {
        file name within the directory
        sha256 of the CSV in lower-case hex
    }
    class Operator {
        readings keyed to the attempt id
        felt and heard
        note
    }
    class Reading {
        label
        value
        unit N or g or C or mm
    }
    class OutputOff {
        token only obtainable with feedback off
    }
    class StopRecord {
        reason from StopReason name
        detail
    }
    SessionDir "1" *-- "1" SessionRecord : session.json
    SessionDir "1" *-- "many" AttemptRecord : NNNN-step.json
    AttemptRecord "1" *-- "1" CsvRef : hashes the sibling CSV
    AttemptRecord "1" *-- "1" Operator : readings typed during or after the run
    AttemptRecord "1" *-- "1" StopRecord : why it ended
    Operator "1" *-- "many" Reading : appended per attempt id
    OutputOff ..> SessionDir : required by every writer
    note for SessionRecord "evidence.rs:286 session.json<br/>evidence.rs:303-308 lift and k_marginal_n_mm, additive fields<br/>evidence.rs:516 write_session<br/>worker.rs:465 write_session_file<br/>worker.rs:482-487 fills lift and k_marginal_n_mm"
    note for AttemptRecord "evidence.rs:201 and evidence.rs:279 stem<br/>evidence.rs:489 write_attempt<br/>evidence.rs:506 rewrite_attempt"
    note for CsvRef "evidence.rs:192 and evidence.rs:496-501<br/>hash taken over the rendered CSV bytes"
    note for OutputOff "evidence.rs:24-34<br/>refuses while motor output is armed"
    note for Operator "evidence.rs:77 and protocol.rs:374-383 Unit N, G, C, Mm<br/>worker.rs:284 unit_allowed refuses a unit the step does not take"
    note for OutputOff "DEBT: every writer takes the OutputOff as _off and never reads it,<br/>so the guarantee rests on the one confirm call site<br/>apps/calibration/src/evidence.rs:490 and apps/calibration/src/worker.rs:493"
    note for Operator "INVARIANT: submit_reading refuses every non-Celsius reading while an attempt runs,<br/>so readings are typed only while output is off; the one exception is the<br/>casing temperature, which feeds the thermal interlock<br/>apps/calibration/src/worker.rs:1010-1014 and apps/calibration/src/worker.rs:1026-1029 and apps/calibration/README.md:68-71"
    note for Operator "OPEN: ADR-0003 states readings are typed only while output is off and does not name the Celsius exception that the README and code carry<br/>docs/adr/ADR-0003-staged-force-verification-ladder.md:52-53 and apps/calibration/README.md:70-71"
    note for AttemptRecord "OPEN: software.commit comes from git rev-parse HEAD at run time<br/>and falls back to unknown outside a checkout<br/>apps/calibration/src/evidence.rs:534-544"
    note for SessionDir "evidence.rs:522 create_session_dir<br/>worker.rs:452 ensure_dir creates it lazily"
```

**What it shows.** A session directory is created lazily, `session.json` is rewritten after each change (it now also carries the lift results, with the gain `m g / (F_m - F_0)`, and the last stable wall stiffness `k_marginal_n_mm`), and each attempt is a CSV plus a JSON whose `csv.sha256` binds the pair. Operator readings are keyed by attempt id, arrive mostly after the attempt ends, and trigger a rewrite of that JSON, where a ladder rung is finally scored.
**Why it is this way.** ADR-0003 wants output off for readings, and the code enforces it for running attempts in `submit_reading` (apps/calibration/src/worker.rs:1010-1014) and, for every file write, by requiring an `OutputOff` (apps/calibration/src/evidence.rs:24-34).

## VT-08.6 Headless path and CLI modes that never open hardware

```mermaid
flowchart TB
    CLI["CLI parse: demo, smoke-test, no-device, help; anything else exits 2<br/>apps/calibration/src/main.rs:228-238"]
    HELP["help prints and returns before any device code<br/>apps/calibration/src/main.rs:235-237"]
    SMOKE["smoke-test: smoke::run in a temp dir, exit 1 on failure<br/>apps/calibration/src/main.rs:239-262"]
    DEMO["demo: Engine over SimHw in the UI thread, no Worker, no shutdown handler<br/>apps/calibration/src/main.rs:60-64 and apps/calibration/src/main.rs:266"]
    NODEV["no-device: Worker spawned, Connect left to the button<br/>apps/calibration/src/main.rs:65-69"]
    REAL["default: Worker sends Connect, Device::connect on the parallel port, output off<br/>apps/calibration/src/main.rs:67-69 and apps/calibration/src/worker.rs:1283"]
    RUN["smoke::run drives the production Engine with simulated true<br/>apps/calibration/src/smoke.rs:99-104"]
    PLANT["SimHw: virtual clock, plant step, the real EncoderState and ForceGuard<br/>apps/calibration/src/sim.rs:292-308"]
    LADDER["liveness, registration, return, gravity, ruler, signs, all rungs, effects, lift, wall, thermal<br/>apps/calibration/src/smoke.rs:105-296"]
    VERIFY["reads back every JSON, checks CSV sha256, schema 2 and at least 25 attempts<br/>apps/calibration/src/smoke.rs:298-332"]
    RECON["reconnect resets the ladder to backdrive and liveness<br/>apps/calibration/src/smoke.rs:335-340"]
    TEST["cargo test runs the same run function<br/>apps/calibration/src/smoke.rs:353-361"]
    PUMP["demo pump steps the engine per frame from the mouse<br/>apps/calibration/src/main.rs:133-160"]
    CLI --> HELP
    CLI --> SMOKE
    CLI --> DEMO
    CLI --> NODEV
    CLI --> REAL
    SMOKE --> RUN
    TEST --> RUN
    DEMO --> PUMP
    PUMP --> PLANT
    RUN --> PLANT
    PLANT --> LADDER
    LADDER --> VERIFY
    VERIFY --> RECON
    N1["DEBT: the simulator arms instantly while the real device runs a zero-only handshake of 5 ms plus four 2 ms preambles, 13 ms of sleeps, so arming latency is never exercised headless<br/>apps/calibration/src/sim.rs:326-332 and crates/hal/src/device.rs:252-275"]
    N2["TENSION: smoke passes held true and demo the real hold, both pass heartbeat age 0, so the stale-heartbeat and release interlocks are covered only by controller unit tests, not by the end-to-end run<br/>apps/calibration/src/smoke.rs:38 and apps/calibration/src/main.rs:157 and apps/calibration/src/controller.rs:486-487"]
    N3["INVARIANT: only worker touches the device, so the whole engine is generic over a Hardware trait the simulator implements<br/>apps/calibration/src/worker.rs:37-62"]
    N4["OPEN: ADR-0003 is staged, its review trigger is a committed owner record, and the smoke run proves the logic against gain 0.6 and weight 0.3 N, not the arm<br/>docs/adr/ADR-0003-staged-force-verification-ladder.md:7-13 and apps/calibration/src/smoke.rs:3-6"]
    PLANT --- N1
    PUMP --- N2
    RUN --- N3
    VERIFY --- N4
```

**What it shows.** The demo and the smoke run are the production `Engine` over a different `Hardware`, so a regression in unlocks, interlocks or evidence fails `cargo test` before anyone stands at the arm, though nothing about real timing or force.
**Why it is this way.** The baseline keeps force claims in operator rows only, so the headless path protects logic and never stands in for the owner session (docs/BASELINE-visiontouch-rs.md:84).

