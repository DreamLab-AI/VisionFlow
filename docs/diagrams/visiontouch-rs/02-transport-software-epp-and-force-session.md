---
id: VT-02
title: Transport, software EPP cycle and force session
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/spec/10-device-readout.md
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
adrs: [visiontouch-rs:ADR-0002]
sources:
  - ../visiontouch-rs/crates/hal/src/device.rs
  - ../visiontouch-rs/docs/spec/10-device-readout.md
  - ../visiontouch-rs/docs/spec/12-force-feedback.md
  - ../visiontouch-rs/docs/adr/ADR-0002-layered-software-force-envelope.md
verified_commit: a9e52b6154b44fb7afc158230c4ec26d4213c015
---

## For developers

Source review and qualifications: [1 November audit](../../estate-review/2026-11-01-visiontouch-software-boundaries.md). The source pins are retained; this review changes claims, not device behaviour.

The `hal` crate owns the native stack’s parallel-port I/O; the separate recorder can observe a reference program’s calls. It claims the ppdev node exclusively and performs IEEE 1284 EPP cycles in software through register ioctls, because the native EPP calls on this host overwrite the INIT bit. On top of that sit three observed protocols: the setup prefix, the 16-byte packet read, and the motor session (a zero-only handshake, 8-byte motor packets with a sequence acknowledgement, and a two-packet shutdown). The public nonzero motor-write method clamps each channel to plus or minus 300 counts; internal lifecycle packets carry zeros. This topic stops at the transport boundary; limits, kinematics and the force guard are in VT-03.

## For the business

This is the part of the system that actually speaks to the arm. It uses software deadlines and treats a failed exchange as a fault, and the arm is only allowed to receive a push after a fixed start-up handshake that sends zeros first. If anything goes wrong the software tries to send the arm a neutral command and let go of the port. It cannot protect against the computer itself freezing or the process being killed, because no timer lives in the hardware that anyone has established. That is why the rest of the stack also keeps force requests small.

## VT-02.1 One software EPP byte cycle

```mermaid
sequenceDiagram
    autonumber
    participant TR as transfer<br/>device.rs:156
    participant WT as wait<br/>device.rs:140
    participant PT as Port ioctls<br/>device.rs:88
    participant DV as the arm on the parallel port
    TR->>PT: set data direction, write idle control<br/>(device.rs:161-162)
    loop each byte, each status wait has its own 5 ms budget
        TR->>PT: write only, put the byte on the data register<br/>(device.rs:165)
        TR->>PT: assert address strobe 8 or data strobe 2<br/>(device.rs:167)
        TR->>WT: expect status bit 7 low
        WT->>PT: poll status up to 1000 times<br/>(device.rs:142-143)
        PT->>DV: register read
        DV-->>WT: ack
        TR->>PT: read only, take the byte from the data register<br/>(device.rs:170)
        TR->>PT: clear strobe, wait status bit 7 high<br/>(device.rs:172-173)
    end
    TR->>PT: always control 0 then direction output<br/>(device.rs:178-179)
    Note over TR,WT: INVARIANT: software checks a deadline and a 1000-poll limit, assuming<br/>each register ioctl returns<br/>device.rs:140-154 and device.rs:115-132
    Note over TR,PT: INVARIANT: cleanup is attempted even when the cycle failed, first error<br/>wins<br/>device.rs:177-180 and 10-device-readout.md:30-31
    Note over TR,DV: DOC-DRIFT: the module doc says this is not a force controller<br/>device.rs:1-2 yet the same file arms motors at device.rs:288
```

The cycle is the public Linux EPP handshake reproduced with plain register reads and writes. Each status wait has its own 5 ms budget measured from the start of that wait, and the device is polled before the deadline is checked (device.rs:135-141). It is not an independent deadline on a blocked ioctl or a guarantee about kernel scheduling.

## VT-02.2 Connect: claim, setup prefix, reply check

```mermaid
sequenceDiagram
    autonumber
    participant CN as DeviceReader::connect<br/>device.rs:218
    participant PT as Port<br/>device.rs:80
    participant TR as transfer<br/>device.rs:156
    participant DV as the arm
    CN->>PT: open read write, nonblocking, must be a character device<br/>(device.rs:219-229)
    CN->>PT: exclusive claim then claim<br/>(device.rs:231-232)
    CN->>PT: control 0<br/>(device.rs:241)
    CN->>TR: select address 0x17 and write 0x01<br/>(device.rs:242-243)
    TR->>DV: active setup write, not a passive query
    CN->>TR: select address 0x0e and read one byte<br/>(device.rs:244-246)
    DV-->>CN: reply must be 0x01 else InvalidData<br/>(device.rs:247-252)
    CN->>TR: select address 0 a second time before the first packet<br/>(device.rs:254)
    Note over CN,DV: INVARIANT: connect does not start a force session, output needs<br/>begin_force_session<br/>device.rs:215-216
    Note over CN,DV: EXTERNAL: the setup bytes and the reply are OBSERVE class facts from the<br/>reference run<br/>10-device-readout.md:37-43 and 10-device-readout.md:50-52
```

Everything here is replicated from a captured reference run. A matching reply is a communication check, not a model identification.

## VT-02.3 Sampling and the motor acknowledgement

```mermaid
sequenceDiagram
    autonumber
    participant SM as DeviceReader::sample<br/>device.rs:266
    participant TR as transfer<br/>device.rs:156
    participant SA as Sample decode<br/>device.rs:190
    SM->>SM: refuse if unhealthy, reconnect needed<br/>(device.rs:267-269)
    SM->>TR: select address 0 then read 16 bytes<br/>(device.rs:271-273)
    SM->>SM: if a motor packet is pending, byte 14 must echo the sequence, byte 13<br/>bit 6 must be set and the fault status bits must be clear<br/>(device.rs:274-276 and device.rs:47-59)
    SM->>SA: raw packet kept whole<br/>(device.rs:278)
    SA-->>SM: six little-endian counters in order 2 0 1 3 4 5<br/>(device.rs:197-198)
    SA-->>SM: stylus button is active low on byte 13 bit 1<br/>(device.rs:193)
    Note over SM: INVARIANT: any failed transaction marks the connection unhealthy for<br/>good<br/>device.rs:280-282
    Note over SM,SA: INVARIANT: checks sequence echo, ready bit and fault status bits, not a fresh-sample<br/>counter<br/>12-force-feedback.md:28-32 and 12-force-feedback.md:195-197
```

The raw 16 bytes are never decoded as a sample unless the full transfer completed. The button, ready bit and sequence byte have observed meanings; the remaining status semantics are unestablished. Sampling twice without a new write can pass when the device returns the same expected echo.

## VT-02.4 Force session: zero handshake, motor packets, shutdown

```mermaid
sequenceDiagram
    autonumber
    participant BG as begin_force_session<br/>device.rs:288
    participant SP as send_motor_packet<br/>device.rs:340
    participant WR as write_motor_offsets<br/>device.rs:361
    participant EN as end_force_session<br/>device.rs:378
    BG->>BG: two samples, select 0x17, write 0x02, sample<br/>(device.rs:295-299)
    BG->>SP: zero packet, control 0xc0, echo last sequence<br/>(device.rs:300)
    BG->>BG: sleep 5 ms then sample<br/>(device.rs:301-303)
    loop four zero preamble packets
        BG->>SP: control 0xce, sequence plus one, sleep 2 ms, sample<br/>(device.rs:305-308)
    end
    BG->>SP: leave a fresh zero packet 0xca pending<br/>(device.rs:311)
    WR->>WR: refuse unless healthy and armed<br/>(device.rs:362-364)
    WR->>SP: clamp each channel to plus or minus the session ceiling, MAX_MOTOR_OFFSET by default, then control 0xca<br/>(device.rs:365-366)
    SP->>SP: word is 2047 plus offset, sequence marked pending BEFORE the I/O<br/>(device.rs:348-349 and device.rs:354)
    EN->>SP: two zero packets with control 0xc5, a read between them<br/>(device.rs:386-399)
    Note over BG,EN: INVARIANT: nonzero offsets are impossible before the handshake completes<br/>device.rs:362 and 12-force-feedback.md:34-36
    Note over WR: INVARIANT: the transport clamp, 300 counts unless a characterisation session raises it,<br/>applies on every write whatever the caller asks<br/>device.rs:19, device.rs:328-333 and device.rs:365
    Note over BG,EN: EXTERNAL: control bytes 0xc0 0xce 0xca 0xc5 and the 2047 centre are<br/>observed lifecycle patterns, their bit meaning is open<br/>12-force-feedback.md:19-26
```

The 5 ms and 2 ms pauses and the trailing pending 0xca packet exist because the first native run rejected a duplicate read, so every servo read must follow a write.

## VT-02.5 Connection state and the cost of a failed shutdown

```mermaid
stateDiagram-v2
    [*] --> Idle: connect ok (device.rs:230-255)
    Idle --> Armed: begin_force_session ok (device.rs:311-312)
    Armed --> Armed: write_motor_offsets ok (device.rs:366-367)
    Armed --> Idle: end_force_session ok (device.rs:400)
    Armed --> Unhealthy: any packet or sample error (device.rs:368-371)
    Idle --> Unhealthy: sample error (device.rs:280-282)
    Unhealthy --> Released: close or Drop attempts shutdown, then release (device.rs:412-419)
    Idle --> Released: close or Drop (device.rs:425-433)
    Released --> [*]
    note right of Unhealthy
        DEBT: end_force_session clears the pending sequence even when shutdown failed,
        so Drop cannot resend neutral packets, it only retries the port release
        device.rs:400 and device.rs:381-383
    end note
    note right of Armed
        INVARIANT: no automatic rearm, a fault ends the session and a new connection is required
        device.rs:289-293 and 12-force-feedback.md:35-36
    end note
```

A failed shutdown cannot be recovered in place. The higher layers (VT-03) therefore make a failed shutdown outrank any other error so the operator reconnects.
