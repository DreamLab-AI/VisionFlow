---
id: VT-06
title: Tools and the evidence path
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/adr/ADR-0005-build-with-quality-retrofit.md
adrs: [visiontouch-rs:ADR-0005]
sources:
  - ../visiontouch-rs/tools/port-recorder/src/lib.rs
  - ../visiontouch-rs/tools/port-recorder/shim.c
  - ../visiontouch-rs/tools/port-recorder/ioctl-entry.S
  - ../visiontouch-rs/tools/port-recorder/README.md
  - ../visiontouch-rs/tools/device-readout/src/main.rs
  - ../visiontouch-rs/tools/port-probe/src/main.rs
  - ../visiontouch-rs/crates/hal/src/lib.rs
  - ../visiontouch-rs/scripts/with-port-access
  - ../visiontouch-rs/scripts/build-recorder
  - ../visiontouch-rs/scripts/quality-gate.sh
  - ../visiontouch-rs/.github/workflows/ci.yml
  - ../visiontouch-rs/docs/adr/ADR-0005-build-with-quality-retrofit.md
  - ../visiontouch-rs/AGENTS.md
verified_commit: a9e52b6154b44fb7afc158230c4ec26d4213c015
---

## For developers

Source review and qualifications: [1 November audit](../../estate-review/2026-11-01-visiontouch-software-boundaries.md). The source pins are retained; this review changes claims, not device behaviour.

Three transport tools, alongside the audio analysis tool, produce the evidence that the spec and provenance rows stand on: a capability probe that only reads host metadata, a bounded readout that performs the active setup and prints raw packets but never sends a force packet, and an LD_PRELOAD recorder that logs the ppdev calls a reference program makes. A launcher refreshes group membership for port access, and one script, `quality-gate.sh`, is both the local pre-commit check and the CI job. Physical tests happen on the owner's machine and come back as provenance rows. The Calibration Studio and its worker are not covered here.

## For the business

The software gate runs on the configured Linux toolchain: formatting, lint, unit tests, documentation and a scan that no vendor file has crept in. None of that touches the arm, by design and by a comment at the top of the script. Anything that depends on the arm itself is done by the owner with the arm connected, and the result enters the repository as a dated evidence record rather than as a pass on a build server. A green build therefore says the code is tidy and consistent, and nothing at all about how the arm feels.

## VT-06.1 From tool to provenance row

```mermaid
flowchart TB
    PR["port-probe: reads host modes, no claim, no negotiation<br/>tools/port-probe/src/main.rs:16<br/>crates/hal/src/lib.rs:52"]
    RO["device-readout: active setup then N raw packets, no force packets<br/>tools/device-readout/src/main.rs:10<br/>tools/device-readout/src/main.rs:22"]
    RC["port-recorder: LD_PRELOAD log of ppdev calls from a reference run<br/>tools/port-recorder/src/lib.rs:1"]
    WPA["with-port-access: group-access check before sudo<br/>with-port-access:10"]
    OWN["owner runs the attended session on the host<br/>AGENTS.md:59-61"]
    EV["P-NNNN row of class OBSERVE or OWNER and evidence under docs/evidence<br/>ADR-0005-build-with-quality-retrofit.md:40-42"]
    CI["CI and quality gate: software claims only<br/>ci.yml:3-5"]
    PR --> OWN
    RO --> OWN
    RC --> OWN
    WPA --> OWN
    OWN --> EV
    CI -. "never opens hardware" .-> EV
    N1["INVARIANT: CI never opens hardware, no gate step touches the device node<br/>quality-gate.sh:3 and AGENTS.md:47-48"]
    N2["INVARIANT: the readout sends no force packets and is bounded to 1 to 600 samples<br/>tools/device-readout/src/main.rs:19-20"]
    CI --- N1
    RO --- N2
```

The three tools are ordered by how much they do to the unit: the probe changes nothing, the readout performs an active setup sequence, and the recorder observes someone else's session.

## VT-06.2 Port recorder: one captured call

```mermaid
sequenceDiagram
    autonumber
    participant AP as reference program
    participant SH as shim read write ioctl<br/>shim.c:30
    participant BG as port_capture_begin<br/>tools/port-recorder/src/lib.rs:229
    participant LC as libc real call
    participant EN as port_capture_end<br/>tools/port-recorder/src/lib.rs:250
    AP->>SH: read, write or ioctl<br/>(ioctl-entry.S:6)
    SH->>SH: nested call on this thread goes straight to the syscall<br/>(shim.c:31)
    SH->>BG: fd, operation, argument, buffer pointer<br/>(shim.c:35)
    BG->>BG: only the canonical PORT_CAPTURE_TARGET descriptor is recorded<br/>(tools/port-recorder/src/lib.rs:170-173)
    BG->>BG: snapshot request buffer by process_vm_readv, capped at 64 KiB<br/>(tools/port-recorder/src/lib.rs:103-117)
    SH->>LC: forward unchanged, errno preserved<br/>(shim.c:37)
    LC-->>SH: result and errno
    SH->>EN: result, error<br/>(shim.c:39)
    EN->>EN: after-snapshot, append one JSON line per event<br/>(tools/port-recorder/src/lib.rs:260-277)
    Note over BG,EN: INVARIANT: panics are caught and reported, never propagated into the<br/>reference program<br/>tools/port-recorder/src/lib.rs:235 and<br/>tools/port-recorder/src/lib.rs:254
    Note over SH: INVARIANT: the third ioctl argument is forwarded untouched, no varargs<br/>read<br/>shim.c:60-62
    Note over BG: EXTERNAL: ppdev request numbers are public Linux UAPI values, matched in<br/>full<br/>tools/port-recorder/src/lib.rs:137-139
    Note over EN: DEBT: a failed log write only prints a diagnostic and the run carries on<br/>with an incomplete capture<br/>tools/port-recorder/src/lib.rs:279-281
```

Capture files are named by session, process and thread and land under the ignored vendor-original directory, so the raw logs never enter the tree.

## VT-06.3 Port access without root

```mermaid
flowchart TB
    S["with-port-access COMMAND<br/>with-port-access:5"]
    A{"parport0 already readable and writable?<br/>with-port-access:10"}
    X["exec the command directly<br/>with-port-access:11"]
    R{"running as root?<br/>with-port-access:15"}
    E["refuse: run as the ordinary user<br/>with-port-access:16-17"]
    SU["sudo -n as the same user, refreshing groups<br/>with-port-access:19"]
    S --> A
    A -->|yes| X
    A -->|no| R
    R -->|yes| E
    R -->|no| SU
    N1["POLICY: do not preload reference software as root, loader variables are set after the launcher<br/>README.md:34-35"]
    SU --- N1
```

The launcher refreshes the owner’s group membership. Its direct-exec branch runs before the root check, so the script does not enforce a universal prohibition on running as root (scripts/with-port-access:10-17).

## VT-06.4 Quality gate and CI

```mermaid
flowchart TB
    G1["ADR ledger frontmatter, edges, staleness<br/>quality-gate.sh:11"]
    G2["cargo fmt --check<br/>quality-gate.sh:14"]
    G3["clippy, all targets, warnings are errors<br/>quality-gate.sh:17"]
    G4["cargo test --workspace<br/>quality-gate.sh:20"]
    G5["strict rustdoc for every crate<br/>quality-gate.sh:23"]
    G6["viewer unit tests and format, skipped by SKIP_VIEWER<br/>quality-gate.sh:30-31"]
    G7["audio analysis tests only<br/>quality-gate.sh:35"]
    G8["no vendor artefact tracked<br/>quality-gate.sh:38"]
    CI["CI runs scripts/quality-gate.sh with full history<br/>ci.yml:44-45"]
    G1 --> G2 --> G3 --> G4 --> G5 --> G6 --> G7 --> G8
    CI --> G1
    N1["INVARIANT: ADR staleness needs full history, so CI checks out with fetch-depth 0<br/>ci.yml:19-22"]
    N2["DEBT: the recorder shim is meant to stay test-covered yet its check.py is run by hand and no gate step runs it<br/>AGENTS.md:53 and README.md:11 vs quality-gate.sh:35"]
    N3["DOC-DRIFT: ADR-0005 lists the gate steps without the vendor scan that the gate and BASELINE include<br/>ADR-0005-build-with-quality-retrofit.md:34-38 vs quality-gate.sh:37-38"]
    N4["INVARIANT: ADR-0005 is accepted, complete, live, matching the wired CI<br/>ADR-0005-build-with-quality-retrofit.md:5-7"]
    CI --- N1
    G7 --- N2
    G8 --- N3
    G1 --- N4
```

Local and hosted CI invoke the same script, but syscall policy and dependencies can differ. The separate recorder integration test is outside this gate; on 1 November its memory snapshots returned EPERM in the audit container, despite a green quality gate.
