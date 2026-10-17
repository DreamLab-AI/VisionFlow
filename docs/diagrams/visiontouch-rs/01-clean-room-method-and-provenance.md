---
id: VT-01
title: Clean-room method and provenance
area: visiontouch-rs
governing:
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/clean-room/POLICY.md
  - ../visiontouch-rs/docs/adr/ADR-0001-observation-first-clean-room.md
adrs: [visiontouch-rs:ADR-0001]
sources:
  - ../visiontouch-rs/docs/clean-room/POLICY.md
  - ../visiontouch-rs/docs/clean-room/PROVENANCE.md
  - ../visiontouch-rs/docs/clean-room/DECOMPILATION-LOG.md
  - ../visiontouch-rs/docs/legal/POSITION.md
  - ../visiontouch-rs/docs/adr/ADR-0001-observation-first-clean-room.md
  - ../visiontouch-rs/docs/BASELINE-visiontouch-rs.md
  - ../visiontouch-rs/docs/research/overload-and-calibration.md
  - ../visiontouch-rs/scripts/quality-gate.sh
  - ../visiontouch-rs/.gitignore
verified_commit: 77898bc42b29d5928b4eae00040ef0c26c3cd0a3
---

## For developers

Every protocol fact in visiontouch-rs must arrive through a ladder: an observation class, a `P-NNNN` provenance row, a spec paragraph in our own words, and only then Rust. Decompilation is the one class that needs a separate dated `D-NNNN` entry first, and whatever it yields is DECOMPILE class and never reaches the paper. The method is enforced by convention and commit order, not by CI: only the "no vendor artefact tracked" step is mechanical. This topic does not cover the legal landscape document or the paper itself.

## For the business

The lab owns one of very few surviving 1990s haptic arms and is studying how to talk to it without copying the maker's software, so the result can be written up as an academic paper and, later, used as an input device for the VisionFlow installation. The method is the product of the study: facts are watched on the wire or learned by using lawfully owned software as a black box, each one is logged with where it came from, and nothing from the maker is stored in the repository. Names such as the arm's trade name are used only to identify the hardware, with no affiliation with the rights holder. If the posture ever changes to release, sale or outside demonstration, the legal review is re-opened first.

## VT-01.1 Source classes and what may reach the paper

```mermaid
flowchart TB
    PUB["PUBLIC<br/>papers, datasheets, standards, kernel source<br/>docs/clean-room/PROVENANCE.md:16"]
    OWN["OWNER<br/>inspection of the owned unit<br/>docs/clean-room/PROVENANCE.md:17"]
    OBS["OBSERVE<br/>passive capture while original software drives the unit<br/>docs/clean-room/PROVENANCE.md:18"]
    BLK["BLACKBOX<br/>inputs to original software or hardware, outputs recorded<br/>docs/clean-room/PROVENANCE.md:19"]
    DES["DESIGN<br/>our own derivation, used on 9 rows but absent from the class table<br/>docs/adr/ADR-0001-observation-first-clean-room.md:34"]
    DEC["DECOMPILE<br/>interoperability only, needs a D-entry<br/>docs/clean-room/PROVENANCE.md:20"]
    ROW["P-NNNN provenance row<br/>docs/clean-room/POLICY.md:7"]
    SPEC["docs/spec paragraph in our own words<br/>docs/clean-room/POLICY.md:6"]
    PAPER["the paper<br/>docs/legal/POSITION.md:33"]
    PUB --> ROW
    OWN --> ROW
    OBS --> ROW
    BLK --> ROW
    DES --> ROW
    DEC --> ROW
    ROW --> SPEC
    PUB --> PAPER
    OWN --> PAPER
    OBS --> PAPER
    BLK --> PAPER
    DEC -. "excluded, CDPA s50B(3)<br/>docs/clean-room/POLICY.md:8" .-> PAPER
    N1["INVARIANT: the paper traces only to PUBLIC, OWNER, OBSERVE or BLACKBOX rows<br/>docs/legal/POSITION.md:33"]
    N2["OPEN: DESIGN is not in the class table nor in the paper-eligible list<br/>docs/clean-room/PROVENANCE.md:14-20 vs docs/legal/POSITION.md:33"]
    DES --- N2
    PAPER --- N1
```

Six classes, one ledger. Five of them may feed the paper (DESIGN pending the open question), DECOMPILE may feed only the spec.

## VT-01.2 Spec before code, provenance before spec

```mermaid
sequenceDiagram
    autonumber
    participant OB as observation<br/>docs/clean-room/POLICY.md:5
    participant PV as PROVENANCE.md<br/>docs/clean-room/PROVENANCE.md:1
    participant SP as docs/spec/*.md<br/>docs/clean-room/POLICY.md:6
    participant RS as Rust crate<br/>docs/BASELINE-visiontouch-rs.md:54
    participant QG as quality-gate.sh<br/>scripts/quality-gate.sh:37
    OB->>PV: append a row with source class (append-only)
    PV->>SP: spec states the behaviour in own words, cites the row
    SP->>RS: code implements the spec, never the reverse
    RS->>QG: gate runs fmt, clippy, tests, rustdoc (scripts/quality-gate.sh:14-23)
    QG->>QG: only the vendor-artefact scan polices this ladder (scripts/quality-gate.sh:38)
    Note over PV,RS: INVARIANT: spec and provenance precede code, evidenced by commit order,<br/>git log --follow docs/spec/10-device-readout.md in docs/adr/ADR-0001-observation-first-clean-room.md:61
    Note over RS,QG: OPEN: invariant 2 at :54 has no mechanical check, which BASELINE now states itself: invariants 2, 3 and 12 are process rules enforced by review and PROVENANCE ordering<br/>docs/BASELINE-visiontouch-rs.md:49
```

The ladder is ordered by the commit history only. Nothing in the gate fails a Rust change that lacks a spec paragraph or a provenance row.

## VT-01.3 The D-NNNN decompilation gate

```mermaid
sequenceDiagram
    autonumber
    participant OW as owner request
    participant DL as DECOMPILATION-LOG.md<br/>docs/clean-room/DECOMPILATION-LOG.md:5
    participant AN as analysis in ignored vendor-original<br/>.gitignore:7
    participant PV as PROVENANCE.md<br/>docs/clean-room/PROVENANCE.md:20
    participant SP as docs/spec
    participant PP as the paper
    OW->>DL: observation failed for this fact (why recorded, docs/clean-room/DECOMPILATION-LOG.md:31)
    DL->>DL: entry names date, artefact SHA-256, ONE interoperability question (docs/clean-room/DECOMPILATION-LOG.md:5-9)
    DL->>AN: analyst examines only the named check and its direct callers (docs/clean-room/DECOMPILATION-LOG.md:39-40)
    AN->>PV: own-word finding recorded as a DECOMPILE row (docs/clean-room/DECOMPILATION-LOG.md:45)
    PV->>SP: finding may shape the spec and conservative policy
    PV--xPP: blocked, DECOMPILE rows never reach the paper (docs/clean-room/POLICY.md:8)
    Note over DL: INVARIANT: no native analysis without a prior D-entry<br/>docs/BASELINE-visiontouch-rs.md:56 and docs/clean-room/DECOMPILATION-LOG.md:11
    Note over DL,PV: DOC-DRIFT: the log header says Initially empty at docs/clean-room/DECOMPILATION-LOG.md:3<br/>yet D-0001 at :11 and D-0002 at :52 follow
    Note over DL,AN: DIVERGENCE: PROVENANCE requires decompilation by the spec team never the implementer<br/>docs/clean-room/PROVENANCE.md:20 while both entries name a coding agent in the specification role<br/>docs/clean-room/DECOMPILATION-LOG.md:42 and :82
    Note over PV,SP: INVARIANT (observed by search): no docs/spec file cites P-0036 or P-0041, the exclusion is held by the research notes<br/>docs/research/overload-and-calibration.md:9 and :52
```

Two bounded entries exist, D-0001 and D-0002, both dated 2026-10-16 and both about release and safety behaviour. Their findings are deliberately not summarised here.

## VT-01.4 What stays out of the tree and what re-opens the legal review

```mermaid
flowchart TB
    R5["rule 5: no binaries, firmware, headers, manuals<br/>docs/clean-room/POLICY.md:9"]
    R6["rule 6: no verbatim vendor strings or tables<br/>docs/clean-room/POLICY.md:10"]
    R7["rule 7: own names, no vendor trademarks<br/>docs/clean-room/POLICY.md:11"]
    GI[".gitignore blocks vendor-original, disasm, sys, vxd, dll, exe, bin<br/>.gitignore:7-13"]
    SC["quality-gate scan of git ls-files<br/>scripts/quality-gate.sh:38"]
    TR["trigger points that re-open the review<br/>docs/legal/POSITION.md:42"]
    T1["release the software, even as source<br/>docs/legal/POSITION.md:44"]
    T2["sell services or hardware based on it<br/>docs/legal/POSITION.md:45"]
    T3["any decompilation of a vendor binary<br/>docs/legal/POSITION.md:46"]
    T4["contact from the rights holder<br/>docs/legal/POSITION.md:47"]
    POS["posture: private lab evaluation, no economic use<br/>docs/BASELINE-visiontouch-rs.md:91-92"]
    R5 --> GI
    GI --> SC
    R6 -. "judgement only" .-> SC
    R7 -. "judgement only" .-> SC
    POS --> TR
    TR --> T1
    TR --> T2
    TR --> T3
    TR --> T4
    N1["INVARIANT: no vendor artefact tracked, scan fails the gate<br/>docs/BASELINE-visiontouch-rs.md:51 and scripts/quality-gate.sh:38"]
    N2["DIVERGENCE: T3 has fired twice (D-0001 and D-0002) but the position document is unchanged<br/>docs/legal/POSITION.md:46 vs docs/clean-room/DECOMPILATION-LOG.md:11"]
    SC --- N1
    T3 --- N2
```

Only rule 5 is machine-checked, and only by file extension and path. Rules 6 and 7 rely on review.
