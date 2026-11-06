---
id: VC-42
title: Codescape command policy and retained runtime artefacts
area: visionclaw
governing:
  - ../project/codescape/docs/BASELINE-codescape.md
  - ../project/codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
adrs: [ADR-0001, ADR-2148]
sources:
  - ../project/codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
  - ../project/codescape/crates/codescape-server/src/voice/jev.rs
  - ../project/codescape/crates/codescape-server/src/config.rs
  - ../project/codescape/crates/codescape-server/src/egress.rs
  - ../project/codescape/crates/codescape-server/src/routes/mod.rs
  - ../project/codescape/crates/codescape-server/tests/isolation.rs
  - ../project/codescape/deploy/codescape-entrypoint.sh
  - ../project/codescape/crates/codescape-layout/src/ptx_source.rs
  - ../project/codescape/crates/codescape-layout/src/cuda.rs
  - ../project/codescape/crates/codescape-layout/src/lib.rs
  - ../project/docker-compose.unified.yml
  - ../project/docs/adr/ADR-2148-enable-local-typed-decisions.md
verified_commit: {codescape: e3b76f63e801c6a8460110b93e88a0b420e9b0b4, visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d}
---

## For developers

This topic distinguishes dependency isolation, endpoint admission, model command selection and runtime artefact reuse. The [audit and executable probe](../../estate-review/2026-10-31-codescape-boundaries.md) compare standalone source with the consumed submodule. Cited at the submodule commit `e3b76f6` that VisionClaw `ec785afa9` records, not at standalone `main`, which is ahead by a formatting-only `jev.rs` change and the presence fence of VC-43. No source pin here establishes current service activation.

## For the business

An enabled model can choose a bounded view action even before anyone calibrates its confidence threshold. Endpoint settings also need a clear privacy contract. A startup message about falling back to CPU does not establish which previously built GPU artefact remains available. These distinctions matter when operators explain what a shared code-view service will do.

## VC-42.1 Extraction separates the build graph from runtime inputs

```mermaid
flowchart TD
    DEC["Separate repository and published wire dependency<br/>ADR-0001-extraction-from-visionclaw.md:28,33"] --> ISO["Check default and cuda resolution<br/>only visionclaw-wire prefix exception<br/>codescape-server/tests/isolation.rs:106,121,139"]
    ISO --> PATH["Reject VisionClaw git sources and external path packages<br/>codescape-server/tests/isolation.rs:215"]
    PATH --> LIC["Inspect normal and build dependency licences<br/>codescape-server/tests/isolation.rs:324"]
    DEC --> GPU["PTX crosses at runtime with manifest and ABI checks<br/>ADR-0001-extraction-from-visionclaw.md:34"]
    GPU --> LOAD["Read pair from configured directory<br/>codescape-layout/src/ptx_source.rs:93,105"]
```

**What it shows.** Cargo isolation and runtime artefact provenance have separate enforcement paths. The audit's eight isolation tests pass, including their forbidden-package control. **Why it is this way.** ADR-0001 removes build-time coupling while retaining a shared wire contract and runtime GPU input. These checks do not establish network isolation or current deployment identity.

## VC-42.2 Explicit endpoints return before the local/cloud gate

```mermaid
flowchart TD
    ON["Jev enabled?<br/>voice/jev.rs:157"] -->|yes| EP["Explicit endpoint configured?<br/>voice/jev.rs:175"]
    ON -->|no| OFF["FlagOff<br/>voice/jev.rs:158"]
    EP -->|yes| RES["Return resolved endpoint<br/>voice/jev.rs:177"]
    EP -->|no| PERM["Local assertion or cloud permission?<br/>voice/jev.rs:172,185"]
    PERM -->|no| NO["NoEndpoint<br/>voice/jev.rs:186"]
    PERM -->|yes| FALL["Try shared proxy then public API key<br/>voice/jev.rs:188,195"]
    FALL -->|found| RES
    FALL -->|none usable| NO
    RES --> LIST["Same resolver supplies allowlist URL<br/>voice/jev.rs:208; codescape-server/src/config.rs:282"]
    LIST --> SEND["Check origin before request and on redirects<br/>voice/jev.rs:279; codescape-server/src/egress.rs:103"]
```

**What it shows.** An explicit remote endpoint can resolve while every local/cloud permission flag is false. The implicit proxy branch is refused under those same permissions. The current host intentionally enables cloud routing (`docker-compose.unified.yml:505`).
**Why it is this way.** Endpoint configuration is trusted earlier than the documented locality gate. The probe establishes resolution and allowlist admission without contacting a remote service.

**Drift:** The local-only promise in `voice/jev.rs:11` and the resolver contract at `voice/jev.rs:154` do not cover the explicit-endpoint return at `voice/jev.rs:175`.

## VC-42.3 Confidence one remains an accepting value

```mermaid
flowchart TD
    GR["Bound command text to 512 characters and parse<br/>codescape-server/src/routes/mod.rs:269,300"] --> MISS["Only NoMatch consults Jev<br/>voice/jev.rs:381"]
    MISS --> SHAPE["Require choice type, finite confidence in range<br/>voice/jev.rs:336,340"]
    SHAPE --> SET["Reject none and unknown command keys<br/>voice/jev.rs:345,348"]
    SET --> COMP["Reject confidence below threshold<br/>default threshold is 1.0<br/>voice/jev.rs:167,171,349"]
    COMP -->|equal or above| VALID["Validate offered argument-free command<br/>voice/jev.rs:89,353"]
    VALID --> RETURN["Replace NoMatch with typed command<br/>voice/jev.rs:382"]
    RETURN --> ROUTE["Route accepts the command normally<br/>codescape-server/src/routes/mod.rs:302"]
```

**What it shows.** An exact-one response passes the default threshold. Lower confidence, an unknown choice or abstention preserves fallback. Grammar hits bypass the model.
**Why it is this way.** The implementation uses an inclusive acceptance threshold rather than an explicit disabled state. Host Compose enables the option and leaves the threshold empty (`docker-compose.unified.yml:504,508`).

**Tension:** Compose calls the enabled model advisory (`docker-compose.unified.yml:507`), while the selection path can return an action at confidence one (`voice/jev.rs:349,354`). This is a bounded view-command path, not an ontology writer.

## VC-42.4 A failed refresh can retain a loader-valid PTX pair

```mermaid
flowchart TD
    BUILD["build_ptx checks GPU source manifest<br/>codescape-entrypoint.sh:53,55"] -->|missing| MSG["Log CPU fallback and return success<br/>codescape-entrypoint.sh:56,57"]
    MSG --> KEEP["Previous output pair is not removed on this branch<br/>codescape-entrypoint.sh:54,57"]
    KEEP --> READ["CUDA constructor reads configured pair<br/>codescape-layout/src/cuda.rs:467,472"]
    READ --> CHECK["Manifest, content tag and ABI acceptance<br/>codescape-layout/src/ptx_source.rs:106,108,117"]
    CHECK --> ENGINE["CPU fallback occurs if CUDA construction fails<br/>codescape-layout/src/lib.rs:377,383"]
    BUILD -->|source present and output produced| SWAP["Stage pair, remove old directory, move replacement<br/>codescape-entrypoint.sh:90,94"]
    SWAP --> READ
```

**What it shows.** Missing source files do not invalidate retained runtime artefacts. The disposable probe executes the verbatim startup function and confirms that the retained pair still passes the actual loader's non-device checks.
**Why it is this way.** Extraction made PTX a separately supplied runtime input. The diagnostic conflates failure to produce a new pair with failure to load the old one.

**Open:** The probe does not load a GPU module. The resulting engine still depends on the CUDA feature, device and artefact; current service activation and device acceptance remain separate evidence obligations.
