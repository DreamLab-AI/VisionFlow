---
id: OM-10
title: Reasoning refusals and derivation-certificate publication
area: ontomark
governing:
  - ../ontomark/docs/adr/ADR-OM-004-owl2-el-as-the-local-reasoning-profile.md
  - ../ontomark/docs/adr/ADR-OM-009-every-published-closure-carries-a-derivation-certificate.md
adrs: [ADR-OM-004, ADR-OM-009]
sources:
  - ../ontomark/crates/ontomark-compile/src/reason.rs
  - ../ontomark/crates/ontomark-compile/src/whelk.rs
  - ../ontomark/crates/ontomark-compile/src/certificate.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/input/vocabulary.rs
  - ../ontomark/crates/ontomark-reason/src/profile.rs
  - ../ontomark/crates/ontomark-reason/src/classify.rs
  - ../ontomark/crates/ontomark-wasm/src/engine.rs
  - ../ontomark/crates/ontomark-wasm/src/reason.rs
  - ../ontomark/crates/rdfs-certificate/src/lib.rs
  - ../ontomark/crates/rdfs-certificate/tests/oo_cert.rs
verified_commit: {ontomark: 0caae5d4b695cb720942b9b9212017c00a80ad41}
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](13-certificate-transaction-and-identity-repairs.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## For developers

The shared typed facade distinguishes unsupported constructs. The compiler's graph extractor selects a narrower input first. The certificate then checks one named-subsumption set, and its files have a separate publication path from the bundle. The [source review and nine executable probes](../../estate-review/2026-11-01-ontomark-reasoning-and-certificates.md) establish these boundaries, including actual execution of the checksum-pinned independent checker.

## For the business

A certificate can let another person recheck a derivation without trusting the classifier. It does not prove that every conclusion is present or that the files belong to the bundle a user is reading. Current failure and path-overlap cases can separate that evidence from the published result. A useful acceptance test follows one inference from authored notes through publication to a reader checking the exact files.

## OM-10.1 A refusal table only sees the input it receives

```mermaid
flowchart TD
    V["Vocabulary mappings checked at load<br/>vocabulary.rs:523,570"] --> G["Compiler graph extraction selects supported shapes<br/>ontomark-compile/src/reason.rs:283"]
    G --> E["Legacy EL input converted to typed axioms<br/>whelk.rs:35"]
    E --> P["Typed profile check<br/>profile.rs:100,103"]
    DIRECT["Caller supplies typed axioms directly<br/>profile.rs:100,103"] --> P
    P --> REF["Outside EL or not implemented<br/>profile.rs:51,52"]
    P --> OK["Accepted axioms reach classifier<br/>classify.rs:222,225"]
    G -.-> DROP["TENSION: unknown shapes and unreadable definitions<br/>can be omitted before refusal check<br/>ontomark-compile/src/reason.rs:283,401"]
    OK --> COMPLETE["Complete only if no refusals or dropped existentials<br/>classify.rs:113,114"]
```

**What it shows.** A raw universal restriction produced zero compiler refusals, while the equivalent typed axiom was refused. A malformed graph definition was also omitted. These are Graph API probes, not a demonstrated bypass of normal vocabulary loading.

**Why it is this way.** The compiler adapter was written for its own emitted fragment. ADR-OM-004's refusal promise needs that boundary stated. The vocabulary's `NON_EL_VOCABULARY` code also covers some EL constructs the backend does not implement; its wording is broader than the typed facade's distinction.

## OM-10.2 Optional parity precedes certificate writes, consistency and publication

```mermaid
flowchart TD
    GRAPH["Build graph and classify<br/>mod.rs:685,689"] --> FLAG{"certificate_out set?<br/>mod.rs:692"}
    FLAG -->|yes| CERT["Certify same graph<br/>mod.rs:695"]
    CERT --> PAR{"Named inferred sets equal?<br/>mod.rs:696; certificate.rs:63,66"}
    PAR -->|no| STOP["CertificateParity refusal<br/>mod.rs:698"]
    PAR -->|yes| WRITE["Write certificate directly to requested directory<br/>mod.rs:715,716"]
    WRITE --> CONS{"Consistent?<br/>mod.rs:743"}
    FLAG -->|no| CONS
    CONS -->|no| UNSAT["Unsatisfiable refusal<br/>old bundle can retain new certificate beside it<br/>mod.rs:744"]
    CONS -->|yes| STAGE["Stage bundle files and markers<br/>mod.rs:763,1015"]
    STAGE --> SWAP["Promote only staged bundle<br/>mod.rs:1018"]
    SWAP -.-> LOST["TENSION: certificate inside output directory<br/>is removed by successful replacement<br/>mod.rs:311,313,316"]
```

**What it shows.** Parity is enforced when requested. A build without the flag has no certificate. Certificate writes occur before the consistency refusal and before bundle promotion. Successful output replacement discards an unstaged certificate placed below that output.

**Why it is this way.** ADR-OM-009 makes certification an option and writes to a caller-supplied directory. There is no shared transaction or output-path exclusion check for the certificate. The probes distinguish a valid parity refusal from these publication defects.

## OM-10.3 Checker soundness is separate from completeness and file identity

```mermaid
flowchart TD
    ASSERT["Asserted triples<br/>rdfs-certificate/src/lib.rs:529"] --> RULES["Six-rule RDFS saturation<br/>rdfs-certificate/src/lib.rs:110,432"]
    RULES --> STEPS["Write rule, conclusion and premises<br/>rdfs-certificate/src/lib.rs:547"]
    ASSERT --> CHECK["Independent checker receives two TSV files<br/>oo_cert.rs:54,55,56"]
    STEPS --> CHECK
    META["scope.tsv and asserted.sha256<br/>rdfs-certificate/src/lib.rs:540,564"] -.-> NO["Not inputs to this checker invocation<br/>oo_cert.rs:54,55,56"]
    CHECK --> SOUND["Accept genuine steps; reject forged conclusion<br/>oo_cert.rs:76,100"]
    STEPS -.-> OMIT["TENSION: omitting all steps still passes<br/>soundness does not establish closure completeness<br/>oo_cert.rs:55,56"]
    WRITE["Four sequential filesystem writes<br/>rdfs-certificate/src/lib.rs:589-592"] -.-> MIX["TENSION: failed second write<br/>can leave new assertions with old digest<br/>rdfs-certificate/src/lib.rs:589,590"]
```

**What it shows.** The checksum-pinned checker accepted a genuine certificate and rejected a forged conclusion. It also accepted an empty derivation file after a one-step certificate was truncated. That is consistent with checking the soundness of supplied steps. Generation metadata and assertion hashes need their own consumer checks.

**Why it is this way.** The checker deliberately stands outside the engine. The host must retain and verify the expected conclusion set and artifact identity if it wants a claim about an entire published closure. The current certificate writer does not atomically replace its four files.

## OM-10.4 Work budgets and completeness are different guarantees

```mermaid
flowchart TD
    START["Try All policy, then bounded Relevant fallback<br/>ontomark-wasm/src/reason.rs:233,389,394"] --> STEP["Default slice is 20000 work units<br/>engine.rs:36,567"]
    STEP --> DONE["Finish classification<br/>ontomark-wasm/src/reason.rs:316"]
    DONE --> INFO["Capability says partial EL<br/>el_complete false<br/>ontomark-wasm/src/reason.rs:221,222"]
    DONE --> RESULT["Per-run complete uses refusal and drop counts<br/>ontomark-wasm/src/reason.rs:370<br/>classify.rs:113,114"]
    DONE --> UNSUP["Carry named refusals to host<br/>ontomark-wasm/src/reason.rs:352,358"]
    STEP -.-> TIME["LIMIT: 25 ms is a recorded native measurement<br/>not a wall-clock deadline<br/>engine.rs:35"]
```

**What it shows.** The engine exposes both backend capability and per-run completeness. The scheduling budget counts saturation work; it is not a timer that interrupts an arbitrarily expensive unit of work after 25 ms.

**Why it is this way.** ADR-OM-004 chooses a tractable profile, but complexity theory and historical timings do not establish UI responsiveness on every ontology or device. The host needs measured latency and a clear incomplete-result state alongside the semantic checks.
