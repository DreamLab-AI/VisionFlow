---
id: VF-08
title: Compatibility matrix and dependency direction across the estate
area: visionflow
governing:
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/repository-map.md
  - docs/BASELINE-visionflow.md
adrs: [ADR-2006, ADR-2007]
sources:
  - docs/architecture/compatibility-matrix.md
  - docs/architecture/repository-map.md
  - docs/architecture/pod-tier-matrix.md
  - docs/architecture/licensing.md
  - docs/architecture/status-reconciliation.md
  - docs/ecosystem-map.md
  - docs/registers/gap-register-v1.1.md
  - docs/registers/gap-register-v1.2.md
  - docs/registers/gap-register-v1.3.md
  - docs/registers/F9-federation-fork-record.md
  - docs/adr/ADR-2006-canon-owns-crossrepo-view-not-implementation.md
  - docs/adr/ADR-2007-estate-closeout-evidence-roadmap.md
  - docs/estate-review/evidence/adr-inventory.json
  - scripts/generate-release-manifest.sh
  - scripts/estate-health/roster.json
  - scripts/drift-counter/allowlist.json
  - docs/releases/candidate-2026-05-22.json
  - docs/BASELINE-visionflow.md
  - MAINTAINERS.md
  - LICENSES/README.md
  - README.md
verified_commit: bec06dc3a
---

## VF-08.1 The nine repositories and the direction of dependency
```mermaid
flowchart TB
    classDef proto fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef app fill:#e4ecf8,stroke:#33559a,color:#222
    classDef canon fill:#fff4d6,stroke:#aa8833,color:#222
    classDef lib fill:#f2eaf6,stroke:#7a4a9a,color:#222

    VF["VisionFlow — coordination docs, website,<br/>release machinery, governing ADRs<br/>repository-map.md:10"]:::canon

    SPR["solid-pod-rs — Solid/JSS foundation:<br/>LDP, WAC, NIP-98, DID:Nostr, git pods<br/>repository-map.md:13"]:::proto
    VW["vowl-wasm — VOWL notation engine, MIT,<br/>outside the AGPL boundary<br/>repository-map.md:16"]:::lib
    KG["knowledgeGraph — published corpus and build<br/>pipeline; the OWL 2 TBox the Loom grounds on<br/>repository-map.md:17"]:::lib

    VC["VisionClaw — knowledge engineering, OWL+SHACL,<br/>PROV-O, GPU graph physics, XR, Judgment Broker<br/>repository-map.md:11"]:::app
    AB["agentbox — sovereign agent runtime, Nix container,<br/>skills and tools, Solid pod and Nostr bridge<br/>repository-map.md:12"]:::app
    NRF["nostr-rust-forum — forum kit, Workers, passkey auth,<br/>relay, governance UI<br/>repository-map.md:14"]:::app
    DLW["dreamlab-ai-website — DreamLab branded deployment<br/>and operator overlay for the forum kit<br/>repository-map.md:15"]:::app
    VG["visionGraph — authoring vault and publishing pipeline<br/>for narrativegoldmine.com<br/>repository-map.md:18"]:::lib

    SPR --> VC
    SPR --> AB
    SPR --> NRF
    NRF --> DLW
    VW --> VG
    VW --> KG
    KG --> VG
    KG --> VC

    VC <--> AB
    VC <--> NRF
    AB <--> NRF

    VF -.-> VC
    VF -.-> AB
    VF -.-> SPR
    VF -.-> NRF
    VF -.-> DLW
    VF -.-> VW
    VF -.-> KG
    VF -.-> VG

    LEG["Solid arrows: a build or runtime dependency, pointing at the dependent.<br/>Double arrows: peer traffic — the Nostr relay mesh, Agent Control Surface,<br/>Judgment Broker, agent events and pod inbox. Dotted: the canon DOCUMENTS<br/>a repository and depends on none of them<br/>repository-map.md:34"]

    N1["INVARIANT: no single repository contains the whole runtime<br/>repository-map.md:6"]
    N2["INVARIANT: VisionFlow holds no substrate implementation and does not assert<br/>implementation status about a sibling — repo-local docs stay authoritative<br/>for their own code<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:30"]

    E1["EXTERNAL: VisionClaw is area visionclaw, VC-01..VC-37"]
    E2["EXTERNAL: agentbox is area agentbox, AB-01..AB-28"]
    E3["EXTERNAL: solid-pod-rs is area solid-pod-rs, SP-NN"]
    E4["EXTERNAL: nostr-rust-forum is area nostr-rust-forum, NF-NN"]
    E5["EXTERNAL: dreamlab-ai-website is area dreamlab-ai-website, DW-NN"]
    E6["EXTERNAL: vowl-wasm is area vowl-wasm, VW-NN"]
    E7["EXTERNAL: knowledgeGraph is area knowledgegraph, KG-NN"]
    E8["EXTERNAL: visionGraph is area visiongraph, VG-NN"]
    E9["EXTERNAL: estate-level topology and interfaces are ES-01..ES-10"]
```

## VF-08.2 The ownership rule — one source of truth per protocol primitive
```mermaid
flowchart LR
    classDef owner fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef open fill:#fff4d6,stroke:#aa8833,color:#222

    RULE["Ownership rule: protocol primitives should have ONE source of truth<br/>repository-map.md:59"]

    RULE --> P1["Solid LDP, WAC, WebID, pod storage"] --> O1["solid-pod-rs<br/>repository-map.md:63"]:::owner
    RULE --> P2["DID:Nostr document and resolution primitives"] --> O2["solid-pod-rs, or a shared crate extracted from it<br/>repository-map.md:64"]:::open
    RULE --> P3["NIP-98 verification and replay contract"] --> O3["a SHARED CRATE that does not exist yet, using the<br/>most complete implementation as reference<br/>repository-map.md:65"]:::open
    RULE --> P4["Agent Control Surface kinds 31400 to 31405"] --> O4["nostr-rust-forum, with the schema consumed<br/>by agentbox and VisionClaw<br/>repository-map.md:66"]:::owner
    RULE --> P5["Judgment Broker domain"] --> O5["VisionClaw<br/>repository-map.md:67"]:::owner
    RULE --> P6["Operator deployment config"] --> O6["dreamlab-ai-website for DreamLab production;<br/>per-operator overlays elsewhere<br/>repository-map.md:68"]:::owner

    O3 --> D1["DIVERGENCE: NIP-98 is REIMPLEMENTED INDEPENDENTLY in four repos<br/>with divergent URL-matching and replay semantics, tracked as G3.<br/>Replay protection exists only in the forum and the CF pod tier<br/>compatibility-matrix.md:10"]:::open
    D1 --> D2["EXTERNAL: the four implementations are VisionClaw VC-NN,<br/>agentbox AB-NN, solid-pod-rs SP-NN and nostr-rust-forum NF-NN.<br/>Convergence is explicitly deferred pending WASM-compatible<br/>shared surfaces<br/>status-reconciliation.md:32"]

    O2 --> D3["EXTERNAL: did:nostr is canonicalised on the Multikey form by<br/>VisionClaw ADR-125; the shared crate has not been extracted<br/>compatibility-matrix.md:10"]:::open

    O5 --> D4["EXTERNAL: on VisionClaw main the broker runs as an inline decide/inbox<br/>handler plus an ElevationActor; the distributed BrokerActor lives on an<br/>unmerged branch. Elevation terminates at PR creation — no closing event<br/>fires on merge<br/>compatibility-matrix.md:13"]

    N["INVARIANT: cross-repo findings are TRACKED here and FIXED upstream.<br/>This repository records the divergence and cannot close it<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:50"]
```

## VF-08.3 Four enumerations of the estate that do not agree
```mermaid
flowchart TB
    classDef drift fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef src fill:#e4ecf8,stroke:#33559a,color:#222

    A["docs/architecture/repository-map.md — NINE repositories<br/>repository-map.md:8"]:::src
    B["scripts/generate-release-manifest.sh ROSTER — FOURTEEN<br/>generate-release-manifest.sh:86"]:::src
    C["scripts/estate-health/roster.json — FOURTEEN, a different fourteen<br/>estate-health/roster.json:4"]:::src
    D["docs/estate-review/evidence/adr-inventory.json — FOURTEEN,<br/>the set the release roster is asserted against<br/>adr-inventory.json:1"]:::src

    A --> A1["VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum,<br/>dreamlab-ai-website, vowl-wasm, knowledgeGraph, visionGraph"]
    B --> B1["the same nine MINUS vowl-wasm, PLUS loom, WasmVOWL, dream-engine,<br/>logseq, ruvector and RuView<br/>generate-release-manifest.sh:93"]
    C --> C1["the same nine PLUS loom, WasmVOWL, prose-sanitiser, diagram-ir<br/>and dream-engine — and it keeps BOTH WasmVOWL and vowl-wasm<br/>estate-health/roster.json:4"]
    D --> D1["keyed by workspace PATH, not name: project, project/agentbox,<br/>project4 and dream-machine rather than VisionClaw, agentbox,<br/>logseq and dream-engine"]

    A1 --> X1["DOC-DRIFT: the release roster names the WASM ontology visualiser<br/>WasmVOWL at path WasmVOWL; repository-map.md:16 and licensing.md:16<br/>name vowl-wasm at ../vowl-wasm. BOTH directories exist in the workspace<br/>and estate-health's roster lists them as SEPARATE repositories with<br/>different roles — a demo shell and a published crate"]:::drift

    B1 --> X2["DOC-DRIFT: ruvector, RuView and logseq appear in the release roster<br/>and the ADR inventory but in NEITHER repository-map.md nor licensing.md,<br/>so two of the estate's fourteen have no stated licence position"]:::drift
    C1 --> X3["DOC-DRIFT: prose-sanitiser and diagram-ir are measured nightly by the<br/>estate-health roster but appear in no release roster and no repository map,<br/>even though licensing.md:16 names them as the pattern vowl-wasm follows"]:::drift

    GUARD["The only automated link between two of these lists:<br/>release-manifest.test.sh asserts the roster size against<br/>adr-inventory.json rather than a hard-coded fourteen, so THOSE two<br/>cannot drift apart<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:127"]

    NOGUARD["DIVERGENCE: nothing reconciles repository-map.md or licensing.md against<br/>either roster. A repository can enter the estate through the nightly<br/>snapshot and never reach the dependency map or the licensing table"]:::drift

    XREF["The nightly snapshot that reads roster.json is VF-03;<br/>the release manifest that reads the ROSTER array is VF-07"]
```

## VF-08.4 The compatibility matrix — five substrates, six areas, one verdict column
```mermaid
classDiagram
    class Identity {
        VisionClaw "NIP-98, NIP-07, DID:Nostr, Solid/WAC; NIP-26 NOT implemented"
        agentbox "bootstrap BIP-340 key becomes the canonical did:nostr"
        nostr_rust_forum "passkey PRF derives Nostr keys; NIP-42 only in the WASM client"
        solid_pod_rs "foundation for DID:Nostr, NIP-98, Solid-OIDC, Tier 1/3 resolution"
        dreamlab_ai_website "WebAuthn PRF plus NIP-98; configured agent identities"
        VERDICT "four independent NIP-98 implementations, G3 open — matrix:10"
    }

    class Mesh {
        VisionClaw "client/subscriber by default unless a local relay is added"
        agentbox "standalone default; embedded relay on 7777 with pod-bridge"
        nostr_rust_forum "nostr-bbs-mesh is SCAFFOLD-ONLY; relay gates by pubkey allowlist"
        solid_pod_rs "standalone with an embedded NIP-01 relay, no NIP-42 AUTH"
        dreamlab_ai_website "CF Workers relay fan-out to public relays"
        VERDICT "standalone-first is the supported mode; federated-by-default is 1 of 4 — matrix:11"
    }

    class Pod {
        VisionClaw "embedded solid-pod-rs on 8484; WAC against did:nostr"
        agentbox "embedded Solid pod plus native overlay via Cloudflare Tunnel"
        nostr_rust_forum "CF Worker pod tier mirrors Solid/JSS; native tier supports git"
        solid_pod_rs "canonical LDP/WAC/WebID/auth/storage implementation"
        dreamlab_ai_website "CF pod tier plus native agentbox pod tier for git"
        VERDICT "strongest area; two-tier behaviour is the main gap — matrix:12"
    }

    class Governance {
        VisionClaw "inline decide/inbox handler plus ElevationActor on main"
        agentbox "publishes and subscribes 31400-31405; 12-tool ontology bridge"
        nostr_rust_forum "full implementation of the kinds; agent registry, cases, roles"
        solid_pod_rs "access-control foundation, not the workflow owner"
        dreamlab_ai_website "governance dashboard renders the kinds, NIP-98 signed"
        VERDICT "shipped inline on main; the closing ConceptElevated event is missing — matrix:13"
    }

    class OntologyBridge {
        VisionClaw "Oxigraph SPARQL store, Whelk-rs OWL 2 EL, canonical KG"
        agentbox "12 MCP tools proxy SPARQL to VisionClaw"
        nostr_rust_forum "not applicable"
        solid_pod_rs "not applicable"
        dreamlab_ai_website "not applicable"
        VERDICT "shipped cross-substrate, bidirectional with a consistency gate — matrix:15"
    }

    class TestsOps {
        VisionClaw "master fixtures relocated to tests/fixtures on 2026-06-29"
        agentbox "upstream vectors with SHA-256 checksums and upstream pins"
        nostr_rust_forum "its own CHECKSUMS.txt is RED, five files fail"
        solid_pod_rs "did-doc frozen pre-refresh, masked by passing self-checksums"
        dreamlab_ai_website "README and test docs conflict; some gates advisory"
        VERDICT "is-envelope is byte-identical everywhere; did-doc has 4-way drift — matrix:16"
    }

    Identity <|-- Mesh
    Mesh <|-- Pod
    Pod <|-- Governance
    Governance <|-- OntologyBridge
    OntologyBridge <|-- TestsOps

    note for Identity "The matrix is the canon's cross-repo VIEW. Each cell is a claim\nabout a sibling repository that the sibling's own docs remain\nauthoritative for. compatibility-matrix.md:6"
    note for TestsOps "DOC-DRIFT: every consumer's sync-fixtures.sh still points at the\ndeleted docs/specs/fixtures path — the same stale default VF-05.6\nfinds in check-fixture-drift.sh. compatibility-matrix.md:16"
    note for OntologyBridge "INVARIANT: the 12-tool figure is policed by the drift counter as the\nmcp-ontology-tools axis, with this file carrying two of its pinned\nsites. See VF-05.3. allowlist.json:45"
    note for Mesh "EXTERNAL: the mesh posture per substrate is VC-NN, AB-NN, NF-NN,\nSP-NN and DW-NN; the estate view is ES-01 and ES-03."
```

## VF-08.5 Stated versions against what the siblings actually pin
```mermaid
flowchart TB
    classDef drift fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef ok fill:#e0f2e4,stroke:#2f7a45,color:#222

    CLAIM["Where the canon states a sibling version"] --> C1["compatibility-matrix.md:12 — solid-pod-rs is canonical for<br/>native pods; the candidate manifest pins v0.4.0-alpha.15<br/>candidate-2026-05-22.json:52"]
    CLAIM --> C2["candidate-2026-05-22.json:54 — cargo test in VisionClaw v0.1.0,<br/>solid-pod-rs v0.4.0-alpha.15 and nostr-rust-forum v3.0-rc3;<br/>npm test in agentbox v0.1.0 and dreamlab-ai-website v1.0.0"]
    CLAIM --> C3["status-reconciliation.md:30 — nostr-rust-forum 3.0.0-rc11<br/>and dreamlab-ai-website default federated"]
    CLAIM --> C4["compatibility-matrix.md:15 — agentbox ships a 12-tool<br/>ontology bridge"]

    C1 --> D1["DOC-DRIFT: EXTERNAL — solid-pod-rs Cargo.toml now declares<br/>0.5.0-alpha.9. The matrix and the committed candidate are two<br/>minor versions behind. See SP-NN"]:::drift
    C2 --> D1
    C2 --> D2["DOC-DRIFT: EXTERNAL — nostr-rust-forum's workspace crates now<br/>declare 1.0.0-beta.10, not 3.0-rc3. See NF-NN"]:::drift
    C3 --> D2
    C3 --> D3["DIVERGENCE: the canon carries THREE forum version strings —<br/>v3.0-rc3, 3.0.0-rc11 and the real 1.0.0-beta.10 — in three<br/>documents, none of which references the others"]:::drift
    C4 --> OK1["CONFIRMED: the ontology-bridge TOOLS registry in the pinned<br/>agentbox checkout counts twelve. This is the one version-shaped<br/>figure the drift counter actually enforces. See VF-05.2"]:::ok

    WHY["WHY only one of these holds: the drift counter policies COUNTS,<br/>not versions. Its three axes are skills, mcp-ontology-tools and<br/>ontology-classes; no axis reads a Cargo.toml or package.json version<br/>allowlist.json:10"]:::drift

    FIX["What would close it: the release manifest already records each<br/>repository's resolved HEAD sha rather than a version string, which is<br/>the drift-proof form. The stale strings live in prose the manifest<br/>does not govern<br/>generate-release-manifest.sh:123"]

    NOTE["INVARIANT breached in spirit: every count the canon asserts must have<br/>one queryable source. A pinned sibling VERSION is a count-shaped claim<br/>with no queryable source and no gate<br/>BASELINE-visionflow.md:232"]:::drift
```

## VF-08.6 The licensing split and the AGPL boundary
```mermaid
flowchart TB
    classDef agpl fill:#e4ecf8,stroke:#33559a,color:#222
    classDef mit fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef open fill:#fff4d6,stroke:#aa8833,color:#222

    subgraph AGPLZONE["AGPL-sensitive — derived from or linked against the JSS/Solid stack"]
        L2["VisionClaw — AGPL-3.0-only in LICENSE and Cargo.toml;<br/>an MPL-2.0 relicense is PROPOSED and LICENSE.MPL ships<br/>in-tree but is NOT operative<br/>licensing.md:11"]:::agpl
        L3["agentbox — AGPL-3.0; networked agent runtime and management API<br/>licensing.md:12"]:::agpl
        L4["solid-pod-rs — AGPL-3.0; Solid/JSS foundation library and server<br/>licensing.md:13"]:::agpl
        L5["nostr-rust-forum — AGPL-3.0 per ecosystem docs; workspace<br/>crates may carry crate-level terms<br/>licensing.md:14"]:::agpl
        L8["knowledgeGraph — ODbL-1.0 DATA, AGPL-3.0 PIPELINE<br/>licensing.md:17"]:::agpl
    end

    subgraph OUTSIDE["Outside the AGPL boundary"]
        L7["vowl-wasm — MIT, a clean-room reimplementation of the VOWL<br/>notation carrying the upstream WebVOWL MIT notice.<br/>Consumed as a crates.io / npm dependency, NEVER linked<br/>into an AGPL substrate<br/>licensing.md:16"]:::mit
    end

    subgraph AMBIG["Stated but unresolved"]
        L1["VisionFlow — AGPL-3.0 with an MPL-2.0 relicense proposed;<br/>practical boundary is documentation and website assets<br/>licensing.md:10"]:::open
        L6["dreamlab-ai-website — a deployment repo that INHERITS<br/>obligations from the kit and components it consumes<br/>licensing.md:15"]:::open
        L9["visionGraph — an authoring vault whose publishing pipeline<br/>sits under the knowledgeGraph terms<br/>licensing.md:18"]:::open
    end

    RULE["BOUNDARY RULE: treat protocol and server components derived from or<br/>linked against the JSS/Solid stack as AGPL-sensitive unless the owning<br/>repository states otherwise. VisionClaw is AGPL-only today, so no MPL/AGPL<br/>import boundary currently applies and the substrates are uniformly<br/>AGPL-sensitive. If the relicense becomes operative, that review returns<br/>licensing.md:22"]

    OPENQ["Four open licensing questions, none of them closed:<br/>which forum crates are for independent publication and under what terms;<br/>whether VisionClaw links AGPL code at build time or over process boundaries;<br/>whether schemas, fixtures and protocol docs are reusable outside AGPL repos;<br/>and what commercial path applies to a combined enterprise deployment<br/>licensing.md:26"]:::open

    GAP1["DOC-DRIFT: LICENSES/ holds a README and no licence files, and the<br/>repository has NO root LICENSE — so the condition attached to reuse is<br/>still unmet while the README badge advertises AGPL-3.0<br/>LICENSES/README.md:5"]:::open
    GAP2["DOC-DRIFT: README.md:265 says AGPL-3.0-only across ALL FOUR code repos.<br/>licensing.md lists NINE, of which at least six carry code, and the release<br/>roster carries fourteen. The four-repo framing predates the wider estate"]:::open
    GAP3["DIVERGENCE: loom, logseq, ruvector and RuView appear in the release roster<br/>with a provenance flag but have no row in licensing.md at all, so the<br/>estate's imported and upstream classes have no stated licence position<br/>generate-release-manifest.sh:97"]:::open

    ROUTE["Escalation route: security and protocol issues go to the OWNING substrate<br/>first; cross-repository architecture issues belong in VisionFlow docs<br/>MAINTAINERS.md:10"]

    E["EXTERNAL, verified against the sibling checkouts: VisionClaw ships both<br/>LICENSE (AGPL) and LICENSE.MPL, solid-pod-rs and nostr-rust-forum ship AGPL,<br/>vowl-wasm declares license = MIT in Cargo.toml, knowledgeGraph ships LICENSE,<br/>LICENSE-DATA and LICENSE-EXPLORER, and dreamlab-ai-website ships none.<br/>See SP-NN, NF-NN, VW-NN, KG-NN and DW-NN"]
```

## VF-08.7 Pod tier matrix — four tiers, one capability envelope
```mermaid
flowchart TB
    classDef cf fill:#fff4d6,stroke:#aa8833,color:#222
    classDef emb fill:#e4ecf8,stroke:#33559a,color:#222
    classDef nat fill:#e0f2e4,stroke:#2f7a45,color:#222

    T1["CF Workers — nostr-bbs-pod-worker on wasm32,<br/>R2 plus Workers KV. Consumers: the forum and<br/>the branded website<br/>pod-tier-matrix.md:11"]:::cf
    T2["Embedded agentbox — the solid-pod-rs library under<br/>Tokio in a Linux container, host filesystem<br/>pod-tier-matrix.md:12"]:::emb
    T3["Native server — the solid-pod-rs-server binary,<br/>filesystem, memory or S3<br/>pod-tier-matrix.md:13"]:::nat
    T4["Git-capable — the native server with the git feature,<br/>filesystem plus a git repository<br/>pod-tier-matrix.md:14"]:::nat

    T1 --> C1["Everything the four tiers share: LDP basic containers,<br/>content negotiation, PATCH, conditional requests, HTTP COPY,<br/>glob GET, .meta sidecars, full WAC, NIP-98 and<br/>did:nostr resolution at Tiers 1 and 3<br/>pod-tier-matrix.md:21"]

    T1 --> M1["CF Workers LACKS: range requests, acl:origin enforcement,<br/>WAC 2.0 conditions, PaymentCondition, Solid-OIDC with DPoP,<br/>every notification channel, all federation, all git,<br/>and every admin surface<br/>pod-tier-matrix.md:28"]:::cf
    T2 --> M2["Embedded agentbox ADDS notifications, webhook signing,<br/>SSRF and path-traversal guards and per-pod quota, but still<br/>lacks WebAuthn, did:key, ActivityPub, CORS allowlist,<br/>PSK provisioning and git<br/>pod-tier-matrix.md:48"]:::emb
    T3 --> M3["Native server ADDS PaymentCondition over HTTP 402 Web Ledgers,<br/>WebAuthn passkeys, Schnorr SSO, an embedded identity provider,<br/>ActivityPub, a pod-resident NIP-05 endpoint and full admin<br/>pod-tier-matrix.md:37"]:::nat
    T4 --> M4["Git-capable ADDS only git: auto-init at provisioning,<br/>HTTP smart transport, and pods as application repositories<br/>pod-tier-matrix.md:57"]:::nat

    WHY["The CF limits are STRUCTURAL, not configuration gaps:<br/>wasm32 has no tokio::process, no POSIX filesystem, no Tokio runtime,<br/>and a 30-second CPU budget hostile to packfile generation<br/>pod-tier-matrix.md:94"]:::cf

    SEL["Tier selection: the forum and the website take CF Workers for edge<br/>deployment; agentbox takes the embedded tier for a full Linux runtime;<br/>standalone operators take the native server; the DreamLab developer<br/>cohort takes git-capable behind a Cloudflare Tunnel<br/>pod-tier-matrix.md:84"]

    MIG["Migration CF to native is ADMIN-DRIVEN, not self-service: provision,<br/>export from R2, PUT each resource with NIP-98, update the WebID's<br/>pod_base_url, optionally tombstone the R2 prefix<br/>pod-tier-matrix.md:116"]
    MIG --> LIM["Limitations stated plainly: no bulk migration tool exists, git history<br/>starts fresh, and NIP-05 may need reconfiguration<br/>pod-tier-matrix.md:142"]

    AUTH["INVARIANT: both tiers verify NIP-98 independently from the SAME<br/>secp256k1 pubkey. No token exchange, no shared session store,<br/>no cross-tier RPC — the signature is self-contained<br/>pod-tier-matrix.md:152"]

    EXT["EXTERNAL: solid-pod-rs owns every tier's implementation (SP-NN); the CF<br/>tier and its ADR-087/089/093/086 rationale live in nostr-rust-forum (NF-NN);<br/>the branded deployment is DW-NN; the embedded tier is AB-NN<br/>pod-tier-matrix.md:163"]
```

## VF-08.8 The status-reconciliation loop — how a stale claim is retired
```mermaid
sequenceDiagram
    autonumber
    participant OLD as "older audit prose in a PRD or ADR"
    participant REC as "docs/architecture/status-reconciliation.md"
    participant CODE as "sibling repository code, read directly"
    participant MAT as "docs/architecture/compatibility-matrix.md"
    participant REG as "docs/registers/gap-register-v1.N.md"

    Note over REC: PURPOSE: keep older audit language from being mistaken for<br/>current runtime status<br/>status-reconciliation.md:6

    OLD->>REC: a claim arrives with a date and an origin
    REC->>CODE: check it against merged code, not against sibling documentation
    alt the claim is now false
        CODE-->>REC: record a reconciled reading rather than deleting the original
        Note over REC: "Treat PRD-010 as historical audit plus target architecture,<br/>not proof that all listed gaps are still open"<br/>status-reconciliation.md:22
    else the claim has been resolved
        CODE-->>REC: move it out of the open list with the evidence
        Note over REC: IS-Envelope ownership: VisionClaw owns the spec, schema and<br/>eleven vectors — agentbox implements decode and dispatch.<br/>"No longer an open item"<br/>status-reconciliation.md:31
    else the claim was mis-stated
        CODE-->>REC: correct the MECHANISM, not just the status
        Note over REC: the gluon is a transient attractive edge on the spring kernel,<br/>NOT a per-node charge modulation<br/>status-reconciliation.md:39
    end

    REC->>MAT: the per-area posture row is updated in place at a dated pass
    Note over MAT: the 2026-07-03 pass corrected the mesh, governance, identity and<br/>ontology-bridge rows against merged code<br/>compatibility-matrix.md:6
    REC->>REG: item-level tier and canary state chain FORWARD into the next register cut

    Note over REC,MAT: DIVERGENCE: status-reconciliation.md carries no update after 2026-05-22<br/>while the matrix was reconciled 2026-07-03 and the registers ran to 2026-07-10.<br/>Its "current high-confidence open items" are the oldest of the three<br/>status-reconciliation.md:4

    Note over REC: DOC-DRIFT: it lists the forum at 3.0.0-rc11 as a current fact — the forum's<br/>crates declare 1.0.0-beta.10 — see VF-08.5<br/>status-reconciliation.md:30

    Note over MAT,REG: INVARIANT: a maturity claim above the tier its evidence supports is a<br/>GOVERNANCE DEFECT, not a footnote. Promoting a tier requires new evidence —<br/>a closure sha or a fired canary — never new prose<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:44
```

## VF-08.9 The register chain — immutable cuts and forward-chained corrections
```mermaid
stateDiagram-v2
    [*] --> v1_0

    state "v1.0 — the meta-PRD consolidated inventory" as v1_0
    state "v1.1 — P0 wave close, 2026-07-08" as v1_1
    state "v1.2 — P1 and P2 close, cuts the sprint" as v1_2
    state "v1.3 — broker-governance code-landing addendum, 2026-07-10" as v1_3
    state "F9 fork record — an immutable side record" as F9

    v1_0 --> v1_1 : "publish and supersede as a STATUS record, not a rescope"
    v1_1 --> v1_2 : "the next wave boundary cuts the next version"
    v1_2 --> v1_3 : "residual re-derivation chains forward into an addendum"
    v1_1 --> F9 : "a fork evaluated mid-sprint chains forward, never edits v1.1"
    F9 --> v1_2 : "carried at planned, with the fork record as its evidence"

    v1_3 --> pending : "the stack-up live-session promotion is STILL OWED"
    state "pending — a later stamp" as pending
    pending --> [*]

    note right of v1_1
        Once published, v1.1 is not edited. A post-publication
        correction is added as a VISIBLE correction note, with the
        original "Cut by" and "Immutability" claims left in place
        as written. gap-register-v1.1.md:33
    end note

    note right of v1_2
        Seven-tier maturity vocabulary: historical, planned, scaffolded,
        standalone, integrated, federation-verified, released. The
        Maturity column is the honest post-fixup tier of landed code;
        the Canary column is a separate liveness state.
        gap-register-v1.2.md:39
    end note

    note right of v1_3
        It records CODE closures against verified trees and explicitly
        does NOT fire the pending-live batch. No tier is promoted to
        integrated from a desk. gap-register-v1.3.md:25
    end note

    note right of F9
        0 of 3 criteria hold, so federation stays planned and the 38xxx
        kinds stay off the forum allow-list. The record states its own
        falsification conditions. EXTERNAL: the forum that would build
        it is NF-NN. F9-federation-fork-record.md:27
    end note

    note right of v1_0
        INVARIANT 5: a published register is immutable. Corrections
        chain forward into the next cut; they never edit an earlier
        register in place. gap-register-v1.2.md:31
    end note
```

## VF-08.10 The canon-only boundary — what VisionFlow may and may not assert
```mermaid
flowchart TB
    classDef may fill:#e0f2e4,stroke:#2f7a45,color:#222
    classDef maynot fill:#ffe0e0,stroke:#aa3333,color:#222
    classDef open fill:#fff4d6,stroke:#aa8833,color:#222

    Q["Is a given claim VisionFlow's to make, and is it backed?<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:26"]

    Q --> MAY["MAY assert — the cross-repo surface"]:::may
    MAY --> M1["the compatibility matrix<br/>compatibility-matrix.md:8"]:::may
    MAY --> M2["the release-evidence manifest and its schema<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:32"]:::may
    MAY --> M3["the shared maturity vocabulary<br/>gap-register-v1.2.md:39"]:::may
    MAY --> M4["the dependency direction and ownership rule<br/>repository-map.md:57"]:::may
    MAY --> M5["counts, each with ONE queryable source, enforced by the drift gate<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:35"]:::may

    Q --> NOT["MAY NOT assert"]:::maynot
    NOT --> N1["substrate implementation status — repo-local docs stay<br/>authoritative for their own code<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:31"]:::maynot
    NOT --> N2["any maturity tier above the tier its evidence supports<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:44"]:::maynot
    NOT --> N3["hand-typed maturity labels — they are READ from the substrates'<br/>own template maturity fields<br/>compatibility-matrix.md:48"]:::maynot
    NOT --> N4["substrate code of any kind: no server, no database, no Rust<br/>BASELINE-visionflow.md:52"]:::maynot

    MAY --> COST["CONSEQUENCE: cross-repo authority is concentrated in a few artefacts,<br/>so their CI gates are LOAD-BEARING, not optional — see VF-05<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:47"]

    OPEN["Still open, stated by the decision's own acceptance annex"]:::open
    OPEN --> O1["binding maturity claims to dated producer and consumer evidence<br/>is NOT done — the matrix sources maturity from the substrates'<br/>template fields with no dated receipt per claim<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:139"]:::open
    OPEN --> O2["the recorded fixture verdict is drift: consumer copies genuinely<br/>differ from the canonical corpus, and reconciling them is substrate<br/>work outside this repository's authority<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:141"]:::open
    OPEN --> O3["fixture-drift.yml stays red on GitHub Actions until sibling-repo<br/>credentials exist or a run is waived — deliberately. See VF-05.5<br/>ADR-2006-canon-owns-crossrepo-view-not-implementation.md:144"]:::open
    OPEN --> O4["ADR-2007 is proposed / partial / staged: it binds estate closeout to<br/>decision lineage and system evidence but does not declare the estate<br/>closed out<br/>ADR-2007-estate-closeout-evidence-roadmap.md:1"]:::open

    ECO["The wider framing this boundary sits inside: the ecosystem map's own<br/>gap register is SUPERSEDED by the closeout, and where the two disagree<br/>the closeout wins — the canon says so in the document itself<br/>ecosystem-map.md:7"]

    XREF["The gates that enforce this boundary are VF-05; the release machinery<br/>is VF-07; the estate-wide governance loop across all nine repositories<br/>is ES-05, and the build/deploy estate is ES-09"]
```
