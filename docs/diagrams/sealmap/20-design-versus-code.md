---
id: SM-20
title: Design versus code after the evidence programme
area: sealmap
governing: [../sealmap/docs/DESIGN.md, ../sealmap/README.md]
adrs: []
sources:
  - ../sealmap/docs/DESIGN.md
  - ../sealmap/README.md
  - ../sealmap/.gitignore
  - ../sealmap/Cargo.toml
  - ../sealmap/.github/workflows/ci.yml
  - ../sealmap/crates/sealmap/Cargo.toml
  - ../sealmap/crates/sealmap/src/main.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/lock.rs
  - ../sealmap/crates/sealmap-corpus/src/seal/check.rs
  - ../sealmap/crates/sealmap-corpus/src/pack.rs
  - ../sealmap/bench/e0/Cargo.toml
verified_commit: eacc59ece21274273c6bf2c83d4642b4fa7acaf5
---
## For developers

`docs/DESIGN.md` was accepted on 2026-10-05 with every recommendation taken
(`docs/DESIGN.md:45`). Steps 1 and 2 of its order of work landed the same day,
and step 3 followed in the tree: the lock, `resolve`, `stale`, `seal-check`,
`verify`, `seal sign`, the retirement of the committed generated corpus,
`sealmap-dense` with its `sealmap dense` subcommand, and `pack`
(`docs/DESIGN.md:340`-`345`). Step 3 closed with the 0.2.0 publish
(`docs/DESIGN.md:344`-`345`), and 0.2.1 followed (`../sealmap/Cargo.toml:6`).

Then the evidence programme ran, and DESIGN now opens with its outcome
(`docs/DESIGN.md:3`-`41`). The code-lens surface is kept: extraction, ids and
hashes, the generated views, `sealmap-dense` and `pack`. The seal surface is
frozen as experimental: every staleness rule finer than per-file lost real
changes for at most about 2× fewer flags. `sealmap-ts` is dropped. The README
says the same to adopters (`README.md:41`-`46`, `README.md:367`-`378`). This
topic is the catalogue of the gap that leaves: what the design still
describes that the code does not do, which steps closed, and what is open.

It is a catalogue, not a plan. The design's order of work
(`docs/DESIGN.md:331`-`357`) is the plan. Where the build settled something
the design left loose, DESIGN now records it as built (`docs/DESIGN.md:114`-`116`,
`docs/DESIGN.md:122`-`136`, `docs/DESIGN.md:155`-`163`, `docs/DESIGN.md:213`-`233`), and the subsystem topics carry the detail: the
seal module in SM-15, the seal commands in SM-16, `generate --check` in
SM-14, `pack` in SM-17.

## For the business

What an adopter gets is a code lens: generated views that show what the
code does, a dense projection an agent can read at about a quarter of the
source's size, and review packs, one bounded, reproducible file holding the
topics to review and the code they cite. The designed variant that holds
diagrams alone, for an outside reviewer who should not see code, is not
built.

The seal workflow also exists, but the headline saving it was built for did
not survive measurement. Seals flagged only slightly fewer topics than
per-file flags and missed real changes, so the cheaper way to keep diagrams
true is per-file flags batched weekly, with a model triaging each one. The
seal commands stay, marked experimental; this repository does not seal its
own diagrams and its CI does not run the seal gate.

## SM-20.1 The design's layers against the tree

```mermaid
flowchart TB
    subgraph AUTH["Authored layer, docs/DESIGN.md:81"]
        A1["topics citing sym ids"]
        A1S["built: citations are found and checked<br/>by sealmap verify, check.rs:359"]
    end
    subgraph SEAL["Sealed layer, docs/DESIGN.md:82"]
        S1["seals.lock plus a sealed pointer<br/>per topic"]
        S1S["built: canonical lock, pointer,<br/>topic_hash, lock.rs:57"]
    end
    subgraph GEN["Generated layer, docs/DESIGN.md:83"]
        G1["model, index, dense projection,<br/>optional 1:1 Mermaid, gitignored"]
        G1S["model, index, 1:1 Mermaid and the dense<br/>projection under a gitignored .sealmap<br/>.gitignore:3, main.rs:51"]
    end
    subgraph PACK["Review pack, docs/DESIGN.md:84"]
        P1["pegged topics, dense slices,<br/>source windows"]
        P1S["built: topics, dense slices, source windows,<br/>pack.rs:274; diagrams-only --review NOT BUILT"]
    end
    A1 --> A1S
    S1 --> S1S
    G1 --> G1S
    P1 --> P1S
    AUTH --> SEAL --> GEN --> PACK
```

**What it shows.** All four layers exist, the generated one no longer
committed; of the review pack, only the diagrams-only mode is missing.

**Why it is this way.** Step 3 split in two: the seal surface first, then
`pack` and `sealmap-dense` (`docs/DESIGN.md:319`-`321`); `sealmap-dense` was
built on a branch and merged after the seal surface. The generated corpus
is rebuilt on demand and never trusted from disk (`docs/DESIGN.md:86`-`90`).

`sealmap-dense` came in under its estimate. The README now reports the
measured 0.17–0.29× source (`README.md:60`) where it quoted the research's
0.45×, and DESIGN records why the built projection is smaller: each callable
is expanded once and each signature printed once (`docs/DESIGN.md:181`-`186`).

**Debt (designed, not built):** `--review`, a pack of diagrams only, is in
the layer table (`docs/DESIGN.md:84`) but not in the binary
(`docs/DESIGN.md:223`, `crates/sealmap/src/main.rs:45`-`65`).

## SM-20.2 The order of work and where the tree stands

```mermaid
flowchart TB
    S1["1 repository: rename, dual licence,<br/>hardening, sealmap-extract<br/>DONE, docs/DESIGN.md:333"]
    S2["2 ids and hashes: sym grammar, sig and body,<br/>injective ids, schema v2<br/>DONE, docs/DESIGN.md:338"]
    S3["3 seal surface, retirement, sealmap-dense<br/>and pack DONE;<br/>published as 0.2.0<br/>docs/DESIGN.md:340"]
    S4["4 E0-R on VisionClaw and agentbox<br/>RAN as E0 to E0d: did not hold<br/>docs/DESIGN.md:346"]
    S5["5 sealmap skill, routing ADR<br/>CLOSED, docs/DESIGN.md:347"]
    S6["6 dogfood: seal this corpus<br/>CLOSED, docs/DESIGN.md:350"]
    S7["7 sealmap-ts<br/>DROPPED, docs/DESIGN.md:352"]
    S8["8 review A/B<br/>RAN as ER, docs/DESIGN.md:354"]
    S9["9 migrate VisionFlow<br/>CLOSED, docs/DESIGN.md:356"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
```

**What it shows.** Steps 1 to 3 are done and published. Step 4, the first
number the design was built to produce, ran as E0 and its successors E0b to
E0d and did not hold; step 8's review comparison ran as ER. The other steps
are closed or dropped (`docs/DESIGN.md:26`-`35`).

**Why it is this way.** The evidence endpoints are a fixed sequence, each
tested only if the previous one holds (`docs/DESIGN.md:304`-`305`). Step 4
failed, so the steps that depended on cheaper upkeep went with it.

The pre-registrations and results now live in the repository, one directory
per experiment under `docs/evidence/` (`docs/DESIGN.md:8`). The design's
older `research/01` to `05` are still cited from a design-session
scratchpad (`docs/DESIGN.md:46`-`47`).

## SM-20.3 Planned commands against present ones

```mermaid
flowchart TB
    subgraph PLAN["designed CLI, docs/DESIGN.md:196-206"]
        RS["resolve"]
        STL["stale since rev"]
        SC["seal-check"]
        VF["verify"]
        SG["seal sign"]
        GC["generate with --check"]
        DN["dense"]
        PK["pack"]
        EX["export --scip, later"]
    end
    subgraph NOW["present CLI, sealmap/src/main.rs:45-65"]
        C1["resolve, seal-check, verify,<br/>stale, seal sign, pack"]
        C2["generate with --check, model, dense"]
    end
    RS --> C1
    STL --> C1
    SC --> C1
    VF --> C1
    SG --> C1
    PK --> C1
    GC --> C2
    DN --> C2
    EX -.->|not built| NOW
```

**What it shows.** Every designed command except the later `export --scip`
exists, and `verify` now means the seal gate only; the old drift check is
`generate --check`.

**Why it is this way.** Git stays out of the libraries: `--since` and
`--diff` export the revision with `git archive` and read it as a second model
(`docs/DESIGN.md:208`-`211`, `crates/sealmap/src/main.rs:608`-`617`). `pack`
was built to its own flags rather than the planned ones: `--diff REV`
against the working tree instead of two named trees, and `--budget` instead
of `--max-bytes`. DESIGN records both (`docs/DESIGN.md:203`,
`docs/DESIGN.md:213`-`222`).

## SM-20.4 A sealed topic's life, as built

```mermaid
stateDiagram-v2
    [*] --> Sealed: seal sign writes the entry and the pointer
    Sealed --> Holds: ids resolve, sig and body match
    Sealed --> Behaviour: body changed
    Sealed --> Contract: signature changed
    Sealed --> Absent: id gone, candidates listed
    Sealed --> Unparsable: a file that may hold it fails to parse
    Sealed --> ProseEdited: topic_hash differs
    Sealed --> LockFault: pointer and lock disagree, or lock not canonical
    Behaviour --> Sealed: cheap cross-family review, sign
    Contract --> Sealed: ADR addendum, re-consolidate, review, sign
    Absent --> Sealed: confirm rename, edit the citation, sign
    ProseEdited --> Sealed: review the edit, sign
    Holds --> [*]
```

**What it shows.** The lifecycle the design gives a sealed topic
(`docs/DESIGN.md:141`-`153`), now with every state reachable from the code;
the review between a failing state and the next seal is the skill's step, not
the crate's.

**Why it is this way.** Code, topic and lock were to land in one commit,
ending the two-commit "change then re-stamp" routine (`docs/DESIGN.md:246`);
`seal sign` re-derives the entry from the code, so the lock is never
hand-edited (`docs/DESIGN.md:122`-`127`). The cycle works as drawn, but
`Holds` means only that no cited body changed: the per-symbol rule caught
0.76 of real staleness (`README.md:81`-`85`), which is why the surface is
frozen.

Reviewer family, evidence and signatures were left to a skill script,
`seal-gate.mjs`, next to `verify` (`docs/DESIGN.md:165`-`166`). With the seal
work frozen that script will not be written, so `seal sign` records whatever
reviewer string it is given and nothing enforces the cross-family rule.

## SM-20.5 Crates planned against crates present

```mermaid
flowchart TB
    subgraph HAVE["in the tree, ../sealmap/Cargo.toml:3"]
        H1["sealmap-model"]
        H2["sealmap-extract"]
        H3["sealmap-rust"]
        H4["sealmap-mermaid"]
        H5["sealmap-corpus: generate, write,<br/>the seal module and pack"]
        H7["sealmap-dense: skeletons, call trees,<br/>index, budgeted slices<br/>../sealmap/crates/sealmap/Cargo.toml:29"]
        H6["sealmap facade: features cli, parallel<br/>../sealmap/crates/sealmap/Cargo.toml:17"]
        H8["bench/e0: the E0 harness,<br/>a member, never published<br/>bench/e0/Cargo.toml:5"]
    end
    subgraph PLANNED["planned, docs/DESIGN.md:170-179"]
        P3["sealmap-ts on oxc, DROPPED<br/>docs/DESIGN.md:34"]
        P4["facade: sealmap-ts behind a default<br/>feature, oxc needs MSRV 1.97"]
    end
    HAVE --> PLANNED
```

**What it shows.** Seven crates exist, `sealmap-dense` the newest, and the
corpus crate has gained `pack`; the one planned crate is dropped. The workspace has an
eighth member, the E0 benchmark harness, which is never published
(`bench/e0/Cargo.toml:5`).

**Why it is this way.** The owner deferred `sealmap-ts` on 2026-10-05, to be
built only if E0-R on the Rust repositories showed the precise-staleness
gain was real (`docs/DESIGN.md:352`-`353`). It did not, so the adapter is
dropped (`docs/DESIGN.md:34`-`35`, `README.md:261`). The crate-surface row
and its default facade feature (`docs/DESIGN.md:175`, `docs/DESIGN.md:179`)
stay in DESIGN as history; the MSRV question they raised no longer arises.

The workspace is at `0.2.1` (`../sealmap/Cargo.toml:6`), so a lock signed by this tree
names `sealmap` 0.2.1; 0.2.0 was the first release with the seal module
(`crates/sealmap-corpus/src/seal/lock.rs:21`).

## SM-20.6 Dogfooding the gate on this repository

```mermaid
flowchart TB
    T["16 topics, path:line citations"]
    V["sealmap verify<br/>check.rs:359"]
    C["topics with no seal and no citation:<br/>coverage only, never a failure<br/>check.rs:391-392"]
    M["SM-11 cites a fixture id in prose:<br/>an unsealed citation, exit 1<br/>check.rs:390"]
    CI["no CI job runs verify, none planned<br/>ci.yml:9-84"]
    T --> V
    V --> C
    V --> M
    V -.-> CI
```

**What it shows.** Run on this repository at `ae478d9`, `sealmap verify` treats
fifteen topics as legacy coverage and fails on one: SM-11 writes an example
id from a fixture crate (`shop`) as an inline code span, which reads as a
citation of a symbol nobody sealed.

**Why it is this way.** Legacy `path:line` topics must not fail the gate, or
a corpus could not migrate one topic at a time (`docs/DESIGN.md:161`-`163`);
a global-shaped id in a code span is always a citation, so a mistyped one
fails rather than vanishing (SM-15).

Step 6, which would have run `verify` in CI here, is closed
(`docs/DESIGN.md:26`-`30`). SM-11's example id would still have to move into
a fenced block or be reworded before `verify` could pass on this repository
(`crates/sealmap-corpus/src/seal/check.rs:390`).
