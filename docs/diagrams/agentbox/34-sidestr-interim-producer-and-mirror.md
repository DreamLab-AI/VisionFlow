---
id: AB-34
title: The interim producer and mirror — announce, the mirror trust rule, and relays as the registry
area: agentbox
governing:
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2098, ADR-2101, ADR-2103, ADR-2105]
sources:
  - ../project/agentbox/config/sidechain/run-producer.sh
  - ../project/agentbox/config/sidechain/mirror-sync.sh
  - ../project/agentbox/config/sidechain/README.md
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/lib.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/tip.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/kinds.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/estate.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/tests/live.rs
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
  - ../project/agentbox/docs/BASELINE-container.md
verified_commit: ec60a8f14f4544520b4b1f6e8f5de2def4cfedcf
---

## For developers

The chain is live but nothing about it is supervised: `run-producer.sh` runs the upstream JS engine from durable checkouts in a tmux window until the specified `[program:sidestr-producer]` exists (`run-producer.sh:2-5`), and `mirror-sync.sh` pushes the block file into a GitHub Pages checkout because Pages already serves open CORS and Range requests, which is all a mirror is (`mirror-sync.sh:2-5`). There is no registry to register with: the relays are it, and any client asking for kind 33333 tagged `t=sidestr` lists every chain that has announced (`sidechain/README.md:59-62`).

## For the business

The settlement chain is reachable by anyone today, through public infrastructure the estate does not own or pay for, and it announces itself rather than being listed anywhere. The arrangement is explicitly temporary: it runs in a terminal window rather than as a managed service, so a container restart stops it, and the record says what must replace it.

## AB-34.1 The interim producer, and what it refuses to start without

```mermaid
flowchart TB
    subgraph pre["Preflight: every one of these must be readable or it exits 1 - run-producer.sh:28-30"]
        P1["/var/lib/agentbox/secrets/sidestr-dreamlab.key<br/>run-producer.sh:20"]
        P2["/var/lib/agentbox/secrets/sidestr-tbtc4.cookie,<br/>the parent node's RPC credential<br/>run-producer.sh:21"]
        P3["the sealed chain document - run-producer.sh:19"]
        P4["three upstream checkouts under WORKSPACE/sidestr/upstream:<br/>spec siding, schema kernel, blaketestnode<br/>run-producer.sh:17"]
    end
    pre --> EXEC["exec node siding.mjs produce<br/>run-producer.sh:33-39"]
    subgraph args["What it is told"]
        A1["port 3450 on loopback, block every 600 s,<br/>10 s with transactions - run-producer.sh:24"]
        A2["five default public relays: nos.lol, damus, primal,<br/>nostr.mom, oxtr.dev - run-producer.sh:26"]
        A3["parent RPC on the LAN testnet4 node, credential by<br/>COOKIE FILE, scanned from the funding height, paid from<br/>wallet sidestr-peg - run-producer.sh:22"]
        A4["--announce-mirror publishes the kind-33333 tip after<br/>every block - sidechain/README.md:59"]
    end
    EXEC --> args
    subgraph rule["The convention the wrapper keeps"]
        R["keys and RPC credentials are FILES under<br/>/var/lib/agentbox/secrets, never arguments<br/>run-producer.sh:4-5"]
    end
    pre --> rule
```

**Debt:** this is a tmux program, not a supervised one — the record it implements specifies `[program:sidestr-node]` on loopback port 9097 and `[program:sidestr-producer]` behind `[sidechain.signer].enabled` (`../project/agentbox/docs/BASELINE-container.md:320-321`), and neither exists, so nothing restarts the chain after a container restart (`../project/agentbox/config/sidechain/run-producer.sh:2-3`).

**Invariant:** the producer never learns a secret from its command line — the key and the parent cookie are paths, checked for readability before `exec` and passed as `--key-file` and `--parent-cookie` (`../project/agentbox/config/sidechain/run-producer.sh:35`, `../project/agentbox/config/sidechain/run-producer.sh:38`).

## AB-34.2 The announcement and the mirror trust rule

```mermaid
sequenceDiagram
    autonumber
    participant PR as the producer<br/>config/sidechain/run-producer.sh:34
    participant RL as five public relays<br/>config/sidechain/run-producer.sh:26
    participant CL as a client that knows only the chain id
    participant MI as the mirror<br/>config/sidechain/mirror-sync.sh:2

    PR->>RL: after every block, a kind-33333 tip: d is the chain id,<br/>content the last headers as hex, u tags naming mirrors<br/>crates/sidestr/sidestr-nostr/src/tip.rs:3-5
    Note over RL: t equals sidestr is what a directory filters on, because<br/>relays index SINGLE-LETTER tags only (tip.rs:9-10)
    CL->>RL: ask for the chain's announcement
    RL-->>CL: the event
    CL->>CL: verify the signature FIRST, decode second<br/>crates/sidestr/sidestr-nostr/src/lib.rs:104-107
    CL->>MI: read the mirror's chain.json
    MI-->>CL: the document, with its signer
    alt the document's signer is the announcement's author
        CL->>CL: accept the mirror (lib.rs:43-47)
        CL->>MI: hold it to the announcement: the header at its tip<br/>must be the announced one
        Note over CL,MI: it may be BEHIND but never AHEAD of the signer,<br/>which is lying (lib.rs:46-47)
    else it is not
        CL->>CL: refuse
    end
    Note over CL: a chain id is a NAME, not a proof: with only the id the<br/>newest announcement wins and the client shows the signer it<br/>settled on. One that already knows the signer takes no<br/>other's (lib.rs:48-50)
```

**Invariant:** nothing from a relay is trusted before `Event::verify` — the signature is checked, and only then must the transaction itself validate (`../project/agentbox/crates/sidestr/sidestr-nostr/src/lib.rs:52-53`).

**Tension (upstream parseTip vs this chain):** upstream accepts only announcement content whose length divides by 328, so every stock-family chain is invisible to the directory, explorer and wallet — the live `sidestr:dreamlab` announcement among them, whose single header is 160 characters (`../project/agentbox/crates/sidestr/sidestr-nostr/src/tip.rs:22-25`); the estate's fix is filed as sidestr/spec PR #7 with three pin bumps to follow its merge (`../project/agentbox/docs/proposals/sovereign-settlement.md:331`).

## AB-34.3 The mirror loop, and why GitHub Pages is enough

```mermaid
sequenceDiagram
    autonumber
    participant M as mirror-sync.sh<br/>config/sidechain/mirror-sync.sh:13
    participant P as the producer on loopback port 3450<br/>config/sidechain/mirror-sync.sh:12
    participant S as the block file directory<br/>config/sidechain/mirror-sync.sh:11
    participant G as a GitHub Pages checkout

    loop every 120 s by default (mirror-sync.sh:13)
        M->>P: curl chain.json with a 10 s cap
        alt it answered
            P-->>M: the served document, moved into place atomically<br/>config/sidechain/mirror-sync.sh:16-17
        else it did not
            Note over M: keep the previous copy, do not fail the loop
        end
        M->>S: copy blocks.dat and blocks.json (mirror-sync.sh:19)
        M->>G: if any of the three changed, or anything is untracked,<br/>read the tip height out of blocks.json<br/>config/sidechain/mirror-sync.sh:20-21
        M->>G: add the three, commit as mirror tip N, push<br/>config/sidechain/mirror-sync.sh:22-23
        alt the push failed
            G-->>M: report it and retry on the next pass (mirror-sync.sh:24)
        end
    end
    Note over G: Pages serves them with open CORS and Range requests,<br/>which is ALL a mirror is (mirror-sync.sh:2-4)
```

**Debt:** the mirror is a public git repository rather than the specified loopback port 9097 behind the nip98 proxy at `/chain/`, and the script says so in its own header (`../project/agentbox/config/sidechain/mirror-sync.sh:4-5`, `../project/agentbox/docs/BASELINE-container.md:320`).

**Open:** the loop pushes whenever anything in the Pages checkout is untracked, not only when the three mirrored files changed, so an unrelated stray file triggers a commit (`../project/agentbox/config/sidechain/mirror-sync.sh:20`).

## AB-34.4 The kind plane as built, and the band it moved into

```mermaid
flowchart TB
    subgraph ext["EXTERNAL, owned by the sidestr spec - crates/sidestr/sidestr-nostr/src/kinds.rs:4-7"]
        E1["23500 transaction, 23501 faucet: EPHEMERAL, so a message<br/>nobody was listening for is GONE - durability is the<br/>publisher's problem, never the relay's<br/>crates/sidestr/sidestr-nostr/src/kinds.rs:9-12"]
        E2["23510 to 23514, the level-2 round - see AB-35.3"]
        E3["33333 tip, 33500 rule, 33501 genesis: ADDRESSABLE,<br/>replaced per kind, pubkey and d<br/>crates/sidestr/sidestr-nostr/src/kinds.rs:13-15"]
        E4["33502 DUAL-SCHEMA peg record or pledge<br/>crates/sidestr/sidestr-nostr/src/kinds.rs:48-50"]
    end
    subgraph est["agentbox's own, from the 38400 to 38499 band"]
        B1["38420 sidestr-account-binding: addressable,<br/>d is chain id and did hex, content the derived spend<br/>pubkey, signed by the IDENTITY key<br/>crates/sidestr/sidestr-nostr/src/estate.rs:7-9"]
        B2["38421 to 38425 the settlement domain events, REGULAR,<br/>because evidence accretes and nothing replaces it<br/>crates/sidestr/sidestr-nostr/src/kinds.rs:15-16"]
    end
    MOVE["The earlier 38110 allocation sat inside the agent-response<br/>reservation and MOVED to 38420 under ADR-2105. Nothing<br/>outside this repo moves to accommodate it<br/>PROTOCOL-registry.md:170"]
    est --> MOVE
    B1 --> PIN["a name is never monetary identity, so the binding PINS the<br/>genesis hash in a genesis tag: re-sealing a chain under the<br/>same name does not carry the old binding over<br/>crates/sidestr/sidestr-nostr/src/estate.rs:11-14"]
    ext --> PROV["the EXTERNAL classification is load-bearing: upstream states<br/>its field names, kinds and document shapes are provisional,<br/>and an upstream change is ADR-2098's review trigger<br/>crates/sidestr/sidestr-nostr/src/kinds.rs:4-7"]
```

**Invariant:** the binding's `d` tag must name the event's own author, because the identity key is what signs it — `parse_account_binding` refuses any other pairing (`../project/agentbox/crates/sidestr/sidestr-nostr/src/estate.rs:10-11`).

**Drift (AB-31.7 and AB-31.8 vs the registry):** those diagrams record the account binding as kind `38110` allocated from the `38106`-`38201` range, which was true at their declared revision and is not true now — ADR-2105 moved it to `38420` (`../project/agentbox/docs/PROTOCOL-registry.md:170`).

## AB-34.5 Interim against specified

```mermaid
flowchart TB
    subgraph now["Running 2026-09-22 - sovereign-settlement.md:331"]
        N1["upstream JS producer in a tmux window<br/>config/sidechain/run-producer.sh:2"]
        N2["announcing kind 33333 to five public relays,<br/>first announcement reached 5 of 5"]
        N3["GitHub Pages mirror dreamlab-ai.github.io/sidestr-dreamlab<br/>sidechain/README.md:62"]
        N4["the relays ARE the registry - sidechain/README.md:60-61"]
    end
    subgraph spec["Specified, and not built - sidechain/README.md:66-71"]
        S1["[sidechain] in agentbox.toml and its schema entry"]
        S2["sidestr-node and sidestr-producer supervised programs"]
        S3["the mirror on loopback port 9097 behind the nip98 proxy<br/>at /chain/"]
        S4["the chain and asset URN kinds"]
        S5["the kind-38420 account binding"]
        S1 ~~~ S2 ~~~ S3 ~~~ S4 ~~~ S5
    end
    now --> GAP["Everything running is outside the container's supervision,<br/>its ingress and its manifest - see AB-32.5 for the schema<br/>fact that blocks the manifest half"]
    spec --> GAP
```

**Open:** none of the five specified pieces has a landing date, and PRD-024's P1 row lists them together with the first peg-in as outstanding (`../project/agentbox/docs/proposals/sovereign-settlement.md:331`).
