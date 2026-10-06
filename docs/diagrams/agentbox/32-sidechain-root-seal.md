---
id: AB-32
title: The first seal — sidestr:dreamlab, its sealed document, key custody, block file and parent node
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2096, ADR-2101, ADR-2103, ADR-2106]
sources:
  - ../project/agentbox/config/sidechain/dreamlab/chain.json
  - ../project/agentbox/config/sidechain/README.md
  - ../project/agentbox/tests/config/sidechain-genesis.test.sh
  - ../project/agentbox/.github/workflows/manifest-validate.yml
  - ../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/agentbox.toml
verified_commit: 6466e39313c3eb4ba0cadfc2efd4e7ffa3ccc296
---

## For developers

On 2026-09-22 the estate stopped proposing a settlement chain and sealed one: `sidestr:dreamlab`, genesis `4db37517…d453dbc0`, one signer, prefix `drm`, no pegs, an 80-byte stock header beside Bitcoin testnet4 (`sidechain/README.md:10`). The committed chain document is the chain's identity, not its configuration — changing a sealed field is a new chain, never an edit (`sidechain/README.md:5-6`). AB-31 is the design this partly implements, AB-33 the crates that can now verify it, AB-34 what is actually running.

**Drift (this topic vs ADR-2112):** the crates this topic points to in AB-33 moved to `DreamLab-AI/sidestr-rs` on 2026-09-23 (ADR-2112); the sealed chain document and its tests stay in agentbox `config/sidechain/`. See SR-01.

**Drift (this topic vs the directory it catalogues):** `sidestr:dreamlab` is no longer the only seal — `sidestr:dreamlab-txbt4` beside BLAKE2b testnet4 was sealed 2026-10-02 and its table ships enabled (`config/sidechain/README.md:11`, `../project/agentbox/agentbox.toml:1648-1649`); the genesis test now runs one pass per committed document, so it covers both chains. Neither chain is anchored: no producer writes checkpoints into its parent (`config/sidechain/README.md:13-18`).

## For the business

The estate owns a settlement chain rather than a plan for one, and it deliberately carries no value: the parent is Bitcoin's test network, nothing was pegged to it at minting, and real money stays behind a legal gate that must exist as working code first. What the seal buys is that a later claim about a payment can be checked by someone outside the estate.

## AB-32.1 The sealed document, field by field

```mermaid
flowchart TB
    subgraph identity["Identity — what a validator refuses to proceed past"]
        ID["id sidestr:dreamlab, name dreamlab<br/>dreamlab/chain.json:2, dreamlab/chain.json:3"]
        GEN["genesisHash 4db37517…d453dbc0<br/>dreamlab/chain.json:18"]
        SIG["signer 7092810a…4c76d62, 32-byte x-only hex<br/>dreamlab/chain.json:17"]
        CH["challenge 5120 concatenated with signer,<br/>the single-key taproot script<br/>dreamlab/chain.json:6"]
    end
    subgraph parent["Parent — the family decides the header (SPEC 3)"]
        P["parent tbtc4, Bitcoin testnet4 on the estate's node<br/>dreamlab/chain.json:4"]
        PL["powLimit 7fff…ffff, bits pinned, never retargets<br/>dreamlab/chain.json:7"]
        PF["stock 80-byte SHA-256d header, version bit 31 clear<br/>ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:197"]
    end
    subgraph econ["Peg and fee parameters"]
        PEGS["pegs is EMPTY: the genesis mints nothing and<br/>needs no parent funds - dreamlab/chain.json:16"]
        PC["pegConfirmations 6, refundBlocks 10000,<br/>pegoutBlocks 144, pegoutMin 10000, minFeeRate 1<br/>dreamlab/chain.json:10"]
        PX["addressPrefix drm, magic d981eab1<br/>dreamlab/chain.json:8"]
    end
    subgraph estate["Estate fields, beyond upstream's document"]
        D["depth 0 - dreamlab/chain.json:19"]
        C["containment: parent, currencyPin tbtc, cashOut false,<br/>protocolProfile sidestr/0.0.2 - dreamlab/chain.json:20"]
        CD["containmentDigest, SHA-256 of the JCS form<br/>dreamlab/chain.json:28"]
    end
    SIG --> CH
    CH --> GEN
    P --> PF
    PL --> PF
    PEGS --> GEN
    C --> CD
    identity --> RULE["A sealed field is a new chain, never a configuration edit<br/>sidechain/README.md:5-6"]
    estate --> RULE
```


**Invariant:** `challenge` is `5120` concatenated with `signer`, so a document cannot name one key and be sealed by another — the genesis test asserts exactly that equality (`../project/agentbox/tests/config/sidechain-genesis.test.sh:83`).

## AB-32.2 What the genesis commits to, and what it does not

```mermaid
flowchart TB
    subgraph committed["Committed by upstream's buildGenesis"]
        C1["the chain id, as the coinbase marker<br/>ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:206"]
        C2["the pegs - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:207"]
        C3["genesisTime - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:207"]
        C4["the signer's witness - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:207"]
    end
    subgraph notcommitted["NOT committed, although the record said they were"]
        N1["parent - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:207-208"]
        N2["comment, signers - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:207"]
        N3["every containment field - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:208"]
    end
    subgraph pin["The pin that would close it"]
        PIN["a coinbase pin: record carrying containmentDigest<br/>is UNBUILT - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:210-211"]
        UP["building it changes genesis construction, so it is an<br/>UPSTREAM proposal, never a local overlay - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:212-214"]
        PIN --> UP
    end
    committed --> BIND["Until the pin exists, the binding of parent and containment is the<br/>committed document plus its genesisHash<br/>sidechain/README.md:77"]
    notcommitted --> BIND
    BIND --> pin
    notcommitted --> TEN["TENSION: ADR-2103 D3 says parent and containment are<br/>bound on-seal. They are not. The record now says so in<br/>its own amendment - ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:208"]
```


**Tension (ADR-2103 D3 vs the sealed chain):** D3 states that `parent`, `headerProfile`, `currencyPin`, `cashOut`, `pegConfirmations` and `refundBlocks` are committed as a `pin:` record in the genesis coinbase (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:73-74`), but the seal established that upstream commits only four things and the pin is unbuilt (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:208`).

**Open:** whether the pin lands at all depends on an upstream proposal to change genesis construction, which has not been filed (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:213`).

## AB-32.3 The engine-free genesis gate, case by case

```mermaid
sequenceDiagram
    autonumber
    participant CI as manifest-validate.yml<br/>.github/workflows/manifest-validate.yml:157
    participant T as sidechain-genesis.test.sh<br/>tests/config/sidechain-genesis.test.sh:58
    participant DOC as config/sidechain/*/chain.json
    participant DAT as blocks.dat under SIDESTR_STATE_ROOT<br/>tests/config/sidechain-genesis.test.sh:29

    CI->>T: bash the test, one run per committed document
    T->>DOC: case 1, required fields present with the right shapes (:72-74)
    T->>DOC: case 1, id equals sidestr plus name, directory equals name (:77-78)
    T->>DOC: case 2, challenge equals 5120 concatenated with signer (:83)
    T->>DOC: case 3, parent is a SPEC 3.2 alias from btc, tbtc4, xbt, txbt4 (:84-85)
    alt the alias is a mainnet one
        T->>DOC: refuse unless p21Receipt is present (:86)
        Note over T: ADR-2103 D4 as far as it is built, a receipt field must<br/>exist. Resolving the receipt is NOT checked
    end
    T->>DOC: case 4, containmentDigest equals sha256 of JCS containment (:91)
    T->>DOC: case 4, containment.parent equals parent, cashOut is false (:92-93)
    T->>T: case 5, no bare 32-byte hex file beside the document (:101)
    alt the block file is reachable
        T->>DAT: case 6, first entry is height 0 (:114)
        DAT-->>T: header hash in the parent's family equals genesisHash (:124)
        DAT-->>T: prev is all zeros, header time equals genesisTime (:125-126)
        DAT-->>T: bit 31 clear on a stock-header chain (:123), set on a<br/>BLAKE2b v2 one (:118)
    else absent in CI
        T->>T: skip, the block file is not in the repository<br/>.github/workflows/manifest-validate.yml:155-156
    end
    T-->>CI: exit non-zero if any case failed (:137, :140)
```


**Invariant:** no key material can sit beside a chain document — case 5 greps the directory for a bare 32-byte hex file and fails the run if it finds one (`../project/agentbox/tests/config/sidechain-genesis.test.sh:101`).

**Debt:** case 3 enforces only that a mainnet parent carries a `p21Receipt` field. ADR-2103 D4 asks for a CI check that the receipt *resolves*, a second check forbidding a level-1 chain on a mainnet parent, and a node that refuses to open one; none is built (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:229-230`).

## AB-32.4 Where the three non-git things live

```mermaid
flowchart TB
    DOCF["chain document<br/>config/sidechain/dreamlab/chain.json, in git<br/>sidechain/README.md:24"]
    KEY["signer key, 32 bytes hex, mode 0400<br/>/var/lib/agentbox/secrets/sidestr-dreamlab.key<br/>sidechain/README.md:25"]
    BLK["block file and index, blocks.dat and blocks.json<br/>under WORKSPACE/sidestr/dreamlab, a host bind<br/>sidechain/README.md:26"]
    NODE["parent node, Bitcoin Core testnet4 on the LAN,<br/>RPC port 48332, wallet sidestr-peg<br/>sidechain/README.md:27"]
    subgraph custody["Custody rules the key obeys (ADR-2101 D3)"]
        K1["agentbox-secrets named volume, survives rebuilds"]
        K2["NEVER in identity.env"]
        K3["NEVER derived from the identity key"]
        K1 ~~~ K2 ~~~ K3
    end
    KEY --> custody
    DOCF --> REPLAY["Replaying the genesis needs the document, the key and<br/>two upstream checkouts, and no npm install<br/>sidechain/README.md:96-104"]
    KEY --> REPLAY
    REPLAY --> OPEN["open refuses a block file whose block 0 does not hash<br/>to the document's genesisHash - sidechain/README.md:104"]
    BLK --> OPEN
    NODE --> PEG["coins enter only by peg-in at pegConfirmations<br/>sidechain/README.md:37-39"]
```


**Invariant:** the block file is reproducible from the document plus the key while the chain is at genesis, which is why the repository can hold the document alone and still let anyone rebuild block 0 (`../project/agentbox/config/sidechain/README.md:26`).

## AB-32.5 What P1 delivered, and what it did not

```mermaid
flowchart TB
    subgraph done["Delivered 2026-09-22"]
        D1["the chain minted and replayed cold by the engine<br/>sovereign-settlement.md:336"]
        D2["block 0's header hashed independently to the document<br/>sovereign-settlement.md:336"]
        D3["the testnet4 peg wallet funded with 0.001 tBTC<br/>sovereign-settlement.md:336"]
        D4["the interim producer live, announcing to five relays<br/>see AB-34.1"]
        D5["2026-09-30: producer, Pages mirror and DREAM faucet supervised<br/>behind [sidechain] - sovereign-settlement.md:337, see AB-34"]
    end
    subgraph missing["Not yet, named by the same row"]
        M1["a native sidestr-node - the supervised producer is still<br/>upstream's JS engine - config/sidechain/README.md:172"]
        M2["the mirror on loopback port 9097 behind the nip98 proxy"]
        M3["the chain and asset URN kinds"]
        M4["the kind-38420 account binding"]
        M5["the first peg-in"]
        M1 ~~~ M2 ~~~ M3 ~~~ M4 ~~~ M5
    end
    subgraph gate["The manifest block, half landed"]
        G1["schema/agentbox.toml.schema.json sets<br/>additionalProperties false at the top level<br/>schema/agentbox.toml.schema.json:7"]
        G2["a sidechain entry now exists for the interim supervision keys only:<br/>enabled, announce_mirror, mirror, mirror_checkout, faucet, faucet_key_file<br/>schema/agentbox.toml.schema.json:3327-3368"]
        G3["it is closed too, additionalProperties false, so D1's parent and<br/>header_profile still cannot be written and the sealed document stays<br/>the only source of the parent - schema/agentbox.toml.schema.json:3329"]
        G1 --> G2 --> G3
    end
    done --> STATUS["implementation_status partial: the seal exists and is<br/>verifiable, D1, D3's pin and boot check, D4's CI receipt<br/>check and the faucet compile-out are not built<br/>ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:228-231"]
    missing --> STATUS
    gate --> STATUS
```


**Drift (ADR-2103 vs the schema):** the record's first-seal snapshot says `schema/agentbox.toml.schema.json` has no `sidechain` entry (`../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:215-217`); since `d0fa1b80b` it has one, for the interim supervision keys only (`../project/agentbox/schema/agentbox.toml.schema.json:3327-3368`). The conclusion still holds for D1, because the new entry is closed and names no `parent` or `header_profile` (`../project/agentbox/schema/agentbox.toml.schema.json:3329`).

**Debt:** the specified `[sidechain]` gate, with its catalogue entry, apply class and supervised program set (`../project/agentbox/docs/BASELINE-container.md:301`, `../project/agentbox/docs/BASELINE-container.md:356`), is still not expressible: what landed is an interim block of six supervision keys, and the specified keys (parent, header profile, signer and bridge children) wait on a schema change (`../project/agentbox/schema/agentbox.toml.schema.json:3329`). See AB-05.13 for where it would appear in the gate catalogue.

**Open:** PRD-024's question 9 is answered — the first seal is `tbtc4` with stock headers, sealed 2026-09-22 (`../project/agentbox/docs/proposals/sovereign-settlement.md:486-491`) — but questions 8, 13, 14 and 15 remain, question 19 (the next chain's parent, beside `txbt4`) was added on 2026-09-30 (`../project/agentbox/docs/proposals/sovereign-settlement.md:527`), and each changes what gets built (`../project/agentbox/docs/proposals/sovereign-settlement.md:495`).
