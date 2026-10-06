---
id: AB-36
title: Role isolation and the identity port
area: agentbox
governing:
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/docs/SECURITY-profiles.md
adrs: [ADR-2122, ADR-2027, ADR-2101, ADR-2078, ADR-2064]
sources:
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/services/agentbox-manifest/src/role_accounts.rs
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/bake-devuser-privilege.sh
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/docker-compose.override.yml
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/config/lib/role-custody.sh
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/config/custody/env-classes.json
  - ../project/agentbox/scripts/ci/env-secret-inventory.js
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/server.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/receipts.rs
  - ../project/agentbox/scripts/activation/role-isolation-rehearsal.sh
  - ../project/agentbox/scripts/activation/role-isolation-rehearsal.host.sh
  - ../project/agentbox/tests/config/role-isolation-rehearsal.test.sh
  - ../project/agentbox/tests/config/role-isolation-supervisor.test.sh
  - ../project/agentbox/docs/estate-closeout/x1-rehearsal-20261003T090708Z.json
verified_commit: 6466e39313c3eb4ba0cadfc2efd4e7ffa3ccc296
---

## For developers

Until this change every secret in the container reached every process. Compose feeds `.env` to PID 1, supervisord passes its environment to every child, and every long-running program ran as devuser (uid 1000), so a 0600 key file protected nothing from a co-resident agent (AB-16.1, AB-16.7). ADR-2122 is X-1 step 1 of the custody programme (swarm `custody-isolation-2026-10-03`). It gives each secret-bearing role its own Unix account, delivers that role's secrets into a root-owned `/run/secrets/<role>/`, and moves the sovereign, operator and JunkieJarvis keys behind one process. That process, `nostr-pod-bridge serve-identity`, signs a closed set of named operations for callers it authorises by kernel peer uid. All of it sits behind `[security].role_isolation` (`agentbox.toml:2227`), which ships **false**. The flag is boot-class. The image always carries both supervisor configs, the accounts, the delivery plan and the port, and the entrypoint picks a shape at its final `exec`. With the flag off the boot is today's. Three changes ship unconditionally, because they close bypasses that would void any boundary: devuser leaves group `root`, the root boot `PATH` drops workspace paths and Stage B runs once per container start (AB-02), and `/run/secrets` becomes its own mount. A fourth bypass, the host Docker socket, only closes under the flag, and only fully once the host narrows the socket's mode (see the Open on AB-36.7).

This topic covers the accounts table, the two supervisor configs, the `/run/secrets` delivery, the identity port's transport, authorisation and receipts, the environment classes, and the activation rehearsal. Boot ordering is in AB-02, the manifest key in AB-05, the compose mount and the browser sidecar's own key (Podkey, K_browser) in AB-06, the secrets inventory and the mirror child key in AB-16, governance signing in AB-14, and the baked producer upstream in AB-34. Not read here: the pods signer cutover in `management-api/lib/pod-signer.js` (AB-14 and AB-16 cover it) and the per-op signing code inside `port.rs` beyond its dispatch. The design (custody design §2, the swarm's working document) is recorded in ADR-2122, which is `proposed` with `activation_status: inactive`. It may move only on a flag-on rehearsal receipt from the owner's rebuild.

## For the business

The assistant container holds keys that speak for the organisation: its Nostr identity, the forum house account, the payment chain's signing keys and the door's emergency credential. Until now any program in the container, including any AI agent's tool, could read all of them. One mistaken or compromised tool could therefore impersonate the organisation or move value. This change gives each of those keys a single owner. Only the program that needs a key can read it, and the identity keys never leave one dedicated signing service. Everyone else asks that service to sign for them. It signs only a fixed list of kinds of action and keeps a receipt of every decision. The agents keep working as before; they just no longer hold the keys. It is switched off today. It switches on only after the owner rebuilds the container and a rehearsal script proves every boundary from both sides. Two things remain outside its reach. Provider credentials (model APIs, GitHub and similar) stay with the agents by deliberate exception, because the agents are their users. And the host machine's Docker control socket has to be tightened on the host itself.

## AB-36.1 The accounts table: one uid per secret-bearing role, baked into the image

```mermaid
flowchart TB
    subgraph TABLE["config/role-accounts.json — the single source (schema line 3)"]
        RANGE["uid_range 960-979 (config/role-accounts.json:4)<br/>965 reserved: the host docker group (config/role-accounts.json:6)"]
        subgraph ROLES["roles — uid = gid, shell /sbin/nologin"]
            ID["ab-identity 960 (config/role-accounts.json:22-23)<br/>programs nostr-relay, serve-identity (config/role-accounts.json:25)<br/>sole holder of sovereign, JunkieJarvis, concierge keys"]
            GW["ab-gateway 961 (config/role-accounts.json:39-40)<br/>holds no key, signs through the port (config/role-accounts.json:41)"]
            IN["ab-ingress 962 (config/role-accounts.json:46-47)<br/>break-glass bearer and session secret (config/role-accounts.json:51-52)"]
            SP["ab-spend 963 (config/role-accounts.json:56-57)<br/>account only, no program: Q6 defers the spend port (config/role-accounts.json:58-59)"]
            CH["ab-sidestr-dreamlab 964, ab-faucet-dreamlab 966<br/>ab-sidestr-dreamlab-txbt4 967, ab-faucet-dreamlab-txbt4 968<br/>(config/role-accounts.json:63, config/role-accounts.json:73, config/role-accounts.json:82, config/role-accounts.json:92)"]
            PK["ab-poker-citizen 970 (config/role-accounts.json:101-102)<br/>program poker-citizen, the forum poker house key (config/role-accounts.json:104)"]
            PK2["ab-poker-citizen-dreamlab-txbt4 971, ab-poker-coach 972<br/>(config/role-accounts.json:110-111, config/role-accounts.json:119-120)<br/>the BLAKES7 house seat and the practice coach, added 2026-10-03"]
            PK -.-> N2["INVARIANT: each poker seat's house key is delivered to its own uid; the programs are<br/>secret-bearing (config/role-accounts.json:16), each source the volume copy with the<br/>workspace file as legacy twin (config/role-accounts.json:106), each ledger a role-owned<br/>2750 dir seeded once (config/role-accounts.json:189-196)"]
        end
        GRP["group ab-identity-port 969<br/>members devuser, ab-identity, ab-gateway (config/role-accounts.json:130-133)"]
        DIR["dir /var/lib/agentbox/events/sign<br/>ab-identity:devuser 2750 (config/role-accounts.json:143-147)"]
    end
    subgraph BUILD["image build — flake.nix"]
        PW["agentbox-manifest role-accounts passwd >> /etc/passwd (flake.nix:3856)"]
        GR["agentbox-manifest role-accounts group >> /etc/group (flake.nix:3869)"]
        ROOTG["root:x:0: — devuser is no longer in group root (flake.nix:3863)<br/>UNCONDITIONAL, both modes, reversed only by a rebuild"]
        ROOTG -.-> N1["INVARIANT: devuser is not in group root in either mode (flake.nix:3863),<br/>probed from devuser by rehearsal check (f) (role-isolation-rehearsal.sh:25-26)"]
    end
    subgraph RUST["services/agentbox-manifest/src/role_accounts.rs"]
        VAL["validate (role_accounts.rs:297)<br/>a reserved uid is refused (role_accounts.rs:359)<br/>root, devuser, wheel, nobody are not role names (role_accounts.rs:541)"]
        LINES["passwd_lines (role_accounts.rs:564)<br/>group_lines: role groups have NO members (role_accounts.rs:581)"]
    end
    TABLE --> VAL --> LINES
    LINES --> PW
    LINES --> GR
    GR --> ROOTG
```

**What it shows.** One JSON table names every role, its uid, its programs and its secrets, and the image build turns it into `/etc/passwd` and `/etc/group` lines through a Rust subcommand that validates it first. The only shared group is the identity port's socket group. Role groups have no members, devuser included. **Why it is this way.** The rootfs is read-only and users are baked by Nix, so accounts cannot be switched at boot. They ship in both modes and are inert while the flag is off (ADR-2122 §1). uid 965 is skipped because a role with gid 965 would sit in the host Docker socket's group, and the Docker daemon is root on the host (`config/role-accounts.json:6`). Removing devuser from group `root` went first and alone, because no program documented a reason for the membership and only a rebuild reverses it.

## AB-36.2 Two supervisor configs from one transform, picked by the flag at exec

```mermaid
flowchart TB
    subgraph BUILD["image build"]
        TODAY["/etc/supervisord.conf — today's, rendered by flake.nix"]
        ISO["agentbox-manifest role-accounts isolate (flake.nix:3836)<br/>writes /etc/supervisord.roles.conf and /etc/agentbox/role-secrets.tsv"]
        TODAY --> ISO
        ISO --> R1["user=devuser becomes user=role (role_accounts.rs:914)"]
        ISO --> R2["environment= gains the role's /run/secrets paths (role_accounts.rs:958)"]
        ISO --> F1["REFUSE: a role program not starting as devuser (role_accounts.rs:906)"]
        ISO --> F2["REFUSE: a secret-bearing program no role runs — the build fails (role_accounts.rs:855)"]
        ISO --> F3["INVARIANT: a user=devuser program outside secret_bearing_programs whose environment<br/>names a key variable fails the build (role_accounts.rs:865-874). Key variable = suffix<br/>_PRIVKEY_HEX, _SK or _KEY_FILE (role_accounts.rs:769) or a ROLE-class name or its file twin<br/>(role_accounts.rs:793-801, role_accounts.rs:773-790); the build passes env-classes.json<br/>to isolate (flake.nix:3836); unit test role_accounts.rs:1241-1250"]
        T["drift test: with user= and environment= removed<br/>the two configs are identical (tests/config/role-isolation-supervisor.test.sh:11-13)"]
        R1 --> T
        R2 --> T
    end
    subgraph BOOT["Stage A, config/entrypoint-unified.sh"]
        FLAG["AGENTBOX_ROLE_ISOLATION from [security].role_isolation (entrypoint-unified.sh:346)"]
        READY{"ab_role_isolation_ready<br/>roles config and plan present (role-custody.sh:59)"}
        UN["ROLE-ISOLATION-UNAVAILABLE: boot as flag off (entrypoint-unified.sh:357)"]
        PICK["ab_supervisord_conf_pick (role-custody.sh:48)<br/>called at entrypoint-unified.sh:1216"]
        EXEC["exec supervisord -c the picked config (entrypoint-unified.sh:1222)"]
        FLAG -->|"flag on"| READY
        READY -->|"no"| UN --> EXEC
        READY -->|"yes"| PICK --> EXEC
        FLAG -->|"flag off: /etc/supervisord.conf (entrypoint-unified.sh:1207)"| EXEC
    end
    BUILD --> BOOT
```

**What it shows.** The isolated config is a pure function of today's rendered config and the role table, computed at build time. It differs only in the role programs' `user=` and `environment=` lines, and the entrypoint chooses between the two configs at its last line. An image that lacks either artefact boots as if the flag were off, and says so loudly. **Why it is this way.** Making the flag boot-class rather than rebuild-class lets the owner flip it and roll it back with a restart (ADR-2122 §2). Deriving the second config instead of hand-writing it means the two cannot drift, and the build itself refuses a new secret-bearing program that has no role, so a new chain cannot ship without an account.

**Debt:** the design's `role-exec` launcher (environment clear, store-only `PATH`, umask 077) does not exist yet. The `only user= and environment= differ` contract is what keeps this step reviewable without it (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:221-223`).

## AB-36.3 /run/secrets: its own root mount, one sealed directory per role

```mermaid
flowchart TB
    MNT["/run/secrets tmpfs mode=711 uid=0 gid=0 noexec nosuid nodev<br/>docker-compose.yml:110, flake.nix:3538<br/>its OWN mount: devuser owns /run but cannot rename a mount point"]
    MNT --> MODE{"[security].role_isolation"}
    MODE -->|"off"| OFF["chown 1000:1000, chmod 0700 as before (entrypoint-unified.sh:388)<br/>devuser's /run/secrets/nostr.key is written as today"]
    MODE -->|"on"| PREP["ab_secrets_root_prepare: root 0711 (role-custody.sh:89)<br/>not a mount point: ROLE-ISOLATION-DEGRADED secrets-mount (role-custody.sh:99)"]
    PREP --> CAP["_ab_role_env_capture: each ROLE variable to /run/secrets/role/NAME 0400,<br/>then unset from PID 1 (entrypoint-unified.sh:489, list at entrypoint-unified.sh:411)"]
    CAP --> DEL["ab_role_secrets_deliver runs the plan (entrypoint-unified.sh:1210, role-custody.sh:218)"]
    subgraph PLAN["plan rows written by isolate"]
        P1["root (role_accounts.rs:981)"]
        P2["role: dir 0500 plus home 0700 (role_accounts.rs:983)"]
        P3["env: write PID 1's VAR, 0400, unset (role_accounts.rs:986)"]
        P4["file: copy an at-rest source, 0400 (role_accounts.rs:988)"]
        P5["sockdir ab-identity-port 0750 (role_accounts.rs:995)"]
        P6["dir: receipts dir outside the mount (role_accounts.rs:1002)"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end
    DEL --> PLAN
    PLAN --> SEAL["seal each role dir last: 0500 owned by the role (role-custody.sh:351-353)"]
    SEAL --> SCRUB["_ab_role_env_scrub: any ROLE variable still present is a LEAK, unset (entrypoint-unified.sh:1221)"]
    SCRUB --> X["exec supervisord (entrypoint-unified.sh:1222)"]
```

**What it shows.** Under the flag, Stage A turns every role secret into a 0400 file in a 0500 directory owned by its role, inside a mount only root can write. The variables it came from leave PID 1's environment before supervisord starts, so no child inherits them. With the flag off, the same mount is handed to devuser as before. **Why it is this way.** `/run` is a tmpfs owned by uid 1000, so devuser can rename anything directly under it, root's entries included. Only a separate mount point is safe (ADR-2122's correction, `ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:328-335`). Delivery runs before supervisord, so no devuser process exists yet to race it, and it fails open with a counted, grep-able marker, leaving a role program with no secret to fail closed under supervisord (`config/lib/role-custody.sh:11-14`).

**Invariant:** under role_isolation, the runtime copy of each role secret is readable only by its role. The directory is sealed `0500` to the role (`config/lib/role-custody.sh:353`) inside a root `0711` mount (`docker-compose.yml:110`).

**Invariant (role_isolation vs the at-rest volume):** the flag confines the at-rest copies as well as the runtime ones. Under it the boot skips the devuser chown of the `agentbox-secrets` volume root (`config/entrypoint-unified.sh:548`) and calls `ab_custody_migrate` on the baked plan (`config/entrypoint-unified.sh:723`). The plan's `atrest` rows are every resolved file source plus each role's `at_rest` entries, and its `atrestdir` rows the table's `at_rest_dirs` (`role_accounts.rs:1011-1028`, `config/role-accounts.json:138-141`). `ab_custody_migrate` (`config/lib/role-custody.sh:447`) copies a missing canonical copy once from its legacy twin, hands every copy to its role at 0400 (`config/lib/role-custody.sh:556`) and each volume dir to root (`config/lib/role-custody.sh:567`); with the flag off `ab_custody_revert` (`config/lib/role-custody.sh:586`) hands them back (`config/entrypoint-unified.sh:725`). The faucets' sources are now the volume copies, the workspace files their legacy twins (`config/role-accounts.json:78`). Those twins are never deleted and stay devuser-readable by design until a manual purge (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:212-215`).

## AB-36.4 The identity port: one request's life

```mermaid
sequenceDiagram
    autonumber
    participant C as caller (devuser 1000 or ab-gateway 961)
    participant S as connection<br/>identity_port/server.rs:104
    participant P as Port.handle<br/>identity_port/port.rs:309
    participant A as Acl.grant<br/>identity_port/acl.rs:244
    participant R as Receipts.write<br/>identity_port/receipts.rs:124
    Note over S: serve-identity binds /run/secrets/ab-identity-port/identity.sock (identity_port/mod.rs:52)<br/>refuses a parent world-writable without sticky (identity_port/server.rs:47)<br/>or owned by neither root nor itself (identity_port/server.rs:64)<br/>socket chgrp ab-identity-port, mode 0660 (identity_port/server.rs:70-73)
    C->>S: one line of JSON, op and params
    S->>S: peer_cred, the kernel's SO_PEERCRED uid (identity_port/server.rs:106)
    S->>P: handle(peer, line) (identity_port/server.rs:137)
    P->>P: uid not in the ACL, refused (identity_port/port.rs:365)
    P->>P: op not in the closed list, refused (identity_port/port.rs:372)
    P->>A: is this op granted to this uid
    A-->>P: grant or none, none is refused (identity_port/port.rs:378)
    P->>P: sign with the held key for the named op only
    P->>R: one JSONL line, admit or refuse, no content and no secret
    R-->>P: written to a 0640 file (identity_port/receipts.rs:133)
    Note over P,R: INVARIANT: an admitted request whose receipt cannot be written<br/>is answered as a refusal, nothing signed off the record (identity_port/port.rs:324)
    P-->>C: one JSON response, the result or refused with a reason
```

**What it shows.** A caller writes one JSON line to a unix socket. The port takes the caller's uid from the kernel, not from anything the caller says, checks it against a static ACL and a closed operation list, signs only what the grant allows, and writes a receipt before it answers. A receipt that cannot be written turns an admission into a refusal. **Why it is this way.** ADR-2101 rules out a generic "sign this payload" port as a bypass, so the port offers named operations only. NIP-98 to a loopback HTTP port was rejected because the caller would need a key to sign the request, which brings back the key it must not hold (ADR-2122 §4). The existing Rust relay daemon became the service because it already loads the key and signs through an allowlist.

**Invariant:** authorisation is the peer uid from `SO_PEERCRED` (`services/nostr-pod-bridge/src/identity_port/server.rs:106`), never the socket mode. The socket group only narrows who can connect (`config/role-accounts.json:132`).

**Open (consumer cutover):** the pods signer moved to the port in W3, but JunkieJarvis, the live-mirror hook, the gateway and dream-engine are owed as W3b (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:224-225`). Under the flag today the gateway holds no key and so signs nothing until then.

## AB-36.5 What each caller may ask for

```mermaid
flowchart TB
    subgraph KEYS["keys held by ab-identity — config/custody/identity-port-acl.json"]
        CORE["core = file AGENTBOX_PRIVKEY_HEX, required, mirror root<br/>nip98 only to manifest:integrations.solid_pod_rs.base_url<br/>relays: loopback port 7777 and the workers relay (identity-port-acl.json:18-26)"]
        JJ["junkiejarvis = file JUNKIEJARVIS_PRIVKEY_HEX, optional (identity-port-acl.json:28-30)"]
    end
    subgraph DEV["caller uid 1000 devuser (identity-port-acl.json:34-43)"]
        D1["pubkey: core, junkiejarvis"]
        D2["nip98: core"]
        D3["sign_event: core, kinds 38410-38413, 38415, 30840, 30841 (identity-port-acl.json:39)"]
        D4["forum_event: junkiejarvis, kinds 1, 42, 31923 (identity-port-acl.json:40)"]
        D5["nip42_auth: core"]
        D6["mirror_key with secret true (identity-port-acl.json:42) — see AB-16"]
    end
    subgraph GWC["caller uid 961 ab-gateway (identity-port-acl.json:45-51)"]
        G1["pubkey: core"]
        G2["nip42_auth: core"]
        G3["mirror_key, public only"]
    end
    NEVER["INVARIANT: NEVER_GRANTABLE_KINDS 27235, 22242, 31400-31405, 38414 (acl.rs:71-73)<br/>an ACL granting one fails to load (acl.rs:267-268)<br/>governance decisions and graduations never carry the container key"]
    CORE --> DEV
    JJ --> DEV
    CORE --> GWC
    NEVER -.->|"checked at load"| DEV
    NEVER -.->|"checked at load"| GWC
```

**What it shows.** Two callers, each with an explicit list of operations, keys and, for raw event signing, kinds. Every other uid is refused before any parsing of the operation. Governance kinds, NIP-98 and NIP-42 kinds can never be reached through `sign_event`, because their dedicated operations carry URL and relay checks that a raw signature would bypass. **Why it is this way.** The ACL is keyed by numeric uid and loaded once at start, and a manifest reference that does not resolve grants nothing (`config/custody/identity-port-acl.json:12`). `dm_unwrap` is not an operation: decryption stays inside the relay process (ADR-2122, the port as built). The `mirror_key` grant to devuser returns the live-mirror child secret, which the design had specified as a `mirror_wrap` operation. That tension is recorded once, in AB-16.

## AB-36.6 Environment classes: what leaves PID 1 and what stays by exception

```mermaid
flowchart TB
    TAB["config/custody/env-classes.json — every env NAME the boot path and the role-secret readers use,<br/>in exactly one class, names only (env-classes.json:2)"]
    TAB --> ROLE["ROLE, 11 names (env-classes.json:33)<br/>sovereign, operator, bridge, JunkieJarvis, concierge, agent keys to ab-identity,<br/>break-glass bearer and session secret to ab-ingress, TAILSCALE_AUTHKEY to root"]
    TAB --> DC["DEVUSER_CLASS, 35 names (env-classes.json:90)<br/>each carries an exception, the Q8 default (env-classes.json:92)"]
    TAB --> NS["NON_SECRET (env-classes.json:232)"]
    ROLE --> EP["_AB_ROLE_ENV_VARS, NAME:role pairs (entrypoint-unified.sh:411)<br/>captured to files and unset under the flag only"]
    DC --> STAY["stay in devuser's ambient environment in BOTH modes"]
    CI["scripts/ci/env-secret-inventory.js --check"]
    CI -->|"entrypoint list must equal the ROLE set (env-secret-inventory.js:261-262)"| EP
    CI -->|"a DEVUSER_CLASS entry must name its exception (env-secret-inventory.js:253)"| DC
```

**What it shows.** The scrub is a denylist of role secrets, not an allowlist of the environment. Eleven names leave PID 1 under the flag. Thirty-five credentials stay in every devuser process by recorded exception. A CI check keeps the entrypoint's list and the table equal and refuses an exception without a reason. **Why it is this way.** A denylist leaves agent tooling working when the flag flips. The agents are the users of provider keys, so hiding those keys from devuser would break the agents and protect nothing (custody design §1, C8). Which of them should leave the ambient environment is owner question Q8, defaulted to "stay" for this step.

**Debt:** 35 DEVUSER_CLASS credentials, among them provider API keys, `BRIDGE_TOKEN` and `MANAGEMENT_API_KEY`, remain in the agent environment by exception Q8 (`config/custody/env-classes.json:90`, `config/custody/env-classes.json:92`, `ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:158-159`).

## AB-36.7 The activation rehearsal: two halves, one receipt, flag still off

```mermaid
flowchart TB
    subgraph IN["container half, as devuser (role-isolation-rehearsal.sh:4-7)"]
        A["(a) role secrets refused to devuser, /run/secrets a root 0711 mount (role-isolation-rehearsal.sh:10-12)"]
        A2["(a) socket dir: devuser can neither rename it nor create beside it (role-isolation-rehearsal.sh:282-289)"]
        B["(b) role programs RUNNING as their uid, environ unreadable (role-isolation-rehearsal.sh:13-14)"]
        C["(c) signing through the port verifies, refusals receipted (role-isolation-rehearsal.sh:15-18)"]
        D["(d) each chain's producer makes a block as its role (role-isolation-rehearsal.sh:19-21)"]
        E["(e) no role secret ambient, by NAME only (role-isolation-rehearsal.sh:22-23)"]
        F["(f) not in group 0, no docker socket, no sudo (role-isolation-rehearsal.sh:25-26)"]
    end
    subgraph HOST["host half, root on the host shell (role-isolation-rehearsal.host.sh:4-6)"]
        H1["root's view of (a), (b), (e)"]
        H2["h-socket: host socket mode RECORDED, not a gate (role-isolation-rehearsal.host.sh:18-19)"]
    end
    V{"verdict (role-isolation-rehearsal.sh:27-32)"}
    IN --> V
    HOST --> V
    V -->|"flag off"| ST["2 STAGED, every row not_applicable"]
    V -->|"flag on, all pass"| PA["0 PASS"]
    V -->|"any row fails"| FA["1 FAIL"]
    ST --> REC["committed receipt: verdict STAGED, flag false<br/>(x1-rehearsal-20261003T090708Z.json:4, x1-rehearsal-20261003T090708Z.json:8)"]
    PA --> ACC["ADR-2122 accepted and activated only on a flag-on PASS receipt<br/>from the owner's rebuild (ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:135-143)"]
```

**What it shows.** The rehearsal decides whether the flag may stay on. Its container half probes as devuser and its host half probes as root. A flag-off run is STAGED, not PASS. The only receipt in the repository is a STAGED run with the flag off. **Why it is this way.** The flag removes root's view from inside the container, so positive probes (a role can read its own file) need the host, and negative probes (devuser cannot) need devuser. Rebuilds and activation belong to the owner. Agents stop at "staged", and peer agent messages are not approval (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:142-143`).

**Open:** `role_isolation` ships off (`agentbox.toml:2227`) and is live only after the owner's boot rehearsal receipt. The decision is accepted only when both halves pass on a rebuilt image with the flag on, and `activation_status` moves only on that receipt (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:135-143`). The committed receipt is STAGED with the flag false (`docs/estate-closeout/x1-rehearsal-20261003T090708Z.json:4`, `docs/estate-closeout/x1-rehearsal-20261003T090708Z.json:8`).

**Open (R2):** the host Docker socket inode stays world-accessible until the host chmods it. The socket is a bind of the host inode that earlier boots widened, and skipping the chmod under the flag does not narrow it (`docker-compose.override.yml:116-117`). The flag-on boot can only detect and record `degraded:docker-socket` (`config/entrypoint-unified.sh:605`). The host half records the mode for the owner without gating on it (`scripts/activation/role-isolation-rehearsal.host.sh:18-19`), while ADR-2122 counts `degraded:docker-socket` as a failure (`ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:142`).

**Invariant:** the identity socket's directory cannot be renamed by devuser. It sits inside the root-owned `/run/secrets` mount (`docker-compose.yml:110`). Check (a) attempts the rename and a create beside it as devuser and passes only if both are refused (`scripts/activation/role-isolation-rehearsal.sh:282-289`). Test case 17 puts the directory under a devuser-writable parent, the `/run/agentbox` case, and shows (a) failing with the directory restored (`tests/config/role-isolation-rehearsal.test.sh:29-30`, `tests/config/role-isolation-rehearsal.test.sh:322`).

**Tension (sudoers vs the no-runtime-sudo design):** the sudo route now follows the BUILD-time flag, not the boot-time one. Built with `role_isolation` off, which is how it ships, the image keeps devuser in `wheel` (`../project/agentbox/flake.nix:3864`) and bakes a `NOPASSWD: ALL` drop-in (`../project/agentbox/config/bake-devuser-privilege.sh:44-47`, called at `../project/agentbox/flake.nix:3878`); built with it on, devuser leaves `wheel` and `root` and gets no drop-in (`../project/agentbox/config/bake-devuser-privilege.sh:29-39`). The same flake says the setuid wrapper is gone and there is no runtime escalation path (`../project/agentbox/flake.nix:3632-3636`). In the shipped image the grant is inert only because `no-new-privileges` is set (`../project/agentbox/docker-compose.yml:129`) and no setuid sudo ships, and an image built off but booted on records `degraded:devuser-sudo` rather than repairing it (`../project/agentbox/config/entrypoint-unified.sh:642`); check (f) probes `sudo -n true` (`../project/agentbox/scripts/activation/role-isolation-rehearsal.sh:591-592`). Reverting either guard on a flag-off build turns devuser back into root, past every role boundary in this topic.

**Drift (committed receipt vs rehearsal):** the committed receipt's uid table is "derived from /etc/agentbox.toml (design §2.2 numbering)" (`docs/estate-closeout/x1-rehearsal-20261003T090708Z.json:26`), from a run at an earlier commit (`docs/estate-closeout/x1-rehearsal-20261003T090708Z.json:16`). The rehearsal at this revision requires the baked table and plan, with no derived numbering (`scripts/activation/role-isolation-rehearsal.sh:41-43`). The receipt predates the rule it would now fail.
