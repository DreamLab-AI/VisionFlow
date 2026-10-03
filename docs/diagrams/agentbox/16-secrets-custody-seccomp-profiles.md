---
id: AB-16
title: Secrets custody, seccomp and runtime profiles
area: agentbox
governing:
  - ../project/agentbox/docs/SECURITY-profiles.md
  - ../project/agentbox/docs/INGRESS-identity.md
adrs: [ADR-2007, ADR-2026, ADR-2027, ADR-2033, ADR-2046, ADR-2122]
sources:
  - ../project/agentbox/config/seccomp-agentbox.json
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/scripts/ci/check-seccomp.sh
  - ../project/agentbox/config/hooks/trust-seed.cjs
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/management-api/lib/agent-identity.js
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/skills/email-search/SKILL.md
  - ../project/agentbox/docs/adr/ADR-2027-secret-custody-rotation-break-glass.md
  - ../project/scripts/backup-secrets.sh
  - ../project/agentbox/services/secret-backup/src/main.rs
  - ../project/agentbox/services/secret-backup/README.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/config/lib/role-custody.sh
  - ../project/agentbox/config/role-accounts.json
  - ../project/agentbox/config/custody/env-classes.json
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/scripts/ci/env-secret-inventory.js
  - ../project/agentbox/config/hooks/lib/operator-key.cjs
  - ../project/agentbox/management-api/lib/role-secret.js
  - ../project/agentbox/services/nostr-pod-bridge/src/role_secret.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/bootstrap.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/lib.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/mirror_key.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/tests/runtime-contract/RC-X1-06.sh
  - ../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md
verified_commit: {agentbox: d03defbeaca6c52d6bf3f7338d3f465a109fcdbf, visionclaw: 7d3ea2edb067432a57e6fe1fd951fd8254380bb8}
---

## AB-16.1 Container hardening posture — what actually confines the box

```mermaid
flowchart TB
    subgraph HOST["docker host"]
        CMP["docker-compose.yml:88-168"]
    end
    subgraph CTR["agentbox container"]
        SUP["supervisord PID 1 as ROOT<br/>required at boot for tmpfs subdirs, cert gen, chown to uid 1000"]
        PROG["every long-running program<br/>user=devuser uid 1000, except role programs<br/>under role_isolation, which run as their own uid 960-968"]
    end
    CMP -->|"read_only: true (compose:94)"| CTR
    CMP -->|"cap_drop: ALL (compose:95-96)"| CTR
    CMP -->|"cap_add: CHOWN FOWNER DAC_OVERRIDE AUDIT_WRITE KILL SETUID SETGID NET_ADMIN (compose:97-104)"| CTR
    CMP -->|"security_opt: no-new-privileges:true (compose:129)"| CTR
    CMP -->|"security_opt: seccomp=./config/seccomp-agentbox.json (compose:130)"| CTR
    CMP -->|"tmpfs: /tmp 8G, /run 256M, /run/secrets 8M root 0711 own mount, /var/log 128M, ~/.cache 4G, ~/.npm 4G, /usr/local/bin 8M exec+suid … (compose:107-127)"| CTR
    CMP -->|"deploy.resources: cpus 56 / memory 256G / pids 32768 limits, nvidia GPU count all reservation (compose:156-168)"| CTR
    SUP -->|"setgroups + setuid demotion"| PROG
    CMP -.-> N1["INVARIANT compose:88-93 — no runtime sudo.<br/>Root-at-boot via supervisord PID 1 is the ONLY elevation.<br/>No agent-facing process runs as root after bootstrap."]
    CMP -.-> N2["SETUID and SETGID are cap_add'ed for privilege DROPPING, not gaining.<br/>supervisord needs CAP_SETGID/CAP_SETUID to demote children to devuser.<br/>no-new-privileges:true neuters setuid FILE BITS at execve, which is a different axis and does not replace these caps."]
    CTR -.-> N3["DIVERGENCE — ADR-2007 governs configuration separation only.<br/>Profile isolation replaced Linux pseudo-user isolation, so profiles are NOT an OS boundary.<br/>SECURITY-profiles states this explicitly. Every profile shares uid 1000."]
    PROG -.-> N4["role_isolation OFF, the shipped default — a same-uid boundary. Every supervised program, MCP server, skill and agent runs as devuser,<br/>so each can read every other's 0600 key file, the AoE daemon token and the mirror key. see AB-16.7"]
    PROG -.-> N6["INVARIANT under role_isolation — each role's runtime secrets sit in /run/secrets/role, a 0500 dir of 0400 files<br/>owned by that role, config/lib/role-custody.sh:104-111, role-custody.sh:282-286. The at-rest volume<br/>is still chowned to devuser in both modes, entrypoint-unified.sh:526, entrypoint-unified.sh:554. see AB-16.13, AB-36"]
    CMP -.-> N5["The GPU/CPU resource envelope (compose:156-168) is a scheduling ceiling,<br/>not a security boundary — it widens the pids limit and grants a whole GPU;<br/>the confinement posture is entirely the caps/read_only/seccomp lines above. see AB-06"]
```

## AB-16.2 The seccomp profile is a supplemental denylist, not a sandbox

```mermaid
classDiagram
    class SeccompProfile {
        <<agentbox/config/seccomp-agentbox.json>>
        +string defaultAction
        +List~string~ architectures
        +List~SyscallRule~ syscalls
    }
    class RuleAfAlg {
        <<syscalls[0]>>
        +names socket
        +action SCMP_ACT_ERRNO
        +errnoRet 1
        +arg index 0 value 38 op SCMP_CMP_EQ
    }
    class RuleHighRisk {
        <<syscalls[1] — 46 names>>
        +action SCMP_ACT_ERRNO
        +errnoRet 1
        +args empty
    }
    class DockerDefault {
        <<overridden by explicit profile selection>>
        +not implicitly stacked
    }
    SeccompProfile --> RuleAfAlg
    SeccompProfile --> RuleHighRisk
    SeccompProfile ..> DockerDefault : explicit security_opt overrides default
    note for SeccompProfile "defaultAction = SCMP_ACT_ALLOW. This is INTENTIONALLY allow-by-default.<br/>The file calls itself supplemental, but explicit Docker security_opt selects it IN PLACE OF the default profile.<br/>An allowlist was REJECTED deliberately — the workload surface (Chromium, CUDA, Godot) is too wide to enumerate safely without breakage.<br/>architectures: SCMP_ARCH_X86_64, X86, AARCH64, ARM."
    note for RuleAfAlg "CVE-2026-31431 'Copy Fail' — blocks AF_ALG (family 38) socket creation,<br/>the algif_aead local privesc via splice(). Arg-indexed, so other socket families are unaffected."
    note for RuleHighRisk "add_key bpf clock_adjtime clock_settime create_module delete_module finit_module get_kernel_syms get_mempolicy<br/>init_module ioperm iopl kcmp kexec_file_load kexec_load keyctl lookup_dcookie mbind mount move_pages nfsservctl<br/>perf_event_open pivot_root process_vm_readv process_vm_writev ptrace query_module quotactl reboot request_key<br/>set_mempolicy setns settimeofday stime swapon swapoff sysfs _sysctl umount umount2 unshare uselib userfaultfd ustat vm86 vm86old"
    note for DockerDefault "DIVERGENCE — reading this file as the confinement boundary is wrong.<br/>Confinement comes from the COMBINATION: cap_drop ALL, read_only root, uid 1000, no-new-privileges,<br/>plus the selected custom seccomp profile and any independently configured host controls. Docker's default is not automatically stacked. see AB-16.1"
```

## AB-16.3 check-seccomp.sh — the invariant gate

```mermaid
sequenceDiagram
    autonumber
    participant CI as CI / pre-rebuild
    participant SH as check-seccomp.sh<br/>agentbox/scripts/ci/check-seccomp.sh
    participant F as seccomp-agentbox.json<br/>agentbox/config/seccomp-agentbox.json
    participant NODE as embedded node heredoc

    CI->>SH: sh scripts/ci/check-seccomp.sh
    SH->>SH: set -eu then ROOT resolve then FILE = $ROOT/config/seccomp-agentbox.json
    alt file missing
        SH-->>CI: fail "missing $FILE" then exit 1
    end
    SH->>NODE: node - FILE BASELINE (all 46 syscall names)
    NODE->>F: JSON.parse(readFileSync)
    alt parse throws
        NODE-->>CI: FAIL — the profile no longer parses
    end
    NODE->>NODE: assert doc.defaultAction === 'SCMP_ACT_ALLOW'
    Note over NODE: the gate asserts the profile STAYS allow-by-default — a well-meaning switch to a denylist default would FAIL this check,<br/>because that would silently turn a supplemental layer into a broken half-allowlist
    loop for each REQUIRED syscall
        NODE->>F: find an SCMP_ACT_ERRNO rule whose names include it
        alt not found
            NODE-->>CI: FAIL — a denial was dropped
        end
    end
    NODE-->>CI: exit 0
    Note over SH,NODE: RESOLVED ADR-2046 — all 46 unconditional syscall denials are gate-protected.<br/>The gate separately verifies socket / argument index 0 / AF_ALG 38 / SCMP_CMP_EQ.<br/>This is a structural source gate, not an attestation of a running container filter.
```

## AB-16.4 Boot privilege drop and the trust-seed hook

```mermaid
sequenceDiagram
    autonumber
    participant D as docker run
    participant SUP as supervisord PID 1 (root)
    participant EP as entrypoint-unified.sh<br/>agentbox/config/entrypoint-unified.sh
    participant TS as trust-seed.cjs<br/>agentbox/config/hooks/trust-seed.cjs
    participant CFG as ~/.claude.json
    participant CC as unattended claude sessions
    participant OP as operator, post-boot worktree

    D->>SUP: PID 1 as root — tmpfs subdir creation, cert generation, chown runtime dirs to uid 1000
    SUP->>EP: run bootstrap
    rect rgb(255,248,235)
    Note over EP,TS: trust pre-acceptance, ONCE at boot — entrypoint-unified.sh:1675-1688
    EP->>TS: node /opt/agentbox/config/hooks/trust-seed.cjs, gated on AGENTBOX_TRUST_SEED != 0 (entrypoint-unified.sh:1686-1687)
    Note over EP: DRIFT (resolved) — trust-seed is deliberately NOT a SessionStart hook any more.<br/>entrypoint-unified.sh:1681-1683: the old registration walked ~1,170 paths (avg 3.2s) per session start<br/>and raced Claude Code's own writes to ~/.claude.json — hooks-reconcile prunes any stale registration.
    end
    TS->>TS: parseArgs — depth default 5, --dry-run, extra paths
    TS->>CFG: read ~/.claude.json projects.<abs path>
    loop workspace root plus every git checkout or worktree under it, bounded by depth
        TS->>CFG: mark trusted, SKIP node_modules target .tmp .cache .venv venv .git dist build
    end
    Note over TS,CFG: INVARIANT trust-seed.cjs:14 — never removes or overwrites other per-project state. Idempotent, safe to run at any time.
    alt any error
        TS-->>EP: one line to stderr then exit 0 — FAIL-OPEN so hooks and boot never stall (trust-seed.cjs:106)
    end
    SUP->>CC: start every long-running program with user=devuser
    CC-->>CC: no "Do you trust the files in this folder?" dialog, for every checkout that existed at boot
    OP->>TS: node /opt/agentbox/config/hooks/trust-seed.cjs <path> — manual, for a worktree made AFTER boot (entrypoint-unified.sh:1684)
    Note over TS,CC: rationale trust-seed.cjs:19 — observed 2026-09-02, ten Opus worker panes sat dead for an hour behind the trust gate.<br/>DIVERGENCE — this pre-accepts a SECURITY prompt for unattended agents. It is a deliberate availability-over-confirmation trade, not a hardening measure.
```

## AB-16.5 Derived keys — HMAC-SHA256 domain separation

```mermaid
sequenceDiagram
    autonumber
    participant ENV as operatorKeyHex<br/>agentbox/config/hooks/lib/operator-key.cjs:55
    participant GW as gateway.cjs<br/>agentbox/config/nostr-gateway/gateway.cjs:186-196
    participant MIR as deriveChildKey<br/>agentbox/config/hooks/nostr-live-mirror.cjs:211
    participant PHONE as operator phone (Amethyst)
    participant RELAY as relays

    rect rgb(235,245,255)
    Note over ENV,GW: gateway identity — gateway.cjs:186-196
    GW->>ENV: operatorKeyHex (gateway.cjs:186)
    alt role_isolation off
        ENV-->>GW: first non-empty of AGENTBOX_PRIVKEY_HEX, AGENTBOX_BRIDGE_SK, OPERATOR_NOSTR_PRIVKEY (operator-key.cjs:58)
    else role_isolation on
        ENV-->>GW: the role-secret file only, a bare variable is a reported leak (operator-key.cjs:59-69)
        Note over GW: as ab-gateway it holds no key file, so it gets none and exits below. ADR-2122 owes the port cutover (W3b)
    end
    alt not 64-hex
        GW-->>GW: log "no operator key (AGENTBOX_PRIVKEY_HEX) — exiting" then process.exit(0) (gateway.cjs:187)
        Note over GW: FAIL-CLOSED by exit — the gateway refuses to run keyless rather than degrading
    end
    GW->>GW: adminPub = getPublicKey(rawSk) — the ONLY pubkey allowed to command (gateway.cjs:189,195)
    alt AGENTBOX_GATEWAY_IDENTITY === 'gateway'
        GW->>GW: sk = HMAC-SHA256(operator_sk, AGENTBOX_GATEWAY_KEY_TAG or 'agentbox-gateway-v1') (gateway.cjs:192)
    else default 'operator'
        GW->>GW: sk = rawSk — the gateway signs AS the operator (gateway.cjs:193)
        Note over GW: DIVERGENCE — the DEFAULT identity is 'operator', so out of the box the gateway holds and signs with the ROOT operator key.<br/>The derived-child mode exists but is opt-in via env.
    end
    GW->>RELAY: auth and receive as getPublicKey(sk), reply to AGENTBOX_GATEWAY_REPLY_TO or adminPub (gateway.cjs:196)
    end
    rect rgb(240,255,240)
    Note over MIR,PHONE: mirror child key — nostr-live-mirror.cjs:211-222, operator hex from operatorKeyHex (nostr-live-mirror.cjs:214)
    MIR->>MIR: cached in _childCache, computed once per process
    alt AGENTBOX_MIRROR_CHILD === '0'
        MIR-->>MIR: null — child mode off, LEGACY operator-self-DM path is used instead
    else operator key not 64-hex, which is the devuser hook's case under role_isolation
        MIR-->>MIR: null — same legacy fallback, sealed under a throwaway sender when no operator key is readable (nostr-live-mirror.cjs:225-233)
    else
        MIR->>MIR: child_sk = HMAC-SHA256(operator_sk, AGENTBOX_MIRROR_KEY_TAG or 'agentbox-mirror-v1')
        MIR->>PHONE: only the CHILD nsec is imported to the device — the root operator key stays off the phone
        Note over MIR,PHONE: rotatable by bumping the tag. The mirror is a self-DM on the child identity, and the child signs nothing of consequence. see AB-13
    end
    end
    Note over MIR,PHONE: TENSION — the identity port grants devuser mirror_key with secret true (identity-port-acl.json:42),<br/>and the port returns the child secret_hex (port.rs:635-637). The ACL comment justifies it by the hook signing the self-DM wrap<br/>with the child (identity-port-acl.json:14-15), so the phone's key becomes devuser-readable, where a wrap-only operation<br/>would have kept it in ab-identity. ADR-2101 calls a generic sign-this-payload port a bypass (ADR-2101:110-111)
    Note over GW,MIR: DIVERGENCE ADR-2027 — neither derived key appears in the SECURITY-profiles provisional custody register.<br/>Both are bespoke HMAC-SHA256 constructions rather than a standard KDF (HKDF), and rotation is "bump the env tag" with no revocation of the previously derived child at any relay.
```

## AB-16.6 Break-glass — the least governed credential

```mermaid
sequenceDiagram
    autonumber
    participant ATK as any LAN client
    participant PX as nip98-proxy verifyIdentity<br/>agentbox/config/nip98-proxy/proxy.mjs:662
    participant CT as constantTimeEqual<br/>agentbox/config/nip98-proxy/proxy.mjs:576
    participant UP as upstream (AoE or mgmt-api)
    participant ADR as ADR-2027 requirement<br/>agentbox/docs/adr/ADR-2027-secret-custody-rotation-break-glass.md

    ATK->>PX: Authorization Bearer <token> on port 9096 over the LAN
    alt BREAK_GLASS unset or unreadable (proxy.mjs:131 roleSecret NIP98_PROXY_ALLOW_BEARER, file-only under role_isolation, proxy.mjs:116-130)
        PX-->>ATK: branch skipped entirely — break-glass disabled
    else configured
        PX->>CT: constantTimeEqual(token, BREAK_GLASS) (proxy.mjs:672)
        alt no match
            PX-->>ATK: fall through to the NIP-98 branch. see AB-10.3
        else match
            CT-->>PX: true
            PX->>PX: breakGlassNotExpired() (proxy.mjs:673) — ADR-2027: matching the token alone is not sufficient
            alt expired
                PX->>PX: breakGlassUse.refused++, log warn break-glass REFUSED (proxy.mjs:674-681)
                PX-->>ATK: ok false, break_glass_expired
            else not expired
                PX->>PX: breakGlassScopeAllows(req.method, req.url) (proxy.mjs:683)
                alt out of scope
                    PX->>PX: breakGlassUse.refused++, log warn break-glass REFUSED (proxy.mjs:684-691)
                    PX-->>ATK: ok false, break_glass_out_of_scope
                else in scope
                    PX->>PX: breakGlassUse.accepted++, log warn break-glass USED with fingerprint, NOT the token (proxy.mjs:693-705)
                    PX->>PX: ok true, pubkey BREAK_GLASS_PUBKEY, mode "break-glass" (proxy.mjs:706)
                    PX->>UP: forward with X-Agentbox-Pubkey = the sentinel
                    Note over PX,UP: the sentinel identity is NOT a real pubkey, so every downstream attribution for this request is a placeholder
                end
            end
        end
    end
    rect rgb(255,240,240)
    Note over ADR,PX: ADR-2027 lifecycle acceptance remains open
    ADR-->>PX: expiry check exists — unset expiry permits unbounded lifetime
    ADR-->>PX: method and path scope checks exist — unset scope is unrestricted
    ADR-->>PX: fingerprinted acceptance/refusal logs exist — durable per-use receipt is unproven
    ADR-->>PX: full lifecycle needs configured bounds, custodians and tested revocation
    end
    Note over ADR: OPEN — ADR-2027 sets verified_paths to the empty list, the record's own escape<br/>hatch for an assertion that is not mechanically checkable, and leaves verified_commit<br/>pinned at an older sha as a provenance pointer rather than bumping it<br/>(docs/adr/ADR-2027-secret-custody-rotation-break-glass.md:193-198).<br/>So the custody claims this diagram leans on are attested by reading, not by a gate
    Note over ATK,PX: DIVERGENCE INGRESS-identity "Break-glass bearer over the LAN" — accepted on port 9096 AND via ?access_token= / ?bearer= on WS upgrades.<br/>A single shared secret bypasses NIP-98 entirely. Documented opt-in, but a full identity bypass while enabled. see AB-10.9
    Note over ADR: ADR-2027 record status: decision proposed, implementation none, activation inactive.<br/>Policy status remains proposed for the complete lifecycle.<br/>Optional bounds and fingerprint logs ARE implemented — lifecycle completion and deployed configuration are not certified.
```

## AB-16.7 Key files at 0600 — what that boundary is and is not

```mermaid
sequenceDiagram
    autonumber
    participant AI as agent-identity loadOrMint<br/>agentbox/management-api/lib/agent-identity.js:128
    participant KF as profile key file
    participant AOE as AoE daemon token<br/>~/.config/agent-of-empires/serve.url
    participant BSK as bridge key<br/>AGENTBOX_BRIDGE_SK_FILE default /run/secrets/nostr.key
    participant PEER as any co-resident devuser process

    AI->>KF: writeFileSync(privHex, mode 0o600) then chmodSync 0o600 (agent-identity.js:165-166)
    Note over AI,KF: comment at agent-identity.js:160-161 — "Persist with 0600 so the key survives a restart of this profile and no other uid can read it"
    AOE-->>AOE: minted by the daemon at launch into an owner-only 0700 directory, not env-settable
    BSK-->>BSK: loaded by load_agent_sk, the legacy environment fallback honoured only while role_isolation is off (lib.rs:179-181)
    rect rgb(255,240,240)
    alt role_isolation off, the shipped default
        Note over PEER,BSK: the residual limit, stated in the governing docs
        PEER->>KF: read — SUCCEEDS, same uid
        PEER->>AOE: read — SUCCEEDS, same uid
        PEER->>BSK: read — SUCCEEDS, the boot writes it 0400 devuser (entrypoint-unified.sh:1019-1028)
    else role_isolation on
        PEER->>BSK: read — REFUSED, the runtime copy is /run/secrets/ab-identity/nostr.key, 0400 ab-identity in a 0500 dir (role-accounts.json:24)
        PEER->>KF: read — still SUCCEEDS, per-agent DID keys stay devuser files and are not in the port's keyring (identity-port-acl.json:17-32)
        PEER->>AOE: read — still SUCCEEDS, the AoE bearer is devuser-class by design
    end
    end
    Note over PEER: DIVERGENCE GOVERNANCE-capabilities item 5 — "a process running as the same devuser can still read the token file.<br/>The token raises the bar but does not isolate same-uid peers. Per-process isolation is future work."<br/>With the flag off the same sentence applies verbatim to the agent key and the bridge key. see AB-16.1
    Note over PEER,BSK: TENSION — under role_isolation the RUNTIME copy is role-only, but the AT-REST copies are not: the boot chowns /var/lib/agentbox/secrets<br/>to devuser in both modes (entrypoint-unified.sh:526, entrypoint-unified.sh:554), and ADR-2122 records that until W2 custody devuser<br/>can still read those copies, so the flag gives no confidentiality on its own (ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:206-209)
    Note over AI,KF: DIVERGENCE SECURITY-profiles custody row — persistence failure is NON-FATAL (agent-identity.js:168-171),<br/>so a container can run a whole session on an in-memory key that vanishes at restart. "Key persistence failure can produce a valid but unstable identity." see AB-11.2
```

## AB-16.8 A secret's lifecycle — implemented versus proposed

```mermaid
stateDiagram-v2
    [*] --> Provisioned
    Provisioned --> InUse : loaded at boot from env, key file or daemon state file
    InUse --> InUse : used per request with no per-use receipt
    InUse --> Derived : HMAC-SHA256 with a domain tag produces a child (gateway, mirror)
    Derived --> InUse
    InUse --> BackedUp : scripts/backup-secrets.sh collects selected config names into a ZIP
    BackedUp --> InUse
    InUse --> RotatedProposed : ADR-2027 rotation cadence
    InUse --> RevokedProposed : ADR-2027 revocation path
    RotatedProposed --> [*]
    RevokedProposed --> [*]
    note right of Provisioned
        IMPLEMENTED. Storage interfaces exist and are source-verified:
        0600 profile key file, /run/secrets/nostr.key, the AoE serve.url
        state file, NIP98_PROXY_ALLOW_BEARER and NIP98_PROXY_SESSION_SECRET
        captured at process start. Under role_isolation the root boot
        copies each role's secrets into its own /run/secrets subdirectory
        at 0400 and unsets the variables before supervisord. see AB-16.13
    end note
    note right of BackedUp
        PARTIAL and unverified. SECURITY-profiles: the backup script invokes
        ordinary zip and unzip integrity testing with NO encryption flags and
        no explicit umask or chmod. Final permissions depend on the invoking
        environment. Integrity testing is not a recovery exercise. The script
        was not run and no backup contents were inspected.
    end note
    note right of RotatedProposed
        NOT IMPLEMENTED. ADR-2027 is proposed / none / inactive.
        No cadence, no custodian, no incident window, no revocation
        procedure and no dated failure-recovery receipt exists for
        any of the seven register rows.
    end note
    note right of RevokedProposed
        NOT IMPLEMENTED, and structurally expensive. The relay allowlist
        is baked at nix build (relayAllowedPubkeysCsv), so publisher
        revocation needs a full rebuild — the compromise window is one
        build-deploy cycle. ADR-2027 Context. see AB-13.3
    end note
```

## AB-16.9 Provisional custody register — seven roles, zero confirmed custodians

```mermaid
flowchart LR
    R["SECURITY-profiles.md<br/>Provisional custody register 2026-09-04"]
    R --> C1["Bridge identity / unwrap key<br/>src: AGENTBOX_BRIDGE_SK_FILE default /run/secrets/nostr.key<br/>under role_isolation held by ab-identity uid 960, role-accounts.json:19-29<br/>legacy env fallback only while the flag is off"]
    R --> C2["Shared server publisher identity<br/>src: relay key list, build-projected<br/>ADR-2012 per-consumer split PENDING"]
    R --> C3["Proxy break-glass bearer<br/>src: NIP98_PROXY_ALLOW_BEARER read at process start, proxy.mjs:131<br/>under role_isolation a file of ab-ingress uid 962, role-accounts.json:39-46<br/>still the same value as BRIDGE_TOKEN until Q4, see AB-16.6"]
    R --> C4["Proxy browser-session signing secret<br/>src: NIP98_PROXY_SESSION_SECRET or per-boot random, proxy.mjs:244-245<br/>under role_isolation a file of ab-ingress, see AB-10.8"]
    R --> C5["AoE daemon token<br/>src: daemon state file read by the proxy with a last-good cache<br/>see AB-10.11"]
    R --> C6["Dream remote-execution identity<br/>src: ssh/scp dispatch on AMBIENT ssh config<br/>no explicit identity file in the inspected calls"]
    R --> C7["Secret backup artefact and recovery access<br/>src: scripts/backup-secrets.sh to workspace/secret-backups ZIP + manifest"]
    C1 --> U["EVERY ROW: custodian UNCONFIRMED, deployed location UNCONFIRMED,<br/>rotation cadence UNCONFIRMED, incident response window UNCONFIRMED"]
    C2 --> U
    C3 --> U
    C4 --> U
    C5 --> U
    C6 --> U
    C7 --> U
    U --> N1["DIVERGENCE — 'Suggested responsible role' in the register is a role TO ASSIGN,<br/>not an assertion that anyone has accepted custody. 'No cadence is invented here.'"]
    U --> N2["DIVERGENCE — these seven rows are a STARTING SET, not completeness certification.<br/>Provider credentials and other estate identities are not yet inventoried.<br/>The two derived HMAC keys in AB-16.5 are absent from the register entirely."]
    R -.-> N4["ROLE ACCOUNTS 2026-10-03, config/role-accounts.json:17-93 — a holder uid per secret-bearing role:<br/>ab-identity 960, ab-gateway 961 holding no key, ab-ingress 962, ab-spend 963 account only,<br/>ab-sidestr-dreamlab 964, ab-faucet-dreamlab 966, txbt4 producer 967 and faucet 968.<br/>965 is reserved for the host docker group, role-accounts.json:5-7.<br/>A holder uid is not a custodian: every row above stays UNCONFIRMED"]
    R -.-> N3["Refer to secret identifiers and custodian ROLES, never secret values.<br/>Do not copy secrets into this register. SECURITY-profiles preamble."]
```

## AB-16.10 email-search — the owner-authorised break-glass tier

```mermaid
sequenceDiagram
    autonumber
    participant AG as agent turn
    participant SK as email-search skill<br/>agentbox/skills/email-search/SKILL.md
    participant GW as email-mcp-gateway:8765<br/>visionclaw_network
    participant IDX as privacy-filtered index
    participant MAIL as raw Proton mail (2 accounts via one Bridge)

    rect rgb(235,245,255)
    Note over AG,GW: TIER 1 — transport bearer, gates ANY call
    AG->>GW: Authorization Bearer <AGENTBOX_EMAIL_GATEWAY_TOKEN>
    alt token absent or wrong
        GW-->>AG: rejected — no tool reachable
    end
    end
    AG->>SK: ask_email(query)
    SK->>GW: ask_email
    GW->>IDX: semantic search
    IDX-->>GW: sanitised evidence[] with ref_id, no real headers
    GW-->>AG: schema-abstracted answer — ALWAYS sanitised regardless of any pubkey passed
    rect rgb(255,240,240)
    Note over AG,MAIL: TIER 2 — capability pubkey, gates the RAW tools on top of tier 1
    AG->>AG: read the operator pubkey from AGENTBOX_X_ONLY_PUBKEY_HEX at call time, never hardcoded in skill source
    AG->>GW: fetch_email_by_ref(ref_id, nostr_pubkey) or fetch_email_raw(query, nostr_pubkey)
    GW->>GW: is nostr_pubkey on PRIVILEGED_NOSTR_PUBKEYS
    Note over GW: accepts bare 64-hex, 0x-prefixed or nostr:-prefixed, case-insensitive. npub1 bech32 is NOT accepted.
    alt not on the allow-list
        GW-->>AG: {"authorized": false, "error": "Nostr pubkey not authorized for raw (unfiltered) access. Use ask_email …"}
        Note over GW: every attempt is logged with an 8-char pubkey fingerprint
    else authorised
        GW->>MAIL: bypass the privacy filter
        MAIL-->>GW: real headers, sender, date, folder, full text
        GW-->>AG: {"authorized":true,"mode":"raw", …} — verbatim mail
    end
    end
    Note over AG,GW: INVARIANT — a Nostr PUBLIC key is the publishable half, an identity/capability token, not a secret.<br/>Passing it as a tool argument is the intended design. The bearer token and the Nostr PRIVATE key are what never leave the box.
    Note over AG,MAIL: DIVERGENCE — this is a real, working break-glass tier with an allow-list, a fingerprinted audit line and a default-sanitised posture,<br/>which is MORE governed than the proxy break-glass bearer in AB-16.6. Neither appears in the ADR-2027 register as an implemented control.
```

## AB-16.11 Egress profiles — mirror versus digest have different contracts

```mermaid
flowchart TB
    S["session content"]
    S --> M["nostr-live-mirror.cjs per-turn hook<br/>agentbox/config/hooks/nostr-live-mirror.cjs"]
    S --> D["kind-30840 SessionEnd digest<br/>[sovereign_mesh.mobile_bridge]"]
    M --> M1["composes UNREDACTED selected text"]
    M1 --> M2["NIP-59 gift-wrap to the derived child identity<br/>see AB-16.5 and AB-13.9"]
    M2 --> M3["cloud worker relay<br/>wss://dreamlab-nostr-relay.solitary-paper-764d.workers.dev"]
    D --> D1["flattens input and sends it to the configured summarisation provider"]
    D1 --> D2["publishes a SEPARATELY SIGNED digest"]
    M3 --> OFF["off switch AGENTBOX_LIVE_MIRROR=0 — fail-open on any publish error"]
    D2 --> OFF2["gated by its own manifest block, different switch"]
    OFF --> X["DIVERGENCE ADR-2026 — proposed / partial / INACTIVE for the complete egress policy.<br/>SECURITY-profiles: 'Their configuration gates and encryption differ.<br/>A shared off/redaction/recipient/retention contract remains open.'"]
    OFF2 --> X
    X --> Y["Consequence: there is no single answer to 'is session content leaving this box, and redacted how'.<br/>Two paths, two gates, two encryption postures, one of them unredacted before wrapping."]
    M3 -.-> Z["TRUST BOUNDARY — this is the only agentbox egress to a NON-LAN destination in this domain.<br/>The email gateway, the Loom and the relay are all LAN or loopback. see AB-13.9"]
```

## Audit qualification — 2026-09-07

AB-16.2 corrects an upstream comment as well as the previous drawing: Docker explicitly selected seccomp profiles override the default; they are not automatically stacked ([Docker seccomp documentation](https://docs.docker.com/engine/security/seccomp/)). No running filter or credential contents were inspected. AB-16.3 passed `check-seccomp.sh` against the current source. The same-UID custody limitation remains. See [audit](../../estate-review/2026-09-07-agentbox-audit.md).


## AB-16.12 agentbox-secret-backup — an age-encrypted archive tool packaged for explicit invocation

```mermaid
flowchart TB
    subgraph cli["agentbox-secret-backup — standalone Cargo bin, services/secret-backup"]
        BACKUP["Backup #123;root, out, recipients, manifest#125;<br/>services/secret-backup/src/main.rs:107-120"]
        RESTORE["Restore #123;archive, dest, identity#125;<br/>services/secret-backup/src/main.rs:123-132"]
        PLAN["Plan #123;root#125;<br/>services/secret-backup/src/main.rs:135-138 — list only, reads no content"]
        SELFTEST["SelfTest<br/>services/secret-backup/src/main.rs:139-143 — round trip on SYNTHETIC data"]
    end
    ENCRYPTOR --> COLLECT["collect#40;root#41; — walks the tree; exclusions apply relative to the requested root<br/>services/secret-backup/src/main.rs:154-167"]
    BACKUP --> ENCRYPTOR["encryptor#40;recipients, passphrase#41;<br/>services/secret-backup/src/main.rs:191 — age::x25519 recipients OR scrypt passphrase, never both silently"]
    COLLECT --> ARCHIVE["tar stream, age-encrypted<br/>services/secret-backup/src/main.rs:224 backup#40;#41;"]
    ARCHIVE --> MANIFEST["optional manifest: file names plus root/archive/encryption/count metadata<br/>services/secret-backup/src/main.rs:264-275 — NEVER file contents"]
    RESTORE --> DECRYPT["age::Decryptor::new_buffered<br/>services/secret-backup/src/main.rs:322 — branches on is_scrypt#40;#41; vs identity file"]
    DECRYPT --> UNSAFE["is_unsafe_entry_path#40;#41; guard<br/>services/secret-backup/src/main.rs:298-307 — rejects absolute paths and .. components<br/>a hand-crafted hostile archive could still carry"]
    UNSAFE --> HARDEN["harden#40;path#41; — chmod 0600 on every extracted file<br/>services/secret-backup/src/main.rs:283-291"]
    subgraph notes["Invariants and drift"]
        direction TB
        N1["INVARIANT: the tool CANNOT write a plaintext archive — age encryption is<br/>mandatory on every Backup path, not optional (README.md)"]
        N2["G17 source closeout: lib/secret-backup.nix packages the locked Rust CLI with checks.<br/>flake.nix includes secretBackupPkg in knowledgeToolPackages.<br/>Explicit operator invocation remains required. Eight synthetic tests pass locally and in the standalone Nix package.<br/>No real-key rotation or recovery is implied.<br/>The active container has not been rebuilt with this source change."]
        N3["ADR-2030: AGPL-3.0-only, publish#61;false, NOT dual-licensed like the sibling<br/>services/ crates #40;MIT OR Apache-2.0#41; — an operator-internal tool, never published"]
        N4["ADR-2027 acceptance gap: retention, off-host placement and the recovery-authority<br/>test remain the operator's responsibility — SelfTest proves restore works on<br/>synthetic data only, never on a real secret"]
        N1 ~~~ N2 ~~~ N3 ~~~ N4
    end
```

## AB-16.13 Role secret delivery — the shipped default versus role_isolation

```mermaid
flowchart TB
    subgraph OFF["role_isolation = false, the shipped default: today's boot, unchanged"]
        direction TB
        O1["/run/secrets handed to devuser at 0700<br/>config/entrypoint-unified.sh:388-391"]
        O2["the bridge key written 0400 devuser and AGENTBOX_BRIDGE_SK unset<br/>entrypoint-unified.sh:1019-1028"]
        O3["capture and scrub return at their first line, so every other ROLE variable stays in PID 1<br/>and reaches every child, entrypoint-unified.sh:424, entrypoint-unified.sh:468"]
        O4["exec supervisord with /etc/supervisord.conf and a byte-identical environment<br/>entrypoint-unified.sh:1144, tests/runtime-contract/RC-X1-06.sh:12"]
        O1 --> O2 --> O3 --> O4
    end
    subgraph ON["role_isolation = true: per-role files under a root-owned mount"]
        direction TB
        N1["/run/secrets is its own root tmpfs, prepared root 0711, a mount point devuser cannot rename<br/>config/lib/role-custody.sh:86-100, entrypoint-unified.sh:352-355"]
        N2["each ROLE variable from the eleven-name list goes to /run/secrets/role/NAME at 0400 and is unset,<br/>before the identity bootstrap runs, entrypoint-unified.sh:411, entrypoint-unified.sh:489"]
        N3["identity.env is rendered public-only: no AGENTBOX_NSEC, no AGENTBOX_BRIDGE_SK,<br/>only the key file's path, services/nostr-pod-bridge/src/bootstrap.rs:230-250"]
        N4["the baked plan /etc/agentbox/role-secrets.tsv is executed row by row<br/>entrypoint-unified.sh:1147, role-custody.sh:160-289"]
        N5["each file lands via a 0400 temp owned by the role, then a rename, role-custody.sh:104-111.<br/>Role dirs are sealed 0500 and role-owned last, role-custody.sh:282-286"]
        N6["final scrub: a ROLE variable still present is logged ROLE-ISOLATION-LEAK and unset,<br/>entrypoint-unified.sh:1158, then exec the roles config, entrypoint-unified.sh:1159"]
        N7["readers take NAME_FILE, then AGENTBOX_SECRETS_DIR/NAME, then a default file,<br/>and the bare variable only with the flag off<br/>services/nostr-pod-bridge/src/role_secret.rs:9-22, management-api/lib/role-secret.js:13-25"]
        N1 --> N2 --> N3 --> N4 --> N5 --> N6 --> N7
    end
    OFF --> ON
    N6 -.-> INV1["INVARIANT under role_isolation: no ROLE variable reaches supervisord's environment.<br/>Captured at entrypoint-unified.sh:489, scrubbed at entrypoint-unified.sh:1158"]
    N5 -.-> INV2["Each chain's signer key and parent credential are delivered to that chain's producer uid only,<br/>role-accounts.json:61-62, role-accounts.json:80-81. The treasury source is still the workspace bind,<br/>role-accounts.json:71, so its at-rest copy stays host-visible"]
    N3 -.-> TEN1["TENSION: role-accounts.json:21 calls ab-identity the sole holder of the sovereign key,<br/>yet the identity file, which carries private_key_hex (identity.rs:71),<br/>is chowned devuser 0600 in BOTH modes, entrypoint-unified.sh:695-699"]
    N4 -.-> TEN2["TENSION: the at-rest volume /var/lib/agentbox/secrets is chowned to devuser in both modes,<br/>entrypoint-unified.sh:526, entrypoint-unified.sh:554. Locking it is owed as W2 custody,<br/>ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:206-209"]
```

**What it shows.** With the flag off the boot is unchanged and every ROLE secret is ambient. With it on the root boot moves each role's secrets into its own 0500 directory of 0400 files under a root-owned mount, writes a public-only `identity.env`, and scrubs the environment twice before `supervisord` starts, so no role program and no devuser process inherits a signing key from PID 1.

**Why it is this way.** ADR-2122 chose per-role Unix accounts over a separate signing container because supervisord already drops privilege per program; the flag is boot-class so the image carries both paths and the owner's rehearsal flips it without a rebuild (see AB-36). The runtime copies are only half of custody: the at-rest volumes are W2's, and until that lands the flag's confidentiality claim is about the environment and `/run/secrets`, not the disk.

## AB-16.14 Environment classes — every secret-shaped name in exactly one class

```mermaid
flowchart TB
    T["config/custody/env-classes.json<br/>names only, never values, env-classes.json:2"]
    T --> ROLE["ROLE, 11 names, env-classes.json:33<br/>a signing, join or session key owned by one role:<br/>ab-identity, ab-ingress, or root for TAILSCALE_AUTHKEY"]
    T --> DEV["DEVUSER_CLASS, 35 names, env-classes.json:90<br/>provider and devuser credentials kept in devuser's environment,<br/>each with a reason and the Q8 exception"]
    T --> NS["NON_SECRET, 612 exact names and no patterns, env-classes.json:232<br/>a secret-shaped name can never land here by pattern"]
    ROLE --> EP["must equal the entrypoint's scrub list, entrypoint-unified.sh:411<br/>scripts/ci/env-secret-inventory.js:259-262"]
    ROLE --> W1["must match W1's delivery plan, every from_env with the same role<br/>env-secret-inventory.js:206-224"]
    T --> CHK["node scripts/ci/env-secret-inventory.js --check fails CI on an unclassified,<br/>ambiguous or secret-shaped-by-pattern name, env-secret-inventory.js:227-238"]
    DEV -.-> DEBT["DEBT: the 35 DEVUSER_CLASS credentials stay in every agent process by exception,<br/>owner disposition Q8 pending, env-classes.json:92. see AB-36"]
```

**What it shows.** One table classifies every environment name the boot path and the role-secret consumers read. Only the ROLE class leaves PID 1 under the flag; the CI inventory keeps that set equal to the entrypoint's scrub list and to the role table's delivery plan, so a new key cannot be added to one and forgotten in another.

**Why it is this way.** Bypass 3 of the custody design was that `.env` reached every process through PID 1. A denylist of role secrets, rather than an allowlist of the environment, lets the flag ship without breaking agent tooling; the price is the DEVUSER_CLASS exception, which the owner has not yet ruled on.

**Invariant (under role_isolation):** the runtime copy of every role secret is readable only by its role — each file is `0400` and role-owned inside a `0500` role-owned directory (`../project/agentbox/config/lib/role-custody.sh:104-111`, `../project/agentbox/config/lib/role-custody.sh:282-286`) on a root-owned mount devuser cannot rename (`../project/agentbox/config/lib/role-custody.sh:86-100`). With the flag off, `/run/secrets` is devuser's as before (`../project/agentbox/config/entrypoint-unified.sh:388-391`).

**Invariant (under role_isolation):** no ROLE variable is in PID 1's environment when `supervisord` starts (`../project/agentbox/config/entrypoint-unified.sh:489`, `../project/agentbox/config/entrypoint-unified.sh:1158`); with the flag off both functions return at their first line and the environment is today's (`../project/agentbox/config/entrypoint-unified.sh:424`, `../project/agentbox/config/entrypoint-unified.sh:468`).

**Tension (role isolation vs the at-rest volumes):** the flag makes the runtime copies role-only, but the boot still chowns `/var/lib/agentbox/secrets` to devuser in both modes (`../project/agentbox/config/entrypoint-unified.sh:526`, `../project/agentbox/config/entrypoint-unified.sh:554`), and ADR-2122 itself records that until W2 custody devuser can read the at-rest copies, so the flag gives no confidentiality on its own (`../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:206-209`).

**Tension (role table vs the identity file):** the role table names `ab-identity` the sole holder of the sovereign key (`../project/agentbox/config/role-accounts.json:21`), but the identity JSON carrying `private_key_hex` (`../project/agentbox/services/nostr-pod-bridge/src/identity.rs:71`) is chowned `devuser 0600` in both modes (`../project/agentbox/config/entrypoint-unified.sh:695-699`).

**Tension (identity port vs the mirror child):** the port hands devuser the live-mirror child secret (`../project/agentbox/config/custody/identity-port-acl.json:42`, `../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs:635-637`) because the hook signs the self-DM wrap with it (`../project/agentbox/config/custody/identity-port-acl.json:14-15`), so the phone's key is a devuser-class credential under the flag; ADR-2101 allows named operations only (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:110-111`). The ACL comment cites the wrap at `nostr-live-mirror.cjs:487`; the signing line is `../project/agentbox/config/hooks/nostr-live-mirror.cjs:488`.

**Open:** under the flag the live-mirror hook, still on its own key read, gets no operator key as devuser (`../project/agentbox/config/hooks/nostr-live-mirror.cjs:214`, `../project/agentbox/config/hooks/lib/operator-key.cjs:59-69`) and does not yet call the port's `mirror_key`, so the mirror falls back to a throwaway sender (`../project/agentbox/config/hooks/nostr-live-mirror.cjs:225-233`) until the W3b cutover (`../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:218-219`).

**Drift (resolved 2026-10-03, secrets inventory vs the sealed chain):** this topic used to omit two files in the `agentbox-secrets` volume — the sidestr signer key at mode 0400, which ADR-2101 D3 requires be neither in `identity.env` nor derived from the identity key, and the parent node's testnet4 RPC cookie. The role table now names both as deliveries to the producer uid (`../project/agentbox/config/role-accounts.json:61-62`, `../project/agentbox/config/role-accounts.json:80-81`) and AB-16.13 draws them. Both are still read by path rather than passed as arguments (see AB-32.4, AB-32.3, AB-34.1).

**Tension (custody boundary vs the faucet key):** since the sidechain faucet became a supervised program (2026-09-30), its treasury signing key is read from the workspace bind, not the `agentbox-secrets` volume — the manifest points at `/home/devuser/workspace/sidestr/agents/treasury.key` (`../project/agentbox/agentbox.toml:1578`), which is also the schema's default when the key is empty — so a spend-capable key sits on a host-visible bind, outside the `agentbox-secrets` volume this topic's custody model inventories. Narrowed, not resolved, on 2026-10-03: under `role_isolation` the runtime copy goes to the faucet uid at 0400, but the role table still sources it from the same bind (`../project/agentbox/config/role-accounts.json:71`).
