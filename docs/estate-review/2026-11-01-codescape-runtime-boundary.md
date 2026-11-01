---
title: Codescape deployment observation and ADR-2143 refresh
date: 2026-11-01
status: in-progress
type: explanation
---

# Codescape deployment observation and ADR-2143 refresh

Codescape’s running container matches the source deployment's bind destinations, read-only flags, client-port publication and capability restrictions. This closes part of the earlier deployment evidence gap. It does **not** establish the stronger claim that its process cannot read a corpus or reach a forbidden network destination. The application allowlist and Docker deployment controls remain separate acceptance surfaces.

This pass refreshes ADR-2143 at VisionClaw `531df5ef74453a2ed320bff968c203a9b91faed7`, carrying forward the [earlier section-by-section assessment](2026-10-31-codescape-presence-and-xr.md#adr-2143-section-dispositions) and its unresolved findings. Codescape standalone remains `58108c0357bd0dbf6588f8b6adf6e044f4c04556`; the consumed checkout remains `e3b76f63e801c6a8460110b93e88a0b420e9b0b4`. No product code, deployment, ADR activation state or springs work is changed.

## What was observed

The [runtime receipt](evidence/2026-11-01/codescape-runtime/runtime.json) records selected Docker metadata at **2026-11-01 13:35:57 UTC**. The container id was unchanged before and after collection. Host source paths are represented by hashes; environment values and secret contents were not collected. This is a point-in-time observation, not an attestation of a future rebuild.

| Surface | Observation | What remains unproved |
| --- | --- | --- |
| Container | Running, health status healthy; started at 12:48:35 UTC; container and image digests retained | Health does not identify the built executable's source revision or prove application acceptance |
| Published ports | Only `127.0.0.1:4100` maps to container 4100; 4101 is absent | The separate nginx route and peer connectivity remain distinct exposure paths; no socket connection was attempted |
| Binds | 78 binds, all with `RW=false`: 77 explicit source/configuration destinations plus the ingest-token file | Directory contents, nested mounts, image contents and the running process's effective view were not inspected |
| Estate | 75 binds beneath `/repos/estate`, covering ten repository labels | Matching labels and destinations does not establish correct host source contents or revision |
| Own writable state | Four named volumes: cargo registry, cargo git, build target and data | Existing volume contents and executable provenance were not inspected |
| Capabilities | `ALL` dropped, `CAP_DAC_READ_SEARCH` added; `no-new-privileges:true`; not privileged | The read-bypass capability is intentional for the token. Its presence does not prove which files are readable |
| Root filesystem/user | Root filesystem is writable; configured user string is empty | No process UID, effective capability or filesystem-access test was run |
| Network | One non-internal bridge, `visionclaw_network`, with no driver options reported | Host firewall rules, DNS answers, service reachability and forbidden-egress denial were not measured |

The [source receipt](evidence/2026-11-01/codescape-runtime/sources.json) binds 36 files to their committed bytes. Its destination comparison finds all 77 explicit read-only binds in the pinned host Compose service and exactly one additional runtime bind, the expected ingest token. This deliberately compares destinations only; it neither resolves host environment interpolation nor treats path hashes as content hashes.

## Isolation claims that still need acceptance

**CR-1: application endpoint admission is not process network isolation.** `egress.rs:36-52,77-84` admits the configured scheme family, hostname and port; HTTP and WebSocket schemes share an origin, and the URL path is not part of that key. `http_client` rechecks redirects (`:98-114`). `config.rs:265-283` builds this list from configured endpoints, including optional Jev resolution. These are useful request controls, but they do not constrain an arbitrary socket opened by a process. The live bridge metadata does not supply the missing deny test. This is an outstanding acceptance gap, not a newly demonstrated connection to RuVector or an authentication bypass.

Docker documents that same-bridge peers communicate by default and that bridge networking normally provides outbound connectivity. External firewall policy can change those defaults; none was inspected here. A loopback publication therefore cannot be used as proof of egress denial. [Docker bridge networking](https://docs.docker.com/engine/network/drivers/bridge/).

**CR-2: mount metadata supports a narrower claim than corpus unreadability.** No explicit `/vault`, Docker socket, whole workspace or secrets-tree destination appears in the captured mount list. The only secret destination is the ingest-token file. However, the service mounts its entire consumed Codescape checkout at `/app/codescape`, all VisionClaw crates at `/app/crates`, and selected estate directories. Read-only directories are not filename allowlists. The shared development image also copies crates, source, docs, client and schema files (`Dockerfile.unified:254,285-293`), and the live image digest has not been linked to a reviewed build receipt. No corpus exposure was demonstrated or ruled out by examining those contents.

Docker's bind documentation distinguishes a mount's read-only flag from propagation and recursive submount handling. The observation contains 56 `rslave` and 22 `rprivate` binds. Without the effective mount tree and kernel behaviour, those flags alone do not prove every descendant is read-only. No writable descendant was observed in this pass. [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).

Codescape’s existing baseline leaves egress and no-corpus-mount enforcement outside its proven invariants. Keep that qualification. Acceptance should use the existing deployment and isolation-test ownership: establish the executable/image identity, inspect effective mounts without reading protected content, and demonstrate allowed connections plus denied forbidden destinations under the actual deployment policy. Any reusable validation should follow the estate's shared tooling conventions. This pass adds no competing runtime checker or new dependency.

## ADR-2143 refresh by section group

The [exact reviewed delta](evidence/2026-11-01/codescape-runtime/reviewed-delta.patch) compares the previous assessment's VisionClaw revision `75b72416bac176ca8fde715cac3bd7c77657b3e7` with `531df5ef7`. ADR-2143 changes only its verification metadata and one appended re-verification section. Its Codescape gitlink is unchanged. The following dispositions supplement every row of the earlier whole-document assessment; historical test receipts keep their original scope.

| Sections | Current disposition |
| --- | --- |
| Metadata, Context, Decisions 1–7 and 9, invariants I1–I7a and I9–I10, Consequences, Alternatives, A1–A2, A5 and the extraction close-out | Substantive ADR text is unchanged. Standalone/consumed Codescape source pins and the previously inspected auth, proxy, origin, presence and baseline files are unchanged. Prior dependency, wire, PTX, command-policy and performance dispositions remain; no tests were rerun or silently promoted to deployment acceptance. |
| Decision 3, Decision 8, I8/I8a, A3–A4, October 30 estate-mount recheck | Source Compose and nginx remain unchanged from the prior receipt. The new live observation supports port/bind/capability configuration at the stated time. CR-1 and CR-2 retain the operational isolation gate. A healthy process does not close authenticated publication, voice, GPU or mounted-data acceptance. |
| Phase plan P0–P6 and prior dated verification sections | Historical results remain historical. P3/P5/P6 and the previously reported presence lifecycle defects remain open. The stale baseline claim that presence carries no avatars is still stale. Partial/inactive owner status is retained. |
| A6, capability gating and XR amendments | `backend_caps.gd` is unchanged. Current `backend_connection.gd:249-274` still gates drag start/update/end before sending; drag end now stops recording an implicit pin. The updated scene creates two connections at `graph_scene.gd:1369-1370`; its grab entry checks volume drag capability at `:3397-3401`. The changed backend-switch test follows the per-hand fields. These source checks preserve Codescape's drag refusal. They do not revalidate every XR path or run GUT/headset acceptance. The separate [grab review](2026-11-01-xr-grab-boundaries.md) covers the changed interaction path and its unresolved findings. |
| Appended “Re-verification — 2026-10-31 (`f1fa1862c`)” | The body says 2026-11-01 and reports merged-tree test counts. Treat those as the author's dated receipt, not tests performed by this audit. The protocol-registry delta adds VisionClaw hold/pin semantics and a global hold invariant; its Codescape producer section is unchanged. Those semantics do not confer drag support on Codescape. The [hold-contract review](2026-11-01-drag-holds-and-acceptance.md) supplies the separate implementation assessment. |

This refresh records a current **document assessment**, not current validation of every historical implementation claim. ADR-2159's separate claim is not rebound by this refresh.

## Diagram correction and validation limits

VC-39's opening paragraph overstated “tested isolation” and “never reaches RuVector”, even though its own gap panel qualifies egress. Its introduction now distinguishes the tested dependency boundary from the intended runtime boundary and links here. Diagram bodies and revision stamps remain unchanged, and this prose correction receives no additional panel review credit. VC-42's existing separation of dependency checks from runtime provenance remains appropriate.

No container was started, stopped, rebuilt or entered. No forbidden network target was probed. No product test was rerun. Fresh checks cover source-byte identity, the unchanged Codescape gitlink, mount-destination comparison, evidence hashes, local documentation links, the completion collector and the diagram citation gate. Previous suite results are referenced with their original dates and limits. The wider Codescape and estate audit remains open.
