---
title: Solid Pod LDP lifecycle and WAC panels
date: 2026-11-02
status: assessed
type: explanation
---

# Solid Pod LDP lifecycle and WAC panels

This review assessed the 13 unreconciled panels in SP-03 (LDP request lifecycle: SP-03.4, .5, .6, .8, .9, .10, .11) and SP-04 (WAC authorisation: SP-04.1, .7, .8, .13, .14, .15) against solid-pod-rs `9b6143ce55b2`. Of the 13, 3 are accurate, 9 corrected and 1 accurate-with-limits.

Both topics had to be restamped from `09ec90f` to `9b6143c`. Upstream commit `daff10d` rewrote the ACL lockout guard and moved every server citation above line 870. Restamping also changes the bodies of 11 panels credited on 2026-11-01, so I re-checked those too: 8 are accurate, 2 corrected and 1 accurate-with-limits. Across all 24 panels: 11 accurate, 11 corrected, 2 accurate-with-limits, none inaccurate-unfixed.

The most important finding is that the credited accounts of the lockout guard (SP-03.3, SP-04.6) described code that no longer exists. The guard now evaluates effective Control, and that new guard refuses every RDF PATCH that would change or create an `.acl`. Second, SP-04.8 said the shipped server ignores `acl:origin`. The server has enabled that gate since July.

## Findings

**LW-01: the lockout guard credited on 2026-11-01 was replaced, and its debt is closed in source.**
- *Trigger:* `daff10d` rewrote `proposed_acl_keeps_caller_control`.
- *Change:* it is now async and returns `true` straight away for a non-ACL (`.meta`) path (`crates/solid-pod-rs-server/src/lib.rs:883`). Before the identity check it fails closed on three things:
  - anything `classify_policy_read` does not return as `Found` (`crates/solid-pod-rs-server/src/lib.rs:889-893`);
  - an unknown condition type (`crates/solid-pod-rs-server/src/lib.rs:894`);
  - a Turtle syntax error (`crates/solid-pod-rs-server/src/lib.rs:901-908`).
- *Identity check:* it evaluates `acl:Control` on the protected resource through `grant_for_identities`, using the caller's balance, linked identities and condition registry (`crates/solid-pod-rs-server/src/lib.rs:910-927`).
- *Consequence:* SP-04.6's old branches are gone. These were "unparseable → true, defer", the `ids_of_acl_field` participant, and the debt "does not check target or conditions" (old `lib.rs:913-934`). So is SP-03.3's debt of the same name, and the SP-04.6 prose debt about a wrong-target Control rule being accepted with 201. The 2026-11-01 HTTP reproduction of that debt predates `daff10d`. The closure here rests on source tracing only.
- *Remaining:* `daff10d` also added `tests/acl_candidate_control.rs`, but I did not run it.
- *To close:* re-run the 2026-11-01 wrong-target reproduction against `9b6143c`.

**LW-02: RDF PATCH cannot change or create an `.acl` (new, source-traced).**
- *Trigger:* the guard now reads candidates through `classify_policy_read`, whose Turtle sniff accepts only a Turtle media type, `@prefix` or the literal `acl:Authorization` (`crates/solid-pod-rs/src/wac/resolver.rs:309-316`). The old guard also recognised `application/n-triples` and full-IRI bodies.
- *Create path:* create-via-PATCH serialises N-Triples with full IRIs and calls the guard with `application/n-triples` (`crates/solid-pod-rs-server/src/lib.rs:2033-2044`), so the candidate is classified Malformed and refused with 409.
- *Existing ACLs:* PATCH of an existing Turtle ACL is already refused at the seed, since Turtle with prefixes is not N-Triples (`crates/solid-pod-rs-server/src/lib.rs:2192`).
- *Consequence:* PUT, POST to a sidecar Slug (`crates/solid-pod-rs-server/src/lib.rs:1835-1865`), COPY onto a sidecar (`crates/solid-pod-rs-server/src/lib.rs:2962-2980`) and JSON Patch still write ACLs; RDF PATCH cannot. The upstream parity note says RDF PATCH limitations "remain open". No route test exercises RDF PATCH on an ACL.
- *To close:* a route test that sends N3 and SPARQL PATCH to an absent `/x.acl` as a Control holder and asserts the intended status. Then either extend the sniff to N-Triples or document the refusal.
- *Recorded in:* SP-03.5 (Debt) and SP-04.6.

**LW-03: the shipped server enforces `acl:origin`; SP-04.8 said it did not.**
- *Evidence:* `crates/solid-pod-rs-server/Cargo.toml:44` enables `acl-origin` on its `solid-pod-rs` dependency, and has done since `51ecbf4` (2026-07-03). SP-01.6 already says this, so SP-04.8 contradicted its sibling.
- *Consequence:* `check_origin` is document-wide (`crates/solid-pod-rs/src/wac/origin.rs:282-304`). One rule with `acl:origin` makes every non-Control grant in that ACL require a matching `Origin`, and requests without an `Origin` header are refused (RejectedNoOrigin).
- *Related drift:* the core manifest comment "unlike `acl-origin`" (`crates/solid-pod-rs/Cargo.toml:142-143`) is stale.
- *To close:* decide whether document-wide scope is intended, and fix the manifest comment upstream.
- *Corrected in:* SP-04.8 (Drift and Tension).

**LW-04: group, client and issuer grants cannot match on the native server.**
- *Groups:* both admission paths build `StaticGroupMembership::default()`, an empty map (`crates/solid-pod-rs-server/src/lib.rs:994`, `:1206`), and so does the lockout guard (`:913`). Nothing in the server crate populates it.
- *Client and issuer:* every `RequestContext` sets `client_id` and `issuer` to `None` (`crates/solid-pod-rs-server/src/lib.rs:1093-1094`, `:1263-1264`).
- *Consequence:* `acl:agentGroup` never grants. Client and issuer conditions grant only through the `foaf:Agent` class (`crates/solid-pod-rs/src/wac/client.rs:61-64`, `crates/solid-pod-rs/src/wac/issuer.rs:57-61`).
- *Panel error:* SP-04.15 called `NoGroupMembership` "the server default". It is crate-private and used only by the WAC 1.x entry point (`crates/solid-pod-rs/src/wac/evaluator.rs:172`).
- *To close:* wire a group resolver, plus client and issuer from the OIDC/DPoP path, or document these as unsupported on the native server.
- *Recorded in:* SP-04.1 (Tension), SP-04.13 (Debt), SP-04.15.

**LW-05: JSON Patch is destructive on non-JSON resources.**
- *Evidence:* a stored body that is not JSON becomes `{}`, and the patched object is written back under the stored content type (`crates/solid-pod-rs-server/src/lib.rs:1949-1952`, `:1978`).
- *Consequence:* SP-03.6's non-destructive invariant holds only for N3 and SPARQL. A JSON Patch that applies to an empty object replaces a Turtle resource's triples.
- *To close:* refuse JSON Patch when the stored body does not parse as JSON (409), mirroring `seed_graph_from_patch_target`.

**LW-06: inner container ACLs are protected but never enforced (Open, known upstream).**
- *Evidence:* `protected_resource_for_acl("/dir/.acl")` is `/dir/` (`crates/solid-pod-rs/src/wac/mod.rs:220`), while the resolver reads `/dir.acl` (`crates/solid-pod-rs/src/wac/resolver.rs:333-338`, `:379`).
- *Status:* `daff10d` changed discovery headers to advertise `/dir.acl` (`crates/solid-pod-rs/src/ldp.rs:152`). It leaves existing inner copies to a migration review (parity row 290).
- *Recorded in:* SP-04.5, as Open.
- *To close:* make protection and resolution agree on one container ACL key, either by probing `/dir/.acl` too or by refusing writes to it, and migrate existing inner copies with a test that a stored `/dir/.acl` either governs or is rejected.

**LW-07: smaller corrections.**
- *SP-03.4:* the guard's 409 applies to `.acl` only, not `.meta`. The sidecar branch does not mint a unique name. `mint_unique_target` is check-then-act, so concurrent same-Slug POSTs can collide (Tension, not exercised).
- *SP-03.5:* the GUARD node cited PUT's guard line, and the JSON Patch edge skipped the guard.
- *SP-03.8:* the destination-sidecar guard runs before the write, not after.
- *SP-03.10:* only PATCH-written resources persist as N-Triples. A Turtle body PUT with prefixes is served verbatim.
- *SP-03.11:* the heading "every LDP response" overstated which headers each handler sets. It is retitled (ID kept) with a Drift note. SP-04.8 is likewise retitled "off in the library, on in the server".
- *SP-04.7:* the origin gate lacked its grant edge.
- *To close:* a concurrent same-Slug POST test (or a create-only storage put) closes the SP-03.4 race. The other items are diagram corrections, closed by this edit.

## Panel dispositions

Credited panels are marked "re-check". The topic restamp changed their bodies, so they were re-assessed at `9b6143c`.

| Panel | Disposition | Evidence |
| --- | --- | --- |
| SP-03.1 (re-check) | accurate | `handle_get` read in full at `lib.rs:1422-1632`. Read gate before bytes (1461), WAC-Allow computed with no document (1463), bare 404 (1629), `handle_options` has no WAC (2304). |
| SP-03.2 (re-check) | accurate | Container branch `lib.rs:1465-1541`; index child gated separately (1481); `mashlib::should_serve` signature `mashlib.rs:122`. |
| SP-03.3 (re-check) | corrected | Guard note rewritten for `daff10d` (LW-01); `.meta` passes the guard (883). The incomplete-condition debt (1668) stands: `validate_acl_document` rejects only unknown types (`conditions.rs:411`). |
| SP-03.4 | corrected | `handle_post` `lib.rs:1793-1882`: Append (1806); sidecar re-enforce (1835); guard (1843); mint (1770-1791); Slug charset (`ldp.rs:197`) and UUID fallback (`ldp.rs:207`). Guard scope and race added (LW-07). |
| SP-03.5 | corrected | `handle_patch` `lib.rs:1884-2065`. JSON Patch guard (1961), RDF guard (1997), absent JSON Patch is 400 (`ldp.rs:2266`). RDF PATCH on an ACL debt (LW-02). |
| SP-03.6 | corrected | Seed (2183-2198), 409 on unparseable (2192), stored as text/turtle (2016). JSON Patch scope debt (LW-05). |
| SP-03.8 | corrected | `handle_copy` `lib.rs:2916-3029`: guard (2962) precedes put (2980); conditional sidecar copy (2990-3021). |
| SP-03.9 | accurate | `handle_glob_get` `lib.rs:3035-3099`: folder gate (3057), per-child gate without charge (3074), filters and 404 as drawn. |
| SP-03.10 | corrected | `rdf_content_negotiate` `lib.rs:2140-2173`: the N-Triples-only parse (2154) means only PATCH-written bodies transcode; other verbatim cases added. |
| SP-03.11 | corrected | Helpers at `lib.rs:1320-1420`; `link_headers` now uses `acl_sidecar_key` (`ldp.rs:152`). Drift: per-handler header subsets. |
| SP-03.12 (re-check) | accurate | `set_cache_policy` `lib.rs:1380-1396`; `cache_control_for_response` `ldp.rs:1981-1989`; mashlib `no-store` (1571). |
| SP-04.1 | accurate | `enforce_write_ctx` `lib.rs:953-1036`, `grant_for_identities` 1078-1102, `guarded_acl_target` 826-841. Restamped. Tension added on empty groups and no client or issuer (LW-04). |
| SP-04.2 (re-check) | accurate | `enforce_read_inner` `lib.rs:1177-1281`: anonymous re-evaluation (1267), Public only if path unchanged (1276), glob no-charge refusal (1224). |
| SP-04.3 (re-check) | accurate | wac sources byte-identical between `09ec90f` and `9b6143c`; resolver lines 65, 153, 249, 333, 344, 401 read. |
| SP-04.4 (re-check) | accurate-with-limits | `classify_policy_read` `resolver.rs:265-326` matches each branch. Limit: N2 concerns an external edge adapter not in this repository. |
| SP-04.5 (re-check) | accurate | `mod.rs:217-257`; wrapper calls 1196 and 972. Open added (LW-06); the panel's mapping claims are unchanged. |
| SP-04.6 (re-check) | corrected | Rewritten for `daff10d` (LW-01); all six call sites re-cited; remaining debt on parser loss, empty groups and RDF PATCH. |
| SP-04.7 | corrected | `evaluator.rs:255-357` read in full; missing GRANT edges for Control and origin pass added (335-336, 344). |
| SP-04.8 | corrected | `crates/solid-pod-rs-server/Cargo.toml:44` enables `acl-origin` (LW-03); `check_origin` `origin.rs:268-305` is document-wide. |
| SP-04.13 | accurate-with-limits | `payment.rs:70-93`, `evaluator.rs:375`, `charge_granted_payment` `lib.rs:1047-1069`, `debit_ledger` 1291-1314 under `PAYMENT_STATE_LOCK`. `WebLedger::charge` first tagged in v0.5.0-alpha.11 (`f7c29f3`). Registry registers all four (`conditions.rs:293-298`). Limit: "never serves unpaid under a race" relies on the lock and was not exercised. Debt LW-04 added. |
| SP-04.14 | accurate | `acl_denial` `lib.rs:1109-1141`; `policy_failure_to_actix` 2244-2275. |
| SP-04.15 | corrected | `evaluator.rs:79-146`. `NoGroupMembership` is not the server default; the server passes an empty `StaticGroupMembership` (LW-04); agentGroup only for an authenticated agent (138). |
| SP-04.9 (re-check) | accurate | `mod.rs:28-112` read; byte and depth caps precede serde. |
| SP-04.10 (re-check) | accurate | `parser.rs` lines 26-50, 93, 118, 150, 242, 287, 317-320, 349, 418, 460, 545 read; unchanged since `09ec90f`. |

## Verification

**Evidence class:** source tracing at the pinned revision, solid-pod-rs `9b6143ce55b242c322c92d8c6bcc0c1a5371b451`. All reads used `git show <sha>:<path>` and `git grep <sha>`.

**Line mapping:** citations were moved from `09ec90f` by aligning the two revisions by content with a sequence matcher, never by offset. The seven guard citations with no counterpart were rewritten by hand from the new function.

**Not done:** no product tests, HTTP probes, builds or deployments were run. No product code was edited, and there were no git writes.

**What changed in the topics:** both had `verified_commit` set to the pinned SHA, and every citation was re-resolved. SP-03 gained `wac/resolver.rs` as a source; SP-04 gained `wac/conditions.rs` and both `Cargo.toml` files. SP-03.11 and SP-04.8 were retitled with their IDs kept, because their old headings stated what the panels now correct. Topic 08 cites SP-03.11 by ID only.

**Generator gate**, run from the worktree root on the final files, with real exit codes. The gate's `citeCheck` walks Mermaid blocks only (`scripts/diagram-index-gen.cjs`, the loop over `t.diagrams`). So the backtick citations in register paragraphs are not gate-checked. I checked every one of them separately: a script printed the cited lines from the pinned blobs and I read each against its claim. That pass led to one tightened anchor (SP-03.5 debt, `2040` → `2035-2044`).

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 03-ldp-request-lifecycle.md --check --cite-check --strict-citations --render
# exit 0 — 126 citations verified at the declared revision, 0 warnings, 11/11 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 04-wac-authorisation.md --check --cite-check --strict-citations --render
# exit 0 — 146 citations verified at the declared revision, 0 warnings, 13/13 rendered
```

**Not verified:**
- the runtime effect of LW-01, LW-02, LW-03 and LW-05, and the POST race (each finding names the test that would close it);
- the SP-04.4 consumer adapter;
- the rendered images, beyond the generator's render step.

On an earlier run, the default JEV advisory pass (120 citations per topic) flagged SP-03.8 `lib.rs:2925`. I re-anchored it to the `enforce_write_ctx` call at 2923. A final run with `--jev-max 8` reviewed all 126 and 146 citations, exited 0 for both topics and raised no advisories.

**Left for the lead:** the new register markers, and the retitled SP-03.11 and SP-04.8 headings, leave the generated `docs/diagrams/COVERAGE.md`, the register and the indexes stale until the lead's full-tree run. I did not regenerate them, since those files are outside my assignment.
