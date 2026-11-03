I found two security-relevant problems (one high, one medium) and four low ones. Nothing in this diff undoes main's repairs or opens a privilege escalation in WAC enforcement.

### Critical
None.

### High

**1. N3 `where` clause blows up combinatorially and can crash the server (OOM), holding the write lock while it runs.** CONFIRMED
- **Where:** `crates/solid-pod-rs/src/n3_patch.rs:284` (`solutions`), called from `N3Patch::apply` (`n3_patch.rs:182`). It runs inside `handle_patch` (`crates/solid-pod-rs-server/src/lib.rs:1970`) after `lock_resource`.
- **Cause:** for each pattern, `solutions` tries every partial binding against every triple, and stores a cloned `Vec` for each match. Nothing limits the number of patterns, the number of partial results, or the CPU time. The work runs directly on the actix worker thread. `web::block` is not used.
- **Scenario:**
  1. An attacker has Write on any resource, for example their own pod.
  2. They PUT a document of about 1,000 triples.
  3. They send `PATCH` with `solid:where { ?a ?b ?c . ?d ?e ?f . ?g ?h ?i . }`.
  4. The partial results grow to 10³, then 10⁶, then 10⁹ cloned bindings. Before failing, `solutions` would also sort them using `format!("{x:?}")`. The process runs out of memory and dies.
  5. Until then, the worker thread is stuck and that lock stripe is held. The stripe is chosen by `DefaultHasher`, which has fixed keys, so an attacker can pick a path that lands on the same stripe as a victim's busy resource.
- **Regression:** main's old `where` only checked that each exact triple was present, which is linear.
- **Fix:**
  - Cap the number of `where` patterns (JSS-style patches use a handful).
  - Abort as soon as `partial.len() > 1`, since only exactly one solution is accepted anyway.
  - Join patterns in order of how selective they are.
  - Run `apply` under `web::block`, or give it a work budget.

### Medium

**2. PATCH with `where` / `deletes` needs only Write, so a writer without Read can probe document contents.** CONFIRMED
- **Where:** `lib.rs:1986`. `enforce_write_ctx(.., AccessMode::Write, ..)` is the only check in the handler.
- **Scenario:**
  - An agent has `acl:Write` but not `acl:Read`, as with a drop-box.
  - They send `where { <#x> <p> ?v . <#x> <p> "guess" }`. They receive 409 "does not match", 409 "matches more than once", or 204, which tells them whether the value is there.
  - A delete that must match an existing triple (new 409 "triple to delete not found") is a second way to probe.
- **What changed:** main already leaked exact-triple membership through `where` (412). Variables and the "more than once" count make the probe much stronger: an attacker can now enumerate values instead of only confirming exact guesses. The Solid Protocol requires Read when `where` or `deletes` is non-empty.
- **Fix:** when `n3_patch.where_clause` or `deletes` is non-empty, run an extra `enforce_write_ctx(.., AccessMode::Read, ..)` before the lock. Apply the same rule to SPARQL `DELETE`/`WHERE`.

**3. The POST name-minting loop amplifies storage probes about 64×, and anonymous callers can trigger it through the public inbox.** CONFIRMED
- **Where:** `lib.rs:260` (`mint_and_lock_target`) together with `lib.rs:1855` (`mint_unique_target`).
- **Cause:** once `slug`, `slug-1` … `slug-9999` exist, `mint_unique_target` falls back to `slug-{DefaultHasher(target)}`. That name is deterministic and an attacker can compute it in advance.
- **Scenario:**
  1. An anonymous attacker uses the public `acl:Append` on `inbox/` (added by `named_pod_acls`) to fill those names, including the hash name by sending it as a Slug.
  2. After that, every `POST /alice/inbox/` with `Slug: note` makes about 64 × 10,001 `exists()` calls before returning 409.
  3. The same inbox can also be filled so that the owner can no longer post under that slug.
- **Fix:**
  - Stop the loop when a candidate repeats.
  - Or mint a random suffix (`uuid`/`getrandom`) under the lock after the first collision, and limit the linear probe to something like 100.

### Low

**4. A PATCH silently re-interprets triples it did not touch, which can activate grants that were inert before.** CONFIRMED
- **Where:** `lib.rs:2312` (`seed_graph_from_patch_target` Turtle fallback) and `lib.rs:2347` (`acl_targets_as_paths`).
- **Scenario:**
  - A Turtle ACL written by PUT contains `acl:accessTo <https://host/alice/secret>` (this matches nothing, because `normalize_path` keeps the full IRI) or a path-form `acl:agent </bob#me>`.
  - Any unrelated owner PATCH rewrites the absolute target to `/alice/secret`, and resolves the path-form agent against a base built from `Host`. Both grants start matching.
  - The lockout guard only checks that the caller keeps Control, so it does not notice the widening. This needs a Control holder.
- **Fix:** only rewrite targets in triples added by this patch. Alternatively, reject the PATCH when its effect on the parsed `AclDocument` goes beyond the inserted or deleted authorisations.

**5. Pod provisioning is not atomic, despite what the commit says.** CONFIRMED
- **Where:** `lib.rs:3049` and `lib.rs:3074`.
- **Cause:** the six `.meta` writes happen before the ACL writes. The comment "before any content" holds only for the profile card.
- **Scenario:**
  - A storage error on the first ACL `put` leaves `/alice/` existing, so a retry returns 409, with no pod ACL. The pod is then governed by the operator's root ACL, which may grant public access.
  - A failure later in the loop is less serious: the missing sidecars fall back to the root pod ACL, which is owner-only.
- **Fix:** write the root ACL first, or delete the partial tree on error.

**6. Any ACL that carries a condition can no longer be PATCHed.** CONFIRMED (this fails closed, so it is availability only)
- **Where:** `lib.rs:2312` and the guard.
- **Cause:** N-Triples serialisation turns `acl:condition [ … ]` into `acl:condition _:b`. `parser.rs` maps `_:b` to `Condition::Unknown`, `validate_acl_document` rejects it, and the guard returns 409.
- **Consequence:** conditions cannot be dropped through PATCH. A PATCH on an ACL that has conditions is simply refused, for example 409 with nothing written.
- **Fix:** document the limitation or teach the parser to resolve blank-node conditions. Do not relax the guard.

**7. The WebID that owns a new pod comes from the request's `Host` header when `public_origin` is not set.**
- **Where:** `lib.rs:2991` and `lib.rs:3074`, via `scheme::request_origin`.
- **Scope:** Turtle injection is not possible. `valid_pod_name` and `valid_authority` restrict the characters, and `effective_scheme` returns `&'static str`. The only effect is that a pod creator can choose the host part of their own owner WebID.
- **Fix:** require `public_origin` whenever registrations are open.

### Checked and not a problem
- **`acl_targets_as_paths`:** `https://host.evil`, a different port, userinfo, `//`, a different letter case, and query or fragment are all left unchanged, so none of them can match a local path. Cross-resource targets cannot escalate, because the resolver only reads this ACL for its own subtree.
- **Guard content type:** the guard and the resolver parse the same bytes under `text/turtle` with the same parser.
- **Named pod ACLs:** the resolver marks ancestor ACLs as `inherited`, and the tests cover `private/` and `settings/`. So the root's public `accessTo` without `default` does not expose them.
- **`create_account` / `create_pod`:** the existence probe now stops on a storage error instead of treating it as "free".
- **Main's repairs:** the payment, export, provenance, mirror and did:nostr code is not touched by this diff. Malformed-ACL denial and condition preservation in `parser.rs` are kept; `add_ids` only makes repeated predicates accumulate.

`★ Insight ─────────────────────────────────────`
- The new `where` matcher works like a small join engine with no join planning. Because only exactly one solution is accepted, it can stop as soon as it finds a second one.
- Previously, every ACL PATCH was refused because the guard checked the result as N-Triples. Switching the guard to `text/turtle` turns ACL PATCH on for the first time, which is why the PATCH findings above (1, 2, 4 and 6) are new here.
- A container-level `accessTo` here also covers the container's direct children. That is safe only because of the `inherited` flag, which makes the evaluator ignore `accessTo` on ACLs found at an ancestor.
`─────────────────────────────────────────────────`
EXIT 0
