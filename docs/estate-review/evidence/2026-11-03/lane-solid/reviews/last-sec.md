No Medium-or-higher findings.

`★ Insight ─────────────────────────────────────`
- **Join correctness:** Stopping at two solutions is safe. Every solution the join produces is distinct: a pattern whose positions are all fixed matches at most one triple, because the graph is a set, and any other pattern differs between matches in a newly bound variable. Two results therefore always mean the clause is ambiguous.
- **Repeated variables (`?x ?p ?x`):** These work. `extend` checks the already-bound value first, then any value bound earlier in the same pattern. The `usize::MAX` placeholder can never equal a real variable index, and a constant missing from the document can never match.
- **u32 overflow:** The ids only collide with `UNBOUND` past about 4.29 billion distinct terms. A stored body capped by `JSS_MAX_REQUEST_BODY` can't hold that many, so this isn't reachable.
- **Patch fallback:** The fallback for stored bodies the Turtle reader refuses is a superset of main, which only read N-Triples. The N-Triples reader (`ldp.rs:826`) is strict per line, so a large Turtle body gets a 409 rather than losing triples silently. Dropping the 413 also stops write-only callers learning a resource's size.
- **Authorisation:** `lib.rs` changes only a comment in `handle_patch`, so the Read checks before PATCH are the same as before.
`─────────────────────────────────────────────────`
EXIT 0
