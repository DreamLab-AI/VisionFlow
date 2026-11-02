# External review reconciliation — 2026-11-02

Source baseline: `0e1572801d03a1af6d8bc8861f928da087d69eed`; forum source inspected at `c39f41f104b114d6a3fa29013daaf6fa58b45080`. This note records source evidence and acceptance gaps. It does not change ADR decision or activation statuses.

| External concern | Current-source disposition and acceptance |
| --- | --- |
| SEO publication after commit (DW-08.5) | Confirmed. `scripts/seo-watch/lib/propose.mjs:288` commits before :298 checks `git diff --cached --name-only`. A normal successful commit empties the index; the expected-file comparison then throws before push/PR. Validate the staged set before commit or the committed tree afterwards; test the real Git lifecycle. |
| SEO error scrubbing (DW-08.6) | Confirmed. `scripts/seo-watch/run.mjs:253` writes advice.error directly into ledger JSON. Scrubbing the Markdown report and PR body does not establish scrubbing of this sink. Exercise secret-shaped synthetic errors across every persisted output. |
| Hand-synchronised configuration | Supported and deliberate in ADR-2005, with check-config-mirrors tests. Such checks only cover named groups. They do not prove complete TOML-to-Worker projection or live deployment equality; missing encryption/template declarations require separate acceptance. |
| Inbox registry confusion (DW-09) | Confirmed. `scripts/seed/provision-website-info.mjs:97` treats public disclosure as registry presence. Forum `relay-worker/src/agent_disclosure.rs:68–102` merges configured ingress into that disclosure, so a configured recipient may be listed without an agent_registry row. Verify durable registration via an authority-specific response and test reset-db. |
| Inbox health privilege check | Confirmed gap. `scripts/inbox-watch/run.mjs:229,293` checks disclosure and admission, not the no-privilege condition claimed for the information identity. Admission and configuration survival do not prove zero cohorts or privileges. Add a distinct privilege check with an explicit failure signal. |
| Digest seen-set truncation | Confirmed. `scripts/inbox-watch/run.mjs:327–335` marks wraps seen before classification and presentation; the digest displays at most 25. Preserve undisplayed verified messages for later presentation and distinguish malformed, probe, displayed and pending IDs. |
| Identity and key split | ADR-2006's ephemeral NIP-42 signing and unverified extension hint are supported. The external five-location admin/governance split is an operational cutover recommendation; this pass does not establish execution, custody or deployed key separation. Do not infer it from a config file. |
| Missing local zone-encryption decision | The existing ADR-2007 describes dual-accept cohorts and encrypted flags; it does not supply a new adopted encryption decision. Record the gap with the instance owners before allocating the proposed ADR-2016 number. This review does not invent or approve a decision. |
| Static SPA, Worker overlay and cron jobs | Supported architecture at the source revision. Cron process liveness does not establish successful SEO publication or inbox completeness. Exact mirror-group counts and model defaults remain revision-specific. |

Historical browser and production receipts remain bounded to their original revisions. No API mutation, publication, key change or deployment was performed in this reconciliation.
