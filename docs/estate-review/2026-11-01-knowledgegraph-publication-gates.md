---
title: knowledgeGraph publication gates and historical invariant succession
date: 2026-11-01
status: assessed-with-open-findings
scope: KG-05 and KG-07, configured visionGraph publisher, knowledgeGraph ADR-2002 to ADR-2004
---

# knowledgeGraph publication gates and historical invariant succession

The configured publisher's final credential scanner misses a synthetic credential-shaped canary after the workflow compresses it. Its layout checker also accepts invalid gzip contents and equal-count substitution of a notes page. These are reproduced limitations of individual gates, not evidence of a real secret leak, corrupted public site or complete deployment bypass.

All five panels in [KG-05](../diagrams/knowledgegraph/05-ci-cd-and-publish.md) and [KG-07](../diagrams/knowledgegraph/07-invariants-register.md) are source-assessed and corrected. All sections of knowledgeGraph ADR-2002, ADR-2003 and ADR-2004 have dispositions below. Together with the [previous ADR-2001 supplement](2026-11-01-knowledgegraph-explorer-contracts.md), this covers the four operative knowledgeGraph records at the cited revision. It does not close the archived decision inventory, KG-01, KG-06, runtime acceptance or the estate goal. No runtime code or decision statuses were changed.

## Evidence and limits

| Evidence | Scope |
|---|---|
| knowledgeGraph | `4ed9ac159daf402b4fb252ce559dbeb894a8d91e`; committed baseline and records, archive marker, historical build and manifest/identity code. Existing dirty README, baseline and licensing edits were preserved. The working baseline already adds a publication-boundary qualification; this pass does not attribute those bytes to the commit. |
| visionGraph | `556a0156b5c326fb61141ed2ab1760dd8d9c4dd6`; full publishing workflow, final scanner, layout checker and compressed-fetch adapter. |
| ontomark | Release tag `v0.7.0` resolves locally to `0caae5d4b695cb720942b9b9212017c00a80ad41`, also checkout HEAD. Inspected Markdown mirror implementation and title/identity fixtures. No new compiler build or release-binary verification was performed. |
| Focused probes | **11 passed**. Actual extracted workflow shell blocks and actual checker scripts executed against temporary synthetic trees. No deployment, real credential or production data was used. |
| Existing publisher tests | **41 passed** via `python3 -m unittest discover -s scripts/tests -v`. These include controlled fake tools and contract tests; they do not build or deploy the real corpus. |
| Scanner self-test | Passed its thirteen-rule fixture. That fixture is uncompressed; its success does not establish coverage of compressed output. |
| Diagram checks | Five renders; 39 revision-verified citations across the two topics, without fallback or exceptions. Whole-corpus validation retained separately. |

[Probe source](evidence/2026-11-01/kg-publication/probes.py), [execution log](evidence/2026-11-01/kg-publication/probes.log), [source hashes](evidence/2026-11-01/kg-publication/sources.json) and [retention receipt](evidence/2026-11-01/kg-publication-retention.json) are the reproducibility boundary. Python is used to drive the existing shell/Python contracts, not as a new runtime implementation. No full Quartz build, browser smoke, live workflow run, complete ontomark audit or downstream activation check is claimed.

## Findings and acceptance conditions

| ID | Evidence and qualification | Consequence and useful acceptance |
|---|---|---|
| KG-P1 | **High, final-scan coverage:** workflow compression at `publish.yml:358-373` precedes the final scanner at line 504. `check-secrets.sh:150` uses `grep -rlIE`; binary data is skipped. Probe 8 rejects the raw canary without printing it; probe 9 runs the real compression step, verifies the canary survives decompression, and observes scan exit 0. | A credential present only in compressed page output can evade this last defence. Source validation and an uncompressed duplicate may still catch it; neither was bypassed here. Scan the final tree with bounded decompression and retain canary controls for each stored representation. Preserve redacted logs. No real secret was searched for or exposed. |
| KG-P2 | **Medium, representation validity:** `check-urls.sh:90-95` accepts existence of page JSON/Markdown or their `.gz` names. Probe 4 supplies non-gzip bytes under both gzip names and still passes the full layout script. Probe 5 removes the JSON path and fails as expected. | Presence is weaker than readable content. Validate gzip integrity, decoded JSON shape and page identity at the same paths the browser uses. This finding does not show that the actual producer emits corrupt bytes or that browser smoke would accept them. |
| KG-P3 | **Medium, notes completeness:** line 115 counts staged Markdown and line 116 counts built HTML; probe 6 has `content/pages/one.md` and only `notes/pages/other.html`, with equal counts and a passing layout result. Probe 7 shows missing staging causes a SKIP and exit 0. The real workflow supplies a staging path, so the latter is a script-level fail-open boundary, not an observed CI omission. | An omitted page can be hidden by an extra page. Compare expected output identities/URLs, with explicit transformations for slugs, and require staging for the release-mode invocation. Report missing and unexpected pages separately. |
| KG-P4 | **Medium, incomplete final-size evidence:** `Finalize website` measures apparent size before building/copying the SPA, static routes and notices. Probe 11 passes the small tree, adds a sparse 1,002 MiB file, and confirms the same gate would reject the enlarged tree if rerun. No such rerun appears after assembly. | The successful measurement does not bound the deployed tree. Measure after all additions. The fixture proves ordering, not the size of the real site. GitHub documents a 1 GB published-site limit; the workflow's `du -sm`/1,000 threshold should also have explicit units. |
| KG-P5 | **Medium, automatic freshness:** `publish.yml:24-30` filters pushes to selected paths, omitting `working/**` and root `assets/**`; the build explicitly uses public working pages and links root assets at lines 155-203. | A push touching only these inputs does not match this workflow's path filter under ordinary diff evaluation. Manual runs or a simultaneous matching change can publish them later. Cover all actual inputs, then test a working-only edit and a referenced-asset-only edit. This is source/configuration evidence, not a deliberately triggered remote run. |
| KG-P6 | **Medium, receipt boundary:** the copy hash comparison occurs before compression. Probe 10 passes the actual copy step, runs actual compression, then finds 1,002 inventory paths absent under their old names, including `data/ontology.json`; their gzip replacements exist. | The copy receipt proves transit into a staging state, not final deployed bytes or raw-path availability. Record a final inventory and content-encoding map after all transformations. Do not claim the old Python manifest is the current producer's format or that its verifier ran in this workflow. |
| KG-P7 | **Medium, decision succession:** knowledgeGraph's accepted/complete/live records describe a root Python build, exact 8,138-class pin and immutable `ontology/pages` locations. The implementation is archived. The configured publisher uses ontomark and different volume/domain checks; canon ADR-2013 still names `VisionClaw/crates/vault` as sole compiler. | Scope or supersede the historical obligations explicitly. Preserve the reasons for independent count/identity checks, but assess their successor enforcement rather than calling old closeout items moot. This review records the tension without silently promoting or rewriting decisions. |

The early machine-artefact gate is deliberately weaker than final acceptance. Probe 1 passes 1,000 index entries and invalid page/binary bodies; probe 2 rejects 999 entries. Probe 3 exposes a wrong-shape index whose undefined `.length` produces a shell integer error yet passes that early gate. The later layout script rejects the same non-array shape. That later refusal is why this report does **not** call it an end-to-end publication bypass.

The publishing viewer supports gzip explicitly (`fetchArtefact.ts:22-38`); the retained knowledgeGraph viewer assessed previously does not share that implementation. The release-tag compiler writes page-ID Markdown names and title aliases only where unambiguous (`mod.rs:842-919`). Its source fixtures cover title/name and slash variants. This supersedes a blanket claim that the current producer writes only slug-form Markdown. It does not certify all alias collisions or a live title-to-prose journey.

## Diagram disposition

| Panel | Correction |
|---|---|
| KG-05.3 | Updated compiler to v0.7.0 and actual step order. Added collision/body gates, compression, pre-SPA size timing, final smoke/scan/layout and post-deploy dispatch. Certificate evidence remains a derivation check, not factual-content certification. |
| KG-05.5 | Replaced whole-repo never-builds claims with explicit archived/source/configured boundaries. Added input-filter gap and the compiler/consumer representation contract. Removed attribution of uncommitted README wording to an old revision. |
| KG-07.1 | Retained all eleven obligations with current committed anchors, clearly historical scope and no claim of verbatim reproduction or present enforcement by the successor. |
| KG-07.2 | Replaced closed/now-moot flow with dated closeout, archive and successor checks. Added equal-count identity loss in the current notes gate. |
| KG-07.3 | Dated old counts and missing lineage; removed today's-8,138 and only-Turtle-is-nondeterministic claims. Distinguished byte receipts from RDF equivalence under blank-node renaming, and exposed ADR-2004/archive tension. |

## knowledgeGraph ADR-2002 section dispositions

| Section | Disposition |
|---|---|
| Metadata/lineage | Accepted/complete/live and d09a476 verification are historical consolidation evidence. The trigger explicitly mentions non-`ontology/pages` authorship; the current frontmatter vault meets the need for a boundary review. |
| Context | Co-locating prose and ontology avoids a separate authored registry; it does not guarantee consistency among derived projections. |
| Decision | Logseq JSON-LD page/class fusion describes the archived corpus. Current publication reads an Obsidian/frontmatter corpus; do not carry the old syntax restriction into the replacement without a successor decision. |
| Consequences | Single-source authorship is useful, but the stated prohibition of non-Logseq authoring has been overtaken. Malformed-fence loss belongs to the old parser's contract. |
| Verification | 6be5871c4, the 8,138 specimens/count and parser line numbers are dated evidence distinct from the frontmatter revision. Not rerun here. |
| Closeout extension | Input census, strict flags, cross-export identity/visibility and separating entailment from factual review remain appropriate acceptance dimensions. |
| Acceptance progress introduction | The three defects declared closed concern that historical implementation, not every replacement or current output representation. |
| Implemented: census | Preserve the coded parser-outcome claim with its revision. Existing historical assessment is not proof that every current staging input is published. |
| Implemented: flags | Literal-boolean publication and missing/non-boolean refusal are old parser safeguards. The present gate tests here neither bypass nor re-certify current source visibility. |
| Implemented: visibility | Known-private reference redaction and deliberate dangling stubs are distinct policies. P1 concerns credential-pattern scanning of final compressed bytes, not a proof that this reference policy fails. |
| Semantic scope | Compiler validation, formal derivation and reviewed factual content must stay separate. Current oo-cert invocation checks a certificate; no factual-content warrant follows. |
| Tests/results | The 85 Python tests are historical. Today's 41 publisher tests and 11 probes cover other surfaces and are not a new combined pipeline total. |
| Governed paths | Listed root `pipeline/` paths have moved under the dated archive. |
| Remaining | Old reasoner absence and upstream URN/HTTP divergence need their own successor evidence; this pass does not prove them resolved or still present in every current consumer. |
| 7 September re-verification | Keep its 96-test/projection/rollback evidence dated. It does not establish final compressed-site acceptance. |

## knowledgeGraph ADR-2003 section dispositions

| Section | Disposition |
|---|---|
| Metadata/lineage | Historical accepted/complete/live and validator-not-run lineage are retained. Archived CI is no longer an active gate merely because the record says live. |
| Context | An independently authored expectation can catch correlated count drift; that rationale remains sound. |
| Decision | Exact `EXPECTED_CLASSES` and both written class counts describe the archived workflow. The current publisher's minimum-volume and domain checks are different guarantees. |
| Consequences | The two-file maintenance obligation was intentional. Removing the old workflow requires explicit disposition of the obligation, not an implicit claim that a minimum threshold is equivalent. |
| Verification | Old 6be5871c4 count/line evidence is not today's active-vault census. |
| Closeout extension | Equal-count substitution, malformed input and visibility changes remain meaningful adversarial cases. P3 applies the identity-versus-count distinction to notes. |
| Acceptance progress introduction / preserved tripwire | Exact preservation is a September claim. It is not true as a description of an active root build at the current revision. |
| Added checks | Archived source still contains count, exact-set, schema and visibility functions. These do not execute merely because they remain in git. |
| Injections | Historical release-gate injection results retain their scope. The new notes substitution probe is a separate current-script counterexample. |
| CI wiring | The six-gate root workflow is now an archived file, not an active GitHub workflow. |
| Tests/results | The reported 85 tests, 8,138 classes and identity digest remain dated. No old whole-corpus build was rerun in this pass. |
| Governed paths | Paths require archive qualification; current publication ownership is visionGraph plus the released compiler. |
| Remaining | Deliberately maintained identity sets remain one possible contract, but the replacement needs its own stated identity acceptance mechanism. |
| 7 September execution qualification | Historical strict input, protected projection and rollback do not prove the final website's scan, identity or size checks. |
| 7 September baseline re-verification | Preserve the 96-test evidence and lack of deployment claim. No status promotion is justified by this review. |

## knowledgeGraph ADR-2004 section dispositions

| Section | Disposition |
|---|---|
| Metadata/lineage | The file-move review trigger applies to the later whole-tree archival transition. Accepted/complete/live alone cannot establish continuing physical-path invariance. |
| Context | Slug-derived identity and inbound references explain why casual renames are dangerous. Whole-tree retirement and per-concept re-keying are distinct operations. |
| Decision | The archive marker contradicts an unqualified permanent `ontology/pages` location. The narrower prohibition of `docs/archive/` is not disproved merely by the existence of root `archive/`. |
| Consequences | Forever-at-original-path wording needs dated scope. Stable semantic IDs require mapping evidence, not inference from source retention. |
| Verification | No-move and banner claims describe the original consolidation, before the later archive. They are not current path assertions. |
| Closeout extension | Mapping page IDs, class IRIs, namespaces and redirects across generations remains the right acceptance obligation. |
| Acceptance progress introduction | The September statement that nothing moved is historical, not a claim about the current tree. |
| Three-generation identity map | Preserve the authored URN → exported HTTP → consumer shape mapping as historical evidence. The current compiler's title aliases are page-content routing, not a substitute for class-identity mapping. |
| Redirects paragraph | Historical none-needed wording is not proof of current URL compatibility. Current workflow stages routes and a legacy explorer redirect; it does not establish every old per-concept URL. |
| Inbound references | The archived committed set detects class substitution in that path. P3 demonstrates that current notes counts do not provide the same identity guarantee. |
| Authoring authority | visionGraph remains the explicitly identified authoring source. The old generation-manifest claim concerns the archived publisher; P6 qualifies the current copy receipt. |
| Tests/results | Keep historical identity and visibility fixtures dated. No current all-IRI migration round trip was executed. |
| Governed paths | Source is preserved under archive, rather than still physically at the locations listed in this annex. |
| Remaining | URN/HTTP and consumer typing obligations need successor-specific verification. They are neither closed nor re-proved by these shell probes. |
| 7 September re-verification | Preserve the shared 96-test evidence and explicit no-activation-promotion stance. |

## Comparison and user utility

GitHub's [path-filter documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow) supports the input-filter finding; its [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) distinguish the published-site size limit from the separate custom-workflow exemption to the build-frequency limit. No hosting migration is recommended from this source review alone.

Gitleaks offers a useful component comparison: its [official documentation](https://github.com/gitleaks/gitleaks#archive-scanning) describes opt-in archive traversal, bounded depth and inner-file paths in findings. Traversal is disabled by default. The current README also says new feature work has moved elsewhere and releases are limited to security patches; this is a comparison of capabilities, not a blanket tool-selection recommendation. The important test is whether a chosen scanner detects the same synthetic canary in each final stored representation, without logging its value.

For corpus users, the proposed acceptance journey is concrete: edit a public working note, observe an automatic build, retrieve its final page and graph generation, and confirm that the same intended page survives every naming and compression step. Measure publication delay, missing/substituted pages, decompression failures, final-site bytes and whether the consumer can identify its generation. Compare the result with a static documentation publisher plus archive-aware scanning before claiming a differentiated governed-knowledge product. These are proposed utility tests, not measured user benefit.
