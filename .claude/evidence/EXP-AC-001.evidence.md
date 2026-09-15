---
expectation_id: EXP-AC-001
git_sha: 22c5065
produced_by: agent:claude-fable-5.1 (queen, this session)
produced_at: 2026-09-14T14:59:15Z
audited_by: agent:claude-sonnet-5 (degraded: same family as producer; codex GPT-6 Astra unavailable — bwrap sandbox refused in container)
audited_at: 2026-09-14T19:18:23Z
auditor_verdict: gaps_closed
auditor_counter_examples_attempted: 9
auditor_counter_examples_found: 0
stabilized_by: tests/gates/augmentation-citations.test.cjs (wired into tests/gates/run-all.sh; npm run check:augmentation)
---

## Scenario 1: the matrix table exists with six rows and every citation resolves

**Command:** `node scripts/check-augmentation-citations.cjs`

**Raw output:**
```
26 citations checked across 6 rows × 3 substrates; 0 failure(s)
```

**Verdict:** ✅ 26 citations across 6 rows × 3 substrates resolve in the sibling checkouts (nostr-rust-forum, agentbox, project=VisionClaw).

## Scenario 2: the gate goes red on each deliberate defect

**Command:** `node tests/gates/augmentation-citations.test.cjs`

**Raw output:**
```
ok - real matrix passes against fixture estate: 26 citations checked across 6 rows × 3 substrates; 0 failure(s)
ok - missing cited path → exit 1: FAIL C2 Meaningful human control/agentbox: missing /home/devuser/workspace/.tmp/ac-gate-vbCKWN/agentbox/management-api/lib/does-not-exist.js
ok - elided path → exit 1
ok - measured citing a document → exit 1
ok - five rows → exit 1
ok - real matrix passes against sibling checkouts: 26 citations checked across 6 rows × 3 substrates; 0 failure(s)
augmentation-citations gate suite OK
```

**Verdict:** ✅ Missing path, elided path, `measured`-cites-document and five-row table each exit 1; the real matrix exits 0 against both the fixture estate and the live siblings.

## Scenario 3: CP-05 / CP-07 name the conditions; terminology defines the four terms

**Command:** `grep -nE "C[1-6]" docs/estate-review/closeout/README.md | cut -c1-120; grep -nE "^- \*\*(augmentation condition|task-property triple|vacuous verification|calibration sample)\*\*" docs/terminology.md | cut -c1-80`

**Raw output:**
```
35:| CP-05 Human judgement and governance | Forum, agentbox, VisionClaw, commercial website | Human intent and rationale
37:| CP-07 Memory and improvement | RuVector consumed modules, agentbox, dream-engine | Compatible embedding/persistence
49:- **augmentation condition** — "one of the six grading questions from arXiv 2
50:- **task-property triple** — "verifiability, reversibility and stakes, declar
51:- **vacuous verification** — "a signed decision made without the proposal, it
52:- **calibration sample** — "a low-risk, reversible request deterministically 
```

**Verdict:** ✅ CP-05 names C2 and C3; CP-07 names C4; all four terms defined with subject-specific glosses.

## Security gate

**Command:** `deepsec-gate.sh --diff-working` → PASS, 0 findings, receipt `.deepsec-gate/reports/20260914T145625Z/receipt.json`.

## Auditor adversarial probes

EDD anti-fox audit against `scripts/check-augmentation-citations.cjs` at HEAD 6714335. Method: an
archive copy of the repo under `/home/devuser/workspace/.tmp/ac-audit/repo` (never the working tree),
driven with `--root` against a fixture estate that materialises every currently-cited path as a
3-line stub file under `nostr-rust-forum/`, `agentbox/`, `project/` — same technique as
`tests/gates/augmentation-citations.test.cjs`. Each probe edits only the scratch matrix copy, runs
the real checker unmodified, and restores the scratch matrix before the next probe. 7 probes run; 5
exposed gaps where EXP-AC-001's stated guarantee does not hold, 2 confirmed correct rejection.

**Probe 1 — citation line number exceeds the cited file's length.** GAP.
Command: rewrote `management-api/lib/execution-projections.js:7` → `...:9999` against a 3-line fixture
file, ran `node scripts/check-augmentation-citations.cjs --root $FIXROOT`.
Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0).
The checker only calls `fs.existsSync(full)` on the path component; it never reads the file or
parses the `:line` suffix, so a citation can point at a file that exists but has far fewer lines
than the cited number. EXP-AC-001 does not explicitly require line-bounds checking, but a
`file:line` citation whose line cannot exist is not meaningfully "grounding" a status and a reader
who spot-checks the citation will find nothing there — this is the closest analogue in-scope to
"citation path existence" the expectation names.

**Probe 2 — a `partial` cell with no citation.** No gap (checker behaves correctly).
Command: replaced the C5/nostr-rust-forum cell with `| C5 Career pathways | \`partial\` — no receipt
yet, mechanism understood |` (status token present, zero backticked citations).
Output: `FAIL C5 Career pathways/nostr-rust-forum: partial without a citation` / `25 citations checked
... 1 failure(s)` (exit 1). Matches `checker.cjs:31` (`status !== 'absent' && cites.length === 0` →
FAIL) and EXP-AC-001's "every non-`absent` cell carries a citation". Correctly enforced.

**Probe 3 — a substrate column header the checker does not map, aliased to an existing key.** GAP.
Command a: renamed the third header to `VisionClawX` (a header with no fixture directory at all) →
correctly fails closed with 10 "missing" failures (the `SUBSTRATE_DIRS[substrate] || substrate`
fallback resolves to a non-existent directory, so citations legitimately fail).
Command b (the real gap): renamed the third header from `VisionClaw` to literally `agentbox` (an
existing `SUBSTRATE_DIRS` key), leaving the VisionClaw cells and their `project/`-relative citations
untouched.
Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0).
The checker takes column identity entirely from `cols[i]` positionally and never validates that the
three header cells are exactly `{nostr-rust-forum, agentbox, VisionClaw}` (in any order or count). A
header edit that accidentally duplicates an existing substrate name silently re-routes that column's
citation checks to the wrong sibling checkout. In the fixture this passes outright because the
fixture materialises every cited path under all three directories; against the real sibling
checkouts a header collision would likely still fail (VisionClaw's `src/actors/...` paths do not
exist under `agentbox/`), but the checker gives no diagnostic naming the header problem itself — it
would report ordinary "missing" failures that point an editor at the wrong root cause.

**Probe 4 — a citation to a directory rather than a file.** GAP.
Command: rewrote `management-api/lib/execution-projections.js:7` → `management-api/lib:7`, where
`management-api/lib/` exists as a directory in the fixture (it contains `authority.js` etc.).
Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0).
`fs.existsSync` is true for directories as well as files; the checker never calls `statSync(...).
isFile()`. A citation that degrades from a precise `file:line` to a bare directory path (e.g. from a
sloppy find-and-replace) passes silently.

**Probe 5 — a row labelled C7.** GAP (the most serious of the five).
Command: inserted `| C7 Rogue condition | \`measured\` — total fabrication (\`does/not/exist.rs:1\`)
| ... | ... |` (three cells, each citing a nonexistent path) ahead of the real C6 row.
Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0) — identical to
baseline; `rows.length` is still 6.
`section.split('\n').filter(l => /^\| C[1-6] /.test(l))` matches only `C1`–`C6`; a `C7` (or any
non-C1–C6-prefixed) row is invisible to the row loop entirely, so nothing in it — including
completely fabricated citations to files that do not exist — is ever checked, and the 6-row count
assertion does not notice the table now has seven data rows. This is the sharpest counter-example:
EXP-AC-001 says "exactly six condition rows (C1–C6)" but the checker only positively matches known
labels rather than asserting the *total* row count in the table body equals 6, so an extra row with
any other label (C7, C0, a typo'd condition) is silently ignored rather than rejected.

**Probe 6 — a citation written in prose outside backticks.** No gap (checker behaves correctly).
Command: replaced the C5/nostr-rust-forum cell with `| C5 Career pathways | \`partial\` — see
crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:738-747 in prose, no backticks used |`
(status token still backticked; the path:line is bare prose, not inside backticks).
Output: `FAIL C5 Career pathways/nostr-rust-forum: partial without a citation` / `25 citations
checked ... 1 failure(s)` (exit 1). The citation regex (`checker.cjs:30`) requires the whole
`path:line` token inside backticks, so a prose-only mention is correctly not recognised as a
citation and the cell correctly fails as uncited. Correctly enforced.

**Probe 7 — a `measured` cell citing a `.md`-content path via a non-`.md`-looking path (e.g.
`README` with no extension).** GAP.
Command: replaced the C1/nostr-rust-forum cell with `| C1 Durable net value | \`measured\` — see the
README (\`crates/nostr-bbs-forum-client/README:1\`)`, and created a prose-only file at that path (no
`.md` extension, plain-text/markdown-style content, no code).
Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0).
The document-detection rule at `checker.cjs:34` is `/\.md$/.test(cite) && status === 'measured'` —
purely an extension sniff. A document renamed or authored without a `.md` suffix (a bare `README`,
`NOTES`, `CHANGELOG`, or a `.txt`/no-extension doc) defeats the check entirely, letting a `measured`
status stand on a document citation, which is exactly the EXP-AC-001 counter-example ("A `measured`
or `partial` cell citing a document rather than code or a receipt") stated as must-not-happen.

### Summary

| # | Probe | Result |
|---|---|---|
| 1 | line number past EOF | **gap** — accepted |
| 2 | `partial` with no citation | correct — rejected |
| 3 | header aliased to existing substrate key | **gap** — accepted |
| 4 | citation to a directory | **gap** — accepted |
| 5 | `C7` row with fabricated citations | **gap** — row invisible, accepted |
| 6 | prose citation outside backticks | correct — rejected |
| 7 | `measured` citing extensionless doc | **gap** — accepted |

5/7 adversarial probes exposed real gaps between EXP-AC-001's stated guarantee and the checker's
actual enforcement. The two counter-examples the expectation explicitly names in its own
"Counter-examples" section (document-cites-code, missing-path) are both defended by the checker's
*obvious*-case handling (`.md`-suffixed paths, exact-match nonexistent paths) but not against the
adjacent cases an adversarial or merely careless edit would produce (extensionless docs, directories,
mislabelled rows, aliased headers, unbounded line numbers). None of the five gaps were fixed as part
of this audit — reporting only, per mandate.

## Auditor re-audit after 4b62289

Commit 4b62289 hardened `scripts/check-augmentation-citations.cjs`: it now opens every cited file
with `statSync`/`isFile()`, reads its content and checks every number in the cited line range against
the actual line count, asserts each row label is exactly one of C1–C6 (each exactly once) plus a
total-row-count assertion, asserts every header column is a known, unique substrate name, and gates
`measured` citations on an evidence-extension allowlist (`.rs .js .cjs .mjs .ts .tsx .sql .toml .json
.sh .yml .yaml .nix`) rather than a `.md` blocklist. Re-audited at HEAD a815c7f (commit `4b62289` is
in the ancestry) using the identical archive-copy + fixture-estate method as the first audit — this
time with 2,500-line fixture stub files (the checker now reads file length, so 3-line stubs from the
first audit would trivially fail every real citation).

**Re-ran all 7 original probes.** All 5 former gaps are now closed; the 2 that were already correct
remain correct:

| # | Probe | First audit | Re-audit |
|---|---|---|---|
| 1 | line number past EOF | gap — accepted | **closed** — `... cites line 999999 but the file has 2501 lines`, exit 1 |
| 2 | `partial` with no citation | correct | correct — unchanged |
| 3 | header aliased to existing substrate key | gap — accepted | **closed** — `FAIL duplicate substrate column "agentbox"`, exit 1 |
| 4 | citation to a directory | gap — accepted | **closed** — `FAIL ... not a regular file ...`, exit 1 |
| 5 | `C7` row with fabricated citations | gap — row invisible, accepted | **closed** — `FAIL row label "C7" is not one of C1..C6` + `expected 6 condition rows, found 7` + the row's own fabricated citations now fail as missing, exit 1 |
| 6 | prose citation outside backticks | correct | correct — unchanged |
| 7 | `measured` citing extensionless doc | gap — accepted | **closed** — `FAIL ... measured must cite code/config/receipt, not "...README"`, exit 1 |

**2 new probes**, chosen to poke at the specific shape of the fix (range-endpoint handling,
multi-citation cells, header normalisation, extension casing) rather than repeat the first round:

**New probe A — a cell with two citations where only the second is bad.** No gap.
Command: in the C2/nostr-rust-forum cell (two citations already present), rewrote only the second
citation's path to a nonexistent file, leaving the first citation untouched and valid.
Output: `FAIL C2 Meaningful human control/nostr-rust-forum: missing .../does-not-exist.rs` / `26
citations checked ... 1 failure(s)` (exit 1). The citation loop (`checker.cjs:62-74`) iterates every
`cites` entry with no short-circuit, so a bad second citation is caught even when the first in the
same cell is fine. Correctly enforced.

**New probe B — a line range whose start exceeds the file length but whose end does not
(`999999-74` against a 2,500-line file).** No gap.
Output: `FAIL C1 Durable net value/VisionClaw: src/services/insight_loop.rs:999999-74 cites line
999999 but the file has 2501 lines` (exit 1). `maxLine` is computed as
`Math.max(...ln.split(',').flatMap(r => r.split('-').map(Number)))` over every number in the whole
range/list, not just the first or the nominal "end" — order within the range is irrelevant, so a
reversed or padded range cannot smuggle an out-of-bounds number past the check. Correctly enforced.

**New probe C — a header cell with surrounding whitespace (`|   agentbox   |`).** Not a gap (correct
by design). Output: `26 citations checked across 6 rows × 3 substrates; 0 failure(s)` (exit 0). Header
cells are split on `|` and `.trim()`-ed before comparison against `SUBSTRATE_DIRS`, so ordinary
ASCII whitespace padding around a legitimate header is correctly normalised away — this is accepting
a still-valid table, not tolerating a defect, so it is not a counter-example. (Not tested: a
zero-width or non-breaking-space character inside the header text, which `String.trim()` may not
strip in all cases — flagged as a follow-up probe, not run here as it strays from the assigned
"surrounding whitespace" case.)

**New probe D — a citation extension in uppercase (`PANEL_REGISTRY.RS` instead of `panel_registry.
rs`).** Not a gap (correct by design). Output: `26 citations checked across 6 rows × 3 substrates; 0
failure(s)` (exit 0). `checker.cjs:65` lowercases the extension (`path.extname(file).toLowerCase()`)
before testing it against `EVIDENCE_EXT`, so a genuinely-uppercase-extensioned source file is still
correctly recognised as code evidence, not a document. Deliberate, correct handling — not a bypass of
the document-citation rule, since the cited artefact really is code.

### Re-audit verdict

**gaps_closed.** 9/9 probes now resolve as intended: the 5 former gaps all fail closed with a
specific, correctly-attributed error message; the 2 already-correct probes are unchanged; the 2 new
probes confirm the hardened checks (multi-citation iteration, range-max computation) have no
early-exit or off-by-position blind spot, and that the two "looks risky but is actually fine" cases
(whitespace-padded headers, uppercase code extensions) are handled deliberately rather than by
accident. No further gaps identified against the mandate's probe set; the one noted follow-up
(non-ASCII/invisible whitespace in header cells) was not run and is not claimed as a finding.

## Iteration after audit (EDD step 6) — 2026-09-14T19:16:40Z

All five auditor gaps closed in commit `4b62289` (checker opens each cited file; regular-file check; line ≤ file length; row labels exactly C1–C6; substrate headers known and unique; `measured` requires a code/config/receipt extension). Each gap is now a red case in `tests/gates/augmentation-citations.test.cjs`.

**Command:** `node tests/gates/augmentation-citations.test.cjs`

**Raw output:**
```
ok - real matrix passes against fixture estate: 26 citations checked across 6 rows × 3 substrates; 0 failure(s)
ok - missing cited path → exit 1: FAIL C2 Meaningful human control/agentbox: missing /home/devuser/workspace/.tmp/ac-gate-pIiN1I/agentbox/management-api/lib/does-not-exist.js
ok - elided path → exit 1
ok - measured citing a document → exit 1
ok - five rows → exit 1
ok - line past EOF → exit 1
ok - duplicate substrate header → exit 1
ok - unknown substrate header → exit 1
ok - directory citation → exit 1
ok - C7 row → exit 1
ok - measured citing extensionless doc → exit 1
ok - real matrix passes against sibling checkouts: 26 citations checked across 6 rows × 3 substrates; 0 failure(s)
augmentation-citations gate suite OK
```

**Verdict:** ✅ 11 defect cases exit 1; real matrix exits 0 against fixture and live siblings. Re-audit of the hardened checker is open for the auditor.
