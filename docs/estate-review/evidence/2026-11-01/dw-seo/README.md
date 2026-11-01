# SEO watch audit fixtures

These are audit-only counterexamples against website revision `b5eb1ecf96599ebe154ebf23bbe7c703d2e653a5`. They intentionally assert observed defects. A passing scenario is evidence for the report, not a product regression fix.

The run reused the website checkout's installed dependencies. `harness.json` records Node, Git, Vitest and jsdom versions; this does not attest a fresh lockfile installation. Source exports are identified by `sources.json` and retained at `/tmp/vf-dw-seo-artifacts/{repo}/{path}`.

To reproduce, create an empty disposable harness under an owned task worktree on an executable filesystem. Export the pinned `scripts/seo-watch/` tree and `scripts/__tests__/seo-watch.test.mjs` into it. Copy `probes.mjs`, `entrypoint-probes.mjs`, `mock-fetch.mjs` and `vitest.config.mjs` from this directory into the harness root; create `package.json` containing `{"type":"module","private":true}` and provide the website's compatible `node_modules`. Create `/tmp/vf-dw-seo-artifacts` for the result outputs. From that harness root, run:

```sh
node_modules/.bin/vitest run --config vitest.config.mjs
node probes.mjs
node entrypoint-probes.mjs
```

Use an empty harness for each run: the scripts intentionally do not overwrite prior fixtures. `probes.mjs` creates a local repository with no remote and `protocol.allow=never`; it substitutes a fake `gh` which accepts only PR listing and records calls. The policy probe tests validation and actual Git application, not the complete build/publish chain. The publisher probe starts with a local modified text file and reaches a real commit before the expected rejection.

`entrypoint-probes.mjs` launches the actual runner with an explicitly constructed environment containing only a synthetic provider key and fixture plumbing. The preload intercepts every fetch. Its Git shim accepts only fetch, revision and source-read commands. Browser and proposal execution are disabled; no real provider or forge is contacted. The four scenarios cover error scrubbing, retry usage, a same-day weekly rerun and a ledger containing a valid spent-budget row followed by malformed JSON.

The first publisher harness attempt used CommonJS `require` inside an extensionless shim resolved as ESM; the fake command failed before logging. The corrected ESM shim passes and records exactly one fake PR-list call. The failed harness log is retained locally and is not counted as product evidence. The first diagram gate also rejected two bad line anchors; the corrected pilot has zero warnings. All six final panels were rendered and visually inspected.

Keep outputs before removing the disposable harness. The runtime receipt is a separate read-only observation of the real scheduler/ledger and does not originate from these fixtures. The source manifest, test results, scope qualifications and recommended acceptance criteria are in the parent report.
