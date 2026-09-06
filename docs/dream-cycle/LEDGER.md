| Date | Deep | Finding | Issue | PR | Evaluated? | Verdict | Effect | Witness | Prior-night fates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | content-integrity | Given a static marketing site (`index.html`, 58 607 B) whose link evaluator repo | NONE | NONE | yes | INCONCLUSIVE |  | 4e70df4766da |  |
| 2026-08-17 | build-pipeline | Given the 2026-08-16 content-integrity night ended INCONCLUSIVE on an environmen | NONE | NONE | yes | ACCEPT |  | 3a02827ffea6 |  |
| 2026-08-17 | build-pipeline | Given a static-site build pipeline (`build.sh`) that copies source assets to `we | NONE | NONE | yes | ACCEPT |  | cd1fb6eed062 |  |
| 2026-08-28 | operator-handoff | OPERATOR FIX: evaluatorEntrypoints with nested double quotes were mangled by the annexe ssh dispatch (bash -lc consumes one escaping level); affected evaluators converted to checked-in scripts (scripts/dream-*.sh) invoked quote-free. Verified passing locally. Dream cycle: trust the script form; never inline double-quoted logic in dream.config.json. RuVector key: dream-evaluator-ssh-quoting-bug-class (patterns ns) | NONE | NONE | n/a | OPERATOR |  | session-018aCYi4 |  |
| 2026-08-29 | build-pipeline | Given the 2026-08-28 operator conversion of evaluators to checked-in scripts (`s | NONE | NONE | yes | ACCEPT |  | 22ffea27b4bf |  |
| 2026-08-30 | webgl-mesh | Given the WebGL2 mesh is self-contained in the hand-written site, when its shade | NONE | NONE | yes | INCONCLUSIVE |  | 330422a39b66 |  |
| 2026-08-31 | seo-and-meta | Given the 2026-08-30 carry-over that CNAME propagation into build output was unv | NONE | NONE | yes | ACCEPT |  | 063e903dfbf7 |  |
| 2026-09-01 | build-pipeline | Given commit `9f024d0` with a static-site pipeline whose only evaluators are `sc | NONE | NONE | yes | ACCEPT |  | 96ab16eaeebd |  |
| 2026-09-02 | webgl-mesh | Given the WebGL2 hero mesh is self-contained hand-written JS with a canvas-fallb | NONE | NONE | yes | ACCEPT |  | 2076c86a8490 |  |
| 2026-09-03 | seo-and-meta | Given the hand-written static site ships zero JSON-LD structured data (SD-SCAN ` | NONE | NONE | yes | ACCEPT |  | 7158dbdda9bf |  |
| 2026-09-04 | content-integrity | Given the meta-scan baseline `twitter-tags: 0` alongside `og-tags: 4` on the han | NONE | NONE | yes | ACCEPT |  | 9db384a9b268 |  |
| 2026-09-05 | build-pipeline | Given the 2026-09-01 sentinel baseline (build.sh emits `BUILD-COMPLETE bytes:N`; | NONE | NONE | yes | ACCEPT |  | f76334fc892e |  |
| 2026-09-06 | webgl-mesh | Given the WebGL2 hero mesh is self-contained inline hand-written JS (ledger 2026 | NONE | NONE | yes | INCONCLUSIVE |  | d9a0030a0b03 |  |
