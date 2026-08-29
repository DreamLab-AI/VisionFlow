| Date | Deep | Finding | Issue | PR | Evaluated? | Verdict | Effect | Witness | Prior-night fates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-16 | content-integrity | Given a static marketing site (`index.html`, 58 607 B) whose link evaluator repo | NONE | NONE | yes | INCONCLUSIVE |  | 4e70df4766da |  |
| 2026-08-17 | build-pipeline | Given the 2026-08-16 content-integrity night ended INCONCLUSIVE on an environmen | NONE | NONE | yes | ACCEPT |  | 3a02827ffea6 |  |
| 2026-08-17 | build-pipeline | Given a static-site build pipeline (`build.sh`) that copies source assets to `we | NONE | NONE | yes | ACCEPT |  | cd1fb6eed062 |  |
| 2026-08-28 | operator-handoff | OPERATOR FIX: evaluatorEntrypoints with nested double quotes were mangled by the annexe ssh dispatch (bash -lc consumes one escaping level); affected evaluators converted to checked-in scripts (scripts/dream-*.sh) invoked quote-free. Verified passing locally. Dream cycle: trust the script form; never inline double-quoted logic in dream.config.json. RuVector key: dream-evaluator-ssh-quoting-bug-class (patterns ns) | NONE | NONE | n/a | OPERATOR |  | session-018aCYi4 |  |
| 2026-08-29 | build-pipeline | Given the 2026-08-28 operator conversion of evaluators to checked-in scripts (`s | NONE | NONE | yes | ACCEPT |  | 22ffea27b4bf |  |
