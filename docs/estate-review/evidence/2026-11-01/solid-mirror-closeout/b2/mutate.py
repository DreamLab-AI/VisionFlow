#!/usr/bin/env python3
"""B2 mutations: copy, mutate, run one test, restore (cp, touch, sha256 check), re-run green."""
import hashlib, os, shutil, subprocess, sys
SRC = "/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src/"
B2 = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.join(os.path.dirname(B2), "b1", "cargo.sh")
BAK = "/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/770d51cb-532b-46ec-b569-b80bea93cc51/scratchpad/b2-mutate.bak"
MUTS = [
 ("M1-trigger-held-not-dropped", "trigger", "migrate_fork.rs", "b2_t7b_",
  "let files: Vec<PathBuf> = triggers.iter().map(|r| r.dropped.clone()).collect();",
  "let files: Vec<PathBuf> = triggers.iter().map(|r| r.held.clone()).collect();"),
 ("M2-fork-filter-off", "inherited-held filter", "migrate_fork.rs", "b2_t8_",
  "if super::reaches_held(source, &sha, ancestor_roots, held).await? {",
  "if super::reaches_held(source, &sha, ancestor_roots, held).await? && false {"),
 ("M3-direct-parent-only", "transitivity", "migrate_fork.rs", "b2_t7_a_fork_of_a_fork",
  "Some(Lineage::Parent(po, pn)) => next = Some((po.clone(), pn.clone())),",
  "Some(Lineage::Parent(_, _)) => {}"),
 ("M4-tmp-sweep-off", "tmp sweep", "migrate_fork.rs", "b2_t19_",
  "    sweep_partial_forks(plugin);\n    let (stores, stores_listed)",
  "    let _ = sweep_partial_forks;\n    let (stores, stores_listed)"),
 ("M5-incomplete-fork-rule-off", "incomplete pass gates forks", "lib.rs", "b2_an_unreadable_fork_config",
  "            if fork_or_unknown {",
  "            if fork_or_unknown && false {"),
 ("M6-trigger-always-fires", "untouched guard", "migrate_fork.rs", "b2_a_fork_with_no_dropped_object",
  "        Ok(false) => Detected::Clean,\n        Err(e) => Detected::Error(e),\n    }\n}",
  "        Ok(false) => Detected::Rebuild,\n        Err(e) => Detected::Error(e),\n    }\n}"),
 ("M7-followed-root-filter-off", "followed fork: inherited filter on roots", "migrate.rs", "b2_a_followed_fork",
  "                if let Some((held, ancestor_roots)) = inherited {",
  "                if let Some((held, ancestor_roots)) = inherited.filter(|_| false) {"),
 ("M8-ancestor-journal-not-gating", "unmigrated ancestor gates", "migrate_fork.rs", "b2_a_fork_of_an_unmigrated_parent",
  "            if journal_exists(plugin, o, n) {\n                plan.gate",
  "            if false && journal_exists(plugin, o, n) {\n                plan.gate"),
 ("M9-gated-fork-journal-rebuilds", "gated plan never rebuilds", "migrate.rs", "b2_a_journalled_fork_with_a_gated",
  "        if let Some(why) = plan.gate() {\n            // The fork's ancestors",
  "        if let Some(why) = plan.gate().filter(|_| false) {\n            // The fork's ancestors"),
 ("M10-rebuilt-against-never", "no re-rebuild of a risk-4 fork", "migrate_fork.rs", "b2_a_rebuilt_fork_is_not_rebuilt_again_while",
  "    if rebuilt_against(plugin, owner, name, &served, &triggers) {",
  "    if false && rebuilt_against(plugin, owner, name, &served, &triggers) {"),
]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def run(test, out):
    with open(out, "w") as f:
        return subprocess.run([CARGO, "test", "--locked", "--offline", "-p", "solid-pod-rs-forge", "--lib", test],
                              stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "RUST_BACKTRACE": "0"}).returncode
only = sys.argv[1:]
summary = []
for name, what, fname, test, old, new in MUTS:
    if only and name not in only: continue
    W = SRC + fname
    shutil.copy2(W, BAK); before = sha(W)
    src = open(W).read(); assert src.count(old) == 1, (name, src.count(old))
    open(W, "w").write(src.replace(old, new))
    red = run(test, os.path.join(B2, f"mutation-{name}.txt"))
    shutil.copy2(BAK, W); os.utime(W, None)
    ok = sha(W) == before
    green = run(test, os.path.join(B2, f"mutation-{name}-restored.txt"))
    summary.append(f"{name} ({what}): file={fname} test={test} mutated_exit={red} ({'caught' if red else 'NOT CAUGHT'}) restored_sha_match={ok} restored_exit={green}")
    print(summary[-1], flush=True)
open(os.path.join(B2, "mutation-summary.txt"), "w").write("\n".join(summary) + "\n")
sys.exit(0 if all("(caught)" in s and "match=True" in s and s.endswith("exit=0") for s in summary) else 1)
