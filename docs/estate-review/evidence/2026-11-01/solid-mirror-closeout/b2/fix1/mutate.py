#!/usr/bin/env python3
"""B2 fix round 1 mutations: copy, mutate, run one test, restore (cp, touch, sha256 check), re-run green."""
import hashlib, os, shutil, subprocess, sys
SRC = "/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src/"
B2 = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.join(os.path.dirname(os.path.dirname(B2)), "b1", "cargo.sh")
BAK = "/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/770d51cb-532b-46ec-b569-b80bea93cc51/scratchpad/b2f1-mutate.bak"
MUTS = [
 ("M1a-unparseable-is-not-a-fork", "item 1: lineage read (pre-fix semantics)", "forks.rs", "b2f1_a_fork_of_a_dot_git_named_mirror",
  "parse_lineage_value(&v).map_or(StoredLineage::Unknown(v), StoredLineage::Parent)",
  "parse_fork_parent(conf).map_or(StoredLineage::None, StoredLineage::Parent)"),
 ("M1b-unknown-is-a-root", "item 1: unknown lineage swept", "migrate_fork.rs", "b2f1_a_fork_of_unknown_lineage_is_checked",
  "Ok(StoredLineage::Unknown(raw)) => Lineage::Unknown(raw),",
  "Ok(StoredLineage::Unknown(_)) => Lineage::Root,"),
 ("M1c-unknown-allows-stamping", "item 1: unknown lineage never allows stamping", "migrate_fork.rs", "b2f1_a_fork_of_unknown_lineage_is_checked",
  "    if plan.gate.is_some() || plan.unknown {",
  "    if plan.gate.is_some() {"),
 ("M1d-incomplete-rule-ignores-unknown", "item 1: incomplete-pass rule", "lib.rs", "b2f1_a_fork_of_unknown_lineage_is_gated",
  "Ok(forks::StoredLineage::None)",
  "Ok(forks::StoredLineage::None | forks::StoredLineage::Unknown(_))"),
 ("M1e-api-fork-writes-any-lineage", "item 1: h_api_fork refusal", "forks.rs", "b2f1_a_fork_whose_lineage_the_parser_rejects",
  "        if !(valid_owner(owner) && valid_repo(name)) {",
  "        if false && !(valid_owner(owner) && valid_repo(name)) {"),
 ("M2-unreachable-check-off", "item 2: unreachable-object trigger", "migrate_fork.rs", "b2f1_a_fork_keeping_an_object_pruned",
  "Ok(false) => match holds_unreachable(&served).await {",
  "Ok(false) => match Ok::<bool, String>(false) {"),
 ("M2b-walk-counts-dropped-refs", "item 2: reachability only through kept refs", "migrate_fork.rs", "b2f1_a_pruned_object_reached_only_by_dropped",
  "            \"--exclude=refs/mirror/*\",\n            \"--exclude=refs/replace/*\",\n",
  ""),
 ("M3-gate-blocks-collected-only", "item 3: a gate blocks every report", "migrate_fork.rs", "a_gate_partway_up_the_lineage",
  "    if plan.gate.is_some() || plan.unknown {",
  "    if plan.unknown {"),
 ("M4-ancestor-journal-hex-only", "item 4: journalled ancestor under any owner", "migrate_fork.rs", "b2f1_a_journalled_ancestor_fork_under_any_owner",
  "        if journal_exists(plugin, o, n) {\n            plan.gate",
  "        if is_hex64(o) && journal_exists(plugin, o, n) {\n            plan.gate"),
 ("M5-gate-after-resume", "item 5: gated-plan check before the resume path", "migrate.rs", "b2f1_a_crash_resumed_fork_with_a_gated_ancestor",
  "        if let Some(why) = plan.gate() {\n            // The fork's ancestors are not all known",
  "        if let Some(why) = plan.gate().filter(|_| false) {\n            // The fork's ancestors are not all known"),
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
open(os.path.join(B2, "mutation-summary" + ("-" + "-".join(only) if only else "") + ".txt"), "w").write("\n".join(summary) + "\n")
sys.exit(0 if all("(caught)" in s and "match=True" in s and s.endswith("exit=0") for s in summary) else 1)
