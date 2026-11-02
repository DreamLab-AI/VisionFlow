#!/usr/bin/env python3
"""B1 fix round 1 mutation driver: for each mutation, back up the files,
apply the edit, run the named test(s), restore with cp, touch, and check
sha256. Output per mutation: mutation-<id>.txt; summary on stdout."""
import hashlib, os, shutil, subprocess, sys, time
W = "/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src/"
FX = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.join(os.path.dirname(FX), "cargo.sh")
SP = "/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/770d51cb-532b-46ec-b569-b80bea93cc51/scratchpad/mut-backup/"
os.makedirs(SP, exist_ok=True)
M = [
 ("M1a-held-set-mirror-only", "item 1", [("migrate.rs", '            "--all",\n            "--stdin",', '            "--glob=refs/mirror",\n            "--stdin",')], ["f1_lost_record", "f1_a_dropped_tag", "f1_force_pushed"]),
 ("M1b-verify-outside-roots-off", "item 1", [("migrate.rs", "    if outside > 0 {", "    if outside > usize::MAX - 1 {")], ["t4_direct"]),
 ("M2a-detection-error-is-none", "item 2", [("migrate.rs", "    error.map_or(Detected::None, Detected::Error)", "    error.map_or(Detected::None, |_| Detected::None)")], ["f2_a_detection_error"]),
 ("M2b-incomplete-flag-off", "item 2", [("lib.rs", "if self.migration_incomplete && !marked", "if false && self.migration_incomplete && !marked")], ["f2_a_panicked"]),
 ("M3a-bad-journal-opens-gate", "item 3", [("migrate.rs", "        Ok(JournalRead::Bad(_)) | Err(_) => true,", "        Ok(JournalRead::Bad(_)) | Err(_) => false,")], ["f3_a_symlinked"]),
 ("M3b-containment-off", "item 3", [("migrate.rs", "    let recovery = match contained(plugin, owner, name, &journal) {", "    let recovery = match Ok::<PathBuf, String>(plugin.join(&journal.recovery)) {")], ["f3_a_journal_recovery"]),
 ("M4-nested-path-to-cgi", "item 4", [("lib.rs", "    let (owner, name) = head.split_once('/')?;", "    let (owner, rest) = head.split_once('/')?;\n    let name = rest.split('/').next()?;"), ("lib.rs", '            path: format!("/{owner}/{name}.git{suffix}"),', '            path: rel_path.to_string(),')], ["f4_smart_http"]),
 ("M5a-served-is-ours-off", "item 5", [("migrate.rs", "    if journal.state == Stage::Published || served_is_ours(&p.served, &journal.id) {", "    if journal.state == Stage::Published {")], ["f5_a_crash_after"]),
 ("M5b-finish-unbound", "item 5", [("migrate.rs", "            Ok(found) if found.iter().any(|r| report_id(r).as_deref() == Some(&id)) => {", "            Ok(found) if !found.is_empty() || !id.is_empty() => {")], ["f5_a_stale_report"]),
 ("M6-attempt-not-counted-at-start", "item 6", [("migrate.rs", "    j.attempts = j.attempts.saturating_add(1);\n    j.state = Stage::Building;", "    j.state = Stage::Building;")], ["f6_three_crashes"]),
 ("M7-lock-wait-unbounded", "item 7", [("migrate.rs", "                    if std::time::Instant::now() >= deadline {", "                    if std::time::Instant::now() >= deadline + Duration::from_secs(86_400) {")], ["f7_a_held_lock"]),
 ("M9-groups-not-killed", "item 9", [("migrate.rs", "            for pid in left {\n                kill_group(Some(pid)).await;\n            }", "            let _ = left;")], ["f9_a_budget"]),
 ("M1c-replace-objects-on", "item 1", [("migrate.rs", '    cmd.env("GIT_NO_REPLACE_OBJECTS", "1");\n    cmd\n', '    cmd\n')], ["f1_a_replace"]),
 ("M3c-published-unbound", "item 3", [("migrate.rs", "    if served_is_ours(&p.served, &journal.id) {", "    if journal.state == Stage::Published || served_is_ours(&p.served, &journal.id) {")], ["f3_a_forged"]),
 ("M11-is-dir-no-follow", "item 1/2", [("migrate.rs", "    std::fs::metadata(p).is_ok_and(|m| m.is_dir())", "    std::fs::symlink_metadata(p).is_ok_and(|m| m.is_dir())")], ["f2_a_symlinked"]),
 ("M10-push-uncapped", "item 10", [("migrate.rs", "    let cap: Option<(u64, DirMeasure)> = Some((size_cap, crate::sizecap::dir_size));", "    let _ = size_cap;\n    let cap: Option<(u64, DirMeasure)> = None;")], ["f10_a_staging"]),
]
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def run(tests, out):
    cmd = [CARGO, "test", "--locked", "--offline", "-p", "solid-pod-rs-forge", "--features", "mirror", "--lib", "--"] + tests
    env = dict(os.environ, RUST_BACKTRACE="0")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=900)
    open(out, "wb").write(r.stdout)
    return r.returncode, r.stdout.decode("utf-8", "replace")
only = sys.argv[1:]
for mid, item, edits, tests in M:
    if only and mid not in only: continue
    files = sorted({f for f, _, _ in edits})
    before = {f: sha(W+f) for f in files}
    for f in files: shutil.copy2(W+f, SP+f)
    try:
        for f, old, new in edits:
            s = open(W+f).read()
            assert s.count(old) == 1, (mid, f, s.count(old), old[:60])
            open(W+f, "w").write(s.replace(old, new))
        code, text = run(tests, os.path.join(FX, f"mutation-{mid}.txt"))
    finally:
        for f in files:
            shutil.copy(SP+f, W+f)
            os.utime(W+f, None)
    after = {f: sha(W+f) for f in files}
    restored = all(before[f] == after[f] for f in files)
    fails = [l for l in text.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
    print(f"{mid} ({item}): mutated exit={code}; failed={len(fails)} {[l.split()[1] for l in fails]}; restored sha ok={restored}")
