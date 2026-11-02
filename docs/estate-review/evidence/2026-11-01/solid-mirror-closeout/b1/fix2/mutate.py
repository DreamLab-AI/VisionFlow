#!/usr/bin/env python3
"""B1 fix round 2 mutation driver: for each mutation, back up the files,
apply the edit, run the named test(s), restore with cp, touch, and check
sha256. Output per mutation: mutation-<id>.txt; summary on stdout."""
import hashlib, os, shutil, subprocess, sys, time
W = "/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src/"
FX = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.join(os.path.dirname(FX), "cargo.sh")
SP = "/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/770d51cb-532b-46ec-b569-b80bea93cc51/scratchpad/mut-backup/"
os.makedirs(SP, exist_ok=True)
M = [
 ("N1-published-opens-gate", "item 1", [("migrate.rs", "    !matches!(\n        tokio::task::spawn_blocking(move || read_journal(&path)).await,\n        Ok(JournalRead::Absent)\n    )", "    !matches!(\n        tokio::task::spawn_blocking(move || read_journal(&path)).await,\n        Ok(JournalRead::Absent) | Ok(JournalRead::Present(Journal { state: Stage::Published, .. }))\n    )")], ["g1_a_published"]),
 ("N2-incomplete-skips-marked", "item 2", [("lib.rs", "        if self.migration_incomplete && mirror_record::is_hex64(owner) {\n            return true;\n        }", "        let marked_early = tokio::fs::metadata(dir.join(PUBLICATION_MARKER)).await.is_ok();\n        if self.migration_incomplete && !marked_early && mirror_record::is_hex64(owner) {\n            return true;\n        }")], ["g2_an_incomplete"]),
 ("N3-enumeration-errors-silent", "item 3", [("migrate.rs", "    ungated == 0 && journals_listed && stores_listed\n", "    let _ = (journals_listed, stores_listed);\n    ungated == 0\n")], ["g3_an_unlistable"]),
 ("N4-probe-error-is-absent", "item 4", [("migrate.rs", "        Ok(false) => return Detected::Skip,\n        Err(e) => return Detected::Error(e),\n    }", "        Ok(false) | Err(_) => return Detected::Clean,\n    }")], ["g4_a_failing"]),
 ("N5-grafts-honoured", "item 5", [("migrate.rs", '    ("GIT_GRAFT_FILE", "/dev/null"),\n', '    ("GIT_GRAFT_FILE_UNUSED", "/dev/null"),\n')], ["g5_grafts"]),
 ("N6-replace-refs-copied", "item 6", [("migrate.rs", '            Some(r) if r.starts_with("refs/replace/") => Some("outside-roots"),\n', '')], ["g6_replace"]),
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
