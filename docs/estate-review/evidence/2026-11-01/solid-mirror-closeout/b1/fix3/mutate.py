#!/usr/bin/env python3
"""B1 fix round 3 mutations: copy, mutate, run one test, restore (cp, touch, sha256 check), re-run green."""
import hashlib, os, shutil, subprocess, sys
W = "/home/devuser/workspace/wt/solid-pod-rs-mirror-publication/crates/solid-pod-rs-forge/src/migrate.rs"
FX = os.path.dirname(os.path.abspath(__file__))
CARGO = os.path.join(os.path.dirname(FX), "cargo.sh")
BAK = "/home/devuser/workspace/.tmp/claude-1000/-home-devuser-workspace-VisionFlow/770d51cb-532b-46ec-b569-b80bea93cc51/scratchpad/migrate.fix3.bak"
MUTS = [
 ("M1-no-early-persist", "h1_a_recovery_probe_error_still_gates_the_store",
  "                if let Err(e) = write_journal(&p.journal, &j) {\n                    tracing::error!(owner = %owner, name = %name, error = %redact(plugin, &e), \"forge: migration journal not written; store not migrated\");\n                    return Outcome::Ungated;\n                }\n                j\n",
  "                j\n"),
 ("M2-no-final-size-check", "h2_a_fast_push_past_the_cap_is_caught_by_the_final_check",
  "        if size > size_cap {\n            return Err(GREW_PAST_CAP.to_string());\n        }\n",
  "        let _ = size;\n"),
]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def run(test, out):
    with open(out, "w") as f:
        r = subprocess.run([CARGO, "test", "--locked", "--offline", "-p", "solid-pod-rs-forge", "--lib", test],
                           stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "RUST_BACKTRACE": "0"})
    return r.returncode
summary = []
for name, test, old, new in MUTS:
    shutil.copy2(W, BAK); before = sha(W)
    src = open(W).read(); assert src.count(old) == 1, name
    open(W, "w").write(src.replace(old, new))
    red = run(test, os.path.join(FX, f"mutation-{name}.txt"))
    shutil.copy2(BAK, W); os.utime(W, None)
    ok = sha(W) == before
    green = run(test, os.path.join(FX, f"mutation-{name}-restored.txt"))
    summary.append(f"{name}: test={test} mutated_exit={red} ({'caught' if red else 'NOT CAUGHT'}) restored_sha_match={ok} restored_exit={green}")
open(os.path.join(FX, "mutation-summary.txt"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary)); sys.exit(0 if all("caught" in s and "match=True" in s and s.endswith("exit=0") for s in summary) else 1)
