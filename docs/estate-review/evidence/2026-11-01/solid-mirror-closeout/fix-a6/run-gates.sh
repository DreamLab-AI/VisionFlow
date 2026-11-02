#!/bin/bash
E=$1; cd /home/devuser/workspace/wt/solid-pod-rs-mirror-publication
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target
out=$E/fix-a6-gates.txt
echo "# Commit A fix round 6 gates, $(date -u +%FT%TZ), CARGO_TARGET_DIR=$CARGO_TARGET_DIR" >> $out
echo "# HEAD $(git rev-parse --short HEAD) + uncommitted diff; $(rustc --version); $(git --version)" >> $out
i=0
run() { i=$((i+1)); echo "\$ $*" >> $out; "$@" > $E/gate-logs/gate$i.log 2>&1; rc=$?; echo "exit=$rc  (log: gate-logs/gate$i.log)" >> $out; grep -E "test result|^error" $E/gate-logs/gate$i.log | sed 's/^/    /' | head -12 >> $out; echo >> $out; }
run cargo fmt --all --check
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror
run cargo test --locked --offline -p solid-pod-rs-forge
run cargo test --locked --offline -p solid-pod-rs-server --features forge-full forge
run cargo clippy --locked --offline -p solid-pod-rs-forge --features mirror --all-targets -- -D warnings
run cargo clippy --locked --offline -p solid-pod-rs-server --features forge-full --all-targets -- -D warnings
run env RUSTDOCFLAGS="-D warnings" cargo doc --locked --offline --no-deps -p solid-pod-rs-forge --features mirror
run cargo test --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce
run cargo clippy --locked --offline -p solid-pod-rs-forge --features mirror,anchoring,announce --all-targets -- -D warnings
echo "# done" >> $out
