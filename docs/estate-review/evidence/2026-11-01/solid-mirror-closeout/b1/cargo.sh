#!/usr/bin/env bash
# B1 cargo wrapper: the brief's environment, --locked --offline added by callers.
export CARGO_HOME=/home/devuser/workspace/.cargo
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/solid-mirror-target
export CARGO_BUILD_JOBS=4
cd /home/devuser/workspace/wt/solid-pod-rs-mirror-publication || exit 2
exec cargo "$@"
