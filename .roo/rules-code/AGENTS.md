# Project Coding Rules (Non-Obvious Only)

- **Type Synchronization**: Run `cargo run --bin generate_types` after modifying Rust structs exposed to frontend.
- **Local Deps**: `whelk` dependency is a local path (`./whelk-rs`), do not change to crates.io version without verifying.
- **GPU Features**: `gpu` and `ontology` features are default in `Cargo.toml`; respect `cfg(feature = "gpu")` guards.
- **Frontend Architecture**: `client/src/features/` architecture (Feature-Sliced Design inspired).
- **Docker Mounts**: `multi-agent-docker/.env` controls volume mounts; changes require `rebuild-agent` (not just restart).
