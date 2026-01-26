# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Test
- **Rust Backend**: `cargo run` (dev), `cargo test` (unit). GPU/Ontology features enabled by default.
- **Frontend**: `cd client && npm install && npm run dev`.
- **Docker**: `./scripts/launch.sh up dev` (preferred over direct `docker compose`).
- **Type Gen**: `cargo run --bin generate_types` updates `client/src/types/` from Rust structs.

## Code Conventions
- **Rust**:
  - `actix-web` for API, `neo4rs` for DB.
  - `whelk-rs` (local path) for ontology reasoning.
  - `generate_types` binary MUST be run after changing API/Data structs.
- **TypeScript**:
  - `client/src/features/` architecture (Feature-Sliced Design inspired).
  - Use `src/types/` for generated types (do not edit manually).

## Project Specifics
- **Multi-Agent**: `multi-agent-docker/` contains independent agent definitions.
- **Orchestration**: `CLAUDE.md` mandates specific "Spawn and Wait" pattern for swarms.
- **Docs**: `docs/` contains architecture, `CLAUDE.md` contains agent behavior rules.
- **Env**: `.env` is ignored; copy from `.env.development.template` or `multi-agent-docker/.env.example`.
