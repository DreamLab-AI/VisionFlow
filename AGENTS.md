# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Test
- **Rust Backend**: `cargo run` (dev), `cargo test` (unit). GPU/Ontology features enabled by default.
- **Frontend**: `cd client && npm install && npm run dev`.
- **Docker**: `./scripts/launch.sh up dev` (preferred over direct `docker compose`).
- **Type Gen**: `cargo run --bin generate_types` updates `client/src/types/` from Rust structs.

## Agent Capabilities

The following agent-facing capabilities are available via MCP tools and REST endpoints:

- **Ontology Discovery**: Semantic search across OWL classes using configurable similarity thresholds.
- **Enriched Note Reading**: Retrieve notes with full axioms, relationships, and metadata.
- **Cypher Query Validation**: Schema-aware query validation with Levenshtein-based hints for typos.
- **Ontology Graph Traversal**: BFS traversal with configurable depth for exploring class hierarchies.
- **Note Proposal**: Create or amend ontology notes with Whelk consistency checks.
- **Quality Scoring**: Automated completeness assessment for ontology entries.
- **GitHub PR Creation**: Automated ontology change PRs via the full GitHub REST API flow.
- **Voice Routing**: Multi-user real-time voice routing with push-to-talk, LiveKit SFU spatial audio, and Turbo-Whisper STT.

## MCP Tools

Seven ontology-focused MCP tools are defined in the MCP server:

1. `ontology_discover` - Semantic search across OWL classes
2. `ontology_read` - Enriched note reading with axioms and relationships
3. `ontology_query` - Schema-aware Cypher query validation
4. `ontology_traverse` - BFS graph traversal with configurable depth
5. `ontology_propose` - Create/amend notes with Whelk consistency checks
6. `ontology_validate` - Automated completeness and quality scoring
7. `ontology_status` - Proposal and PR lifecycle tracking

## Code Conventions
- **Rust**:
  - `actix-web` for API, `neo4rs` for graph DB.
  - `whelk-rs` (local path) for ontology reasoning.
  - `generate_types` binary MUST be run after changing API/Data structs.
  - `OntologyRepository` uses in-memory `Arc<RwLock<HashMap>>` for proposal state.
  - `OntologyQueryService` and `OntologyMutationService` are the agent-facing API layer for ontology operations.
- **TypeScript**:
  - `client/src/features/` architecture (Feature-Sliced Design inspired).
  - Use `src/types/` for generated types (do not edit manually).

## Project Specifics
- **Multi-Agent**: `multi-agent-docker/` contains independent agent definitions.
- **MCP Server**: `multi-agent-docker/mcp-infrastructure/servers/mcp-server.js` has MCP tool definitions (including the 7 ontology tools).
- **Orchestration**: `CLAUDE.md` mandates specific "Spawn and Wait" pattern for swarms.
- **Docs**: `docs/` contains architecture, `CLAUDE.md` contains agent behavior rules.
- **Ontology Tests**: `tests/ontology_agent_integration_test.rs` contains 13 integration tests for the ontology pipeline.
- **Env**: `.env` is ignored; copy from `.env.development.template` or `multi-agent-docker/.env.example`.
