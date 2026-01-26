# Project Architecture Rules (Non-Obvious Only)

- **Hybrid Compute**: Rust backend handles heavy lift (GPU/Ontology), Node.js agents handle orchestration/reasoning.
- **Unified Memory**: `ruvector` (PostgreSQL + pgvector) is the shared memory backbone for all agents.
- **Hexagonal**: Backend aims for Hexagonal architecture (see `docs/architecture/HEXAGONAL_ARCHITECTURE_STATUS.md`).
- **Swarm Topology**: Defaults to `hierarchical` (Queen -> Workers) to prevent drift (see `CLAUDE.md`).
