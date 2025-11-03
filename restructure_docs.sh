#!/bin/bash
set -e
echo "Starting documentation restructuring..."

# Create new directories
mkdir -p docs/concepts
mkdir -p docs/tutorials
mkdir -p docs/how-to-guides
mkdir -p docs/reference

# Move files to new directories
mv docs/architecture/00-ARCHITECTURE-OVERVIEW.md docs/concepts/00_architecture-overview.md
mv docs/architecture/hexagonal-cqrs-architecture.md docs/concepts/01_hexagonal-cqrs-architecture.md
mv docs/architecture/semantic-physics-system.md docs/concepts/02_semantic-physics-system.md
mv docs/architecture/ontology-reasoning-pipeline.md docs/concepts/03_ontology-reasoning-pipeline.md
mv docs/architecture/gpu-semantic-forces.md docs/concepts/04_gpu-semantic-forces.md
mv docs/getting-started/01-installation.md docs/tutorials/00_installation.md
mv docs/getting-started/02-first-graph-and-agents.md docs/tutorials/01_first-graph-and-agents.md
mv docs/guides/developer/01-development-setup.md docs/how-to-guides/00_development-setup.md
mv docs/guides/developer/02-project-structure.md docs/how-to-guides/01_project-structure.md
mv docs/guides/developer/04-adding-features.md docs/how-to-guides/02_adding-features.md
mv docs/guides/developer/05-testing.md docs/how-to-guides/03_testing.md
mv docs/guides/deployment.md docs/how-to-guides/04_deployment.md
mv docs/guides/configuration.md docs/reference/00_configuration.md
mv docs/api/rest-api-complete.md docs/reference/01_rest-api.md
mv docs/api/03-websocket.md docs/reference/02_websocket-api.md
mv docs/architecture/04-database-schemas.md docs/reference/03_database-schemas.md

# Remove old directories
rm -rf docs/architecture
rm -rf docs/getting-started
rm -rf docs/guides
rm -rf docs/api

echo "Restructuring complete."