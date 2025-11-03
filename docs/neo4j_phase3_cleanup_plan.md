# Neo4j Phase 3 Cleanup - Remaining Issues

## Files Still Referencing rusqlite

These files need to be updated to remove rusqlite dependencies:

1. `/home/devuser/workspace/project/src/reasoning/horned_integration.rs`
2. `/home/devuser/workspace/project/src/reasoning/inference_cache.rs`
3. `/home/devuser/workspace/project/src/repositories/query_builder.rs`
4. `/home/devuser/workspace/project/src/repositories/unified_ontology_repository.rs`
5. `/home/devuser/workspace/project/src/services/github_sync_service.rs`
6. `/home/devuser/workspace/project/src/utils/result_mappers.rs`

## Strategy

These files appear to be:
- **reasoning/** - May need Neo4j-based inference caching
- **repositories/** - Should use Neo4j query builder
- **services/** - GitHub sync may need refactoring
- **utils/** - Result mappers need generic error types

## Recommendation

Phase 3 cleanup reveals that the migration is incomplete. These files represent core functionality that still depends on SQLite.

**Options:**
1. Continue Phase 3 to migrate these files to Neo4j
2. Keep rusqlite as a dependency temporarily for these specific modules
3. Refactor to use Neo4j-based alternatives

The safest approach is to keep rusqlite for now and plan Phase 4 to migrate these remaining components.
