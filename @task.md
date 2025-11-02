# Critical Issue: Only 40 Nodes Loading Instead of 990 Ontology Nodes

## Problem Statement
The visionflow container is rendering only 40 nodes when the system should display 900+ ontology nodes from the mainKnowledgeGraph/pages directory.

**User Feedback:** "40 nodes is very wrong. the ontology is over 900 nodes"

## Root Cause Analysis

### GitHub Repository Structure
- **Repository:** jjohare/logseq
- **Source Path:** mainKnowledgeGraph/pages
- **Total Markdown Files:** 990 files available
- **Currently Loading:** 40 rb-* (robotics) files

### Current Architecture Issue
The system uses KnowledgeGraphParser which only extracts nodes from files containing `public:: true` metadata:
- Only 40 rb-* files have `public:: true` metadata
- The remaining 950 markdown files are ignored
- Database schema shows: 40 graph_nodes, 1 owl_class (not 900+)

### Why This Breaks "Fully Migrated to Ontology-Based Nodes"
The GitHub sync service (github_sync_service.rs) processes files through KnowledgeGraphParser first, and only creates ontology classes as a fallback when the parser fails. Since rb-* files parse successfully as KG nodes, the fallback condition (`kg_nodes_added == 0`) never triggers.

**Key Code Location:** src/services/github_sync_service.rs, lines 279-321

## Required Solution

### Architecture Change Needed
Instead of:
```
For each markdown file:
  1. Try KnowledgeGraphParser (requires `public:: true` metadata)
  2. If parser fails, create ontology class
```

Implement:
```
For each markdown file:
  1. Create OWL ontology class (mandatory for all files)
  2. Optionally parse as KG nodes if `public:: true` metadata exists
```

### Implementation Steps
1. **Modify GitHub sync pipeline** to create ontology classes for ALL 990 markdown files regardless of metadata
2. **Extract markdown content** including title, headers, and description for class IRI and labels
3. **Build ontology hierarchy** from markdown structure (headers → parent classes)
4. **Test full sync** verifying 990+ nodes load into database
5. **Verify WebSocket** InitialGraphLoad sends all nodes with metadata
6. **Test GPU physics** stress majorization with 900+ nodes

### Database Requirements
- graph_nodes table: 40 rb-* files (existing KG nodes)
- owl_classes table: 990 total markdown files (all as ontology classes)
- **Total items:** 1030 (40 + 990)

## Investigation Files Created
- `/tmp/check_schema.py` - Database schema inspection
- `/tmp/check_node_source.py` - Node metadata inspection
- `/tmp/check_github.py` - GitHub API verification (confirms 990 files exist)

## Status
- ✅ Identified root cause: KG parser filter limiting to 40 `public:: true` files
- ✅ Located critical code: github_sync_service.rs lines 279-321
- ✅ Verified GitHub API returns 990 available files
- ❌ **IN PROGRESS:** Redesign GitHub sync to load all 990 as ontology classes

## Next Critical Actions
1. Update github_sync_service.rs to create ontology classes for ALL markdown files
2. Remove dependency on `public:: true` metadata for ontology loading
3. Build, test, and verify 900+ nodes sync to database
4. Monitor WebSocket traffic to confirm all nodes sent to client
5. Verify GPU physics can handle 900+ node rendering
