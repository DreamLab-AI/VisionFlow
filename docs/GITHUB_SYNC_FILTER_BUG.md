# GitHub Sync Filter Bug Analysis

**Date**: 2025-10-26
**Status**: ROOT CAUSE IDENTIFIED
**Severity**: CRITICAL

## Problem Statement

The knowledge graph contains 529 nodes, but **495 nodes (93.6%) don't have "public" in their metadata**. This violates the design requirement that only files with `public:: true` should be included in the knowledge graph.

## Investigation Results

### API Query Results
```
Total nodes: 529
Nodes WITHOUT public metadata: 495 (93.6%)
Nodes WITH public metadata: 34 (6.4%)
```

### Sample Nodes Without Public Metadata
- `king1966fisher` (id: 2636) - type: "linked_page"
- `Variational Autoencoders` (id: 4283)
- `costigan2018world` (id: 9177)
- `json` (id: 10245)
- `Diagrams as Code` (id: 12252)

## Root Cause Analysis

### Issue #1: Parser Intentionally Removes "public" Property

**File**: `src/services/parsers/knowledge_graph_parser.rs:181-184`

```rust
fn extract_metadata_store(&self, content: &str) -> MetadataStore {
    // ... property extraction ...
    for cap in prop_pattern.captures_iter(content) {
        if let (Some(key), Some(value)) = (cap.get(1), cap.get(2)) {
            let key_str = key.as_str().to_string();
            let value_str = value.as_str().trim().to_string();

            // ❌ BUG: Skip the "public" property as it's just a marker
            if key_str != "public" {
                properties.insert(key_str, value_str);
            }
        }
    }
}
```

**Impact**: Even though files are correctly filtered for "public:: true", the property is not stored in node metadata, making it impossible to verify later which nodes came from public files.

### Issue #2: Linked Pages Bypass Public Filter

**File**: `src/services/parsers/knowledge_graph_parser.rs:137-150`

```rust
fn extract_links(&self, content: &str, source_id: &u32) -> (Vec<Node>, Vec<Edge>) {
    // ... link extraction ...
    for cap in link_pattern.captures_iter(content) {
        if let Some(link_match) = cap.get(1) {
            let target_page = link_match.as_str().trim().to_string();
            let target_id = self.page_name_to_id(&target_page);

            // ❌ BUG: Create node WITHOUT checking if target page has "public:: true"
            let mut metadata = HashMap::new();
            metadata.insert("type".to_string(), "linked_page".to_string());

            nodes.push(Node {
                id: target_id,
                metadata_id: target_page.clone(),
                label: target_page.clone(),
                // ... only has type: "linked_page" metadata
            });
        }
    }
}
```

**Impact**: When a public page contains `[[WikiLink]]`, a node is created for "WikiLink" WITHOUT verifying that "WikiLink.md" has "public:: true". This allows private pages to enter the graph as "linked_page" nodes.

## Data Flow Analysis

```
1. GitHub Sync fetches all .md files
2. detect_file_type() checks for "public:: true" ✅ WORKING
   - Returns FileType::KnowledgeGraph if found
   - Returns FileType::Skip if not found
3. process_knowledge_graph_file() calls kg_parser.parse() ✅ WORKING
4. Parser creates TWO types of nodes:
   a) Main page node (from file itself) ✅ Filtered correctly
   b) Linked page nodes (from [[links]] in content) ❌ NOT FILTERED
5. extract_metadata_store() removes "public" property ❌ LOST
6. All nodes saved to database WITHOUT public indicator ❌ NO WAY TO VERIFY
```

## Examples

### Example 1: Legitimate Public Page
```markdown
File: "Deep Learning.md"
Content:
  public:: true
  tags:: AI, ML

  # Deep Learning
  See also [[Neural Networks]]
```

**Result**:
- Node 1: "Deep Learning" (type: "page") ✅ Correct
- Node 2: "Neural Networks" (type: "linked_page") ❌ Added without checking if Neural Networks.md has public:: true

### Example 2: Private Page Referenced
```markdown
File: "Research Notes.md" (public:: true)
Content:
  public:: true

  # Research Notes
  Draft ideas in [[Private Thoughts]]
```

```markdown
File: "Private Thoughts.md" (NO public:: true)
Content:
  # Private Thoughts
  Confidential research ideas...
```

**Result**:
- Node 1: "Research Notes" (type: "page") ✅ Correct
- Node 2: "Private Thoughts" (type: "linked_page") ❌ PRIVACY VIOLATION - private page added to public graph!

## Impact Assessment

### Security Impact: HIGH
- Private pages can leak into public knowledge graph through [[links]]
- No way to audit which nodes came from public vs private files
- Metadata loss makes it impossible to verify data provenance

### Data Integrity Impact: HIGH
- 93.6% of nodes lack public indicator
- Graph contains mix of verified and unverified nodes
- Cannot distinguish legitimate nodes from linked references

### User Experience Impact: MEDIUM
- Users see pages in graph that don't have public:: true
- Confusion about what data is included
- Graph may contain incomplete/broken references

## Proposed Solutions

### Solution A: Conservative (Recommended)
**Only include nodes that have corresponding public files**

1. **Preserve "public" metadata**:
   - Remove line 182 condition: `if key_str != "public"`
   - Store all properties including "public:: true"

2. **Filter linked pages**:
   - Maintain a HashSet of all page names that passed public filter
   - In `extract_links()`, only create linked_page nodes for pages in the set
   - OR mark them as "reference_only" with different color/size

**Pros**:
- Secure - no private page leakage
- Clean separation of public vs private
- Easy to audit

**Cons**:
- Might miss some legitimate relationships
- Requires tracking public page set

### Solution B: Hybrid (Flexible)
**Distinguish between verified and referenced nodes**

1. **Preserve "public" metadata** (same as Solution A)

2. **Create two node types**:
   - `type: "page"` - Files with public:: true ✅
   - `type: "reference"` - Mentioned in [[links]] but not verified ⚠️

3. **Visual distinction**:
   - Public pages: Solid cyan (#4A90E2)
   - References: Translucent purple (#7C3AED) with dotted border

4. **Add metadata**:
   ```rust
   metadata.insert("verified_public".to_string(), "false".to_string());
   ```

**Pros**:
- Preserves all relationships
- Clear visual/data distinction
- Users can filter by verified status

**Cons**:
- More complex logic
- Might still expose private page names

### Solution C: Minimal (Quick Fix)
**Just preserve the metadata**

1. Remove line 182: `if key_str != "public"`
2. Keep current linked page logic
3. Add post-processing to flag nodes without public:: true

**Pros**:
- Minimal code change
- Backwards compatible

**Cons**:
- Doesn't fix the core filtering issue
- Still leaks private page references

## Recommended Implementation: Solution A

### Phase 1: Preserve Public Metadata
**File**: `src/services/parsers/knowledge_graph_parser.rs:168-195`

```rust
fn extract_metadata_store(&self, content: &str) -> MetadataStore {
    let mut store = MetadataStore::new();
    let prop_pattern = regex::Regex::new(r"([a-zA-Z_]+)::\s*(.+)").unwrap();

    let mut properties = HashMap::new();
    for cap in prop_pattern.captures_iter(content) {
        if let (Some(key), Some(value)) = (cap.get(1), cap.get(2)) {
            let key_str = key.as_str().to_string();
            let value_str = value.as_str().trim().to_string();

            // ✅ FIX: Store ALL properties including "public"
            properties.insert(key_str, value_str);
        }
    }

    store.insert("properties".to_string(), serde_json::to_value(&properties).unwrap_or_default());
    store
}
```

**Also update main page node creation to include public property**:
```rust
fn create_page_node(&self, page_name: &str, content: &str) -> Node {
    let mut metadata = HashMap::new();
    metadata.insert("type".to_string(), "page".to_string());
    metadata.insert("source_file".to_string(), format!("{}.md", page_name));
    metadata.insert("public".to_string(), "true".to_string()); // ✅ ADD THIS

    // ... rest of node creation
}
```

### Phase 2: Filter Linked Pages
**File**: `src/services/github_sync_service.rs:75-88`

```rust
pub async fn sync_graphs(&self) -> Result<SyncStatistics, String> {
    // ... existing setup ...

    let mut accumulated_nodes: HashMap<u32, Node> = HashMap::new();
    let mut accumulated_edges: HashMap<String, Edge> = HashMap::new();

    // ✅ NEW: Track which pages have public:: true
    let mut public_page_names: HashSet<String> = HashSet::new();

    // ... rest of sync logic
}
```

**File**: `src/services/github_sync_service.rs:250-285`

```rust
async fn process_knowledge_graph_file(
    &self,
    file: &GitHubFileBasicMetadata,
    content: &str,
    accumulated_nodes: &mut HashMap<u32, Node>,
    accumulated_edges: &mut HashMap<String, Edge>,
    public_page_names: &mut HashSet<String>, // ✅ NEW PARAMETER
) -> FileProcessResult {
    // Add this page to public set
    let page_name = file.name.strip_suffix(".md").unwrap_or(&file.name);
    public_page_names.insert(page_name.to_string());

    let graph_data = match self.kg_parser.parse(content, &file.name) {
        Ok(data) => data,
        Err(e) => return FileProcessResult::Error { error: format!("Parse error: {}", e) },
    };

    // ✅ NEW: Filter nodes to only include public pages
    for node in graph_data.nodes {
        // Only add if node is main page OR linked page is also public
        let should_add = match node.metadata.get("type").map(|s| s.as_str()) {
            Some("page") => true, // Main page nodes always included
            Some("linked_page") => public_page_names.contains(&node.metadata_id), // ✅ CHECK
            _ => true,
        };

        if should_add {
            accumulated_nodes.insert(node.id, node);
        }
    }

    // ... rest of processing
}
```

### Phase 3: Database Query Validation

Add API endpoint to check public status:

```rust
// GET /api/graph/data?public_only=true
async fn get_graph_data(query: Query<GraphQueryParams>) -> impl Responder {
    let graph = load_graph().await?;

    if query.public_only.unwrap_or(true) {
        // Filter to only nodes with public:: true
        let filtered_nodes: Vec<Node> = graph.nodes.into_iter()
            .filter(|n| n.metadata.get("public") == Some(&"true".to_string()))
            .collect();

        // ... return filtered graph
    }
}
```

## Testing Plan

### Unit Tests

1. **Test public metadata preservation**:
   ```rust
   #[test]
   fn test_public_metadata_preserved() {
       let content = "public:: true\ntags:: test\n# Test";
       let parser = KnowledgeGraphParser::new();
       let graph = parser.parse(content, "test.md").unwrap();

       let metadata = graph.metadata.get("properties").unwrap();
       assert!(metadata.contains_key("public"));
       assert_eq!(metadata.get("public"), Some(&"true".to_string()));
   }
   ```

2. **Test linked page filtering**:
   ```rust
   #[test]
   fn test_linked_pages_filtered() {
       // Test that linked pages without public:: true are not included
   }
   ```

### Integration Tests

1. **Sync with mixed public/private files**
2. **Verify node count matches public files**
3. **Check all nodes have public metadata**

## Migration Path

1. **Deploy fix to staging**
2. **Clear knowledge_graph.db**
3. **Run full sync**
4. **Verify all nodes have public:: true in metadata**
5. **Check node count == expected public files**
6. **Deploy to production**

## Success Criteria

- ✅ 100% of nodes have "public" in metadata
- ✅ No private pages in knowledge graph
- ✅ Node count matches public file count ± linked duplicates
- ✅ API can filter by public status
- ✅ Visual distinction between verified and unverified nodes (if Solution B)

## Timeline

- **Phase 1** (Preserve metadata): 30 minutes
- **Phase 2** (Filter linked pages): 1-2 hours
- **Phase 3** (API validation): 30 minutes
- **Testing**: 1 hour
- **Deployment**: 30 minutes

**Total**: ~4 hours
