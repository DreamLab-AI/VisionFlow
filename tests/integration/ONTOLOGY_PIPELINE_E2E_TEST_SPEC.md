# Ontology Pipeline End-to-End Integration Test Specification

## Overview

This document specifies the comprehensive end-to-end integration test for the VisionFlow ontology processing pipeline, validating data richness and transformation quality through all pipeline stages.

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ONTOLOGY PIPELINE E2E TEST FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

Input Files (8 samples)
    ├─ AI Domain (5): AI Governance, AI-0416-Differential-Privacy,
    │                 AI Agent System, AI Alignment, AI Ethics Board
    └─ Blockchain Domain (1): 51 Percent Attack

↓ STAGE 1: PARSING ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ ontology_parser.rs - Enhanced Parser                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Extract Tier 1 Properties (Required):                                     │
│   • term-id, preferred-term, definition                                     │
│   • owl:class, owl:physicality, owl:role                                    │
│   • source-domain, status, public-access, last-updated                      │
│   • is-subclass-of relationships                                            │
│                                                                             │
│ ✓ Extract Tier 2 Properties (Recommended):                                  │
│   • alt-terms, version, quality-score, cross-domain-links                   │
│   • maturity, source, authority-score                                       │
│   • belongs-to-domain, uses, has-part, enables relationships                │
│                                                                             │
│ ✓ Extract Tier 3 Properties (Optional):                                     │
│   • bridges-to/from relationships, OWL axioms                               │
│   • Domain-specific extensions, metadata                                    │
└─────────────────────────────────────────────────────────────────────────────┘
    Metrics: Properties Captured, Relationships Extracted, Parse Time

↓ STAGE 2: CONTENT ANALYSIS ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ ontology_content_analyzer.rs - Domain & Quality Detection                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Detect source domain from term-id prefix (AI-, BC-, MV-)                  │
│ ✓ Extract topics from topic:: markers                                       │
│ ✓ Count relationships and OWL class definitions                             │
│ ✓ Detect public:: true flag                                                 │
│ ✓ Validate ontology block structure                                         │
└─────────────────────────────────────────────────────────────────────────────┘
    Metrics: Domain Detection Accuracy, Topic Coverage, Block Detection Rate

↓ STAGE 3: SQLITE STORAGE ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ sqlite_ontology_repository.rs - Rich Metadata Storage                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Store all Tier 1, 2, 3 properties in structured schema                    │
│ ✓ Persist relationships with confidence scores                              │
│ ✓ Track source files with SHA1 hashes                                       │
│ ✓ Maintain markdown content for full-text search                            │
│ ✓ Record last_synced timestamps                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    Metrics: Data Richness Score, Quality/Authority Averages, Storage Time

↓ STAGE 4: NEO4J SYNC (Optional) ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ neo4j_ontology_repository.rs - Graph Database Sync                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Create OwlClass nodes with rich properties                                │
│ ✓ Create relationship edges with semantic types                             │
│ ✓ Apply domain classifications                                              │
│ ✓ Index by IRI, term-id, domain                                             │
└─────────────────────────────────────────────────────────────────────────────┘
    Metrics: Nodes Created, Edges Created, Sync Time

↓ STAGE 5: SEMANTIC FORCES ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ ontology_constraints.rs - Physics Constraint Generation                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Convert relationships to attraction/repulsion forces                      │
│ ✓ Apply hierarchy-based positioning (subclass-of)                           │
│ ✓ Set force strengths based on quality/authority scores                     │
│ ✓ Generate ConstraintSet with ConstraintKind::Semantic                      │
└─────────────────────────────────────────────────────────────────────────────┘
    Metrics: Constraints Generated, Force Strength Distribution

↓ VALIDATION & REPORTING ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Comprehensive Data Quality Validation                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Tier 1 Completeness >= 70% (critical properties)                          │
│ ✓ Tier 2 Completeness >= 50% (recommended properties)                       │
│ ✓ Tier 3 Completeness >= 30% (optional properties)                          │
│ ✓ Domain Detection Accuracy >= 60%                                          │
│ ✓ Relationship Extraction Rate >= 50%                                       │
│ ✓ Overall Data Richness >= 60%                                              │
│ ✓ No data loss between stages                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Test Data

### Selected Ontologies (8 Files)

| File | Domain | Term ID | Key Features |
|------|--------|---------|--------------|
| AI Governance.md | AI | AI-0091 | Mature, high authority (0.95), comprehensive definition |
| AI-0416-Differential-Privacy.md | AI | AI-0416 | OWL classification, cross-domain bridges, quality score 0.95 |
| AI Agent System.md | AI | AI-0600 | Rich relationships (17 types), quality score 0.92 |
| 51 Percent Attack.md | Blockchain | BC-0077 | Cross-domain links, maturity: mature, authority 0.95 |
| AI Alignment.md | AI | - | Multiple topics, UK context examples |
| AI Ethics Board.md | AI | - | Governance relationships |
| AI Model Card.md | AI | - | Technical documentation standards |
| AI Risk.md | AI | - | Risk assessment properties |

### Coverage Rationale

- **Domain Diversity**: AI (7), Blockchain (1) - validates domain detection
- **Maturity Levels**: Mature (4), Complete (1), Draft (3) - tests status handling
- **Property Richness**: High (4), Medium (3), Low (1) - validates tier extraction
- **Relationship Complexity**: Rich (3), Moderate (4), Minimal (1) - tests edge extraction

## Data Richness Metrics

### Tier 1 Properties (Weight: 3x - Critical)
```
Required Properties (8 total):
├─ term-id          :: Unique identifier (AI-XXXX, BC-XXXX)
├─ preferred-term   :: Human-readable name
├─ definition       :: Full semantic definition
├─ source-domain    :: Domain classification (ai, blockchain, etc.)
├─ status           :: Lifecycle status (draft, in-progress, complete)
├─ owl:class        :: OWL class IRI
├─ owl:physicality  :: Physical nature (VirtualEntity, PhysicalEntity)
├─ owl:role         :: Semantic role (Process, Agent, System)
└─ is-subclass-of   :: Parent class relationships

Target: >= 70% completeness across all ontologies
```

### Tier 2 Properties (Weight: 2x - Recommended)
```
Recommended Properties (6 total):
├─ version          :: Ontology version (semver)
├─ quality-score    :: Data quality metric (0.0-1.0)
├─ maturity         :: Maturity level (draft, mature, stable)
├─ authority-score  :: Source authority (0.0-1.0)
├─ belongs-to-domain:: Domain membership
└─ public-access    :: Visibility flag (true/false)

Target: >= 50% completeness across all ontologies
```

### Tier 3 Properties (Weight: 1x - Optional)
```
Optional Properties (5 total):
├─ bridges-to/from  :: Cross-domain bridges
├─ source-file      :: File path tracking
├─ file-sha1        :: Content hash
├─ markdown-content :: Full source preservation
└─ properties (map) :: Extended metadata

Target: >= 30% completeness across all ontologies
```

### Data Richness Calculation

```rust
data_richness = (tier1_completeness * 0.5) +
                (tier2_completeness * 0.3) +
                (tier3_completeness * 0.2)

// Per-ontology richness:
richness_score = (captured_tier1 * 3.0 +
                  captured_tier2 * 2.0 +
                  captured_tier3 * 1.0) /
                 (total_tier1 * 3.0 +
                  total_tier2 * 2.0 +
                  total_tier3 * 1.0)
```

## Expected Test Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║          ONTOLOGY PIPELINE END-TO-END INTEGRATION TEST                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔄 STAGE 1: Loading and Parsing Ontology Files...
  📄 Parsing: AI Governance.md
    ✓ Found OntologyBlock: AI Governance (props: 10, rels: 1)
  📄 Parsing: AI-0416-Differential-Privacy.md
    ✓ Found OntologyBlock: Differential Privacy (props: 12, rels: 4)
  📄 Parsing: AI Agent System.md
    ✓ Found OntologyBlock: AI Agent System (props: 11, rels: 17)
  📄 Parsing: 51 Percent Attack.md
    ✓ Found OntologyBlock: 51 Percent Attack (props: 9, rels: 3)
  ✓ Parsing Complete: 8 blocks, 82 properties, 28 relationships in 45ms

🔍 STAGE 2: Analyzing Content...
  📊 Analysis for AI Governance.md:
    - Has OntologyBlock: true
    - Domain: Some("Artificial Intelligence")
    - Topics: 4
    - Relationships: 1
  📊 Analysis for AI-0416-Differential-Privacy.md:
    - Has OntologyBlock: true
    - Domain: Some("AI")
    - Topics: 2
    - Relationships: 7
  ✓ Analysis Complete: 87.5% domain detection, 75.0% quality metrics in 12ms

💾 STAGE 3: Storing in SQLite...
  💽 Storing: AI Governance (richness: 82.3%)
  💽 Storing: Differential Privacy (richness: 89.1%)
  💽 Storing: AI Agent System (richness: 85.7%)
  💽 Storing: 51 Percent Attack (richness: 78.4%)
  ✓ Storage Complete: 8 classes stored in 28ms

✅ STAGE 4: Validating Data Richness...
  ✓ Validation Complete:
    - Tier 1: 84.2%
    - Tier 2: 68.5%
    - Tier 3: 42.3%
    - Relationship Extraction: 92.8%

╔══════════════════════════════════════════════════════════════════════════════╗
║                  ONTOLOGY PIPELINE E2E TEST REPORT                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL METRICS
  ├─ Total Files Processed: 8
  ├─ Total Duration: 127ms
  ├─ Overall Data Richness: 76.4%
  └─ Pipeline Status: ✓ EXCELLENT

📋 TIER COMPLETENESS
  ├─ Tier 1 (Required):    84.2% ✓
  ├─ Tier 2 (Recommended): 68.5%
  └─ Tier 3 (Optional):    42.3%

🔍 PARSING STAGE
  ├─ Duration: 45ms
  ├─ Items Processed: 8
  ├─ Properties Captured: 82
  ├─ Relationships Captured: 28
  └─ Data Richness: 85.4%

📊 ANALYSIS STAGE
  ├─ Duration: 12ms
  ├─ Items Processed: 8
  ├─ Domain Detection Accuracy: 87.5%
  ├─ Quality Metrics Coverage: 75.0%
  └─ Data Richness: 81.2%

💾 STORAGE STAGE
  ├─ Duration: 28ms
  ├─ Items Stored: 8
  ├─ Avg Quality Score: 0.93
  ├─ Avg Authority Score: 0.95
  └─ Data Richness: 83.9%

✅ VALIDATION STAGE
  ├─ Duration: 42ms
  ├─ Items Validated: 8
  ├─ Relationship Extraction Rate: 92.8%
  └─ Data Richness: 83.9%

📈 DATA FLOW ANALYSIS
  ├─ Properties: Parsing → Analysis → Storage
  │  └─ Retention: 82 → 78 → 78 (95.1% retention)
  ├─ Relationships: Parsing → Storage
  │  └─ Retention: 28 → 26 (92.8% retention)
  └─ Data Loss: 4.9%

🎯 KEY FINDINGS
  ✓ All Tier 1 properties captured: YES ✓
  ✓ Domain detection working: YES ✓
  ✓ Quality scores populated: YES ✓
  ✓ Relationships extracted: YES ✓
  ✓ OWL properties captured: YES ✓

✅ All assertions passed! Pipeline validation complete.
```

## Validation Assertions

### Critical Assertions (Must Pass)

```rust
// Tier 1 completeness is crucial - these are required properties
assert!(
    report.tier1_completeness >= 0.70,
    "Tier 1 completeness should be >= 70%, got {:.1}%",
    report.tier1_completeness * 100.0
);

// Domain detection must work for proper classification
assert!(
    report.domain_detection_accuracy >= 0.60,
    "Domain detection should be >= 60%, got {:.1}%",
    report.domain_detection_accuracy * 100.0
);

// Relationships are core to graph structure
assert!(
    report.relationship_extraction_rate >= 0.50,
    "Relationship extraction should be >= 50%, got {:.1}%",
    report.relationship_extraction_rate * 100.0
);

// Overall pipeline quality
assert!(
    report.overall_data_richness >= 0.60,
    "Overall data richness should be >= 60%, got {:.1}%",
    report.overall_data_richness * 100.0
);

// Performance requirement
assert!(
    report.total_duration_ms < 5000,
    "Pipeline should complete in < 5s, took {}ms",
    report.total_duration_ms
);
```

### Data Quality Assertions

```rust
// No duplicate term IDs
let term_ids: HashSet<_> = stored_classes.iter()
    .filter_map(|c| c.term_id.as_ref())
    .collect();
assert_eq!(
    term_ids.len(),
    stored_classes.iter().filter(|c| c.term_id.is_some()).count(),
    "term-id values must be unique"
);

// Domain prefixes match content
for owl_class in &stored_classes {
    if let Some(term_id) = &owl_class.term_id {
        if term_id.starts_with("AI-") {
            assert!(
                owl_class.source_domain.as_ref()
                    .map(|d| d.to_lowercase().contains("ai"))
                    .unwrap_or(false),
                "AI- prefix should correspond to AI domain"
            );
        }
    }
}

// Quality scores are in valid range [0.0, 1.0]
for owl_class in &stored_classes {
    if let Some(qs) = owl_class.quality_score {
        assert!(
            qs >= 0.0 && qs <= 1.0,
            "quality_score must be in [0.0, 1.0], got {}",
            qs
        );
    }
    if let Some(as_) = owl_class.authority_score {
        assert!(
            as_ >= 0.0 && as_ <= 1.0,
            "authority_score must be in [0.0, 1.0], got {}",
            as_
        );
    }
}
```

## Test Implementation Status

### ✅ Completed Components

- [x] Test file structure created: `/tests/integration/ontology_pipeline_e2e_test.rs`
- [x] Comprehensive metrics framework designed
- [x] Data richness calculation formulas defined
- [x] Sample ontology selection (8 diverse files)
- [x] Validation assertions specified
- [x] Report generation format designed
- [x] Documentation created

### ⚠️ Dependencies Required

- [ ] Fix library compilation errors (GPU feature gating)
- [ ] Ensure SQLite repository compiles without GPU features
- [ ] Add integration test to `tests/integration/mod.rs` (✅ Done)
- [ ] Verify sample ontology files exist and are accessible

### 🔄 Next Steps

1. **Fix Compilation**: Resolve GPU feature conditional compilation issues in main library
2. **Run Test**: Execute `cargo test ontology_pipeline_e2e --features ontology -- --nocapture`
3. **Validate Metrics**: Ensure all stages produce expected metrics
4. **Tune Thresholds**: Adjust assertion thresholds based on actual data quality
5. **Add Neo4j**: Include Neo4j sync validation when available
6. **Performance Baseline**: Establish performance baselines for each stage

## Benefits

### Data Quality Assurance

- **Comprehensive Coverage**: Tests all 19 tier properties across 3 levels
- **Relationship Validation**: Verifies 8+ relationship types are extracted
- **Domain Classification**: Ensures proper categorization (AI, BC, MV, etc.)
- **Quality Metrics**: Validates authority and quality score population

### Pipeline Health Monitoring

- **Stage-by-Stage Metrics**: Tracks data flow through each transformation
- **Data Loss Detection**: Identifies where properties are dropped
- **Performance Tracking**: Measures duration of each stage
- **Regression Prevention**: Catches quality degradation early

### Development Confidence

- **Refactoring Safety**: Validates that changes don't break data richness
- **Feature Validation**: Tests new property extraction features
- **Schema Migration**: Ensures data migration preserves quality
- **Documentation**: Living specification of expected behavior

## Usage

### Running the Test

```bash
# Run the complete E2E test
cargo test ontology_pipeline_e2e --features ontology -- --nocapture

# Run specific sub-tests
cargo test test_tier1_properties_comprehensive -- --nocapture
cargo test test_relationship_extraction_comprehensive -- --nocapture

# Run with timing details
cargo test ontology_pipeline_e2e --features ontology -- --nocapture --test-threads=1
```

### Interpreting Results

- **Overall Data Richness >= 70%**: Excellent quality
- **Overall Data Richness 60-70%**: Good quality
- **Overall Data Richness < 60%**: Needs improvement

- **Tier 1 Completeness >= 80%**: Meets critical requirements
- **Tier 2 Completeness >= 50%**: Good metadata coverage
- **Tier 3 Completeness >= 30%**: Adequate extended metadata

### Debugging Failures

If assertions fail, check the detailed report for:

1. **Stage-specific issues**: Which stage has low data richness?
2. **Property gaps**: Which tier properties are missing?
3. **Relationship extraction**: Are specific relationship types not captured?
4. **Domain detection**: Are prefixes correctly mapped to domains?
5. **Data loss**: Is data being dropped between stages?

## Maintenance

### Updating Test Data

When adding new ontology files:

1. Add to `select_test_ontologies()` function
2. Specify expected term-id and domain
3. Update total file count in assertions
4. Re-baseline metrics if needed

### Adjusting Thresholds

If data quality improves/degrades systematically:

1. Review metrics across multiple runs
2. Adjust assertion thresholds in test
3. Document rationale in this specification
4. Update expected output examples

### Extending Coverage

To test additional features:

1. Add new stage metrics to `StageMetrics`
2. Implement validation in appropriate stage
3. Add assertions for new properties
4. Update report generation logic

## References

- **Parser Spec**: `/home/user/VisionFlow/src/services/parsers/ontology_parser.rs`
- **Analyzer Spec**: `/home/user/VisionFlow/src/services/ontology_content_analyzer.rs`
- **SQLite Schema**: `/home/user/VisionFlow/src/adapters/sqlite_ontology_repository.rs`
- **Neo4j Sync**: `/home/user/VisionFlow/src/adapters/neo4j_ontology_repository.rs`
- **Ontology Spec**: `/home/user/VisionFlow/docs/canonical-ontology-block.md`

---

**Test Created**: 2025-11-22
**Test Version**: 1.0.0
**Maintainer**: VisionFlow Development Team
