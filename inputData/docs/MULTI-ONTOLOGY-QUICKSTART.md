# Multi-Ontology Quick Start Guide

## Quick Reference: 6 Domains

| Domain | Namespace | Prefix | Key Properties |
|--------|-----------|--------|----------------|
| **Artificial Intelligence** | `ai:` | `AI-` | algorithm-type, computational-complexity |
| **Metaverse** | `mv:` | `MV-` | immersion-level, interaction-mode |
| **Telecollaboration** | `tc:` | `TC-` | collaboration-type, communication-mode |
| **Robotics** | `rb:` | `RB-` | physicality, autonomy-level |
| **Disruptive Technologies** | `dt:` | `DT-` | disruption-level, maturity-stage |
| **Blockchain** | `bc:` | `BC-` | consensus-mechanism, decentralization-level |

## Common Commands

```bash
# List all 6 domains with current statistics
node cli.js domains

# Show detailed stats for a specific domain
node cli.js domain-stats ai
node cli.js domain-stats tc

# Validate a specific domain
node cli.js validate-domain rb

# Process a specific domain (dry-run)
node cli.js process-domain bc

# Process a specific domain (live)
node cli.js process-domain ai --live

# Analyze cross-domain links
node cli.js cross-domain-links

# Detect domain for a file
node cli.js detect-domain path/to/file.md

# Run test suite
node test-multi-ontology.js
```

## Quick Domain Detection

The system automatically detects domains from:

1. **Filename prefix:** `AI-001-*.md` → ai
2. **Term ID:** `TC-005` → tc
3. **Namespace:** `rb:ClassName` → rb
4. **Content keywords:** Analyzes text for domain-specific terms

## Example: Creating a Telecollaboration Term

```markdown
- ### OntologyBlock
  id:: remote-collaboration-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: TC-001
    - preferred-term:: Remote Collaboration
    - source-domain:: tc
    - status:: draft
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-21

  - **Definition**
    - definition:: Collaborative work performed by geographically distributed team members
    - maturity:: emerging

  - **Domain Extensions**
    - collaboration-type:: synchronous
    - communication-mode:: multimodal
    - platform:: virtual-workspace
    - synchronicity:: real-time
    - participant-count:: 2-10

  - **Semantic Classification**
    - owl:class:: tc:RemoteCollaboration
    - owl:physicality:: ConceptualEntity
    - owl:role:: Process

  - #### Relationships
    - is-subclass-of:: [[Collaboration]]
    - requires:: [[Communication Technology]]
    - enables:: [[Distributed Work]]
```

## Validation Rules

### Core (All Domains)
✅ Must have: ontology, term-id, preferred-term, definition, source-domain, owl:class

### Domain-Specific
✅ AI: Must have algorithm-type, computational-complexity
✅ MV: Must have immersion-level, interaction-mode
✅ TC: Must have collaboration-type, communication-mode
✅ RB: Must have physicality, autonomy-level
✅ DT: Must have disruption-level, maturity-stage
✅ BC: Must have consensus-mechanism, decentralization-level

### Namespace
✅ Namespace must match domain (ai: for AI, tc: for TC, etc.)
✅ Class names must be CamelCase
✅ Physicality and role must be appropriate for domain

## Testing

```bash
# Run full test suite (16 tests)
node test-multi-ontology.js

# Test domain detection
node domain-detector.js detect path/to/file.md

# Test single file transformation
node cli.js test path/to/file.md
```

## Processing Pipeline

1. **Scan:** `node cli.js scan`
   - Detects all 6 domains
   - Classifies sub-domains
   - Identifies issues

2. **Validate:** `node cli.js validate-domain [domain]`
   - Checks core properties
   - Validates domain-specific properties
   - Verifies namespace correctness

3. **Process:** `node cli.js process-domain [domain] --live`
   - Generates canonical blocks
   - Adds domain extension properties
   - Fixes namespace issues
   - Creates backups

4. **Analyze:** `node cli.js cross-domain-links`
   - Finds cross-domain references
   - Suggests bridges
   - Maps domain interactions

## Cross-Domain Bridges

The system recognizes natural connections between domains:

- **AI ↔ MV:** virtual-agents, procedural-generation, ai-npcs
- **AI ↔ RB:** robot-perception, autonomous-navigation, learning-control
- **MV ↔ TC:** virtual-meetings, collaborative-spaces, shared-environments
- **RB ↔ TC:** telepresence-robots, remote-operation, teleoperation
- **BC ↔ DT:** tokenization, decentralized-innovation, crypto-disruption

## File Organization

```
scripts/ontology-migration/
├── domain-config.json          # Configuration for all 6 domains
├── domain-detector.js          # Domain detection utilities
├── generator.js                # Block generation (updated)
├── validator.js                # Validation logic (updated)
├── scanner.js                  # File scanning (updated)
├── cli.js                      # Command-line interface (updated)
├── test-multi-ontology.js      # Test suite
└── ...

docs/
├── MULTI-ONTOLOGY.md           # Comprehensive guide
└── MULTI-ONTOLOGY-QUICKSTART.md # This file
```

## Common Issues

### Issue: Wrong namespace for domain
**Solution:** Pipeline automatically fixes (e.g., mv: → rb: for robotics files)

### Issue: Missing domain-specific properties
**Solution:** Pipeline adds placeholders for required properties

### Issue: Domain detection ambiguous
**Solution:** Explicitly set `source-domain:: [domain-key]` in block

### Issue: Sub-domain not detected
**Solution:** Add domain-specific keywords to content

## Migration Strategy

For each domain:

1. **Identify files** belonging to domain
2. **Validate current state** with domain-specific rules
3. **Generate canonical blocks** with extension properties
4. **Fix namespace issues** (especially robotics mv: → rb:)
5. **Add sub-domain classification** where applicable
6. **Document cross-domain links** in CrossDomainBridges section

## Architecture Benefits

✅ **Federated:** Each domain maintains independence
✅ **Extensible:** Add domain properties without affecting others
✅ **Interoperable:** Cross-domain bridges enable knowledge linking
✅ **Validated:** Domain-specific validation ensures consistency
✅ **Scalable:** Pattern supports future domain additions
✅ **Flexible:** Sub-domains provide fine-grained organization

## Next Steps

1. Run `node cli.js scan` to inventory all files
2. Run `node cli.js domains` to see domain distribution
3. Pick a domain and run `node cli.js domain-stats [domain]`
4. Validate: `node cli.js validate-domain [domain]`
5. Process (dry-run first): `node cli.js process-domain [domain]`
6. Process (live): `node cli.js process-domain [domain] --live`

## Resources

- Full Guide: `docs/MULTI-ONTOLOGY.md`
- Domain Config: `scripts/ontology-migration/domain-config.json`
- CLI Help: `node scripts/ontology-migration/cli.js help`
- Tests: `node scripts/ontology-migration/test-multi-ontology.js`

---

**Last Updated:** 2025-11-21
**Framework Version:** Multi-Ontology v2.0 (6 domains)
