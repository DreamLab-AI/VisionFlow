# Ontology API Documentation

## Overview

The Ontology API generators create JSON data for the React application, providing comprehensive ontology metadata with full IRI support across all 6 domains.

**Updated:** 2025-11-21
**Tools Version:** 2.0
**Library:** ontology_block_parser

---

## Generators

### 1. Page API Generator (`generate_page_api.py`)

Generates individual JSON files for each ontology page plus a domain index.

**Usage:**
```bash
python3 generate_page_api.py <pages-dir> <output-dir>
```

**Output Structure:**
- `/pages/*.json` - Individual page files
- `/domain-index.json` - Domain organization index

---

### 2. Search Index Generator (`generate_search_index.py`)

Generates a comprehensive search index with facets and metadata.

**Usage:**
```bash
python3 generate_search_index.py <pages-dir> <output-file>
```

**Output:**
- Single JSON file with all searchable documents
- Faceted metadata for filtering
- Optimized for fuzzy search

---

## API Format Specifications

### Page API Format

Each page JSON file (`/pages/{page-id}.json`) contains:

```json
{
  "id": "51 Percent Attack",
  "title": "51 Percent Attack",
  "content": "Full markdown content...",
  "backlinks": ["Blockchain Attack", "Double-Spending"],
  "wiki_links": ["blockchain", "hash rate", "Proof-of-Work"],
  "ontology": {
    // Core Identification
    "term_id": "BC-0077",
    "preferred_term": "51 Percent Attack",
    "alt_terms": ["Majority Attack", "51% Attack"],
    "iri": "http://narrativegoldmine.com/blockchain#51PercentAttack",

    // Domain Classification
    "source_domain": "blockchain",
    "domain": "bc",
    "domain_full_name": "Blockchain",

    // Definition
    "definition": "Attack on PoW blockchain networks...",
    "scope_note": "Primarily affects smaller networks...",

    // Status & Quality
    "status": "complete",
    "maturity": "mature",
    "version": "1.1.0",
    "public_access": true,
    "last_updated": "2025-11-13",
    "authority_score": 0.95,
    "quality_score": 0.95,
    "cross_domain_links": 3,

    // OWL Semantic Classification
    "owl_class": "bc:51PercentAttack",
    "owl_physicality": "VirtualEntity",
    "owl_role": "Threat",
    "owl_inferred_class": "bc:SecurityThreat",

    // Relationships
    "is_subclass_of": ["Blockchain Attack", "Consensus Attack"],
    "has_part": [],
    "is_part_of": [],
    "requires": ["Hash Rate", "Mining Pool"],
    "depends_on": ["Proof-of-Work"],
    "enables": ["Double-Spending", "Transaction Censorship"],
    "relates_to": ["Selfish Mining", "Eclipse Attack"],

    // Cross-Domain Bridges
    "bridges_to": [],
    "bridges_from": [],

    // Domain-Specific Extensions
    "domain_extensions": {
      "consensus-mechanism": "Proof-of-Work",
      "decentralization-level": "varies"
    },

    // Additional Metadata
    "belongs_to_domain": ["CryptographicDomain", "BlockchainSecurity"],
    "implemented_in_layer": ["ConsensusLayer"],
    "source": ["ISO/IEC 23257:2021", "IEEE 2418.1"],

    // Validation
    "validation": {
      "is_valid": true,
      "errors": []
    }
  }
}
```

---

### Domain Index Format

The domain index (`/domain-index.json`) provides domain organization:

```json
{
  "domains": {
    "ai": {
      "name": "Artificial Intelligence",
      "prefix": "AI-",
      "namespace": "http://narrativegoldmine.com/ai#",
      "pages": [
        {
          "id": "AI Agent System",
          "title": "AI Agent System",
          "term_id": "AI-0600",
          "iri": "http://narrativegoldmine.com/ai#AIAgentSystem"
        }
      ],
      "count": 314
    },
    "bc": {
      "name": "Blockchain",
      "prefix": "BC-",
      "namespace": "http://narrativegoldmine.com/blockchain#",
      "pages": [...],
      "count": 195
    },
    "rb": {
      "name": "Robotics",
      "prefix": "RB-",
      "namespace": "http://narrativegoldmine.com/robotics#",
      "pages": [...],
      "count": 53
    },
    "mv": {
      "name": "Metaverse",
      "prefix": "MV-",
      "namespace": "http://narrativegoldmine.com/metaverse#",
      "pages": [...],
      "count": 5
    },
    "tc": {
      "name": "Telecollaboration",
      "prefix": "TC-",
      "namespace": "http://narrativegoldmine.com/telecollaboration#",
      "pages": [...],
      "count": 0
    },
    "dt": {
      "name": "Disruptive Technologies",
      "prefix": "DT-",
      "namespace": "http://narrativegoldmine.com/disruptive-tech#",
      "pages": [...],
      "count": 1
    }
  },
  "uncategorized": {
    "pages": [...],
    "count": 955
  },
  "total_pages": 1684
}
```

---

### Search Index Format

The search index (`/search-index.json`) provides comprehensive search capabilities:

```json
{
  "version": "2.0",
  "generated_at": null,
  "total_documents": 1523,

  // Search Facets for Filtering
  "facets": {
    "domains": {
      "ai": {
        "name": "Artificial Intelligence",
        "count": 314
      },
      "bc": {
        "name": "Blockchain",
        "count": 195
      }
    },
    "physicality": {
      "ConceptualEntity": 457,
      "VirtualEntity": 249,
      "PhysicalEntity": 10
    },
    "roles": {
      "Concept": 426,
      "Object": 208,
      "Process": 66
    },
    "maturity": {
      "draft": 378,
      "mature": 324,
      "emerging": 22
    },
    "status": {
      "draft": 588,
      "complete": 263,
      "active": 62
    }
  },

  // Search Documents
  "documents": [
    {
      "id": "51 Percent Attack",
      "title": "51 Percent Attack",
      "content": "Preview text for search results...",

      // Core Ontology Data
      "term_id": "BC-0077",
      "preferred_term": "51 Percent Attack",
      "alt_terms": ["Majority Attack"],
      "definition": "Attack on PoW blockchain networks...",

      // Full IRI for Linking
      "iri": "http://narrativegoldmine.com/blockchain#51PercentAttack",

      // Domain Facets
      "domain": "bc",
      "domain_name": "Blockchain",
      "source_domain": "blockchain",

      // Search Optimization
      "search_terms": [
        "51 percent attack",
        "51",
        "percent",
        "attack",
        "bc-0077",
        "0077",
        "blockchain",
        "bc",
        "majority",
        "consensus",
        "proof-of-work"
      ],

      // Quality for Ranking
      "authority_score": 0.95,
      "quality_score": 0.95,
      "maturity": "mature",
      "status": "complete",

      // Classification for Faceted Search
      "owl_class": "bc:51PercentAttack",
      "owl_physicality": "VirtualEntity",
      "owl_role": "Threat",

      // Relationships for Context
      "is_subclass_of": ["Blockchain Attack"],
      "relates_to": ["Selfish Mining", "Eclipse Attack"],

      // Cross-Domain
      "cross_domain_links": 3,
      "belongs_to_domain": ["CryptographicDomain"],

      // Metadata
      "last_updated": "2025-11-13",
      "public_access": true
    }
  ]
}
```

---

## Key Features

### Full IRI Support

All terms include their complete IRI (Internationalized Resource Identifier):

```
term_id: "BC-0077"
owl_class: "bc:51PercentAttack"
iri: "http://narrativegoldmine.com/blockchain#51PercentAttack"
```

The IRI is constructed using:
- Domain namespace from DOMAIN_CONFIG
- Local name from owl:class property
- Supports all 6 domains plus standard namespaces (OWL, RDF, SKOS, etc.)

### Domain Organization

**6 Supported Domains:**
1. **AI** - Artificial Intelligence (`http://narrativegoldmine.com/ai#`)
2. **BC** - Blockchain (`http://narrativegoldmine.com/blockchain#`)
3. **RB** - Robotics (`http://narrativegoldmine.com/robotics#`)
4. **MV** - Metaverse (`http://narrativegoldmine.com/metaverse#`)
5. **TC** - Telecollaboration (`http://narrativegoldmine.com/telecollaboration#`)
6. **DT** - Disruptive Technologies (`http://narrativegoldmine.com/disruptive-tech#`)

### Search Optimization

**Fuzzy Search Support:**
- Comprehensive search_terms array
- Includes preferred term, alt terms, term ID
- Domain names and key definition words
- Both full terms and individual words

**Faceted Filtering:**
- Domain facets
- Physicality categories (Virtual, Physical, Conceptual, Hybrid)
- Role categories (Concept, Object, Process, etc.)
- Maturity levels (draft, emerging, mature)
- Status values (draft, complete, active)

### Quality Metrics

**Ranking Signals:**
- `authority_score` (0.0-1.0) - Source credibility
- `quality_score` (0.0-1.0) - Content quality
- `maturity` - Development stage
- `status` - Completion status
- `cross_domain_links` - Integration level

---

## Usage Examples

### Fetching a Page

```javascript
// Fetch single page
const page = await fetch('/api/pages/51 Percent Attack.json').then(r => r.json());

// Access ontology data
console.log(page.ontology.iri);
// => "http://narrativegoldmine.com/blockchain#51PercentAttack"

console.log(page.ontology.domain_full_name);
// => "Blockchain"
```

### Domain Navigation

```javascript
// Fetch domain index
const domains = await fetch('/api/domain-index.json').then(r => r.json());

// Get all blockchain pages
const bcPages = domains.domains.bc.pages;

// Navigate by namespace
const bcNamespace = domains.domains.bc.namespace;
// => "http://narrativegoldmine.com/blockchain#"
```

### Search Implementation

```javascript
// Fetch search index
const searchData = await fetch('/api/search-index.json').then(r => r.json());

// Fuzzy search
function fuzzySearch(query) {
  const q = query.toLowerCase();
  return searchData.documents.filter(doc =>
    doc.search_terms.some(term => term.includes(q))
  ).sort((a, b) =>
    (b.authority_score + b.quality_score) - (a.authority_score + a.quality_score)
  );
}

// Faceted filtering
function filterByDomain(domain) {
  return searchData.documents.filter(doc => doc.domain === domain);
}

function filterByPhysicality(type) {
  return searchData.documents.filter(doc => doc.owl_physicality === type);
}
```

---

## Test Results

### Page API Generation

**Test Run:** 2025-11-21

```
Source: /home/user/logseq/mainKnowledgeGraph/pages
Output: /tmp/test_output

Statistics:
- Total pages processed: 1,684
- Errors: 0
- Processing time: ~30s

Domain Distribution:
- Artificial Intelligence: 314 pages
- Blockchain: 195 pages
- Robotics: 53 pages
- Metaverse: 5 pages
- Telecollaboration: 0 pages (in directory)
- Disruptive Technologies: 1 page
- Uncategorized: 955 pages

Output:
- Individual pages: /tmp/test_output/pages/*.json (1,684 files)
- Domain index: /tmp/test_output/domain-index.json
```

### Search Index Generation

**Test Run:** 2025-11-21

```
Source: /home/user/logseq/mainKnowledgeGraph/pages
Output: /tmp/test_output/search-index.json

Statistics:
- Total files scanned: 1,684
- Documents indexed: 1,523
- Skipped (no ontology): 161
- File size: 2.2 MB

Quality Metrics:
- Average authority score: 0.60
- Average quality score: 0.50

Facets Generated:
- Domains: 20 different domain values
- Physicality types: 9 categories
- Roles: 22 different roles
- Maturity levels: 9 stages
- Status values: 11 states
```

---

## Performance Characteristics

### Page API
- **Generation time:** ~30s for 1,684 pages
- **Output size:** ~1,684 JSON files + 1 index
- **Individual file size:** 1-50 KB average
- **Total size:** ~15-20 MB

### Search Index
- **Generation time:** ~35s for 1,523 documents
- **Output size:** 2.2 MB (single file)
- **Compression:** Recommend gzip (reduces to ~300 KB)
- **Load time:** <100ms for modern browsers

### Recommendations

1. **Page API:** Use for individual page views, lazy loading
2. **Search Index:** Load once on app initialization, cache in memory
3. **Domain Index:** Use for navigation menus, domain filters
4. **CDN:** Serve all JSON files through CDN for best performance
5. **Caching:** Set long cache headers (immutable after generation)

---

## Integration with ontology_block_parser

Both tools now use the shared `ontology_block_parser` library:

**Import:**
```python
from ontology_block_parser import OntologyBlockParser, OntologyBlock, DOMAIN_CONFIG
```

**Usage:**
```python
# Initialize parser
parser = OntologyBlockParser()

# Parse file
block = parser.parse_file(md_file)

# Access properties
iri = block.get_full_iri()
domain = block.get_domain()
validation_errors = block.validate()
```

**Benefits:**
- Consistent parsing across all tools
- Automatic IRI generation
- Domain detection
- Validation support
- All 6 domains supported
- Domain-specific extensions

---

## Version History

### v2.0 (2025-11-21)
- Migrated to shared ontology_block_parser library
- Added full IRI support
- Added domain-organized structure
- Enhanced search index with facets
- Added fuzzy search optimization
- Support for all 6 domains
- Added validation metadata

### v1.0 (Previous)
- Basic page API generation
- Simple search index
- Manual parsing
- Limited domain support

---

## See Also

- `/home/user/logseq/Ontology-Tools/tools/lib/ontology_block_parser.py` - Shared parser library
- `/home/user/logseq/Ontology-Tools/tools/converters/generate_page_api.py` - Page API generator
- `/home/user/logseq/Ontology-Tools/tools/converters/generate_search_index.py` - Search index generator
