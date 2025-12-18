---
name: ontology-core
description: Foundation library for ontology manipulation with zero data loss
version: 1.0.0
author: Claude Code
tags: [ontology, owl2, validation, parsing, library]
---

# Ontology Core Library

**Foundation for ontology manipulation with ZERO data loss guarantee.**

## Overview

This shared library provides production-quality ontology manipulation with complete field preservation, OWL2 validation, and automatic rollback capabilities.

## Features

- ✅ **Complete Field Extraction** - ALL 17+ fields preserved
- ✅ **Zero Data Loss** - Round-trip identity guaranteed
- ✅ **Unknown Field Preservation** - Forward compatibility
- ✅ **OWL2 Validation** - Namespace and syntax checks
- ✅ **Safe Modifications** - Automatic rollback on failure
- ✅ **Immutable Operations** - Never mutate original data

## Quick Start

```python
from ontology_core.ontology_parser import OntologyParser
from ontology_core.ontology_modifier import OntologyModifier

# Parse with complete field preservation
parser = OntologyParser()
block = parser.parse_ontology_block(content)

# Safe modification with validation
modifier = OntologyModifier()
result = modifier.modify_file(
    file_path,
    updates={"status": "complete"},
    validate=True,
    backup=True
)
```

## Complete Field Set

### Core (17+ fields)
- id, term-id, preferred-term
- ontology, type, source-domain, version
- status, maturity, quality-score, authority-score, public-access
- definition, source, collapsed
- owl:class, owl:physicality, owl:role
- belongsToDomain, bridges-to-domain

### Additional
- Relationships, OWL Axioms, Cross-References
- Unknown fields via `additional_fields`

## API Reference

See implementation files for complete documentation.

## Testing

```bash
python tests/test_field_preservation.py
python tests/test_owl2_validator.py
python tests/test_ontology_modifier.py
```

## References

- Architecture: `/home/devuser/workspace/logseq/docs/ONTOLOGY-SKILL-ARCHITECTURE.md`
- Audit: `/home/devuser/workspace/logseq/docs/ONTOLOGY-ENRICHMENT-SKILL-AUDIT.md`
