# Ontology System Migration Guide

## Overview

This guide helps you migrate from the ontology stub configuration (`ontology.toml`) to the full-featured ontology validation system. The new system provides semantic validation, logical inference, and physics-based constraint visualization.

## What's Changed

### Before (Stub Implementation)
- Simple `ontology.toml` configuration file
- No validation or reasoning capabilities
- Manual constraint management
- Limited semantic features

### After (Full Implementation)
- Complete OWL/RDF validation engine
- Automated logical inference
- Physics constraint integration
- Real-time validation updates
- WebSocket support for live feedback

## Migration Steps

### Step 1: Enable the Feature Flag

The ontology validation system is feature-flagged for safe rollout.

**Enable via API:**
```bash
curl -X POST "/api/analytics/feature-flags" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "ontology_validation": true
  }'
```

**Enable via Configuration:**
```toml
# config.toml
[features]
ontology_validation = true
```

### Step 2: Prepare Your Ontology File

Convert your simple ontology configuration to a proper OWL/RDF ontology.

**Old Format (ontology.toml):**
```toml
[classes]
Person = "A human individual"
Company = "A business organization"

[properties]
employs = { domain = "Company", range = "Person" }
```

**New Format (domain.owl - Turtle):**
```turtle
@prefix ex: <http://example.org/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Person a owl:Class ;
    rdfs:label "Person" ;
    rdfs:comment "A human individual" .

ex:Company a owl:Class ;
    rdfs:label "Company" ;
    rdfs:comment "A business organization" .

ex:Person owl:disjointWith ex:Company .

ex:employs a owl:ObjectProperty ;
    rdfs:domain ex:Company ;
    rdfs:range ex:Person ;
    owl:inverseOf ex:worksFor .

ex:worksFor a owl:ObjectProperty ;
    rdfs:domain ex:Person ;
    rdfs:range ex:Company .
```

### Step 3: Update Mapping Configuration

Create or update `ontology/mapping.toml` to map your property graph schema to RDF semantics.

```toml
[global]
base_iri = "https://yourcompany.com/graph#"
default_class = "ex:Thing"

# Map node types to OWL classes
[classes.node_type]
"person" = "ex:Person"
"employee" = "ex:Employee"
"company" = "ex:Company"
"department" = "ex:Department"

# Map edge types to OWL properties
[properties.edge_type]
"employs" = "ex:employs"
"worksFor" = "ex:worksFor"
"manages" = "ex:manages"
"knows" = "foaf:knows"

# Map node metadata to data properties
[properties.metadata]
"email" = "foaf:mbox"
"name" = "foaf:name"
"age" = "ex:age"
"created" = "dcterms:created"

# Define inverse relationships
[inverses]
"ex:employs" = "ex:worksFor"
"ex:manages" = "ex:managedBy"

# IRI generation templates
[templates]
node_iri = "ex:node/{id}"
edge_iri = "ex:edge/{source}-{target}"

# Domain and range constraints
[constraints.domain]
"ex:employs" = "ex:Company"
"ex:manages" = "ex:Manager"

[constraints.range]
"ex:employs" = "ex:Employee"
"ex:manages" = "ex:Employee"
```

### Step 4: Load Your Ontology

**Via REST API:**
```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "source": "/path/to/your/domain.owl",
    "format": "turtle",
    "validateImmediately": false
  }'
```

**Response:**
```json
{
  "ontologyId": "ontology_abc123",
  "loadedAt": "2024-01-15T10:30:00Z",
  "axiomCount": 45,
  "loadingTimeMs": 150
}
```

Save the `ontologyId` for subsequent operations.

### Step 5: Configure Validation Behavior

Set validation parameters to match your requirements.

```bash
curl -X POST "/api/ontology/mapping" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "config": {
      "enableReasoning": true,
      "reasoningTimeoutSeconds": 30,
      "enableInference": true,
      "maxInferenceDepth": 3,
      "enableCaching": true,
      "cacheTtlSeconds": 3600,
      "validateCardinality": true,
      "validateDomainsRanges": true,
      "validateDisjointClasses": true
    }
  }'
```

### Step 6: Run Initial Validation

Validate your existing graph against the loaded ontology.

```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "ontologyId": "ontology_abc123",
    "mode": "full",
    "priority": 8
  }'
```

### Step 7: Review Validation Results

Check the validation report for any violations.

```bash
curl "/api/ontology/report" \
  -H "Authorization: Bearer <token>"
```

**Sample Report:**
```json
{
  "id": "report_def456",
  "timestamp": "2024-01-15T10:32:00Z",
  "durationMs": 450,
  "violations": [
    {
      "severity": "Error",
      "rule": "DisjointClasses",
      "message": "Node 'john_123' cannot be both Person and Company",
      "subject": "ex:john_123"
    }
  ],
  "inferredTriples": [
    {
      "subject": "ex:alice",
      "predicate": "ex:worksFor",
      "object": "ex:acme_corp"
    }
  ],
  "statistics": {
    "classesChecked": 12,
    "propertiesChecked": 25,
    "individualsChecked": 450
  }
}
```

### Step 8: Fix Violations

Address any violations found in the validation report:

1. **Disjoint Class Violations**: Correct nodes that have conflicting types
2. **Domain/Range Violations**: Ensure properties connect appropriate node types
3. **Cardinality Violations**: Fix nodes with too many or too few property values

### Step 9: Apply Inferences (Optional)

If the reasoner discovered new relationships, you can apply them:

```bash
curl -X POST "/api/ontology/apply" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "rdfTriples": [
      {
        "subject": "ex:alice",
        "predicate": "ex:worksFor",
        "object": "ex:acme_corp",
        "isLiteral": false
      }
    ],
    "updateGraph": true
  }'
```

### Step 10: Set Up Real-Time Monitoring

Enable WebSocket updates for live validation feedback.

```javascript
const ws = new WebSocket('/api/ontology/ws?client_id=my_app');

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'validation_complete':
      console.log('Validation completed:', message.payload);
      handleValidationResults(message.payload);
      break;

    case 'constraint_update':
      console.log('Physics constraints updated:', message.payload);
      updateVisualization(message.payload);
      break;
  }
};

// Trigger validation with real-time updates
fetch('/api/ontology/validate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    ontologyId: 'ontology_abc123',
    mode: 'full',
    enableWebsocketUpdates: true,
    clientId: 'my_app'
  })
});
```

## Migration Checklist

- [ ] Feature flag enabled
- [ ] Ontology file created in OWL/Turtle format
- [ ] Mapping configuration updated (`ontology/mapping.toml`)
- [ ] Ontology loaded via API
- [ ] Validation configuration set
- [ ] Initial validation run
- [ ] Violations reviewed and fixed
- [ ] Inferences applied (if desired)
- [ ] Real-time monitoring configured
- [ ] Team trained on new features

## Common Migration Issues

### Issue 1: Ontology Parse Errors

**Problem:** Ontology fails to load with syntax errors.

**Solution:**
1. Validate your OWL/Turtle syntax using online validators
2. Check for missing namespace declarations
3. Ensure proper IRI formats
4. Use a simpler format (Turtle) if having issues with RDF/XML

### Issue 2: Excessive Violations

**Problem:** Too many validation violations after migration.

**Solution:**
1. Start with lenient validation settings
2. Fix violations incrementally
3. Use `mode: "quick"` for faster iteration
4. Temporarily disable strict checks:
```json
{
  "config": {
    "validateCardinality": false,
    "validateDomainsRanges": false
  }
}
```

### Issue 3: Performance Degradation

**Problem:** Validation slows down graph operations.

**Solution:**
1. Use incremental validation mode for updates
2. Increase cache TTL
3. Reduce inference depth
4. Run full validation on schedule, not on every change:
```json
{
  "config": {
    "maxInferenceDepth": 1,
    "cacheTtlSeconds": 7200
  }
}
```

### Issue 4: Mapping Conflicts

**Problem:** Graph data doesn't map cleanly to RDF.

**Solution:**
1. Add fallback mappings in `mapping.toml`
2. Use `default_class` for unmapped node types
3. Create custom IRI templates
4. Add exception rules for edge cases

## Rollback Procedure

If you need to roll back to the stub implementation:

1. **Disable the feature flag:**
```bash
curl -X POST "/api/analytics/feature-flags" \
  -H "Content-Type: application/json" \
  -d '{"ontology_validation": false}'
```

2. **Clear ontology caches:**
```bash
curl -X DELETE "/api/ontology/cache" \
  -H "Authorization: Bearer <token>"
```

3. **Revert to stub configuration:**
   - Restore `ontology.toml` if needed
   - Remove `ontology/mapping.toml` customizations

The system will seamlessly fall back to stub behavior with no data loss.

## Gradual Migration Strategy

For large deployments, consider a phased approach:

### Phase 1: Read-Only Validation (Week 1-2)
- Enable feature flag
- Load ontology
- Run validation in monitoring mode only
- Do NOT apply inferences or constraints
- Collect violation statistics

### Phase 2: Fix Critical Violations (Week 3-4)
- Address Error-level violations
- Update graph data to comply with ontology
- Re-run validation to verify fixes

### Phase 3: Enable Inference (Week 5-6)
- Enable reasoning and inference
- Apply inferred relationships gradually
- Monitor graph quality improvements

### Phase 4: Enable Physics Constraints (Week 7-8)
- Activate ontology-driven physics constraints
- Observe layout improvements
- Fine-tune constraint strengths

### Phase 5: Real-Time Validation (Week 9+)
- Enable WebSocket updates
- Implement client-side violation handling
- Train users on new capabilities

## Performance Tuning

Optimize validation performance for your workload:

### For Development
```json
{
  "config": {
    "enableReasoning": true,
    "enableInference": false,
    "enableCaching": false,
    "validateCardinality": false
  }
}
```

### For Testing
```json
{
  "config": {
    "enableReasoning": true,
    "enableInference": true,
    "maxInferenceDepth": 3,
    "enableCaching": true,
    "validateCardinality": true
  }
}
```

### For Production
```json
{
  "config": {
    "enableReasoning": true,
    "enableInference": true,
    "maxInferenceDepth": 5,
    "enableCaching": true,
    "cacheTtlSeconds": 7200,
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true
  }
}
```

## New Features Available

After migration, you gain access to:

1. **Logical Consistency Checking**
   - Automatic detection of logical contradictions
   - Domain/range constraint validation
   - Cardinality enforcement

2. **Automated Inference**
   - Inverse property inference (employs ↔ worksFor)
   - Transitive relationship chains
   - Symmetric relationship completion
   - Class hierarchy propagation

3. **Physics-Based Visualization**
   - Disjoint classes repel each other
   - Hierarchical relationships cluster together
   - Same-as individuals merge visually
   - Cardinality creates natural boundaries

4. **Real-Time Validation**
   - WebSocket updates as validation progresses
   - Live violation notifications
   - Immediate constraint application

5. **Comprehensive Reporting**
   - Detailed violation descriptions
   - Suggested fixes with confidence scores
   - Performance metrics and statistics
   - Cache hit rate monitoring

## Getting Help

- **Documentation**: `/docs/specialized/ontology/`
- **API Reference**: `ontology-api-reference.md`
- **User Guide**: `ontology-user-guide.md`
- **System Overview**: `ontology-system-overview.md`
- **Examples**: `/examples/ontology_*.rs`

## Next Steps

After successful migration:

1. **Explore Advanced Features**
   - Custom inference rules
   - Complex ontology patterns
   - Multi-ontology support

2. **Optimize Performance**
   - Fine-tune caching strategies
   - Adjust validation frequency
   - Configure incremental updates

3. **Enhance Ontology**
   - Add domain-specific classes
   - Define custom properties
   - Create richer constraints

4. **Monitor and Improve**
   - Track validation metrics
   - Analyze inference patterns
   - Refine mapping configuration

---

**Migration Support:** For assistance with migration, consult the comprehensive documentation or review the example implementations in `/examples/`.
