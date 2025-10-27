# Ontology System Quick Start Guide

Get up and running with the VisionFlow Ontology System in 10 minutes.

## Prerequisites

- VisionFlow server running
- Valid API credentials
- Basic understanding of knowledge graphs
- Optional: Familiarity with OWL/RDF concepts

## Step 1: Enable Ontology Validation (2 minutes)

### Check Feature Status

```bash
curl "/api/ontology/health"
```

**Expected Response:**
```json
{
  "status": "healthy",
  "ontologyValidationEnabled": true,
  "health": {
    "loadedOntologies": 0,
    "cachedReports": 0,
    "validationQueueSize": 0
  }
}
```

### Enable the Feature (if disabled)

```bash
curl -X POST "/api/analytics/feature-flags" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "ontology_validation": true
  }'
```

## Step 2: Create a Simple Ontology (3 minutes)

Create a file `company-ontology.owl`:

```turtle
@prefix : <http://example.com/company#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Ontology Declaration
: rdf:type owl:Ontology ;
  rdfs:label "Company Ontology"@en ;
  rdfs:comment "A simple ontology for corporate knowledge"@en .

# Classes
:Person rdf:type owl:Class ;
    rdfs:label "Person"@en ;
    rdfs:comment "An individual person"@en .

:Employee rdf:type owl:Class ;
    rdfs:subClassOf :Person ;
    rdfs:label "Employee"@en .

:Manager rdf:type owl:Class ;
    rdfs:subClassOf :Employee ;
    rdfs:label "Manager"@en .

:Company rdf:type owl:Class ;
    rdfs:label "Company"@en ;
    rdfs:comment "A business organization"@en .

:Department rdf:type owl:Class ;
    rdfs:label "Department"@en ;
    rdfs:comment "An organizational unit"@en .

# Disjoint Classes (People are not companies)
:Person owl:disjointWith :Company .
:Employee owl:disjointWith :Department .

# Object Properties
:employs rdf:type owl:ObjectProperty ;
    rdfs:domain :Company ;
    rdfs:range :Employee ;
    rdfs:label "employs"@en ;
    owl:inverseOf :worksFor .

:worksFor rdf:type owl:ObjectProperty ;
    rdfs:domain :Employee ;
    rdfs:range :Company ;
    rdfs:label "works for"@en .

:manages rdf:type owl:ObjectProperty ;
    rdfs:domain :Manager ;
    rdfs:range :Employee ;
    rdfs:label "manages"@en ;
    owl:inverseOf :managedBy .

:managedBy rdf:type owl:ObjectProperty ;
    rdfs:domain :Employee ;
    rdfs:range :Manager ;
    rdf:type owl:FunctionalProperty ;
    rdfs:label "managed by"@en .

:memberOf rdf:type owl:ObjectProperty ;
    rdfs:domain :Employee ;
    rdfs:range :Department ;
    rdfs:label "member of"@en .

# Data Properties
:hasName rdf:type owl:DatatypeProperty ;
    rdfs:domain :Person ;
    rdfs:range xsd:string ;
    rdfs:label "has name"@en .

:hasEmail rdf:type owl:DatatypeProperty ;
    rdfs:domain :Employee ;
    rdfs:range xsd:string ;
    rdf:type owl:FunctionalProperty ;
    rdfs:label "has email"@en .

:employeeId rdf:type owl:DatatypeProperty ;
    rdfs:domain :Employee ;
    rdfs:range xsd:string ;
    rdf:type owl:FunctionalProperty ;
    rdfs:label "employee ID"@en .

# Cardinality Constraints
:Employee rdfs:subClassOf [
    rdf:type owl:Restriction ;
    owl:onProperty :employeeId ;
    owl:cardinality 1
] .

:Company rdfs:subClassOf [
    rdf:type owl:Restriction ;
    owl:onProperty :employs ;
    owl:minCardinality 1
] .
```

## Step 3: Load Your Ontology (1 minute)

```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "source": "file:///path/to/company-ontology.owl",
    "format": "turtle",
    "validateImmediately": false
  }'
```

**Expected Response:**
```json
{
  "ontologyId": "ontology_abc123",
  "loadedAt": "2025-10-27T14:30:00Z",
  "axiomCount": 45,
  "loadingTimeMs": 856
}
```

💡 **Tip**: Save the `ontologyId` for the next steps!

### Alternative: Load from URL

```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://example.com/ontology.owl",
    "format": "rdf-xml"
  }'
```

### Alternative: Inline Content

```bash
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "@prefix ex: <http://example.org/> . ex:Person a owl:Class .",
    "format": "turtle"
  }'
```

## Step 4: Configure Graph Mapping (2 minutes)

Create mapping configuration that links your property graph to the ontology:

```bash
curl -X POST "/api/ontology/mapping" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
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
    },
    "applyToAll": true
  }'
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Mapping configuration updated",
  "timestamp": "2025-10-27T14:31:00Z"
}
```

## Step 5: Run Validation (1 minute)

### Quick Validation (fastest)

```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "ontologyId": "ontology_abc123",
    "mode": "quick",
    "priority": 5
  }'
```

### Full Validation (recommended)

```bash
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "ontologyId": "ontology_abc123",
    "mode": "full",
    "priority": 8
  }'
```

**Expected Response:**
```json
{
  "jobId": "job_validation_xyz789",
  "status": "queued",
  "estimatedCompletion": "2025-10-27T14:31:05Z",
  "queuePosition": 1
}
```

## Step 6: Review Results (1 minute)

### Get Validation Report

```bash
curl "/api/ontology/report" \
  -H "Authorization: Bearer <your-token>"
```

**Example Response:**
```json
{
  "id": "report_def456",
  "timestamp": "2025-10-27T14:31:05Z",
  "durationMs": 1850,
  "graphSignature": "blake3_hash_abc123",
  "totalTriples": 1250,
  "violations": [
    {
      "id": "violation_001",
      "severity": "Error",
      "rule": "DisjointClasses",
      "message": "Individual ex:john cannot be both Person and Company",
      "subject": "ex:john",
      "timestamp": "2025-10-27T14:31:05Z"
    }
  ],
  "inferredTriples": [
    {
      "subject": "ex:alice",
      "predicate": "ex:worksFor",
      "object": "ex:acme_corp",
      "isLiteral": false
    }
  ],
  "statistics": {
    "classesChecked": 28,
    "propertiesChecked": 45,
    "individualsChecked": 892,
    "constraintsEvaluated": 156,
    "inferenceRulesApplied": 67,
    "cacheHits": 234,
    "cacheMisses": 15
  }
}
```

### Understanding Results

**Violations** - Logical errors that need fixing:
- ❌ **Error**: Serious constraint violation (disjoint classes, cardinality, etc.)
- ⚠️ **Warning**: Potential issue (missing recommended properties)
- ℹ️ **Info**: Informational notice

**Inferred Triples** - New knowledge discovered:
- Automatically generated relationships based on ontology rules
- Can be applied to enrich your graph

### Check Specific Report

```bash
curl "/api/ontology/report?report_id=report_def456" \
  -H "Authorization: Bearer <your-token>"
```

## Step 7: Apply Inferences (Optional)

If validation discovered new relationships, you can apply them:

```bash
curl -X POST "/api/ontology/apply" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
    "rdfTriples": [
      {
        "subject": "ex:alice",
        "predicate": "ex:worksFor",
        "object": "ex:acme_corp",
        "isLiteral": false
      }
    ],
    "maxDepth": 5,
    "updateGraph": true
  }'
```

## Next Steps

### 🎓 Learn More

- **[User Guide](./ontology-user-guide.md)** - Comprehensive documentation
- **[Ontology Fundamentals](./ontology-fundamentals.md)** - Learn OWL/RDF concepts
- **[Use Cases](./use-cases-examples.md)** - Real-world examples

### 🏗️ Build Your Ontology

- **[Semantic Modeling](./semantic-modeling.md)** - Design principles
- **[Entity Types](./entity-types-relationships.md)** - Available constructs
- **[Best Practices](./best-practices.md)** - Recommendations

### ⚙️ Advanced Features

- **[Real-time Updates](./ontology-user-guide.md#real-time-updates-with-websockets)** - WebSocket integration
- **[Physics Integration](./physics-integration.md)** - Semantic spatial constraints
- **[Performance Tuning](./performance-optimization.md)** - Optimization guide

### 🐛 Troubleshooting

- **[Troubleshooting Guide](./troubleshooting-guide.md)** - Common issues
- **[Error Codes](./error-codes.md)** - Error reference
- **[Configuration](./configuration-reference.md)** - Settings guide

## Common Commands Cheat Sheet

```bash
# Check system health
curl "/api/ontology/health"

# Load ontology from file
curl -X POST "/api/ontology/load-axioms" \
  -H "Content-Type: application/json" \
  -d '{"source": "file:///path/to/ontology.owl", "format": "turtle"}'

# Quick validation
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{"ontologyId": "ont_123", "mode": "quick"}'

# Full validation
curl -X POST "/api/ontology/validate" \
  -H "Content-Type: application/json" \
  -d '{"ontologyId": "ont_123", "mode": "full"}'

# Get latest report
curl "/api/ontology/report"

# Clear caches
curl -X DELETE "/api/ontology/cache"
```

## Quick Tips

✅ **Start Simple** - Begin with basic class hierarchies
✅ **Test Early** - Validate frequently during development
✅ **Use Templates** - Leverage existing domain ontologies
✅ **Monitor Performance** - Check `/api/ontology/health` regularly
✅ **Cache Results** - Enable caching for better performance

## Getting Help

**Have questions?**
- Review the [User Guide](./ontology-user-guide.md)
- Check [Troubleshooting](./troubleshooting-guide.md)
- See [Use Cases](./use-cases-examples.md) for examples

**Found a bug?**
- Check [Error Codes Reference](./error-codes.md)
- Review system logs
- Verify configuration settings

---

**Congratulations!** 🎉 You've successfully set up ontology validation. Continue exploring the [full documentation](./README.md) to unlock more advanced features.
