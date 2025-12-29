---
title: Solid Pod Schema
description: Complete Solid pod structure and ACL patterns for VisionFlow decentralized data storage
category: reference
tags:
  - database
  - solid
  - decentralized
  - backend
updated-date: 2025-12-29
difficulty-level: intermediate
---


# Solid Pod Schema

## Overview

VisionFlow uses Solid pods for decentralized user data storage, enabling users to own their data while participating in the collaborative ontology ecosystem. This document defines the complete pod structure, JSON-LD contexts, and ACL patterns.

---

## Pod Structure

```
/pods/{npub}/
  ├── profile/
  │   └── card                    # WebID document
  ├── ontology/
  │   ├── contributions/          # User additions
  │   ├── proposals/              # Pending proposals
  │   └── annotations/            # Comments
  ├── preferences/                # App settings
  ├── agent-memory/               # AI agent storage
  │   ├── sessions/               # Session memory
  │   └── long-term/              # Persistent memory
  └── inbox/                      # Notifications
```

### Directory Descriptions

| Directory | Purpose | Access Level |
|-----------|---------|--------------|
| `profile/card` | WebID identity document | Public read |
| `ontology/contributions/` | User-submitted ontology terms | Owner + reviewers |
| `ontology/proposals/` | Pending proposals for review | Owner + reviewers |
| `ontology/annotations/` | Comments on ontology terms | Public read |
| `preferences/` | User application settings | Owner only |
| `agent-memory/sessions/` | Short-term AI agent memory | Owner only |
| `agent-memory/long-term/` | Persistent AI agent memory | Owner only |
| `inbox/` | Notifications and messages | Owner + authorized senders |

---

## JSON-LD Contexts

### agent-memory.jsonld

Context for AI agent memory storage in Solid pods.

```json
{
  "@context": {
    "@vocab": "https://visionflow.io/vocab/",
    "@version": 1.1,
    "memory": "https://visionflow.io/vocab/memory#",
    "agent": "https://visionflow.io/vocab/agent#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dcterms": "http://purl.org/dc/terms/",

    "AgentMemory": "memory:AgentMemory",
    "SessionMemory": "memory:SessionMemory",
    "LongTermMemory": "memory:LongTermMemory",
    "EpisodicMemory": "memory:EpisodicMemory",
    "SemanticMemory": "memory:SemanticMemory",

    "sessionId": {
      "@id": "memory:sessionId",
      "@type": "xsd:string"
    },
    "agentType": {
      "@id": "agent:type",
      "@type": "xsd:string"
    },
    "agentId": {
      "@id": "agent:id",
      "@type": "xsd:string"
    },
    "memoryType": {
      "@id": "memory:memoryType",
      "@type": "@vocab"
    },
    "content": {
      "@id": "memory:content",
      "@type": "xsd:string"
    },
    "embedding": {
      "@id": "memory:embedding",
      "@container": "@list"
    },
    "priority": {
      "@id": "memory:priority",
      "@type": "xsd:float"
    },
    "createdAt": {
      "@id": "dcterms:created",
      "@type": "xsd:dateTime"
    },
    "expiresAt": {
      "@id": "memory:expiresAt",
      "@type": "xsd:dateTime"
    },
    "accessCount": {
      "@id": "memory:accessCount",
      "@type": "xsd:integer"
    },
    "lastAccessed": {
      "@id": "memory:lastAccessed",
      "@type": "xsd:dateTime"
    },
    "relatedMemories": {
      "@id": "memory:relatedTo",
      "@type": "@id",
      "@container": "@set"
    }
  }
}
```

**Example Memory Document**:

```json
{
  "@context": "https://visionflow.io/contexts/agent-memory.jsonld",
  "@id": "urn:uuid:550e8400-e29b-41d4-a716-446655440000",
  "@type": "EpisodicMemory",
  "sessionId": "session-2024-12-29-001",
  "agentType": "researcher",
  "agentId": "agent-researcher-alpha",
  "content": "User inquired about blockchain consensus mechanisms. Provided analysis of PoW, PoS, and BFT variants.",
  "embedding": [0.123, -0.456, 0.789, 0.234, -0.567],
  "priority": 0.85,
  "createdAt": "2024-12-29T10:30:00Z",
  "expiresAt": "2025-03-29T10:30:00Z",
  "accessCount": 5,
  "lastAccessed": "2024-12-29T14:22:00Z",
  "relatedMemories": [
    "urn:uuid:660e8400-e29b-41d4-a716-446655440001",
    "urn:uuid:770e8400-e29b-41d4-a716-446655440002"
  ]
}
```

---

### ontology-contribution.jsonld

Context for user ontology contributions.

```json
{
  "@context": {
    "@vocab": "https://visionflow.io/vocab/",
    "@version": 1.1,
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dcterms": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "contrib": "https://visionflow.io/vocab/contribution#",

    "OntologyContribution": "contrib:OntologyContribution",

    "termId": {
      "@id": "dcterms:identifier",
      "@type": "xsd:string"
    },
    "preferredTerm": {
      "@id": "skos:prefLabel",
      "@language": "en"
    },
    "alternateTerms": {
      "@id": "skos:altLabel",
      "@language": "en",
      "@container": "@set"
    },
    "definition": {
      "@id": "skos:definition",
      "@language": "en"
    },
    "sourceDomain": {
      "@id": "dcterms:subject",
      "@type": "@id"
    },
    "parentClass": {
      "@id": "rdfs:subClassOf",
      "@type": "@id"
    },
    "relatedClasses": {
      "@id": "rdfs:seeAlso",
      "@type": "@id",
      "@container": "@set"
    },
    "contributor": {
      "@id": "dcterms:contributor",
      "@type": "@id"
    },
    "status": {
      "@id": "contrib:status",
      "@type": "xsd:string"
    },
    "submittedAt": {
      "@id": "dcterms:created",
      "@type": "xsd:dateTime"
    },
    "modifiedAt": {
      "@id": "dcterms:modified",
      "@type": "xsd:dateTime"
    },
    "references": {
      "@id": "dcterms:references",
      "@type": "@id",
      "@container": "@set"
    },
    "justification": {
      "@id": "contrib:justification",
      "@type": "xsd:string"
    }
  }
}
```

**Example Contribution Document**:

```json
{
  "@context": "https://visionflow.io/contexts/ontology-contribution.jsonld",
  "@id": "urn:contribution:BC-0501",
  "@type": "OntologyContribution",
  "termId": "BC-0501",
  "preferredTerm": "Rollup",
  "alternateTerms": ["Layer 2 Rollup", "L2 Rollup"],
  "definition": "A scaling solution that executes transactions outside the main blockchain while posting transaction data on-chain.",
  "sourceDomain": "https://visionflow.io/ontology/domain#Blockchain",
  "parentClass": "https://visionflow.io/ontology/blockchain#ScalingSolution",
  "relatedClasses": [
    "https://visionflow.io/ontology/blockchain#ZKRollup",
    "https://visionflow.io/ontology/blockchain#OptimisticRollup"
  ],
  "contributor": "did:nostr:npub1abc123...",
  "status": "draft",
  "submittedAt": "2024-12-29T10:00:00Z",
  "modifiedAt": "2024-12-29T11:30:00Z",
  "references": [
    "https://ethereum.org/en/developers/docs/scaling/layer-2-rollups/",
    "https://doi.org/10.1145/3548606.3560578"
  ],
  "justification": "Rollups are a critical scaling technology for Ethereum and other blockchains, warranting formal ontological definition."
}
```

---

### user-preferences.jsonld

Context for user application preferences.

```json
{
  "@context": {
    "@vocab": "https://visionflow.io/vocab/",
    "@version": 1.1,
    "prefs": "https://visionflow.io/vocab/preferences#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",

    "UserPreferences": "prefs:UserPreferences",
    "GraphPreferences": "prefs:GraphPreferences",
    "FilterPreferences": "prefs:FilterPreferences",
    "UIPreferences": "prefs:UIPreferences",

    "graphLayout": {
      "@id": "prefs:graphLayout",
      "@type": "xsd:string"
    },
    "graphPhysicsEnabled": {
      "@id": "prefs:physicsEnabled",
      "@type": "xsd:boolean"
    },
    "graphNodeSize": {
      "@id": "prefs:nodeSize",
      "@type": "xsd:float"
    },
    "graphEdgeOpacity": {
      "@id": "prefs:edgeOpacity",
      "@type": "xsd:float"
    },
    "filterEnabled": {
      "@id": "prefs:filterEnabled",
      "@type": "xsd:boolean"
    },
    "filterQualityThreshold": {
      "@id": "prefs:qualityThreshold",
      "@type": "xsd:float"
    },
    "filterAuthorityThreshold": {
      "@id": "prefs:authorityThreshold",
      "@type": "xsd:float"
    },
    "filterMode": {
      "@id": "prefs:filterMode",
      "@type": "xsd:string"
    },
    "theme": {
      "@id": "prefs:theme",
      "@type": "xsd:string"
    },
    "defaultView": {
      "@id": "prefs:defaultView",
      "@type": "xsd:string"
    },
    "sidebarCollapsed": {
      "@id": "prefs:sidebarCollapsed",
      "@type": "xsd:boolean"
    },
    "updatedAt": {
      "@id": "prefs:updatedAt",
      "@type": "xsd:dateTime"
    }
  }
}
```

**Example Preferences Document**:

```json
{
  "@context": "https://visionflow.io/contexts/user-preferences.jsonld",
  "@id": "urn:preferences:npub1abc123",
  "@type": "UserPreferences",
  "graphLayout": "force-directed",
  "graphPhysicsEnabled": true,
  "graphNodeSize": 12.0,
  "graphEdgeOpacity": 0.6,
  "filterEnabled": true,
  "filterQualityThreshold": 0.7,
  "filterAuthorityThreshold": 0.5,
  "filterMode": "or",
  "theme": "dark",
  "defaultView": "graph",
  "sidebarCollapsed": false,
  "updatedAt": "2024-12-29T10:00:00Z"
}
```

---

## ACL Patterns

VisionFlow uses Web Access Control (WAC) for Solid pod authorization.

### Default Pod ACLs

Base ACL for the pod root (`.acl`):

```turtle
@prefix acl: <http://www.w3.org/ns/auth/acl#>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.

# Owner has full control
<#owner>
    a acl:Authorization;
    acl:agent <https://visionflow.io/pods/{npub}/profile/card#me>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read, acl:Write, acl:Control.

# VisionFlow app has read access to non-private containers
<#app-read>
    a acl:Authorization;
    acl:origin <https://visionflow.io>;
    acl:accessTo <./>;
    acl:mode acl:Read.
```

### Public Ontology ACLs

ACL for public-readable ontology annotations (`ontology/annotations/.acl`):

```turtle
@prefix acl: <http://www.w3.org/ns/auth/acl#>.
@prefix foaf: <http://xmlns.com/foaf/0.1/>.

# Owner has full control
<#owner>
    a acl:Authorization;
    acl:agent <https://visionflow.io/pods/{npub}/profile/card#me>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read, acl:Write, acl:Control.

# Public can read annotations
<#public-read>
    a acl:Authorization;
    acl:agentClass foaf:Agent;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read.
```

### Agent Memory ACLs

ACL for private agent memory (`agent-memory/.acl`):

```turtle
@prefix acl: <http://www.w3.org/ns/auth/acl#>.

# Only owner has access to agent memory
<#owner>
    a acl:Authorization;
    acl:agent <https://visionflow.io/pods/{npub}/profile/card#me>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read, acl:Write, acl:Control.

# VisionFlow agents can read/write with user consent
<#visionflow-agents>
    a acl:Authorization;
    acl:origin <https://visionflow.io>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read, acl:Write.
```

### Proposal Review ACLs

ACL for proposals under review (`ontology/proposals/.acl`):

```turtle
@prefix acl: <http://www.w3.org/ns/auth/acl#>.

# Owner has full control
<#owner>
    a acl:Authorization;
    acl:agent <https://visionflow.io/pods/{npub}/profile/card#me>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read, acl:Write, acl:Control.

# Reviewers can read proposals
<#reviewers>
    a acl:Authorization;
    acl:agentGroup <https://visionflow.io/groups/reviewers>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read.

# VisionFlow backend can read for processing
<#visionflow-backend>
    a acl:Authorization;
    acl:origin <https://visionflow.io>;
    acl:accessTo <./>;
    acl:default <./>;
    acl:mode acl:Read.
```

---

## Integration with VisionFlow

### Syncing Pod Data

```rust
use crate::adapters::solid_client::SolidClient;

pub async fn sync_user_preferences(
    client: &SolidClient,
    npub: &str,
) -> Result<UserPreferences, SolidError> {
    let pod_url = format!("https://visionflow.io/pods/{}/preferences/graph-settings.jsonld", npub);
    let prefs = client.get_jsonld::<UserPreferences>(&pod_url).await?;
    Ok(prefs)
}

pub async fn save_agent_memory(
    client: &SolidClient,
    npub: &str,
    memory: &AgentMemory,
) -> Result<(), SolidError> {
    let pod_url = format!(
        "https://visionflow.io/pods/{}/agent-memory/sessions/{}.jsonld",
        npub,
        memory.session_id
    );
    client.put_jsonld(&pod_url, memory).await?;
    Ok(())
}
```

### Querying Pod Data with SPARQL

```sparql
PREFIX memory: <https://visionflow.io/vocab/memory#>
PREFIX agent: <https://visionflow.io/vocab/agent#>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?memory ?content ?priority ?created
WHERE {
  ?memory a memory:EpisodicMemory ;
          memory:content ?content ;
          memory:priority ?priority ;
          dcterms:created ?created ;
          agent:type "researcher" .
  FILTER (?priority >= 0.7)
}
ORDER BY DESC(?priority)
LIMIT 10
```

---

## Security Considerations

1. **Private Data**: Agent memory and preferences are owner-only by default
2. **Origin Restrictions**: VisionFlow origin (`https://visionflow.io`) is explicitly authorized
3. **Group Membership**: Reviewers group membership is managed centrally
4. **Token Validation**: Solid access tokens are validated for each request
5. **Encryption**: Sensitive memory content should be encrypted at rest

---

## Related Documentation

- [Database Schema Reference](../DATABASE_SCHEMA_REFERENCE.md)
- [Unified Database Schema](./schemas.md)
- [Neo4j Ontology Schema V2](./ontology-schema-v2.md)
- [User Settings Schema](./user-settings-schema.md)

---

**Schema Version**: 1.0
**Last Updated**: December 29, 2025
**Maintainer**: VisionFlow Database Team
