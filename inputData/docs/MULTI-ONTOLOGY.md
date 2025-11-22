# Multi-Ontology Architecture Guide

## Overview

The ontology framework implements a **federated multi-ontology architecture** with 6 independent but interconnected domains. Each domain has its own namespace, core properties, and domain-specific extension properties.

## The 6 Domains

### 1. Artificial Intelligence (ai:)

**Namespace:** `ai:`
**Prefix:** `AI-`
**Description:** AI domain covering machine learning, neural networks, and intelligent systems

**Required Extension Properties:**
- `algorithm-type` - Type of AI algorithm (e.g., supervised, unsupervised, reinforcement)
- `computational-complexity` - Computational complexity notation (e.g., O(n^2))

**Optional Extension Properties:**
- `training-data` - Type or source of training data
- `model-architecture` - Architecture of the AI model
- `inference-mode` - Mode of inference (batch, real-time, streaming)
- `learning-paradigm` - Learning approach used
- `accuracy-metrics` - Performance metrics

**Valid Physicalities:** ConceptualEntity, AbstractEntity, VirtualEntity
**Valid Roles:** Process, Agent, Concept, Quality

**Sub-domains:**
- `machine-learning:` (supervised, unsupervised, semi-supervised, reinforcement)
- `nlp:` (text-analysis, speech, translation, generation)
- `computer-vision:` (image-recognition, object-detection, segmentation)
- `deep-learning:`
- `reinforcement-learning:`

---

### 2. Metaverse (mv:)

**Namespace:** `mv:`
**Prefix:** `MV-`
**Description:** Metaverse domain covering virtual worlds, immersive experiences, and digital spaces

**Required Extension Properties:**
- `immersion-level` - Level of immersion (low, medium, high, full)
- `interaction-mode` - Mode of interaction (point-click, gesture, voice, haptic)

**Optional Extension Properties:**
- `rendering-engine` - 3D rendering engine used
- `platform-support` - Platforms supported (desktop, mobile, VR, AR)
- `spatial-dimensionality` - 2D, 2.5D, 3D, 4D
- `user-presence` - Representation of user presence
- `world-persistence` - Whether world state persists
- `social-features` - Social interaction capabilities

**Valid Physicalities:** VirtualEntity, HybridEntity, ConceptualEntity
**Valid Roles:** Object, Process, Concept, Relation

**Sub-domains:**
- `virtual-worlds:` (gaming, social, enterprise)
- `ar:` (mobile, wearable, spatial)
- `vr:` (immersive, desktop, mobile)
- `xr:` (extended reality)
- `spatial-computing:`
- `digital-twins:`

---

### 3. Telecollaboration (tc:)

**Namespace:** `tc:`
**Prefix:** `TC-`
**Description:** Telecollaboration domain covering remote work, distributed teams, and virtual collaboration

**Required Extension Properties:**
- `collaboration-type` - Type of collaboration (synchronous, asynchronous, hybrid)
- `communication-mode` - Primary communication mode (text, voice, video, multimodal)

**Optional Extension Properties:**
- `platform` - Collaboration platform used
- `synchronicity` - Level of real-time interaction required
- `participant-count` - Typical number of participants
- `interaction-model` - Model of interaction (peer-to-peer, hub-spoke, mesh)
- `media-richness` - Richness of communication media
- `coordination-mechanism` - How coordination is achieved

**Valid Physicalities:** ConceptualEntity, VirtualEntity, HybridEntity
**Valid Roles:** Process, Relation, Agent, Concept

**Sub-domains:**
- `remote-work:` (asynchronous, synchronous, hybrid)
- `education:` (k12, higher-ed, corporate-training)
- `healthcare:` (telemedicine, remote-monitoring, consultation)
- `virtual-teams:`
- `telepresence:`
- `distance-learning:`

---

### 4. Robotics (rb:)

**Namespace:** `rb:`
**Prefix:** `RB-`
**Description:** Robotics domain covering autonomous systems, sensors, actuators, and robot control

**Required Extension Properties:**
- `physicality` - Physical nature (embodied, virtual, hybrid, conceptual)
- `autonomy-level` - Level of autonomy (teleoperated, semi-autonomous, fully-autonomous)

**Optional Extension Properties:**
- `sensing-modality` - Types of sensors (vision, lidar, tactile, proprioceptive)
- `actuation-type` - Type of actuation (electric, hydraulic, pneumatic)
- `control-architecture` - Control system architecture
- `mobility-type` - Type of mobility (wheeled, legged, aerial, aquatic)
- `task-domain` - Domain of tasks performed
- `human-robot-interaction` - Type of HRI

**Valid Physicalities:** PhysicalEntity, HybridEntity, ConceptualEntity
**Valid Roles:** Object, Process, Agent, Quality

**Sub-domains:**
- `autonomous-systems:` (aerial, ground, marine, space)
- `sensors:` (vision, proximity, force, environmental)
- `actuators:` (electric, hydraulic, pneumatic)
- `control-systems:`
- `navigation:`
- `manipulation:`

---

### 5. Disruptive Technologies (dt:)

**Namespace:** `dt:`
**Prefix:** `DT-`
**Description:** Disruptive Technologies domain covering emerging innovations and transformative tech

**Required Extension Properties:**
- `disruption-level` - Level of disruption (incremental, sustaining, disruptive, radical)
- `maturity-stage` - Stage of technology maturity (concept, prototype, early, growth, mature)

**Optional Extension Properties:**
- `adoption-rate` - Rate of technology adoption
- `market-impact` - Impact on markets (niche, segment, industry, cross-industry)
- `innovation-type` - Type of innovation (product, process, business-model)
- `transformation-scope` - Scope of transformation
- `risk-level` - Risk assessment
- `time-to-adoption` - Expected time to mainstream adoption

**Valid Physicalities:** ConceptualEntity, AbstractEntity, PhysicalEntity, HybridEntity
**Valid Roles:** Concept, Process, Object, Quality

**Sub-domains:**
- `emerging:` (early-stage, prototype, pilot)
- `transformative:` (industry-wide, societal, economic)
- `disruptive:` (market-creating, low-end, new-market)
- `innovative:`
- `exponential:`

---

### 6. Blockchain (bc:)

**Namespace:** `bc:`
**Prefix:** `BC-`
**Description:** Blockchain domain covering distributed ledgers, smart contracts, and decentralized systems

**Required Extension Properties:**
- `consensus-mechanism` - Consensus algorithm (PoW, PoS, PBFT, Raft, etc.)
- `decentralization-level` - Degree of decentralization (centralized, federated, decentralized)

**Optional Extension Properties:**
- `blockchain-type` - Type of blockchain (public, private, consortium)
- `scalability` - Scalability characteristics
- `transaction-throughput` - Transactions per second
- `finality-time` - Time to transaction finality
- `security-model` - Security approach
- `token-economics` - Economic model for tokens

**Valid Physicalities:** ConceptualEntity, VirtualEntity, AbstractEntity
**Valid Roles:** Process, Object, Concept, Relation

**Sub-domains:**
- `cryptocurrency:` (bitcoin, ethereum, altcoins)
- `smart-contracts:` (solidity, vyper, rust)
- `consensus:` (pow, pos, pbft, raft)
- `defi:` (decentralized finance)
- `nft:` (non-fungible tokens)
- `dao:` (decentralized autonomous organizations)

---

## Core Properties (Universal)

All ontology blocks across all domains share these core properties:

### Identification
- `ontology` - Always `true`
- `term-id` - Domain-prefixed ID (e.g., AI-001, TC-005)
- `preferred-term` - Primary term name
- `source-domain` - One of: ai, mv, tc, rb, dt, bc
- `status` - draft | in-progress | complete | deprecated
- `public-access` - true | false
- `version` - Semantic version
- `last-updated` - ISO date

### Definition
- `definition` - Clear definition of the term
- `maturity` - draft | emerging | mature | established

### Semantic Classification
- `owl:class` - Namespace:ClassName (e.g., ai:MachineLearning)
- `owl:physicality` - PhysicalEntity | VirtualEntity | ConceptualEntity | AbstractEntity | HybridEntity
- `owl:role` - Object | Process | Agent | Quality | Relation | Concept

### Recommended
- `alt-terms` - Alternative terms
- `authority-score` - Authority score (0-100)
- `quality-score` - Quality score (0-100)
- `source` - Source reference
- `belongsToDomain` - Domain classification
- `implementedInLayer` - Implementation layer

---

## Block Structure

```markdown
- ### OntologyBlock
  id:: [generated-id]
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: [DOMAIN-###]
    - preferred-term:: [Term Name]
    - source-domain:: [domain-key]
    - status:: [status]
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: [YYYY-MM-DD]

  - **Definition**
    - definition:: [Clear definition]
    - maturity:: [maturity-level]

  - **Domain Extensions**
    - [domain-required-prop-1]:: [value]
    - [domain-required-prop-2]:: [value]
    - [domain-optional-prop]:: [value]
    - sub-domain:: [sub-domain-classification]

  - **Semantic Classification**
    - owl:class:: [namespace]:[ClassName]
    - owl:physicality:: [physicality]
    - owl:role:: [role]

  - #### Relationships
    - is-subclass-of:: [[ParentConcept]]
    - has-part:: [[Component]]
    - requires:: [[Dependency]]
    - enables:: [[EnabledConcept]]

  - #### CrossDomainBridges
    - bridges-to:: [[OtherDomainConcept]]
    - bridges-from:: [[SourceDomainConcept]]
```

---

## Cross-Domain Bridges

The framework defines recommended bridges between domains:

### AI ↔ Metaverse
- virtual-agents
- procedural-generation
- ai-npcs

### AI ↔ Robotics
- robot-perception
- autonomous-navigation
- learning-control

### AI ↔ Blockchain
- ai-driven-trading
- predictive-analytics
- fraud-detection

### Metaverse ↔ Telecollaboration
- virtual-meetings
- collaborative-spaces
- shared-environments

### Robotics ↔ Telecollaboration
- telepresence-robots
- remote-operation
- teleoperation

### Blockchain ↔ Disruptive Technologies
- tokenization
- decentralized-innovation
- crypto-disruption

---

## Usage Examples

### Processing Specific Domain

```bash
# Scan and identify domains
node cli.js scan

# List all 6 domains with statistics
node cli.js domains

# Show statistics for AI domain
node cli.js domain-stats ai

# Validate Telecollaboration domain files
node cli.js validate-domain tc

# Process Blockchain domain files
node cli.js process-domain bc --live

# Analyze cross-domain links
node cli.js cross-domain-links

# Detect domain for specific file
node cli.js detect-domain path/to/file.md
```

### Domain Detection

The pipeline automatically detects domains using multiple strategies:

1. **From filename:** `AI-001-*.md` → ai
2. **From term-id:** `TC-005` → tc
3. **From namespace:** `rb:` → rb
4. **From content:** Keywords and patterns

### Validation

Domain-specific validation checks:

- Core properties present (universal)
- Domain-specific required properties present
- Namespace matches domain (ai: for AI, tc: for TC, etc.)
- Extension properties conform to domain schema
- Physicality and role appropriate for domain
- Cross-domain references valid

---

## Migration Strategy by Domain

### AI Domain
1. Detect AI files (AI- prefix or ai: namespace)
2. Add required: algorithm-type, computational-complexity
3. Add optional: training-data, model-architecture, etc.
4. Validate namespace is ai:
5. Check sub-domain classification

### Metaverse Domain
1. Detect MV files (MV- prefix or mv: namespace)
2. Add required: immersion-level, interaction-mode
3. Add optional: rendering-engine, platform-support, etc.
4. Validate namespace is mv:
5. Classify sub-domains (ar, vr, xr, etc.)

### Telecollaboration Domain (NEW)
1. Detect TC files (TC- prefix or content keywords)
2. Add required: collaboration-type, communication-mode
3. Add optional: platform, synchronicity, etc.
4. Assign namespace tc:
5. Classify sub-domains (remote-work, education, healthcare)

### Robotics Domain
1. Detect RB files (RB- prefix)
2. Fix namespace issues (mv: → rb:)
3. Add required: physicality, autonomy-level
4. Add optional: sensing-modality, actuation-type, etc.
5. Validate namespace is rb:

### Disruptive Technologies Domain
1. Detect DT files (DT- prefix or dt: namespace)
2. Add required: disruption-level, maturity-stage
3. Add optional: adoption-rate, market-impact, etc.
4. Validate namespace is dt:
5. Classify innovation type

### Blockchain Domain
1. Detect BC files (BC- prefix or bc: namespace)
2. Add required: consensus-mechanism, decentralization-level
3. Add optional: blockchain-type, scalability, etc.
4. Validate namespace is bc:
5. Classify sub-domains (DeFi, NFT, DAO, etc.)

---

## Testing Multi-Ontology Processing

```bash
# Run comprehensive test suite
node scripts/ontology-migration/test-multi-ontology.js

# Test specific domain detection
node scripts/ontology-migration/domain-detector.js detect path/to/file.md

# List all domains
node scripts/ontology-migration/domain-detector.js list
```

Test coverage includes:
- Domain detection from multiple sources
- Extension property generation
- Namespace validation
- Sub-domain classification
- Cross-domain link analysis
- Domain-specific physicality/role validation

---

## Best Practices

1. **Always specify source-domain** explicitly in ontology blocks
2. **Use correct namespace** for each domain (ai:, mv:, tc:, rb:, dt:, bc:)
3. **Include domain-specific required properties** - they are mandatory
4. **Document cross-domain links** using CrossDomainBridges section
5. **Classify sub-domains** when applicable for better organization
6. **Validate before committing** using domain-specific validation
7. **Respect domain boundaries** - don't mix extension properties across domains

---

## Tools and Utilities

### Domain Detector (`domain-detector.js`)
- Detects domain from multiple sources
- Classifies sub-domains
- Validates namespace alignment
- Analyzes cross-domain links

### Generator (`generator.js`)
- Generates canonical ontology blocks
- Adds domain-specific extension properties
- Includes sub-domain classification
- Preserves cross-domain references

### Validator (`validator.js`)
- Validates core properties (universal)
- Validates domain-specific properties
- Checks namespace correctness
- Verifies physicality/role for domain

### Scanner (`scanner.js`)
- Scans all 6 domains
- Detects sub-domains
- Identifies cross-domain links
- Generates comprehensive reports

---

## Architecture Principles

1. **Federated:** Each domain is independent with its own schema
2. **Extensible:** Domains can add extension properties without affecting others
3. **Interoperable:** Cross-domain bridges enable knowledge linking
4. **Validated:** Domain-specific validation ensures consistency
5. **Scalable:** New domains can be added following the same pattern
6. **Flexible:** Sub-domains provide fine-grained classification

---

## Support and Documentation

- **Configuration:** `scripts/ontology-migration/domain-config.json`
- **Tests:** `scripts/ontology-migration/test-multi-ontology.js`
- **CLI Help:** `node scripts/ontology-migration/cli.js help`
- **Issue Tracking:** Use `node cli.js stats` to see current status

For questions or issues with the multi-ontology architecture, consult this guide or run the test suite to verify functionality.
