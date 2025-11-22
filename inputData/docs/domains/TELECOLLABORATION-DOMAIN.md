# Telecollaboration Domain Documentation

**Version:** 2.0.0
**Date:** 2025-11-21
**Domain:** Telecollaboration (tc:)
**Base URI:** `http://narrativegoldmine.com/telecollaboration#`
**Term Prefix:** TC-XXXX
**Status:** Active
**Authority:** Telecollaboration Ontology Working Group

---

## Overview

The **Telecollaboration Domain** encompasses technologies, practices, and competencies that enable distributed collaboration, remote interaction, and shared work across geographic and temporal distance. This domain integrates synchronous and asynchronous communication, knowledge creation, and embodied presence technologies.

### Domain Scope

The telecollaboration domain covers:
- **Synchronous Collaboration** technologies (video conferencing, real-time communication)
- **Asynchronous Collaboration** tools (document sharing, project management)
- **Embodied Presence** systems (telepresence robots, avatar-based interaction)
- **Knowledge Co-Creation** processes (collaborative learning, distributed research)
- **Pedagogical Approaches** for remote teaching and learning
- **Cultural and Intercultural** competencies in distributed teams
- **Digital Literacy** and online community engagement

### Domain Boundaries

The telecollaboration domain:
- **Includes:** Technologies and practices enabling remote interaction and collaboration
- **Excludes:** General information technology infrastructure (covered by IT domain)
- **Interfaces with:**
  - **Robotics (RB):** Through telepresence robot systems
  - **Artificial Intelligence (AI):** Through automated translation, transcription, and assistance
  - **Metaverse (MV):** Through virtual collaboration spaces and immersive environments
  - **Disruptive Technologies (DT):** Through emerging communication paradigms
  - **Blockchain (BC):** Through credential verification and decentralized systems

---

## Domain Configuration

### Namespace and Prefix

```json
{
  "namespace": "tc:",
  "prefix": "TC-",
  "baseURI": "http://narrativegoldmine.com/telecollaboration#",
  "name": "Telecollaboration",
  "description": "Remote work, distributed teams, and virtual collaboration"
}
```

### Required Properties

All telecollaboration concepts must include:

1. **collaboration-type** (enum)
   - `synchronous`: Real-time interaction
   - `asynchronous`: Time-shifted interaction
   - `hybrid`: Mix of synchronous and asynchronous

2. **communication-mode** (enum)
   - `text`: Written/chat-based
   - `audio`: Voice-based
   - `video`: Visual communication
   - `embodied`: Physical presence (robot/avatar)
   - `multimodal`: Combination of modes

### Optional Properties

Telecollaboration concepts may include:

- `platform`: Technology platform used
- `synchronicity`: Temporal structure
- `participant-count`: Number of participants
- `interaction-model`: Type of interaction pattern
- `media-richness`: Range of media supported
- `coordination-mechanism`: How collaboration is coordinated
- `cultural-contexts`: Cultural diversity represented
- `geographic-distribution`: Geographic span

---

## Sub-Domains

### 1. Remote Work (tc:rw:)

**Focus:** Distributed workplace collaboration and professional practice

**Key Concepts:**
- Asynchronous and synchronous meeting practices
- Distributed team coordination
- Remote management and supervision
- Work-life balance in remote contexts
- Home office and workspace setup

**Example Terms:**
- TC-0101: Distributed Team Management
- TC-0102: Asynchronous Work Practices
- TC-0103: Remote Meeting Protocols

**Cross-Domain Links:**
- HR and organizational management
- Workplace technology and digital tools
- Business process optimization

### 2. Education (tc:ed:)

**Focus:** Teaching and learning in distributed contexts

**Key Concepts:**
- Distance learning and online education
- Synchronous online instruction
- Asynchronous learning environments
- Virtual classroom design
- Online assessment and proctoring
- Student engagement in distance learning

**Example Terms:**
- TC-0201: Online Learning Environment Design
- TC-0202: Synchronous Distance Instruction
- TC-0203: Online Assessment Practices

**Cross-Domain Links:**
- Pedagogical theory and instructional design
- Learning sciences and educational technology
- Assessment and evaluation

### 3. Healthcare (tc:hc:)

**Focus:** Remote medical and mental health services

**Key Concepts:**
- Telemedicine and remote consultation
- Remote patient monitoring
- Mental health counseling
- Medical education and training
- Patient-provider communication
- Clinical decision support

**Example Terms:**
- TC-0301: Telemedicine Practice
- TC-0302: Remote Patient Monitoring
- TC-0303: Telehealth Infrastructure

**Cross-Domain Links:**
- Healthcare systems and clinical practice
- Medical informatics
- Health and wellness

### 4. Telepresence (tc:tp:)

**Focus:** Embodied and immersive presence technologies

**Key Concepts:**
- Telepresence robots and platforms
- Avatar-based interaction
- Haptic feedback systems
- Virtual presence and immersion
- Spatial collaboration spaces
- Holographic communication

**Example Terms:**
- TC-0003: Telepresence Robot
- TC-0401: Avatar-Based Collaboration
- TC-0402: Haptic Feedback Systems
- TC-0403: Virtual Collaboration Spaces

**Cross-Domain Links:**
- Robotics and automation (RB domain)
- Metaverse and virtual worlds (MV domain)
- Immersive technologies and XR

### 5. Virtual Teams (tc:vt:)

**Focus:** Distributed team dynamics and organizational behavior

**Key Concepts:**
- Team cohesion in virtual contexts
- Trust building in distributed teams
- Communication patterns and norms
- Diversity and inclusion in virtual teams
- Remote team motivation and engagement
- Virtual organizational culture

**Example Terms:**
- TC-0501: Virtual Team Dynamics
- TC-0502: Trust Building in Remote Teams
- TC-0503: Virtual Team Communication Norms

**Cross-Domain Links:**
- Organizational psychology
- Group dynamics and team science
- Intercultural communication

### 6. Distance Learning (tc:dl:)

**Focus:** Educational practices and technologies for distributed learning

**Key Concepts:**
- Curriculum design for distance education
- Learner support services
- Learning analytics for online courses
- Community of inquiry in distance contexts
- Student persistence and retention
- Accessibility in distance learning

**Example Terms:**
- TC-0601: Distance Education Curriculum Design
- TC-0602: Learning Analytics in Distance Contexts
- TC-0603: Community of Inquiry Framework

**Cross-Domain Links:**
- Educational design and technology
- Learning sciences
- Educational assessment

---

## Core Concepts (Example Implementations)

### TC-0001: Video Conferencing

**Classification:** Synchronous Communication Technology
**Physicality:** VirtualEntity
**Role:** Process

**Key Properties:**
- Enables face-to-face interaction at distance
- Real-time audio and video transmission
- Supports non-verbal communication cues
- Platform examples: Zoom, Teams, Google Meet, Webex

**Relationships:**
- Enables: Team collaboration, distance learning, telemedicine
- Uses-technology: Audio/video codecs, network protocols
- Bridges-to: Real-time translation (AI), Virtual meeting spaces (MV)

### TC-0002: Collaborative Document Editing

**Classification:** Knowledge Co-Creation Technology
**Physicality:** VirtualEntity
**Role:** Process

**Key Properties:**
- Multi-user real-time or asynchronous editing
- Version control and change tracking
- Comment and feedback mechanisms
- Hybrid synchronous-asynchronous model

**Relationships:**
- Enables: Distributed teamwork, peer learning, knowledge building
- Uses-technology: Operational transformation, CRDT, version control
- Bridges-to: Smart writing suggestions (AI), Immutable records (BC)

### TC-0003: Telepresence Robot

**Classification:** Embodied Telepresence Technology
**Physicality:** HybridEntity
**Role:** Agent

**Key Properties:**
- Physical embodied presence at distance
- Real-time teleoperation
- Multi-camera and audio system
- Synchronous interaction with manipulation

**Relationships:**
- Enables: Remote task execution, physical presence, environmental interaction
- Bridges-to: Autonomous navigation (RB), Computer vision (AI), Virtual avatars (MV)
- Cross-domain: Robotics sensors and control systems

---

## Telecollaboration-Specific Properties

### Collaboration Properties

**tc:collaboration-type** (Required)
- Values: `synchronous`, `asynchronous`, `hybrid`
- Example: `tc:collaboration-type:: hybrid`

**tc:communication-mode** (Required)
- Values: `text`, `audio`, `video`, `embodied`, `multimodal`
- Example: `tc:communication-mode:: video`

**tc:participant-count** (Optional)
- Range: 1 to unlimited
- Example: `tc:participant-count:: 50`

**tc:duration** (Optional)
- Format: String with units (minutes, hours, days, weeks)
- Example: `tc:duration:: 90 minutes`

**tc:cultural-contexts** (Optional)
- Format: Page link list
- Example: `tc:cultural-contexts:: [[North America]], [[Europe]], [[Asia]]`

**tc:geographic-distribution** (Optional)
- Format: List of locations
- Example: `tc:geographic-distribution:: Global, 15+ countries`

### Technology Properties

**tc:platform** (Optional)
- Format: Page link list
- Example: `tc:platform:: [[Zoom]], [[Microsoft Teams]], [[Moodle]]`

**tc:collaboration-tools** (Optional)
- Format: Page link list
- Example: `tc:collaboration-tools:: [[Screen Sharing]], [[Chat]], [[Recording]]`

**tc:media-richness** (Optional)
- Values: `low`, `medium`, `high`, `very-high`
- Example: `tc:media-richness:: high`

**tc:synchronicity** (Optional)
- Values: `synchronous`, `asynchronous`, `hybrid`
- Example: `tc:synchronicity:: synchronous`

**tc:accessibility-features** (Optional)
- Format: Page link list
- Example: `tc:accessibility-features:: [[Captions]], [[Screen Reader]], [[Translation]]`

### Pedagogical Properties

**tc:learning-model** (Optional)
- Format: Page link list
- Example: `tc:learning-model:: [[Community of Inquiry]], [[Constructivism]]`

**tc:instructional-design** (Optional)
- Format: Page link list
- Example: `tc:instructional-design:: [[Project-Based Learning]], [[Task-Based Learning]]`

**tc:assessment-methods** (Optional)
- Format: Page link list
- Example: `tc:assessment-methods:: [[Peer Assessment]], [[Portfolio Assessment]]`

**tc:transversal-skills** (Optional)
- Format: Page link list
- Example: `tc:transversal-skills:: [[Critical Thinking]], [[Communication]], [[Collaboration]]`

**tc:intercultural-competencies** (Optional)
- Format: Page link list
- Example: `tc:intercultural-competencies:: [[Cultural Awareness]], [[Perspective Taking]]`

### Outcome Properties

**tc:social-presence** (Optional)
- Values: `low`, `medium`, `high`
- Definition: Extent to which participants feel socially connected
- Example: `tc:social-presence:: high`

**tc:cognitive-presence** (Optional)
- Values: `low`, `medium`, `high`
- Definition: Extent of intellectual engagement
- Example: `tc:cognitive-presence:: high`

**tc:teaching-presence** (Optional)
- Values: `low`, `medium`, `high`
- Definition: Level of facilitation and structure
- Example: `tc:teaching-presence:: medium`

**tc:knowledge-creation-approach** (Optional)
- Format: Page link list
- Example: `tc:knowledge-creation-approach:: [[Social Constructivism]], [[Connectivism]]`

**tc:learning-outcomes** (Optional)
- Format: Page link list
- Example: `tc:learning-outcomes:: [[Intercultural Competence]], [[Digital Skills]]`

---

## Telecollaboration-Specific Relationships

### Pedagogical Relationships

**tc:facilitates** - Learning outcomes or competencies facilitated
- Domain: Telecollaboration concept
- Range: Learning outcome, competency
- Example: `tc:facilitates:: [[Intercultural Communication]], [[Language Learning]]`

**tc:involves-cultures** - Cultural groups engaged
- Domain: Telecollaboration concept
- Range: Cultural context, group
- Example: `tc:involves-cultures:: [[Western Culture]], [[Eastern Culture]]`

**tc:supports-pedagogy** - Pedagogical approaches supported
- Domain: Technology or method
- Range: Pedagogical approach
- Example: `tc:supports-pedagogy:: [[Constructivism]], [[Collaborative Learning]]`

### Technology Relationships

**tc:uses-technology** - Technologies employed
- Domain: Telecollaboration concept
- Range: Technology component
- Example: `tc:uses-technology:: [[Video Encoding]], [[Network Protocol]]`

**tc:integrates-with** - Systems or platforms integrated
- Domain: Telecollaboration system
- Range: External system, platform
- Example: `tc:integrates-with:: [[LMS]], [[Analytics Platform]]`

### Outcome Relationships

**tc:develops-competency** - Competencies being developed
- Domain: Learning activity, program
- Range: Competency
- Example: `tc:develops-competency:: [[Remote Communication]], [[Digital Collaboration]]`

**tc:creates-knowledge** - Knowledge artifacts created
- Domain: Collaboration activity
- Range: Knowledge artifact, output
- Example: `tc:creates-knowledge:: [[Collaborative Reports]], [[Shared Understanding]]`

**tc:builds-community** - Types of communities formed
- Domain: Telecollaboration initiative
- Range: Community type
- Example: `tc:builds-community:: [[Learning Community]], [[Professional Network]]`

---

## Valid Physicalities and Roles

### Valid Physicalities (OWL)

For telecollaboration concepts:
- **ConceptualEntity**: Abstract concepts (collaboration models, pedagogies)
- **VirtualEntity**: Software-based tools (video conferencing, document editing)
- **HybridEntity**: Combined physical-virtual systems (telepresence robots)

### Valid Roles (OWL)

For telecollaboration concepts:
- **Process**: Collaborative activities and interactions
- **Relation**: Connections and relationships between participants
- **Agent**: Participants, facilitators, robots, systems with agency
- **Concept**: Abstract ideas and frameworks

---

## Cross-Domain Bridges

### TC ↔ AI (Artificial Intelligence)

**Bridge Concepts:**
- Real-Time Translation: Breaking language barriers in synchronous collaboration
- Sentiment Analysis: Analyzing engagement and emotional tone in interactions
- Automated Transcription: Converting speech to text during meetings
- Virtual Meeting Assistant: AI summarizing key decisions and action items
- Intelligent Routing: AI directing messages and collaboration tools

**Pattern:** `tc:uses-technology:: [[AI Translation]], [[AI Transcription]]`

### TC ↔ MV (Metaverse)

**Bridge Concepts:**
- Virtual Collaboration Spaces: 3D virtual rooms for meetings
- Avatar-Based Interaction: Representing participants as avatars
- Immersive Learning Environments: Extended reality for distance education
- Spatial Communication: Using 3D space for organizing collaboration
- Mixed Reality Meetings: Blending physical and virtual participants

**Pattern:** `tc:integrates-with:: [[Virtual World]], [[Metaverse Platform]]`

### TC ↔ RB (Robotics)

**Bridge Concepts:**
- Telepresence Robots: Mobile robots providing embodied presence
- Remote Operation of Industrial Systems: Controlling robots from distance
- Autonomous Team Members: Semi-autonomous robots in distributed teams
- Haptic Feedback: Tactile sensation in remote manipulation
- Multi-Robot Collaboration: Distributed robot swarms with remote oversight

**Pattern:** `tc:uses-technology:: [[Telepresence Robot]], [[Teleoperation System]]`

### TC ↔ DT (Disruptive Technologies)

**Bridge Concepts:**
- 5G/6G Low-Latency Networks: Enabling responsive remote interaction
- Edge Computing: Processing collaboration data at network edge
- Holographic Displays: Advanced visualization for meetings
- Quantum Computing: Encryption for secure collaboration
- Brain-Computer Interfaces: Direct neural control of collaboration systems

**Pattern:** `tc:enables-by:: [[5G Network]], [[Quantum Encryption]]`

### TC ↔ BC (Blockchain)

**Bridge Concepts:**
- Credential Verification: Verifying educational credentials in distance learning
- Digital Badges: Blockchain-based achievement recognition
- Decentralized Learning Records: Immutable educational transcripts
- Smart Contracts: Automated course completion and payment
- Decentralized Collaboration Networks: P2P collaboration platforms

**Pattern:** `tc:secures-via:: [[Blockchain Certificate]], [[Smart Contract]]`

---

## Standard Ontology Template

### Structure for New Telecollaboration Concepts

```markdown
- ### [Concept Name]
  id:: tc-[concept-slug]-ontology
  collapsed:: true

  - **Identification** [CORE - Tier 1]
    - ontology:: true
    - term-id:: TC-XXXX
    - preferred-term:: [Human Readable Name]
    - alt-terms:: [[Alternative 1]], [[Alternative 2]]
    - source-domain:: telecollaboration
    - status:: [draft | in-progress | complete | deprecated]
    - public-access:: [true | false]
    - version:: [M.m.p]
    - last-updated:: [YYYY-MM-DD]

  - **Definition** [CORE - Tier 1]
    - definition:: [Comprehensive 2-5 sentence definition]
    - maturity:: [draft | emerging | mature | established]
    - source:: [[UNICollaboration]], [[Academic Source]]
    - authority-score:: [0.0-1.0]

  - **Semantic Classification** [CORE - Tier 1]
    - owl:class:: tc:[ClassName]
    - owl:physicality:: [ConceptualEntity | VirtualEntity | HybridEntity]
    - owl:role:: [Process | Relation | Agent | Concept]
    - belongsToDomain:: [[TelechollaborationDomain]]
    - belongsToSubDomain:: [[SubDomain 1]], [[SubDomain 2]]

  - **Collaboration Properties** [TC EXTENSION]
    - tc:collaboration-type:: [synchronous | asynchronous | hybrid]
    - tc:communication-mode:: [text | audio | video | embodied | multimodal]
    - tc:participant-count:: [number]
    - tc:platform:: [[Platform1]], [[Platform2]]

  - **Pedagogical Properties** [TC EXTENSION - if applicable]
    - tc:learning-model:: [[Model1]], [[Model2]]
    - tc:instructional-design:: [[Approach1]], [[Approach2]]
    - tc:assessment-methods:: [[Method1]], [[Method2]]
    - tc:transversal-skills:: [[Skill1]], [[Skill2]]

  - **Technology Properties** [TC EXTENSION]
    - tc:platform-used:: [[Platform1]], [[Platform2]]
    - tc:collaboration-tools:: [[Tool1]], [[Tool2]]
    - tc:media-richness:: [low | medium | high | very-high]
    - tc:accessibility-features:: [[Feature1]], [[Feature2]]

  - **Outcomes Properties** [TC EXTENSION]
    - tc:social-presence:: [low | medium | high]
    - tc:cognitive-presence:: [low | medium | high]
    - tc:teaching-presence:: [low | medium | high]
    - tc:learning-outcomes:: [[Outcome1]], [[Outcome2]]

  - #### Relationships [CORE - Tier 1]
    - is-subclass-of:: [[ParentClass1]], [[ParentClass2]]
    - has-part:: [[Component1]], [[Component2]]
    - enables:: [[Capability1]], [[Capability2]]

  - #### Telecollaboration-Specific Relationships [TC EXTENSION]
    - tc:facilitates:: [[Outcome1]], [[Outcome2]]
    - tc:uses-technology:: [[Technology1]], [[Technology2]]
    - tc:develops-competency:: [[Competency1]], [[Competency2]]
    - tc:supports-pedagogy:: [[Pedagogy1]], [[Pedagogy2]]

  - #### CrossDomainBridges [CORE - Tier 3]
    - bridges-to:: [[AI Translation]] via ai-language-processing
    - bridges-to:: [[Virtual Space]] via metaverse-integration
    - bridges-to:: [[Teleoperated Robot]] via robotics-embodiment
```

---

## Validation Rules

### TC-Specific Validations

1. **Collaboration Type Consistency**
   - If `synchronous`: should have low latency requirements
   - If `asynchronous`: should support time-shifted access
   - If `hybrid`: should specify both modes

2. **Pedagogical Alignment**
   - Learning model should align with instructional design
   - Assessment methods should match learning outcomes
   - Presence levels should be consistent with program goals

3. **Community of Inquiry Framework**
   - If using CoI model: must specify social, cognitive, teaching presence
   - Presence levels should be justified by program characteristics
   - Facilitation should support all three presences

4. **Cultural Sensitivity**
   - If multiple cultural contexts: should document intercultural competencies
   - Accessibility features should be adequate for participant diversity
   - Language support should match participant needs

5. **Technology-Pedagogy Alignment**
   - Platform capabilities should support instructional design
   - Tools should enable stated learning outcomes
   - Media richness should match pedagogical requirements

---

## Existing Telecollaboration Concepts

### Current Implementations

- **TC-0001: Video Conferencing** - Synchronous face-to-face communication
- **TC-0002: Collaborative Document Editing** - Hybrid knowledge co-creation
- **TC-0003: Telepresence Robot** - Embodied remote presence

### Concept Development Status

| Term ID | Concept | Status | Maturity | Authority |
|---------|---------|--------|----------|-----------|
| TC-0001 | Video Conferencing | Draft | Mature | 0.96 |
| TC-0002 | Collaborative Document Editing | Draft | Mature | 0.94 |
| TC-0003 | Telepresence Robot | Draft | Emerging | 0.87 |

---

## Research and Development Priorities

### High Priority Concepts Needed

1. **Virtual Exchange Programs** - International collaborative learning initiatives
2. **Community of Inquiry Framework** - Theoretical foundation for online learning
3. **Intercultural Competence Development** - Cultural learning outcomes
4. **Digital Literacy Frameworks** - Competency models for technology use
5. **Collaborative Pedagogy Approaches** - Teaching methods for distance education
6. **Online Learning Communities** - Community formation and sustainability
7. **Telecollaboration Platforms** - Technology infrastructures
8. **Asynchronous Collaboration Practices** - Time-shifted work methods
9. **Hybrid Work Models** - Blending remote and in-person work
10. **Team Dynamics in Virtual Contexts** - Group behavior and cohesion

### Emerging Research Areas

- Artificial intelligence and automated collaboration support
- Brain-computer interfaces for enhanced presence
- Quantum-secure communication for sensitive collaboration
- Decentralized and peer-to-peer collaboration networks
- Holographic and advanced spatial presence
- Neurocognitive aspects of remote presence and embodiment
- Post-pandemic evolution of hybrid work and learning

---

## Authority and Governance

### Domain Coordination

- **Domain Lead**: Telecollaboration Ontology Working Group
- **Authority Score Range**: 0.75-1.0 for validated concepts
- **Review Cycle**: Annual review and update

### Quality Assurance

- All concepts must include definitions and relationships
- Authority scores based on source documentation and validation
- Status tracking from draft through completion
- Regular validation against UNICollaboration and CoI literature

### Contribution Guidelines

Contributions should include:
- Clear definition with examples
- Appropriate semantic classification (owl:class, physicality, role)
- Relevant telecollaboration properties
- Relationships to other concepts
- Documentation of cross-domain bridges
- Source citations and references

---

## Research Sources and References

### Primary Authorities

- **UNICollaboration**: International organization for telecollaboration research
- **Community of Inquiry Framework**: Garrison, Anderson, Archer
- **Stevens Initiative**: Virtual exchange programs
- **Erasmus+ Virtual Exchange**: European research initiatives
- **EVOLVE Project**: European virtual exchange research

### Standards and Frameworks

- **ACCESSIBLE**: Accessibility standards for online learning
- **Quality Matters**: Rubric for distance education design
- **UNESCO**: Guidelines for distance learning and remote education
- **IEEE LTSC**: Learning Technology Standards Committee

### Key Publications

- Garrison, D.R. (2011). E-learning in the 21st century
- Weller, M. (2018). 25 Years of Ed Tech
- Moore, M.G., & Kearsley, G. (2011). Distance Education: A Systems View
- Salmon, G., & Edirisingha, P. (2008). Whitepapers on e-learning
- Garrison, Anderson, & Archer (2010). The first decade of the Community of Inquiry framework

---

## Document Control

**Version**: 2.0.0
**Status**: Active
**Last Updated**: 2025-11-21
**Next Review**: 2026-01-21
**Maintainer**: Telecollaboration Ontology Working Group
**License**: CC BY 4.0

**Revision History:**
- 2.0.0 (2025-11-21): Initial comprehensive domain documentation with 3 example implementations
- 1.0.0 (2025-11-01): Schema definition and property specification

---

## Appendix: Quick Reference

### Common Abbreviations

- **TC**: Telecollaboration
- **CoI**: Community of Inquiry
- **LMS**: Learning Management System
- **IRI**: Internationalized Resource Identifier
- **OWL**: Web Ontology Language
- **RDF**: Resource Description Framework
- **RTP**: Real-time Transport Protocol

### Related Domains

- **RB**: Robotics Domain
- **AI**: Artificial Intelligence Domain
- **MV**: Metaverse Domain
- **DT**: Disruptive Technologies Domain
- **BC**: Blockchain Domain

### Common Links

- Domain Configuration: `/scripts/ontology-migration/domain-config.json`
- Telecollaboration Extension Schema: `/docs/ontology-migration/schemas/domain-extensions/telecollaboration-extension.md`
- Example Files: `/mainKnowledgeGraph/pages/tc-0*.md`
