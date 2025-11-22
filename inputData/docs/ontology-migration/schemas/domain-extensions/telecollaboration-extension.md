# Telecollaboration Domain Extension Schema

**Version:** 2.0.0
**Date:** 2025-11-21
**Domain:** Telecollaboration (tc:)
**Base URI:** `http://narrativegoldmine.com/telecollaboration#`
**Term Prefix:** TC-XXXX

---

## Domain Overview

The Telecollaboration domain covers virtual exchange, intercultural learning, collaborative pedagogy, digital literacy, knowledge creation, and online learning communities. Based on academic frameworks including UNICollaboration, Community of Inquiry model, and virtual exchange research, this domain extension defines telecollaboration-specific properties and patterns that extend the core ontology schema.

---

## Sub-Domains

| Sub-Domain | Namespace | Description | Example Concepts |
|------------|-----------|-------------|------------------|
| Virtual Exchange | `tc:ve:` | Cross-cultural online collaboration programs | COIL, Telecollaborative Projects, International Partnerships |
| Collaborative Pedagogy | `tc:cp:` | Teaching methodologies for online collaboration | Inquiry-Based Learning, Project-Based Learning, Collaborative Inquiry |
| Digital Literacy | `tc:dl:` | Technology and information competencies | Digital Citizenship, Information Literacy, ICT Skills |
| Intercultural Learning | `tc:il:` | Cross-cultural competency development | Cultural Awareness, Global Citizenship, Intercultural Communication |
| Knowledge Creation | `tc:kc:` | Collaborative knowledge building processes | Community of Inquiry, Collective Intelligence, Knowledge Co-Creation |
| Online Communities | `tc:oc:` | Virtual learning and practice communities | Community of Practice, Learning Network, Professional Learning Community |

---

## Telecollaboration-Specific Properties

### Collaboration Properties

**tc:collaboration-type** (enum)
- **Purpose**: Temporal structure of collaboration
- **Values**: synchronous, asynchronous, hybrid
- **Example**: `tc:collaboration-type:: hybrid`

**tc:participant-count** (integer)
- **Purpose**: Number of participants in collaboration
- **Format**: Integer
- **Example**: `tc:participant-count:: 60`

**tc:duration** (string)
- **Purpose**: Length of collaboration period
- **Format**: String with units (weeks, months, semester)
- **Example**: `tc:duration:: 8 weeks`

**tc:cultural-contexts** (page link list)
- **Purpose**: Cultural backgrounds represented
- **Example**: `tc:cultural-contexts:: [[North American]], [[European]], [[Asian]], [[Latin American]], [[African]]`

**tc:languages-used** (list)
- **Purpose**: Languages employed in collaboration
- **Example**: `tc:languages-used:: English, Spanish, Mandarin, French`

**tc:geographic-distribution** (list)
- **Purpose**: Geographic locations of participants
- **Example**: `tc:geographic-distribution:: United States, Spain, China, Mexico, Kenya`

**tc:partner-institutions** (page link list)
- **Purpose**: Educational institutions involved
- **Example**: `tc:partner-institutions:: [[University A]], [[University B]], [[School C]]`

### Pedagogical Properties

**tc:learning-model** (page link list)
- **Purpose**: Theoretical learning framework(s)
- **Example**: `tc:learning-model:: [[Community of Inquiry]], [[Constructivism]], [[Connectivism]], [[CSCL]]`

**tc:instructional-design** (page link list)
- **Purpose**: Pedagogical approach to instruction
- **Example**: `tc:instructional-design:: [[Project-Based Learning]], [[Task-Based Learning]], [[Problem-Based Learning]], [[Case-Based Learning]]`

**tc:assessment-methods** (page link list)
- **Purpose**: Methods for evaluating learning
- **Example**: `tc:assessment-methods:: [[Peer Assessment]], [[Portfolio Assessment]], [[Reflection]], [[Rubric-Based Evaluation]]`

**tc:transversal-skills** (page link list)
- **Purpose**: Cross-cutting competencies developed
- **Example**: `tc:transversal-skills:: [[Critical Thinking]], [[Communication]], [[Collaboration]], [[Creativity]], [[Problem Solving]]`

**tc:intercultural-competencies** (page link list)
- **Purpose**: Intercultural abilities fostered
- **Example**: `tc:intercultural-competencies:: [[Cultural Self-Awareness]], [[Perspective Taking]], [[Empathy]], [[Intercultural Communication]], [[Global Citizenship]]`

**tc:discipline-focus** (list)
- **Purpose**: Academic disciplines involved
- **Example**: `tc:discipline-focus:: Language Education, Business, Engineering, Social Sciences`

### Technology Properties

**tc:platform-used** (page link list)
- **Purpose**: Primary communication/collaboration platforms
- **Example**: `tc:platform-used:: [[Zoom]], [[Microsoft Teams]], [[Moodle]], [[Canvas]], [[Custom LMS]]`

**tc:collaboration-tools** (page link list)
- **Purpose**: Specific tools for collaboration
- **Example**: `tc:collaboration-tools:: [[Google Docs]], [[Padlet]], [[Flipgrid]], [[Miro]], [[Wiki]], [[Discussion Forum]]`

**tc:semantic-interoperability** (boolean)
- **Purpose**: Systems use semantic standards for data exchange
- **Values**: true, false
- **Example**: `tc:semantic-interoperability:: true`

**tc:data-exchange-format** (list)
- **Purpose**: Data formats for system integration
- **Example**: `tc:data-exchange-format:: JSON, XML, RDF, IMS LTI, xAPI`

**tc:accessibility-features** (page link list)
- **Purpose**: Accessibility accommodations provided
- **Example**: `tc:accessibility-features:: [[Closed Captions]], [[Screen Reader Support]], [[Translation]], [[Alternative Formats]]`

### Outcomes Properties

**tc:knowledge-creation-approach** (page link list)
- **Purpose**: Epistemological approach to knowledge building
- **Example**: `tc:knowledge-creation-approach:: [[Social Constructivism]], [[Connectivism]], [[Constructionism]]`

**tc:digital-literacy-outcomes** (page link list)
- **Purpose**: Digital competencies achieved
- **Example**: `tc:digital-literacy-outcomes:: [[Information Literacy]], [[Media Literacy]], [[ICT Skills]], [[Digital Citizenship]], [[Online Collaboration]]`

**tc:social-presence** (enum)
- **Purpose**: Level of social presence in Community of Inquiry
- **Values**: low, medium, high
- **Example**: `tc:social-presence:: high`

**tc:cognitive-presence** (enum)
- **Purpose**: Level of cognitive engagement in Community of Inquiry
- **Values**: low, medium, high
- **Example**: `tc:cognitive-presence:: high`

**tc:teaching-presence** (enum)
- **Purpose**: Level of teaching/facilitation presence
- **Values**: low, medium, high
- **Example**: `tc:teaching-presence:: medium`

**tc:learning-outcomes** (page link list)
- **Purpose**: Specific learning outcomes achieved
- **Example**: `tc:learning-outcomes:: [[Intercultural Competence]], [[Language Proficiency]], [[Digital Skills]], [[Global Awareness]]`

### Program Management Properties

**tc:facilitation-model** (enum)
- **Purpose**: How collaboration is facilitated
- **Values**: instructor-led, student-led, peer-facilitated, minimal-facilitation
- **Example**: `tc:facilitation-model:: instructor-led`

**tc:institutional-support** (page link list)
- **Purpose**: Institutional support provided
- **Example**: `tc:institutional-support:: [[Faculty Training]], [[Technical Support]], [[Funding]], [[Recognition]]`

**tc:sustainability-approach** (string)
- **Purpose**: Strategy for program continuation
- **Example**: `tc:sustainability-approach:: Embedded in curriculum, faculty champions, institutional partnership`

---

## Telecollaboration-Specific Relationships

### Pedagogical Relationships

**tc:facilitates** (page link list)
- **Purpose**: Learning outcomes or competencies facilitated
- **Example**: `tc:facilitates:: [[Intercultural Competence Development]], [[Language Learning]], [[Critical Thinking]]`

**tc:involves-cultures** (page link list)
- **Purpose**: Cultural groups engaged
- **Example**: `tc:involves-cultures:: [[Western Culture]], [[Eastern Culture]], [[Indigenous Cultures]], [[Global South]]`

**tc:supports-pedagogy** (page link list)
- **Purpose**: Pedagogical approaches supported or enabled
- **Example**: `tc:supports-pedagogy:: [[Constructivism]], [[Experiential Learning]], [[Collaborative Learning]]`

### Technology Relationships

**tc:uses-technology** (page link list)
- **Purpose**: Technologies employed
- **Example**: `tc:uses-technology:: [[Video Conferencing]], [[Collaborative Writing]], [[LMS]], [[Translation Tools]]`

**tc:integrates-with** (page link list)
- **Purpose**: Systems or platforms integrated
- **Example**: `tc:integrates-with:: [[Student Information System]], [[Learning Analytics]], [[Badge Platform]]`

### Outcome Relationships

**tc:develops-competency** (page link list)
- **Purpose**: Competencies being developed
- **Example**: `tc:develops-competency:: [[Cross-Cultural Communication]], [[Digital Collaboration]], [[Critical Reflection]]`

**tc:creates-knowledge** (page link list)
- **Purpose**: Knowledge artifacts or products created
- **Example**: `tc:creates-knowledge:: [[Collaborative Reports]], [[Intercultural Case Studies]], [[Digital Portfolios]], [[Multimedia Projects]]`

**tc:builds-community** (page link list)
- **Purpose**: Types of communities formed
- **Example**: `tc:builds-community:: [[Learning Community]], [[Community of Practice]], [[Professional Network]]`

---

## Extended Template for Telecollaboration Domain

```markdown
- ### [Telecollaboration Concept Name]
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
    - quality-score:: [0.0-1.0]
    - cross-domain-links:: [number]

  - **Definition** [CORE - Tier 1]
    - definition:: [2-5 sentence comprehensive definition with [[concept links]]]
    - maturity:: [draft | emerging | mature | established]
    - source:: [[UNICollaboration]], [[Academic Source]], etc.
    - authority-score:: [0.0-1.0]

  - **Semantic Classification** [CORE - Tier 1]
    - owl:class:: tc:[ClassName]
    - owl:physicality:: [VirtualEntity | AbstractEntity most common]
    - owl:role:: [Process | Concept | Agent]
    - owl:inferred-class:: tc:[PhysicalityRole]
    - belongsToDomain:: [[EducationDomain]], [[CommunicationDomain]]
    - belongsToSubDomain:: [[Virtual Exchange]], [[Collaborative Pedagogy]], etc.

  - **Collaboration Properties** [TC EXTENSION]
    - tc:collaboration-type:: [synchronous | asynchronous | hybrid]
    - tc:participant-count:: [integer]
    - tc:duration:: [string with units]
    - tc:cultural-contexts:: [[Context1]], [[Context2]]
    - tc:languages-used:: [list of languages]
    - tc:geographic-distribution:: [list of locations]

  - **Pedagogical Properties** [TC EXTENSION]
    - tc:learning-model:: [[Model1]], [[Model2]]
    - tc:instructional-design:: [[Approach1]], [[Approach2]]
    - tc:assessment-methods:: [[Method1]], [[Method2]]
    - tc:transversal-skills:: [[Skill1]], [[Skill2]]
    - tc:intercultural-competencies:: [[Competency1]], [[Competency2]]

  - **Technology Properties** [TC EXTENSION]
    - tc:platform-used:: [[Platform1]], [[Platform2]]
    - tc:collaboration-tools:: [[Tool1]], [[Tool2]]
    - tc:semantic-interoperability:: [true | false]
    - tc:accessibility-features:: [[Feature1]], [[Feature2]]

  - **Outcomes Properties** [TC EXTENSION]
    - tc:knowledge-creation-approach:: [[Approach1]], [[Approach2]]
    - tc:digital-literacy-outcomes:: [[Outcome1]], [[Outcome2]]
    - tc:social-presence:: [low | medium | high]
    - tc:cognitive-presence:: [low | medium | high]
    - tc:teaching-presence:: [low | medium | high]

  - #### Relationships [CORE - Tier 1]
    id:: tc-[concept-slug]-relationships

    - is-subclass-of:: [[ParentClass1]], [[ParentClass2]]
    - has-part:: [[Component1]], [[Component2]]
    - requires:: [[Requirement1]]
    - enables:: [[Capability1]]

  - #### Telecollaboration-Specific Relationships [TC EXTENSION]
    - tc:facilitates:: [[Outcome1]], [[Outcome2]]
    - tc:involves-cultures:: [[Culture1]], [[Culture2]]
    - tc:uses-technology:: [[Technology1]], [[Technology2]]
    - tc:develops-competency:: [[Competency1]], [[Competency2]]
    - tc:supports-pedagogy:: [[Pedagogy1]], [[Pedagogy2]]
    - tc:creates-knowledge:: [[Artifact1]], [[Artifact2]]

  - #### CrossDomainBridges [CORE - Tier 3]
    - bridges-to:: [[AI Translation]] via uses (TC → AI)
    - bridges-to:: [[Virtual Collaboration Space]] via requires (TC → MV)
    - bridges-to:: [[Platform Innovation]] via exemplifies (TC → DT)
```

---

## Common Telecollaboration Patterns

### Pattern 1: Virtual Exchange Program

```markdown
- ### [Program Name]
  - **Semantic Classification**
    - owl:class:: tc:[ProgramName]
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - belongsToSubDomain:: [[Virtual Exchange]]

  - **Collaboration Properties**
    - tc:collaboration-type:: hybrid
    - tc:participant-count:: [number]
    - tc:cultural-contexts:: [[Multiple cultures]]
    - tc:duration:: [weeks/months]

  - #### Relationships
    - is-subclass-of:: [[Virtual Exchange]], [[Collaborative Learning]]
    - tc:facilitates:: [[Intercultural Competence]], [[Language Learning]]
```

### Pattern 2: Collaborative Pedagogy Method

```markdown
- ### [Pedagogy Name]
  - **Semantic Classification**
    - owl:class:: tc:[PedagogyName]
    - owl:physicality:: AbstractEntity
    - owl:role:: Concept
    - belongsToSubDomain:: [[Collaborative Pedagogy]]

  - **Pedagogical Properties**
    - tc:learning-model:: [[Theoretical Framework]]
    - tc:instructional-design:: [[Design Approach]]
    - tc:transversal-skills:: [[Skills Developed]]

  - #### Relationships
    - is-subclass-of:: [[Pedagogical Approach]], [[Collaborative Learning]]
    - tc:supports-pedagogy:: [[Related Pedagogies]]
```

### Pattern 3: Digital Literacy Competency

```markdown
- ### [Competency Name]
  - **Semantic Classification**
    - owl:class:: tc:[CompetencyName]
    - owl:physicality:: AbstractEntity
    - owl:role:: Quality
    - belongsToSubDomain:: [[Digital Literacy]]

  - **Outcomes Properties**
    - tc:digital-literacy-outcomes:: [[Specific Skills]]

  - #### Relationships
    - is-subclass-of:: [[Digital Literacy]], [[21st Century Skill]]
    - is-part-of:: [[Digital Competence Framework]]
```

---

## Cross-Domain Bridge Patterns

### TC → AI

```markdown
- bridges-to:: [[Real-Time Translation]] via uses (TC → AI)
- bridges-to:: [[Sentiment Analysis]] via uses (TC → AI)
- bridges-to:: [[Automated Facilitation]] via enabled-by (TC → AI)
- bridges-to:: [[Learning Analytics]] via uses (TC → AI)
```

### TC → Metaverse

```markdown
- bridges-to:: [[Virtual Collaboration Space]] via requires (TC → MV)
- bridges-to:: [[Virtual Classroom]] via hosted-in (TC → MV)
- bridges-to:: [[Immersive Learning Environment]] via uses (TC → MV)
```

### TC → Disruptive Technologies

```markdown
- bridges-to:: [[Disruptive Education Model]] via exemplifies (TC → DT)
- bridges-to:: [[Platform-Based Learning]] via implements (TC → DT)
- bridges-from:: [[Innovation Assessment]] via evaluated-by (DT → TC)
```

### TC → Blockchain

```markdown
- bridges-to:: [[Credential Verification]] via uses (TC → BC)
- bridges-to:: [[Digital Badges]] via secured-by (TC → BC)
- bridges-to:: [[Decentralized Learning Records]] via stores-in (TC → BC)
```

---

## Validation Rules for Telecollaboration Domain

### TC-Specific Validations

1. **Collaboration Consistency**
   - If cultural-contexts specified, should have multiple cultures
   - Participant count should align with program scale

2. **Pedagogical Alignment**
   - Learning model should align with instructional design
   - Assessment methods should match learning outcomes

3. **Community of Inquiry**
   - If CoI model used, should specify social, cognitive, teaching presence
   - Presence levels should be consistent with program goals

4. **Cultural Sensitivity**
   - Multiple cultural contexts should be represented in virtual exchange
   - Intercultural competencies should align with cultural contexts

---

## Migration Notes

### Creating Telecollaboration Blocks (New Domain)

1. **Identify TC Concepts** from existing education/collaboration content
2. **Add TC Properties** to collaborative learning concepts
3. **Specify Sub-Domain** for all telecollaboration concepts
4. **Add tc:facilitates** relationships for learning outcomes
5. **Document Cross-Domain** connections to AI, MV, DT

### Priority Telecollaboration Concepts to Create

- Virtual Exchange Programs (COIL, etc.)
- Community of Inquiry Model
- Intercultural Competence Development
- Digital Literacy Frameworks
- Collaborative Pedagogy Approaches
- Online Learning Communities
- Telecollaboration Platforms and Tools

---

## Research Sources

- **UNICollaboration**: Cross-disciplinary organization for telecollaboration and virtual exchange
- **Community of Inquiry Framework**: Garrison, Anderson, and Archer
- **EVOLVE Project**: European virtual exchange research
- **Stevens Initiative**: Virtual exchange programs
- **Erasmus+ Virtual Exchange**: European Commission virtual exchange initiative

---

**Document Control:**
- **Version**: 2.0.0
- **Status**: Authoritative
- **Domain Coordinator**: TBD
- **Last Updated**: 2025-11-21
- **Next Review**: 2026-01-21
