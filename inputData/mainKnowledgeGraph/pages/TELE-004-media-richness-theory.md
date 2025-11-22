# Media Richness Theory

- ### OntologyBlock
  id:: media-richness-theory-ontology
  collapsed:: true
  - ontology:: true
  - term-id:: TELE-004
  - preferred-term:: Media Richness Theory
  - alternate-terms::
  - MRT
  - Information Richness Theory
  - Channel Richness Theory
  - source-domain:: tele
  - status:: active
  - public-access:: true
  - definition:: "A theoretical framework positing that communication media vary in their capacity to convey rich information through multiple cues, immediate feedback, language variety, and personal focus, predicting that richer media enable more effective communication of complex, ambiguous information whilst leaner media suffice for routine, unambiguous messages."
  - maturity:: mature
  - authority-score:: 0.93
  - owl:class:: tele:MediaRichnessTheory
  - owl:physicality:: ConceptualEntity
  - owl:role:: Concept
  - belongsToDomain::
  - [[TELE-0000-telepresence-domain]]
  - [[TelepresenceFoundations]]
  - bridges-to::
  - [[AIDomain]]


#### Relationships
id:: media-richness-theory-relationships
- is-subclass-of:: [[CommunicationTheory]], [[OrganisationalTheory]]
- enables:: [[MediaSelection]], [[TaskMediaFit]], [[CommunicationEffectiveness]]
- related-to:: [[TELE-003-social-presence-theory]], [[TELE-001-telepresence]], [[InformationProcessing]]

#### OWL Axioms
id:: media-richness-theory-owl-axioms
collapsed:: true
- ```clojure
  Declaration(Class(tele:MediaRichnessTheory))

  SubClassOf(tele:MediaRichnessTheory tele:FoundationalConcept)
  SubClassOf(tele:MediaRichnessTheory tele:ConceptualEntity)

  SubClassOf(tele:MediaRichnessTheory
    ObjectSomeValuesFrom(tele:belongsToDomain tele:TelecollaborationDomain)
  )

  AnnotationAssertion(rdfs:label tele:MediaRichnessTheory "Media Richness Theory"@en-GB)
  AnnotationAssertion(rdfs:comment tele:MediaRichnessTheory "Theory of communication media information-carrying capacity"@en-GB)
  AnnotationAssertion(dcterms:identifier tele:MediaRichnessTheory "TELE-004"^^xsd:string)
  AnnotationAssertion(dcterms:created tele:MediaRichnessTheory "2025-11-16"^^xsd:date)
  ```

## Definition

**Media Richness Theory** (MRT), formulated by Daft and Lengel (1986), classifies communication media along a spectrum from "lean" (low information-carrying capacity) to "rich" (high capacity) based on four criteria: ability to transmit multiple cues simultaneously, enable rapid feedback, support natural language variety, and convey personal focus. The theory predicts that task-media fit determines communication effectiveness: equivocal (ambiguous, interpretive) tasks require rich media like face-to-face meetings or [[TELE-020-virtual-reality-telepresence]], whilst routine data transmission succeeds with lean media like email or forms.

MRT identifies face-to-face communication as richest (multimodal cues, instant feedback, nuanced language, personalised), followed descending by video calls, telephone, email, and formal documents (leanest). Modern telepresence technologies ([[TELE-001-telepresence]]) approach face-to-face richness through immersive environments, spatial audio, and photorealistic avatars ([[TELE-100-ai-avatars]]), enabling effective communication of complex, emotionally charged, or culturally nuanced information across distance.

## Media Richness Hierarchy (2025 Updated)

### Richest Media
1. **Face-to-Face** (benchmark: 100% richness)
2. **Immersive VR Telepresence** ([[TELE-020-virtual-reality-telepresence]]): 85-90% richness
3. **Video Conferencing** (Zoom, Teams): 70-75%
4. **Telephone/VoIP**: 50-60%
5. **Instant Messaging**: 30-40%
6. **Email**: 20-30%
7. **Formal Documents**: 10-20%

### Leanest Media

## Related Concepts

- [[TELE-003-social-presence-theory]]
- [[TELE-001-telepresence]]
- [[TELE-020-virtual-reality-telepresence]]

## References

1. Daft, R. L., & Lengel, R. H. (1986). "Organizational Information Requirements, Media Richness and Structural Design". *Management Science*, 32(5), 554-571.

## Metadata

- **Term-ID**: TELE-004
- **Last Updated**: 2025-11-16
- **Maturity**: Mature
- **Authority Score**: 0.93
