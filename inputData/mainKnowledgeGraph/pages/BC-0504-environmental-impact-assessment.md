- ### OntologyBlock
  id:: bc-0504-environmental-impact-assessment-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0504
	- preferred-term:: Environmental Impact Assessment
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:EnvironmentalImpactAssessment
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :EnvironmentalImpactAssessment))

;; Annotations
(AnnotationAssertion rdfs:label :EnvironmentalImpactAssessment "Environmental Impact Assessment"@en)
(AnnotationAssertion rdfs:comment :EnvironmentalImpactAssessment "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :EnvironmentalImpactAssessment "BC-0504"^^xsd:string)
```
