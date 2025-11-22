- ### OntologyBlock
  id:: bc-0498-carbon-footprint-measurement-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0498
	- preferred-term:: Carbon Footprint Measurement
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:CarbonFootprintMeasurement
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :CarbonFootprintMeasurement))

;; Annotations
(AnnotationAssertion rdfs:label :CarbonFootprintMeasurement "Carbon Footprint Measurement"@en)
(AnnotationAssertion rdfs:comment :CarbonFootprintMeasurement "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :CarbonFootprintMeasurement "BC-0498"^^xsd:string)
```
