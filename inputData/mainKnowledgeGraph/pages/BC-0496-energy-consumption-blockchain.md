- ### OntologyBlock
  id:: bc-0496-energy-consumption-blockchain-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0496
	- preferred-term:: Energy Consumption Blockchain
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:EnergyConsumptionBlockchain
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :EnergyConsumptionBlockchain))

;; Annotations
(AnnotationAssertion rdfs:label :EnergyConsumptionBlockchain "Energy Consumption Blockchain"@en)
(AnnotationAssertion rdfs:comment :EnergyConsumptionBlockchain "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :EnergyConsumptionBlockchain "BC-0496"^^xsd:string)
```
