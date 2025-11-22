- ### OntologyBlock
  id:: bc-0505-carbon-neutral-blockchain-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0505
	- preferred-term:: Carbon Neutral Blockchain
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:CarbonNeutralBlockchain
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :CarbonNeutralBlockchain))

;; Annotations
(AnnotationAssertion rdfs:label :CarbonNeutralBlockchain "Carbon Neutral Blockchain"@en)
(AnnotationAssertion rdfs:comment :CarbonNeutralBlockchain "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :CarbonNeutralBlockchain "BC-0505"^^xsd:string)
```
