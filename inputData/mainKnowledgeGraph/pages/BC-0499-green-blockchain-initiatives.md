- ### OntologyBlock
  id:: bc-0499-green-blockchain-initiatives-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0499
	- preferred-term:: Green Blockchain Initiatives
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:GreenBlockchainInitiatives
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :GreenBlockchainInitiatives))

;; Annotations
(AnnotationAssertion rdfs:label :GreenBlockchainInitiatives "Green Blockchain Initiatives"@en)
(AnnotationAssertion rdfs:comment :GreenBlockchainInitiatives "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :GreenBlockchainInitiatives "BC-0499"^^xsd:string)
```
