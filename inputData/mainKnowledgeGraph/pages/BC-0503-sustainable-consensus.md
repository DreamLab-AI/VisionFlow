- ### OntologyBlock
  id:: bc-0503-sustainable-consensus-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0503
	- preferred-term:: Sustainable Consensus
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:SustainableConsensus
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Consensus Mechanism]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :SustainableConsensus))

;; Annotations
(AnnotationAssertion rdfs:label :SustainableConsensus "Sustainable Consensus"@en)
(AnnotationAssertion rdfs:comment :SustainableConsensus "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :SustainableConsensus "BC-0503"^^xsd:string)
```
