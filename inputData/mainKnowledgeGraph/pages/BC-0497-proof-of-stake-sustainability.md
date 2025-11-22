- ### OntologyBlock
  id:: bc-0497-proof-of-stake-sustainability-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0497
	- preferred-term:: Proof Of Stake Sustainability
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:ProofOfStakeSustainability
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :ProofOfStakeSustainability))

;; Annotations
(AnnotationAssertion rdfs:label :ProofOfStakeSustainability "Proof Of Stake Sustainability"@en)
(AnnotationAssertion rdfs:comment :ProofOfStakeSustainability "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :ProofOfStakeSustainability "BC-0497"^^xsd:string)
```
