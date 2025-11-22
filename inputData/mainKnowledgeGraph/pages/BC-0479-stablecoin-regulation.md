- ### OntologyBlock
  id:: bc-0479-stablecoin-regulation-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0479
	- preferred-term:: Stablecoin Regulation
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:StablecoinRegulation
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :StablecoinRegulation))

;; Annotations
(AnnotationAssertion rdfs:label :StablecoinRegulation "Stablecoin Regulation"@en)
(AnnotationAssertion rdfs:comment :StablecoinRegulation "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :StablecoinRegulation "BC-0479"^^xsd:string)
```
