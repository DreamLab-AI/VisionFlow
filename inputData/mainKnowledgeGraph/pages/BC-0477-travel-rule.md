- ### OntologyBlock
  id:: bc-0477-travel-rule-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0477
	- preferred-term:: Travel Rule
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:TravelRule
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :TravelRule))

;; Annotations
(AnnotationAssertion rdfs:label :TravelRule "Travel Rule"@en)
(AnnotationAssertion rdfs:comment :TravelRule "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :TravelRule "BC-0477"^^xsd:string)
```
