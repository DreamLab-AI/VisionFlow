- ### OntologyBlock
  id:: bc-0480-cbdc-frameworks-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0480
	- preferred-term:: CBDC Frameworks
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:CBDCFrameworks
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :CBDCFrameworks))

;; Annotations
(AnnotationAssertion rdfs:label :CBDCFrameworks "CBDC Frameworks"@en)
(AnnotationAssertion rdfs:comment :CBDCFrameworks "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :CBDCFrameworks "BC-0480"^^xsd:string)
```
