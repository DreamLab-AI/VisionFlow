- ### OntologyBlock
  id:: bc-0478-securities-regulation-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0478
	- preferred-term:: Securities Regulation
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:SecuritiesRegulation
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :SecuritiesRegulation))

;; Annotations
(AnnotationAssertion rdfs:label :SecuritiesRegulation "Securities Regulation"@en)
(AnnotationAssertion rdfs:comment :SecuritiesRegulation "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :SecuritiesRegulation "BC-0478"^^xsd:string)
```
