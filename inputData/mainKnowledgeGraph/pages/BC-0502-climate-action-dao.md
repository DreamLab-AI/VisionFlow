- ### OntologyBlock
  id:: bc-0502-climate-action-dao-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0502
	- preferred-term:: Climate Action DAO
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:ClimateActionDAO
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Decentralized Autonomous Organization]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :ClimateActionDAO))

;; Annotations
(AnnotationAssertion rdfs:label :ClimateActionDAO "Climate Action DAO"@en)
(AnnotationAssertion rdfs:comment :ClimateActionDAO "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :ClimateActionDAO "BC-0502"^^xsd:string)
```
