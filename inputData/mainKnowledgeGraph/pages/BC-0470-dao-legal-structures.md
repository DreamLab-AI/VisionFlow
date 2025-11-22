- ### OntologyBlock
  id:: bc-0470-dao-legal-structures-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0470
	- preferred-term:: DAO Legal Structures
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:DAOLegalStructures
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Decentralized Autonomous Organization]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :DAOLegalStructures))

;; Annotations
(AnnotationAssertion rdfs:label :DAOLegalStructures "DAO Legal Structures"@en)
(AnnotationAssertion rdfs:comment :DAOLegalStructures "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :DAOLegalStructures "BC-0470"^^xsd:string)
```
