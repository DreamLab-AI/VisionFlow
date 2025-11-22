- ### OntologyBlock
  id:: bc-0469-snapshot-voting-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0469
	- preferred-term:: Snapshot Voting
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:SnapshotVoting
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[On-Chain Voting]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :SnapshotVoting))

;; Annotations
(AnnotationAssertion rdfs:label :SnapshotVoting "Snapshot Voting"@en)
(AnnotationAssertion rdfs:comment :SnapshotVoting "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :SnapshotVoting "BC-0469"^^xsd:string)
```
