- ### OntologyBlock
  id:: consensusalgorithm-ontology
  collapsed:: true
	- ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
	- term-id:: BC-0002
	- preferred-term:: ConsensusAlgorithm
	- source-domain:: blockchain
	- status:: active
	- public-access:: true
	- definition:: A protocol used by distributed systems to achieve agreement on a single data value or state among multiple nodes, ensuring network integrity and preventing double-spending.
	- maturity:: mature
	- owl:class:: bc:ConsensusAlgorithm
	- owl:equivalentClass:: mv:ConsensusAlgorithm
	- owl:physicality:: ConceptualEntity
	- owl:role:: Process
	- belongsToDomain:: [[BlockchainDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Consensusalgorithm))

;; Annotations
(AnnotationAssertion rdfs:label :Consensusalgorithm "ConsensusAlgorithm"@en)
(AnnotationAssertion rdfs:comment :Consensusalgorithm "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Consensusalgorithm "mv-1761742247908"^^xsd:string)
```
