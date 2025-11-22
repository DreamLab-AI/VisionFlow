- ### OntologyBlock
  id:: bc-0501-esg-reporting-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0501
	- preferred-term:: ESG Reporting
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:ESGReporting
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Sustainability]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :ESGReporting))

;; Annotations
(AnnotationAssertion rdfs:label :ESGReporting "ESG Reporting"@en)
(AnnotationAssertion rdfs:comment :ESGReporting "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :ESGReporting "BC-0501"^^xsd:string)
```
