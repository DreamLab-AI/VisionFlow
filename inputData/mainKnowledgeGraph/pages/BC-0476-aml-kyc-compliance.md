- ### OntologyBlock
  id:: bc-0476-aml-kyc-compliance-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0476
	- preferred-term:: AML KYC Compliance
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:AMLKYCCompliance
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :AMLKYCCompliance))

;; Annotations
(AnnotationAssertion rdfs:label :AMLKYCCompliance "AML KYC Compliance"@en)
(AnnotationAssertion rdfs:comment :AMLKYCCompliance "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :AMLKYCCompliance "BC-0476"^^xsd:string)
```
