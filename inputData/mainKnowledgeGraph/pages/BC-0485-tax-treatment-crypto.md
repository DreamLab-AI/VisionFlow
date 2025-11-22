- ### OntologyBlock
  id:: bc-0485-tax-treatment-crypto-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: BC-0485
	- preferred-term:: Tax Treatment Crypto
	- source-domain:: blockchain
	- status:: stub-needs-content
	- content-status:: minimal-placeholder-requires-authoring
	- definition:: A component of the blockchain ecosystem.
	- maturity:: draft
	- owl:class:: bc:TaxTreatmentCrypto
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[BlockchainDomain]]

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Compliance]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :TaxTreatmentCrypto))

;; Annotations
(AnnotationAssertion rdfs:label :TaxTreatmentCrypto "Tax Treatment Crypto"@en)
(AnnotationAssertion rdfs:comment :TaxTreatmentCrypto "A component of the blockchain ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :TaxTreatmentCrypto "BC-0485"^^xsd:string)
```
