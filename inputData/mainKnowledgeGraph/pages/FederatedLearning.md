- ### OntologyBlock
  id:: federatedlearning-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: AI-0504
	- preferred-term:: FederatedLearning
	- source-domain:: artificial-intelligence
	- status:: active
	- public-access:: true
	- definition:: A distributed machine learning approach where models are trained across multiple decentralized devices or servers holding local data, without exchanging raw data to preserve privacy.
	- maturity:: mature
	- owl:class:: ai:FederatedLearning
	- owl:equivalentClass:: mv:FederatedLearning
	- owl:physicality:: ConceptualEntity
	- owl:role:: Technique
	- belongsToDomain:: [[AIDomain]]


### Relationships
- is-subclass-of:: [[MachineLearning]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :FederatedLearning))

;; Annotations
(AnnotationAssertion rdfs:label :FederatedLearning "FederatedLearning"@en)
(AnnotationAssertion rdfs:comment :FederatedLearning "A distributed machine learning approach where models are trained across multiple decentralized devices or servers holding local data, without exchanging raw data to preserve privacy."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :FederatedLearning "ai-0504"^^xsd:string)
```
