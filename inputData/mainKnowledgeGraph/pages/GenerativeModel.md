- ### OntologyBlock
  id:: generativemodel-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0037
	- preferred-term:: GenerativeModel
	- source-domain:: artificial-intelligence
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: ai:GenerativeModel
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- belongsToDomain:: [[AIDomain]]


### Relationships
- is-subclass-of:: [[DeepLearning]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :GenerativeModel))

;; Annotations
(AnnotationAssertion rdfs:label :GenerativeModel "GenerativeModel"@en)
(AnnotationAssertion rdfs:comment :GenerativeModel "Machine learning models capable of generating new content such as images, text, audio, or 3D models based on training data."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :GenerativeModel "ai-0037"^^xsd:string)
```
