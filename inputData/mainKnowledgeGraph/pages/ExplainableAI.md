- ### OntologyBlock
  id:: explainableai-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0035
	- preferred-term:: ExplainableAI
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: mv:ExplainableAI
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :ExplainableAI))

;; Annotations
(AnnotationAssertion rdfs:label :ExplainableAI "ExplainableAI"@en)
(AnnotationAssertion rdfs:comment :ExplainableAI "Artificial intelligence systems that provide transparent and interpretable explanations of their decision-making processes."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :ExplainableAI "mv-0035"^^xsd:string)
```
