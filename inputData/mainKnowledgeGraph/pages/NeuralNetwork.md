- ### OntologyBlock
  id:: neuralnetwork-ontology
  collapsed:: true
	- ontology:: true
    - is-subclass-of:: [[ArtificialIntelligenceTechnology]]
	- term-id:: AI-0501
	- preferred-term:: NeuralNetwork
	- source-domain:: artificial-intelligence
	- status:: active
	- public-access:: true
	- definition:: A computational system inspired by biological neural networks, consisting of interconnected nodes that process information through weighted connections to learn patterns and make predictions.
	- maturity:: mature
	- owl:class:: ai:NeuralNetwork
	- owl:equivalentClass:: ai:NeuralNetwork
	- owl:physicality:: ConceptualEntity
	- owl:role:: Model
	- belongsToDomain:: [[AIDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Neuralnetwork))

;; Annotations
(AnnotationAssertion rdfs:label :Neuralnetwork "NeuralNetwork"@en)
(AnnotationAssertion rdfs:comment :Neuralnetwork "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Neuralnetwork "mv-1761742247950"^^xsd:string)
```
