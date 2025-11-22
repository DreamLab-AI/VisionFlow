- ### OntologyBlock
  id:: softrobotics-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0056
	- preferred-term:: SoftRobotics
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: mv:SoftRobotics
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Softrobotics))

;; Annotations
(AnnotationAssertion rdfs:label :Softrobotics "SoftRobotics"@en)
(AnnotationAssertion rdfs:comment :Softrobotics "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Softrobotics "mv-1761742247969"^^xsd:string)
```
