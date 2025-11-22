- ### OntologyBlock
  id:: slam-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0052
	- preferred-term:: SLAM
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: mv:SLAM
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Slam))

;; Annotations
(AnnotationAssertion rdfs:label :Slam "SLAM"@en)
(AnnotationAssertion rdfs:comment :Slam "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Slam "mv-1761742247966"^^xsd:string)
```
