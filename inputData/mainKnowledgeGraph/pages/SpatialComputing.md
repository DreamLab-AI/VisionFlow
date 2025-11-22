- ### OntologyBlock
  id:: spatialcomputing-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0006
	- preferred-term:: SpatialComputing
	- source-domain:: metaverse
	- status:: active
	- public-access:: true
	- definition:: Computing paradigm that blends digital content with the physical world, enabling natural interaction with 3D data through gestures, voice, and gaze in physical space.
	- maturity:: mature
	- owl:class:: mv:SpatialComputing
	- owl:physicality:: Technology
	- owl:role:: System
	- is-subclass-of:: [[Metaverse]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Spatialcomputing))

;; Annotations
(AnnotationAssertion rdfs:label :Spatialcomputing "SpatialComputing"@en)
(AnnotationAssertion rdfs:comment :Spatialcomputing "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Spatialcomputing "mv-1761742247973"^^xsd:string)
```
