- ### OntologyBlock
  id:: virtualworld-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0003
	- preferred-term:: VirtualWorld
	- source-domain:: metaverse
	- status:: active
	- public-access:: true
	- definition:: A computer-generated, immersive 3D environment where users can interact with digital objects and other participants through avatars, forming the foundational space of metaverse experiences.
	- maturity:: mature
	- owl:class:: mv:VirtualWorld
	- owl:physicality:: VirtualEntity
	- owl:role:: Environment
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]
	- #### CrossDomainBridges
		- dt:requires:: [[Physics Engine]]
		- dt:uses:: [[Computer Vision]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Virtualworld))

;; Annotations
(AnnotationAssertion rdfs:label :Virtualworld "VirtualWorld"@en)
(AnnotationAssertion rdfs:comment :Virtualworld "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Virtualworld "mv-1761742247979"^^xsd:string)
```
