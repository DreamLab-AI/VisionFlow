- ### OntologyBlock
  id:: digitalavatar-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0005
	- preferred-term:: DigitalAvatar
	- source-domain:: metaverse
	- status:: active
	- public-access:: true
	- definition:: A virtual representation of a user in digital environments, ranging from simple 2D icons to sophisticated 3D models that can express emotions and interact with virtual worlds.
	- maturity:: mature
	- owl:class:: mv:DigitalAvatar
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- is-subclass-of:: [[ArtificialIntelligence]]
	- belongsToDomain:: [[MetaverseDomain]]
	- #### CrossDomainBridges
		- dt:uses:: [[Computer Vision]]
		- dt:uses:: [[Machine Learning]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Digitalavatar))

;; Annotations
(AnnotationAssertion rdfs:label :Digitalavatar "DigitalAvatar"@en)
(AnnotationAssertion rdfs:comment :Digitalavatar "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Digitalavatar "mv-1761742247916"^^xsd:string)
```
