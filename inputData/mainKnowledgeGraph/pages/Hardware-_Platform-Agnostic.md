- ### OntologyBlock
  id:: hardware-_platform-agnostic-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20055
	- preferred-term:: Hardware _Platform Agnostic
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: capability of a metaverse system to operate across multiple hardware or software platforms without dependency.
maturity:: 3
source:: [[Metaverse 101]]
	- maturity:: draft
	- owl:class:: mv:Hardware_PlatformAgnostic
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :HardwarePlatformAgnostic))

;; Annotations
(AnnotationAssertion rdfs:label :HardwarePlatformAgnostic "Hardware _Platform Agnostic"@en)
(AnnotationAssertion rdfs:comment :HardwarePlatformAgnostic "capability of a metaverse system to operate across multiple hardware or software platforms without dependency."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :HardwarePlatformAgnostic "20055"^^xsd:string)
```
