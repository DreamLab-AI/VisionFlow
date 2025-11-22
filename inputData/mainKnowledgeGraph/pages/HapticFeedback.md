- ### OntologyBlock
  id:: hapticfeedback-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0040
	- preferred-term:: HapticFeedback
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: mv:HapticFeedback
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Blockchain]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Hapticfeedback))

;; Annotations
(AnnotationAssertion rdfs:label :Hapticfeedback "HapticFeedback"@en)
(AnnotationAssertion rdfs:comment :Hapticfeedback "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Hapticfeedback "mv-1761742247930"^^xsd:string)
```
