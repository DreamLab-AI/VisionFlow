- ### OntologyBlock
  id:: socialvr-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: MV-0055
	- preferred-term:: SocialVR
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- definition:: A component of the metaverse ecosystem.
	- maturity:: draft
	- owl:class:: mv:SocialVR
	- owl:physicality:: ConceptualEntity
	- owl:role:: Concept
	- is-subclass-of:: [[Extended Reality (XR)]]
	- belongsToDomain:: [[MetaverseDomain]]

## OWL Formal Semantics

```clojure
;; OWL Functional Syntax

(Declaration (Class :Socialvr))

;; Annotations
(AnnotationAssertion rdfs:label :Socialvr "SocialVR"@en)
(AnnotationAssertion rdfs:comment :Socialvr "A component of the metaverse ecosystem."@en)

;; Data Properties
(AnnotationAssertion dcterms:identifier :Socialvr "mv-1761742247969"^^xsd:string)
```
