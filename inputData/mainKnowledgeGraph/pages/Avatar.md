- ### OntologyBlock
  id:: avatar-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20067
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Avatar
	- definition:: Digital representation of a person or agent used to interact within a virtual environment.
	- maturity:: mature
	- source:: [[ACM + Web3D HAnim]]
	- owl:class:: mv:Avatar
	- owl:physicality:: VirtualEntity
	- owl:role:: Agent
	- owl:inferred-class:: mv:VirtualAgent
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InteractionDomain]]
	- implementedInLayer:: [[UserExperienceLayer]]
	- #### Relationships
	  id:: avatar-relationships
		- is-subclass-of:: [[ArtificialIntelligence]]
		- is-part-of:: [[Metaverse]]
		- has-part:: [[Visual Mesh]], [[Animation Rig]]
		- requires:: [[3D Rendering Engine]]
		- enables:: [[User Embodiment]], [[Social Presence]]
	- #### CrossDomainBridges
		- dt:uses:: [[Computer Vision]]
		- dt:uses:: [[Machine Learning]]
	- #### OWL Axioms
	  id:: avatar-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Avatar))

		  # Classification
		  SubClassOf(mv:Avatar mv:VirtualEntity)
		  SubClassOf(mv:Avatar mv:Agent)

		  # Constraints
		  SubClassOf(mv:Avatar
		    ObjectExactCardinality(1 mv:represents mv:Agent)
		  )

		  # Domain Classification
		  SubClassOf(mv:Avatar
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InteractionDomain)
		  )
		  SubClassOf(mv:Avatar
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:UserExperienceLayer)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
