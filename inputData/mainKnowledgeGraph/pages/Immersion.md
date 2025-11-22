- ### OntologyBlock
  id:: immersion-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20255
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Immersion
	- definition:: Subjective experience of psychological engagement and sense of presence within a virtual environment, characterized by reduced awareness of physical surroundings and absorption in virtual context.
	- maturity:: mature
	- source:: [[ACM]]
	- owl:class:: mv:Immersion
	- owl:physicality:: VirtualEntity
	- owl:role:: Process
	- owl:inferred-class:: mv:VirtualProcess
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InteractionDomain]]
	- implementedInLayer:: [[ApplicationLayer]], [[PresentationLayer]]
	- #### Relationships
	  id:: immersion-relationships
		- is-subclass-of:: [[Metaverse Infrastructure]]
		- has-part:: [[Sensory Immersion]], [[Emotional Immersion]], [[Cognitive Immersion]]
		- requires:: [[Display Technology]], [[Interaction Mechanism]], [[Content Quality]]
		- enables:: [[Presence]], [[Flow State]], [[User Engagement]]
		- depends-on:: [[Visual Fidelity]], [[Audio Spatialization]], [[Haptic Feedback]]
	- #### OWL Axioms
	  id:: immersion-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Immersion))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Immersion mv:VirtualEntity)
		  SubClassOf(mv:Immersion mv:Process)

		  # Compositional structure - three primary dimensions
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:hasPart mv:SensoryImmersion)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:hasPart mv:EmotionalImmersion)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:hasPart mv:CognitiveImmersion)
		  )

		  # Domain classification
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InteractionDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:PresentationLayer)
		  )

		  # Technical requirements
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:requires mv:DisplayTechnology)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:requires mv:InteractionMechanism)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:requires mv:ContentQuality)
		  )

		  # Quality dependencies
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:dependsOn mv:VisualFidelity)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:dependsOn mv:AudioSpatialization)
		  )

		  # Enabled experiences
		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:enables mv:Presence)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:enables mv:FlowState)
		  )

		  SubClassOf(mv:Immersion
		    ObjectSomeValuesFrom(mv:enables mv:UserEngagement)
		  )

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)
```
