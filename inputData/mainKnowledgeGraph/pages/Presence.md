- ### OntologyBlock
  id:: presence-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20256
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Presence
	- definition:: Perceptual state in which a user feels located inside a virtual or mixed environment, experiencing spatial, social, and self presence.
	- maturity:: mature
	- source:: [[ACM]]
	- owl:class:: mv:Presence
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InteractionDomain]]
	- implementedInLayer:: [[ComputeLayer]]
	- #### Relationships
	  id:: presence-relationships
		- is-subclass-of:: [[Extended Reality (XR)]]
		- is-enabled-by:: [[Extended Reality (XR)]], [[Immersion]], [[Human Interface Layer (HIL)]], [[Experience Layer]]
		- has-part:: [[Spatial Presence]], [[Social Presence]], [[Self Presence]]
		- is-part-of:: [[Immersive Experience]]
		- requires:: [[Sensory Feedback]], [[Haptic Device]], [[Visual Display]]
		- depends-on:: [[Latency]], [[Frame Rate]], [[Field of View]]
		- enables:: [[Engagement]], [[Embodiment]], [[Social Connection]]
	- #### OWL Axioms
	  id:: presence-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Presence))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Presence mv:VirtualEntity)
		  SubClassOf(mv:Presence mv:Object)

		  # Presence is a psychological construct that can be measured
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:hasMeasurableAttribute mv:PresenceMetric)
		  )

		  # Presence has distinct dimensional components
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:hasPart mv:SpatialPresence)
		  )

		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:hasPart mv:SocialPresence)
		  )

		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:hasPart mv:SelfPresence)
		  )

		  # Presence requires sensory feedback to manifest
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:requires mv:SensoryFeedback)
		  )

		  # Presence depends on technical performance factors
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:dependsOn mv:Latency)
		  )

		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:dependsOn mv:FrameRate)
		  )

		  # Presence enables immersive experiences
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:enables mv:Immersion)
		  )

		  # Presence is part of immersive experiences
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:isPartOf mv:ImmersiveExperience)
		  )

		  # Domain classification
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InteractionDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:Presence
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ComputeLayer)
		  )

  # Property characteristics
  AsymmetricObjectProperty(dt:isenabledby)

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
