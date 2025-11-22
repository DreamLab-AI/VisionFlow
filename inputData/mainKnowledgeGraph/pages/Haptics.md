- ### OntologyBlock
  id:: haptics-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20153
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Haptics
	- definition:: Physical hardware systems that simulate tactile sensations and force feedback within virtual environments through actuators and sensors.
	- maturity:: mature
	- source:: [[ISO 9241-960]]
	- owl:class:: mv:Haptics
	- owl:physicality:: PhysicalEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:PhysicalObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InteractionDomain]]
	- implementedInLayer:: [[PhysicalLayer]]
	- #### Relationships
	  id:: haptics-relationships
		- is-subclass-of:: [[Metaverse Infrastructure]]
		- has-part:: [[Force Feedback Actuators]], [[Tactile Actuators]], [[Vibration Motors]], [[Piezoelectric Sensors]]
		- is-part-of:: [[Human Interface Device]]
		- requires:: [[Power Supply]], [[Signal Processing Unit]], [[Driver Software]]
		- depends-on:: [[Real-time Processing]], [[Low Latency Communication]]
		- enables:: [[Tactile Feedback]], [[Force Feedback]], [[Texture Simulation]], [[Physical Presence]]
	- #### OWL Axioms
	  id:: haptics-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Haptics))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Haptics mv:PhysicalEntity)
		  SubClassOf(mv:Haptics mv:Object)

		  # Hardware component requirements
		  SubClassOf(mv:Haptics
		    ObjectSomeValuesFrom(mv:hasPart mv:ForceFeedbackActuators)
		  )
		  SubClassOf(mv:Haptics
		    ObjectSomeValuesFrom(mv:hasPart mv:TactileActuators)
		  )

		  # Domain classification
		  SubClassOf(mv:Haptics
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InteractionDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:Haptics
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:PhysicalLayer)
		  )

		  # Functional requirements
		  SubClassOf(mv:Haptics
		    ObjectSomeValuesFrom(mv:requires mv:RealTimeProcessing)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
