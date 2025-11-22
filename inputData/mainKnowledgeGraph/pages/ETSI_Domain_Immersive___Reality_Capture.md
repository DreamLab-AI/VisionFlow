- ### OntologyBlock
  id:: etsi-domain-immersive-reality-capture-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20360
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: ETSI Domain Immersive + Reality Capture Crossover
	- definition:: Domain categorization marker indicating metaverse systems operating at the intersection of immersive interaction capabilities and reality capture technologies for photorealistic virtual environment creation.
	- maturity:: mature
	- source:: [[ETSI GR MEC 032]], [[ISO 23257]]
	- owl:class:: mv:ETSIDomainImmersiveRealityCaptureMarker
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InteractionDomain]], [[CreativeMediaDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: etsi-domain-immersive-reality-capture-relationships
		- is-subclass-of:: [[Metaverse]]
		- is-part-of:: [[ETSI Metaverse Domain Model]]
		- requires:: [[ETSI Domain Immersive]], [[ETSI Domain Reality Capture]]
		- enables:: [[Photorealistic Immersion]], [[Volumetric Capture]], [[Spatial Interaction]]
	- #### OWL Axioms
	  id:: etsi-domain-immersive-reality-capture-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:ETSIDomainImmersiveRealityCaptureMarker))

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker mv:VirtualEntity)
		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker mv:Object)

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:categorizesDomain mv:InteractionDomain)
		  )

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:categorizesDomain mv:CreativeMediaDomain)
		  )

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:requiresDomain mv:ETSIDomainImmersive)
		  )

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:requiresDomain mv:ETSIDomainRealityCapture)
		  )

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

		  SubClassOf(mv:ETSIDomainImmersiveRealityCaptureMarker
		    ObjectSomeValuesFrom(mv:enablesCapability mv:PhotorealisticImmersion)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
