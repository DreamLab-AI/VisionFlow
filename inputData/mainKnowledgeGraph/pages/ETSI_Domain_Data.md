- ### OntologyBlock
  id:: etsi-domain-data-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20374
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: ETSI Domain Data
	- definition:: Domain categorization for data management, storage, analytics, AI/ML systems, and intelligence capabilities processing information in metaverse environments.
	- maturity:: mature
	- source:: [[ETSI GR MEC 032]], [[ISO 23257]]
	- owl:class:: mv:ETSIDomainData
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[ComputationAndIntelligenceDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: etsi-domain-data-relationships
		- is-subclass-of:: [[ArtificialIntelligence]]
		- is-part-of:: [[ETSI Metaverse Domain Model]]
		- has-part:: [[Data Storage]], [[Data Analytics]], [[Machine Learning]], [[AI Systems]]
		- enables:: [[Data Processing]], [[Intelligence]], [[Predictive Analytics]]
	- #### OWL Axioms
	  id:: etsi-domain-data-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:ETSIDomainData))

		  SubClassOf(mv:ETSIDomainData mv:VirtualEntity)
		  SubClassOf(mv:ETSIDomainData mv:Object)

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:ComputationAndIntelligenceDomain)
		  )

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:hasPart mv:DataStorage)
		  )

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:hasPart ai:MachineLearning)
		  )

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:enablesCapability mv:DataProcessing)
		  )

		  SubClassOf(mv:ETSIDomainData
		    ObjectSomeValuesFrom(mv:enablesCapability mv:Intelligence)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
