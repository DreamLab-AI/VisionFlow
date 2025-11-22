- ### OntologyBlock
  id:: etsi-domain-ai-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20330
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: ETSI Domain AI
	- definition:: Domain marker concept for categorizing metaverse components related to artificial intelligence, machine learning, and computational intelligence capabilities.
	- maturity:: mature
	- source:: [[ETSI GS MEC]]
	- owl:class:: mv:ETSIDomainAI
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[ComputationAndIntelligenceDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: etsi-domain-ai-relationships
		- is-subclass-of:: [[ArtificialIntelligence]]
		- is-dependency-of:: [[ETSI_Domain_AI___Governance]], [[ETSI_Domain_AI___Data_Mgmt]], [[ETSI_Domain_AI___Human_Interface]], [[ETSI_Domain_AI___Creative_Media]]
		- is-part-of:: [[ETSI Domain Taxonomy]]
		- enables:: [[AI Service Classification]], [[Intelligence Layer Categorization]]
		- depends-on:: [[ComputationAndIntelligenceDomain]]
		- has-part:: [[ETSI Domain AI Creative Media]], [[ETSI Domain AI Data Mgmt]], [[ETSI Domain AI Governance]], [[ETSI Domain AI Human Interface]]
	- #### OWL Axioms
	  id:: etsi-domain-ai-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:ETSIDomainAI))

		  # Classification along two primary dimensions
		  SubClassOf(mv:ETSIDomainAI mv:VirtualEntity)
		  SubClassOf(mv:ETSIDomainAI mv:Object)

		  # Domain marker classification
		  SubClassOf(mv:ETSIDomainAI mv:DomainMarker)
		  SubClassOf(mv:ETSIDomainAI mv:TaxonomyNode)

		  # Domain classification
		  SubClassOf(mv:ETSIDomainAI
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:ComputationAndIntelligenceDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:ETSIDomainAI
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

		  # Taxonomy relationships
		  SubClassOf(mv:ETSIDomainAI
		    ObjectSomeValuesFrom(mv:categorizes mv:AIComponent)
		  )

  # Property characteristics
  AsymmetricObjectProperty(dt:isdependencyof)

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)
```
