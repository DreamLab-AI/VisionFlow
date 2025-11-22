- ### OntologyBlock
  id:: etsi-domain-application-industry-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20338
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: ETSI Domain Application + Industry
	- definition:: Cross-domain marker for metaverse application components focused on industrial applications including manufacturing simulations, industrial digital twins, predictive maintenance, remote operations, and industrial training systems.
	- maturity:: mature
	- source:: [[ETSI GS MEC]]
	- owl:class:: mv:ETSIDomainApplicationIndustry
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InfrastructureDomain]], [[VirtualEconomyDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: etsi-domain-application-industry-relationships
		- is-subclass-of:: [[ArtificialIntelligence]]
		- is-part-of:: [[ETSI Domain Taxonomy]]
		- depends-on:: [[InfrastructureDomain]], [[VirtualEconomyDomain]]
		- enables:: [[Industrial Application Classification]], [[Manufacturing Platform Categorization]]
		- categorizes:: [[Industrial Digital Twin]], [[Manufacturing Simulation]], [[Predictive Maintenance]], [[Remote Operations]]
	- #### OWL Axioms
	  id:: etsi-domain-application-industry-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:ETSIDomainApplicationIndustry))

		  # Classification along two primary dimensions
		  SubClassOf(mv:ETSIDomainApplicationIndustry mv:VirtualEntity)
		  SubClassOf(mv:ETSIDomainApplicationIndustry mv:Object)

		  # Cross-domain marker classification
		  SubClassOf(mv:ETSIDomainApplicationIndustry mv:DomainMarker)
		  SubClassOf(mv:ETSIDomainApplicationIndustry mv:CrossDomainMarker)

		  # Multiple domain classification
		  SubClassOf(mv:ETSIDomainApplicationIndustry
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InfrastructureDomain)
		  )
		  SubClassOf(mv:ETSIDomainApplicationIndustry
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:VirtualEconomyDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:ETSIDomainApplicationIndustry
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
