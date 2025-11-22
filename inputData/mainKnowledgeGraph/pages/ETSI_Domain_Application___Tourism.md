- ### OntologyBlock
  id:: etsi-domain-application-tourism-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20339
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: ETSI Domain Application + Tourism
	- definition:: Cross-domain marker for metaverse application components focused on tourism and hospitality including virtual tours, destination previews, cultural heritage experiences, and travel planning platforms in immersive environments.
	- maturity:: mature
	- source:: [[ETSI GS MEC]]
	- owl:class:: mv:ETSIDomainApplicationTourism
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InfrastructureDomain]], [[VirtualSocietyDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: etsi-domain-application-tourism-relationships
		- is-subclass-of:: [[Extended Reality (XR)]]
		- is-part-of:: [[ETSI Domain Taxonomy]]
		- depends-on:: [[InfrastructureDomain]], [[VirtualSocietyDomain]]
		- enables:: [[Tourism Application Classification]], [[Cultural Experience Categorization]]
		- categorizes:: [[Virtual Tour]], [[Destination Preview]], [[Cultural Heritage Experience]], [[Travel Planning Platform]]
	- #### OWL Axioms
	  id:: etsi-domain-application-tourism-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:ETSIDomainApplicationTourism))

		  # Classification along two primary dimensions
		  SubClassOf(mv:ETSIDomainApplicationTourism mv:VirtualEntity)
		  SubClassOf(mv:ETSIDomainApplicationTourism mv:Object)

		  # Cross-domain marker classification
		  SubClassOf(mv:ETSIDomainApplicationTourism mv:DomainMarker)
		  SubClassOf(mv:ETSIDomainApplicationTourism mv:CrossDomainMarker)

		  # Multiple domain classification
		  SubClassOf(mv:ETSIDomainApplicationTourism
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InfrastructureDomain)
		  )
		  SubClassOf(mv:ETSIDomainApplicationTourism
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:VirtualSocietyDomain)
		  )

		  # Layer classification
		  SubClassOf(mv:ETSIDomainApplicationTourism
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
