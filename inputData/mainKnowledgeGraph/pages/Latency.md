- ### OntologyBlock
  id:: latency-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20148
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Latency
	- definition:: Virtual performance metric representing the time delay between a user action and corresponding system response within networked immersive environments.
	- maturity:: mature
	- source:: [[ETSI ARF 010]]
	- owl:class:: mv:Latency
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InfrastructureDomain]]
	- implementedInLayer:: [[Physical Layer]], [[Network Layer]]
	- #### Relationships
	  id:: latency-relationships
		- is-subclass-of:: [[Metaverse Infrastructure]]
		- is-dependency-of:: [[Presence]]
		- is-part-of:: [[Network Performance Metrics]], [[Quality of Service]]
		- depends-on:: [[Network Infrastructure]], [[Routing Protocol]], [[Bandwidth]], [[Processing Delay]], [[Propagation Delay]]
		- requires:: [[Measurement Tools]], [[Monitoring System]], [[Timestamp Synchronization]]
		- enables:: [[Performance Optimization]], [[Quality Assessment]], [[SLA Monitoring]], [[User Experience Tuning]]
	- #### OWL Axioms
	  id:: latency-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Latency))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Latency mv:VirtualEntity)
		  SubClassOf(mv:Latency mv:Object)

		  # Domain classification
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InfrastructureDomain)
		  )

		  # Layer classification - spans physical and network layers
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:PhysicalLayer)
		  )

		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:NetworkLayer)
		  )

		  # Part of performance metrics
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:isPartOf mv:NetworkPerformanceMetrics)
		  )

		  # Depends on network infrastructure
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:dependsOn mv:NetworkInfrastructure)
		  )

		  # Has measurable value in milliseconds
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:hasValue xsd:decimal)
		  )

		  # Has time unit (milliseconds)
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:hasUnit mv:Millisecond)
		  )

		  # Enables performance optimization
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:enables mv:PerformanceOptimization)
		  )

		  # Virtual metric with no physical form
		  SubClassOf(mv:Latency
		    ObjectComplementOf(mv:PhysicalEntity)
		  )

		  # Data-based measurement requiring timestamp synchronization
		  SubClassOf(mv:Latency
		    ObjectSomeValuesFrom(mv:requires mv:TimestampSynchronization)
		  )

  # Property characteristics
  AsymmetricObjectProperty(dt:isdependencyof)

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
