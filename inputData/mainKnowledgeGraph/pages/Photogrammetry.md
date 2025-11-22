- ### OntologyBlock
  id:: photogrammetry-ontology
  collapsed:: true
	- ontology:: true
	- term-id:: 20073
	- source-domain:: metaverse
	- status:: draft
    - public-access:: true
	- preferred-term:: Photogrammetry
	- definition:: A computational technique for reconstructing 3D geometry from overlapping photographic images through mathematical analysis of correspondences, camera poses, and geometric transformations to extract spatial information from 2D image data.
	- maturity:: mature
	- source:: [[Siemens + ACM]]
	- owl:class:: mv:Photogrammetry
	- owl:physicality:: VirtualEntity
	- owl:role:: Process
	- owl:inferred-class:: mv:VirtualProcess
	- owl:functional-syntax:: true
	- belongsToDomain:: [[CreativeMediaDomain]], [[RealityCaptureDomain]]
	- implementedInLayer:: [[Application Layer]]
	- #### Relationships
	  id:: photogrammetry-relationships
		- is-subclass-of:: [[Metaverse]]
		- is-dependency-of:: [[Tourism Metaverse]], [[Cultural Heritage XR Experience]]
		- has-part:: [[Feature Detection]], [[Camera Calibration]], [[Image Matching]], [[Triangulation]], [[Point Cloud Generation]], [[Mesh Reconstruction]]
		- is-part-of:: [[3D Reconstruction Pipeline]], [[Reality Capture Workflow]]
		- requires:: [[Camera]], [[Multiple Images]], [[Overlapping Coverage]], [[Image Processing Software]], [[Computational Resources]]
		- depends-on:: [[Computer Vision Algorithms]], [[Structure from Motion]], [[Multi-View Geometry]], [[Camera Models]]
		- enables:: [[3D Model Creation]], [[Digital Twin Construction]], [[Terrain Mapping]], [[Asset Digitization]], [[Spatial Measurement]]
	- #### OWL Axioms
	  id:: photogrammetry-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Photogrammetry))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Photogrammetry mv:VirtualEntity)
		  SubClassOf(mv:Photogrammetry mv:Process)

		  # Domain-specific constraints
		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:CreativeMediaDomain)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:RealityCaptureDomain)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:implementedInLayer mv:ApplicationLayer)
		  )

		  # Process input requirements
		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:requires mv:Camera)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:requires mv:MultipleImages)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:requires mv:OverlappingCoverage)
		  )

		  # Core sub-processes
		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:hasPart mv:FeatureDetection)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:hasPart mv:CameraCalibration)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:hasPart mv:ImageMatching)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:hasPart mv:Triangulation)
		  )

		  # Process outputs
		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:enables mv:3DModelCreation)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:enables mv:DigitalTwinConstruction)
		  )

		  # Dependency constraints
		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:dependsOn mv:ComputerVisionAlgorithms)
		  )

		  SubClassOf(mv:Photogrammetry
		    ObjectSomeValuesFrom(mv:dependsOn mv:StructureFromMotion)
		  )

  # Property characteristics
  AsymmetricObjectProperty(dt:isdependencyof)

  # Property characteristics
  TransitiveObjectProperty(dt:ispartof)

  # Property characteristics
  AsymmetricObjectProperty(dt:requires)

  # Property characteristics
  AsymmetricObjectProperty(dt:dependson)

  # Property characteristics
  AsymmetricObjectProperty(dt:enables)
```
