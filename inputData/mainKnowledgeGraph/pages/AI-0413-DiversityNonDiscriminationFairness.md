- ### OntologyBlock
  id:: 0413-diversitynondiscriminationfairness-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[ArtificialIntelligenceTechnology]]
    - term-id:: AI-0413
    - preferred-term:: Diversity, Non-Discrimination, and Fairness
    - source-domain:: ai-grounded
    - status:: in-progress
    - version:: 1.0
    - last-updated:: 2025-10-29

  - **Definition**
    - definition:: Diversity Non-Discrimination and Fairness is a trustworthiness dimension ensuring AI systems avoid unfair bias, ensure equitable treatment across demographic groups, implement accessibility and universal design, and enable inclusive stakeholder participation throughout development and deployment. This dimension encompasses three core components: unfair bias avoidance (identifying bias affecting protected characteristics including sex, racial or ethnic origin, religion, disability, age, and sexual orientation per EU Charter Article 21, implementing bias mitigation through pre-processing data corrections, in-processing fairness constraints, and post-processing prediction adjustments, and continuously monitoring fairness metrics including demographic parity requiring equal selection rates across groups, equalized odds ensuring equal true positive and false positive rates, equal opportunity guaranteeing equal true positive rates, and individual fairness treating similar individuals similarly), accessibility and universal design (complying with Web Content Accessibility Guidelines WCAG ensuring perceivable, operable, understandable, and robust interfaces, implementing European Accessibility Act requirements, and applying universal design principles creating systems usable by people with diverse abilities without specialized adaptation), and stakeholder participation (involving diverse stakeholders including end users, affected communities, domain experts, and civil society throughout development lifecycle, implementing participatory design methodologies enabling co-creation with affected populations, and ensuring representative development teams reflecting diversity of deployment contexts and user populations). Legal frameworks including the EU AI Act mandate high-risk systems implement data governance ensuring training, validation, and testing datasets are relevant, representative, accurate, complete, and free from errors, with potential biases identified and mitigated. The 2024-2025 period marked transition from voluntary fairness practices to legally mandated requirements with enforcement mechanisms across jurisdictions including EU AI Act penalties reaching EUR 35 million or 7% of worldwide annual turnover, U.S. state-level legislation including Colorado AI Act and New York City Bias Audit Law, and international standards including ISO/IEC TR 24027:2021 for bias detection and ISO/IEC 42001:2023 for AI risk management, with regulatory sandboxes enabling deliberate testing to expose unwanted bias before deployment.
    - maturity:: mature
    - source:: [[EU AI Act]], [[EU Charter Article 21]], [[ISO/IEC TR 24027]], [[WCAG]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: aigo:DiversityNonDiscriminationFairness
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - owl:inferred-class:: aigo:VirtualProcess
    - belongsToDomain:: [[AIEthicsDomain]]
    - implementedInLayer:: [[ConceptualLayer]]

  - #### Relationships
    id:: 0413-diversitynondiscriminationfairness-relationships

  - #### OWL Axioms
    id:: 0413-diversitynondiscriminationfairness-owl-axioms
    collapsed:: true
    - ```clojure
      (Declaration (Class :DiversityNonDiscriminationFairness))
(SubClassOf :DiversityNonDiscriminationFairness :TrustworthinessDimension)
(SubClassOf :DiversityNonDiscriminationFairness :FundamentalRightsRequirement)

;; Three core components
(Declaration (Class :UnfairBiasAvoidance))
(Declaration (Class :AccessibilityUniversalDesign))
(Declaration (Class :StakeholderParticipation))

(SubClassOf :UnfairBiasAvoidance :DiversityNonDiscriminationFairness)
(SubClassOf :AccessibilityUniversalDesign :DiversityNonDiscriminationFairness)
(SubClassOf :StakeholderParticipation :DiversityNonDiscriminationFairness)

;; Bias avoidance requirements
(SubClassOf :UnfairBiasAvoidance
  (ObjectSomeValuesFrom :identifiesBias :ProtectedCharacteristic))
(SubClassOf :UnfairBiasAvoidance
  (ObjectSomeValuesFrom :mitigates :AlgorithmicBias))
(SubClassOf :UnfairBiasAvoidance
  (ObjectSomeValuesFrom :monitors :FairnessMetric))

;; Protected characteristics (EU Charter Article 21)
(Declaration (Class :ProtectedCharacteristic))
(Declaration (Class :Sex))
(Declaration (Class :RacialEthnicOrigin))
(Declaration (Class :Religion))
(Declaration (Class :Disability))
(Declaration (Class :Age))
(Declaration (Class :SexualOrientation))

(SubClassOf :Sex :ProtectedCharacteristic)
(SubClassOf :RacialEthnicOrigin :ProtectedCharacteristic)
(SubClassOf :Religion :ProtectedCharacteristic)
(SubClassOf :Disability :ProtectedCharacteristic)
(SubClassOf :Age :ProtectedCharacteristic)
(SubClassOf :SexualOrientation :ProtectedCharacteristic)

;; Fairness definitions
(Declaration (Class :FairnessDefinition))
(Declaration (Class :DemographicParity))
(Declaration (Class :EqualOpportunity))
(Declaration (Class :EqualOdds))
(Declaration (Class :IndividualFairness))

(SubClassOf :DemographicParity :FairnessDefinition)
(SubClassOf :EqualOpportunity :FairnessDefinition)
(SubClassOf :EqualOdds :FairnessDefinition)
(SubClassOf :IndividualFairness :FairnessDefinition)

;; Accessibility requirements
(SubClassOf :AccessibilityUniversalDesign
  (ObjectSomeValuesFrom :compliesWith :WCAG))
(SubClassOf :AccessibilityUniversalDesign
  (ObjectSomeValuesFrom :compliesWith :EuropeanAccessibilityAct))
(SubClassOf :AccessibilityUniversalDesign
  (ObjectSomeValuesFrom :implements :UniversalDesignPrinciple))

;; Stakeholder participation requirements
(SubClassOf :StakeholderParticipation
  (ObjectSomeValuesFrom :involves :DiverseStakeholders))
(SubClassOf :StakeholderParticipation
  (ObjectSomeValuesFrom :implements :ParticipatoryDesign))
(SubClassOf :StakeholderParticipation
  (ObjectSomeValuesFrom :ensures :RepresentativeDevelopmentTeam))

(DisjointClasses :DiversityNonDiscriminationFairness :DiscriminatorySystem)
(DisjointClasses :DiversityNonDiscriminationFairness :BiasedSystem)
      ```
