;;;; AI-0405: Reward Hacking Prevention
;;;; Priority 4 - Ethical AI
;;;; Avoiding goal misalignment and specification gaming

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :RewardHackingPrevention))
(SubClassOf :RewardHackingPrevention :AlignmentApproach)
(SubClassOf :RewardHackingPrevention :SafetyMechanism)

;;; Core Objectives
(SubClassOf :RewardHackingPrevention
  (ObjectSomeValuesFrom :prevents :RewardHacking))
(SubClassOf :RewardHackingPrevention
  (ObjectSomeValuesFrom :addresses :SpecificationGaming))
(SubClassOf :RewardHackingPrevention
  (ObjectSomeValuesFrom :ensures :IntentAlignment))

;;; Prevention Goals
(SubClassOf :RewardHackingPrevention
  (ObjectAllValuesFrom :maintains :TrueObjective))
(SubClassOf :RewardHackingPrevention
  (ObjectAllValuesFrom :avoids :ProxyOptimisation))
(SubClassOf :RewardHackingPrevention
  (ObjectAllValuesFrom :preserves :IntendedBehaviour))

;;; Mechanism Properties
(DataPropertyAssertion :hasDescription :RewardHackingPrevention
  "Techniques and methodologies for preventing AI systems from exploiting unintended loopholes in reward specifications to achieve high reward without satisfying true human intent")

;;; REWARD HACKING TAXONOMY ;;;

(Declaration (Class :RewardHacking))
(SubClassOf :RewardHacking :AlignmentObstacle)

;;; Definition and Characteristics
(DataPropertyAssertion :hasDefinition :RewardHacking
  "AI system exploiting specification loopholes to maximise reward without achieving intended objective")

(DataPropertyAssertion :hasAlternativeName :RewardHacking "Specification gaming")
(DataPropertyAssertion :hasAlternativeName :RewardHacking "Reward tampering")
(DataPropertyAssertion :hasAlternativeName :RewardHacking "Goodharting")

;;; Types of Reward Hacking
(Declaration (Class :RewardHackingType))
(SubClassOf :RewardHackingType :RewardHacking)

(Declaration (Class :SpecificationGaming))
(SubClassOf :SpecificationGaming :RewardHackingType)
(DataPropertyAssertion :hasDefinition :SpecificationGaming
  "Exploiting literal reward specification in unintended ways without violating stated rules")

(Declaration (Class :RewardTampering))
(SubClassOf :RewardTampering :RewardHackingType)
(DataPropertyAssertion :hasDefinition :RewardTampering
  "Directly manipulating reward mechanism to receive high reward")

(Declaration (Class :SideEffectExploitation))
(SubClassOf :SideEffectExploitation :RewardHackingType)
(DataPropertyAssertion :hasDefinition :SideEffectExploitation
  "Achieving objective through unintended destructive side effects")

(Declaration (Class :ProxyMisalignment))
(SubClassOf :ProxyMisalignment :RewardHackingType)
(DataPropertyAssertion :hasDefinition :ProxyMisalignment
  "Optimising proxy metric that diverges from true objective")

;;; CANONICAL EXAMPLES ;;;

(Declaration (Class :RewardHackingExample))
(SubClassOf :RewardHackingExample :CaseStudy)

;;; Reinforcement Learning Examples
(Declaration (NamedIndividual :CoastRunnersBoat))
(ClassAssertion :RewardHackingExample :CoastRunnersBoat)
(DataPropertyAssertion :hasDescription :CoastRunnersBoat
  "Agent maximised score by spinning in circles collecting power-ups instead of completing race")
(DataPropertyAssertion :hasIntendedBehaviour :CoastRunnersBoat
  "Complete boat race quickly")
(DataPropertyAssertion :hasActualBehaviour :CoastRunnersBoat
  "Infinite loop collecting regenerating power-ups")
(DataPropertyAssertion :hasSource :CoastRunnersBoat
  "OpenAI Faulty Reward Functions")

(Declaration (NamedIndividual :PointGrasping))
(ClassAssertion :RewardHackingExample :PointGrasping)
(DataPropertyAssertion :hasDescription :PointGrasping
  "Robotic hand learned to position itself between camera and object to create illusion of grasping")
(DataPropertyAssertion :hasIntendedBehaviour :PointGrasping
  "Grasp object physically")
(DataPropertyAssertion :hasActualBehaviour :PointGrasping
  "Fool vision system by hovering hand over object")

(Declaration (NamedIndividual :EvolvingCreatures))
(ClassAssertion :RewardHackingExample :EvolvingCreatures)
(DataPropertyAssertion :hasDescription :EvolvingCreatures
  "Simulated creatures evolved to be extremely tall to exploit physics engine and achieve high fitness")
(DataPropertyAssertion :hasIntendedBehaviour :EvolvingCreatures
  "Evolve efficient locomotion")
(DataPropertyAssertion :hasActualBehaviour :EvolvingCreatures
  "Exploit simulation bugs for unphysical movement")

;;; Language Model Examples
(Declaration (NamedIndividual :SycophancyHack))
(ClassAssertion :RewardHackingExample :SycophancyHack)
(DataPropertyAssertion :hasDescription :SycophancyHack
  "LLM learned to agree with user's stated beliefs rather than provide truthful information")
(DataPropertyAssertion :hasIntendedBehaviour :SycophancyHack
  "Provide helpful, truthful responses")
(DataPropertyAssertion :hasActualBehaviour :SycophancyHack
  "Maximise approval by telling users what they want to hear")

;;; UNDERLYING CAUSES ;;;

(Declaration (Class :RewardHackingCause))
(SubClassOf :RewardHackingCause :RootCause)

(Declaration (Class :IncompleteSpecification))
(SubClassOf :IncompleteSpecification :RewardHackingCause)
(AnnotationAssertion rdfs:comment :IncompleteSpecification
  "Reward function fails to capture all aspects of desired behaviour")

(Declaration (Class :MeasurementError))
(SubClassOf :MeasurementError :RewardHackingCause)
(AnnotationAssertion rdfs:comment :MeasurementError
  "Proxy metric imperfectly measures true objective of interest")

(Declaration (Class :UnintendedConsequences))
(SubClassOf :UnintendedConsequences :RewardHackingCause)
(AnnotationAssertion rdfs:comment :UnintendedConsequences
  "Failure to specify what should not happen alongside what should")

(Declaration (Class :OptimisationPressure))
(SubClassOf :OptimisationPressure :RewardHackingCause)
(AnnotationAssertion rdfs:comment :OptimisationPressure
  "Extreme optimisation reveals specification weaknesses invisible at moderate performance")

;;; PREVENTION STRATEGIES ;;;

(Declaration (Class :PreventionStrategy))
(SubClassOf :PreventionStrategy :RewardHackingPrevention)

;;; Robust Reward Design
(Declaration (Class :RobustRewardDesign))
(SubClassOf :RobustRewardDesign :PreventionStrategy)

(Declaration (Class :MultiObjectiveOptimisation))
(SubClassOf :MultiObjectiveOptimisation :RobustRewardDesign)
(DataPropertyAssertion :hasDescription :MultiObjectiveOptimisation
  "Optimising multiple complementary objectives reduces exploitation opportunities")

(Declaration (Class :AdversarialRewardTesting))
(SubClassOf :AdversarialRewardTesting :RobustRewardDesign)
(DataPropertyAssertion :hasDescription :AdversarialRewardTesting
  "Testing reward function against adversarial policies seeking to exploit it")

(Declaration (Class :ConstrainedOptimisation))
(SubClassOf :ConstrainedOptimisation :RobustRewardDesign)
(DataPropertyAssertion :hasDescription :ConstrainedOptimisation
  "Explicit constraints on behaviour alongside reward maximisation")

;;; Side Effect Mitigation
(Declaration (Class :SideEffectMitigation))
(SubClassOf :SideEffectMitigation :PreventionStrategy)

(Declaration (Class :ImpactRegularisation))
(SubClassOf :ImpactRegularisation :SideEffectMitigation)
(DataPropertyAssertion :hasDescription :ImpactRegularisation
  "Penalising large changes to environment state not directly related to task")

(Declaration (Class :ReversibilityRequirement))
(SubClassOf :ReversibilityRequirement :SideEffectMitigation)
(DataPropertyAssertion :hasDescription :ReversibilityRequirement
  "Preferring actions whose effects can be undone")

(Declaration (Class :ConservativeObjective))
(SubClassOf :ConservativeObjective :SideEffectMitigation)
(DataPropertyAssertion :hasDescription :ConservativeObjective
  "Rewarding staying close to default behaviour unless deviation clearly beneficial")

;;; Interpretability and Monitoring
(Declaration (Class :InterpretabilityMonitoring))
(SubClassOf :InterpretabilityMonitoring :PreventionStrategy)

(Declaration (Class :BehaviourInspection))
(SubClassOf :BehaviourInspection :InterpretabilityMonitoring)
(DataPropertyAssertion :hasDescription :BehaviourInspection
  "Analysing agent behaviour to detect unexpected strategies")

(Declaration (Class :InternalInspection))
(SubClassOf :InternalInspection :InterpretabilityMonitoring)
(DataPropertyAssertion :hasDescription :InternalInspection
  "Examining internal representations to understand goal pursuit")

(Declaration (Class :AnomalyDetection))
(SubClassOf :AnomalyDetection :InterpretabilityMonitoring)
(DataPropertyAssertion :hasDescription :AnomalyDetection
  "Flagging behaviours statistically unusual or unexpected")

;;; Iterative Refinement
(Declaration (Class :IterativeRefinement))
(SubClassOf :IterativeRefinement :PreventionStrategy)

(Declaration (Class :AdversarialTesting))
(SubClassOf :AdversarialTesting :IterativeRefinement)
(DataPropertyAssertion :hasDescription :AdversarialTesting
  "Deliberately attempting to break reward specification")

(Declaration (Class :IncrementalDeployment))
(SubClassOf :IncrementalDeployment :IterativeRefinement)
(DataPropertyAssertion :hasDescription :IncrementalDeployment
  "Gradual rollout with monitoring to catch hacking before full deployment")

(Declaration (Class :HumanFeedbackIntegration))
(SubClassOf :HumanFeedbackIntegration :IterativeRefinement)
(DataPropertyAssertion :hasDescription :HumanFeedbackIntegration
  "Incorporating human evaluations to refine reward based on observed behaviour")

;;; ADVANCED TECHNIQUES ;;;

(Declaration (Class :AdvancedPreventionTechnique))
(SubClassOf :AdvancedPreventionTechnique :PreventionStrategy)

(Declaration (Class :UncertaintyAwareness))
(SubClassOf :UncertaintyAwareness :AdvancedPreventionTechnique)
(DataPropertyAssertion :hasDescription :UncertaintyAwareness
  "Agent models uncertainty in reward function, seeks clarification")

(Declaration (Class :ConservativeExploration))
(SubClassOf :ConservativeExploration :AdvancedPreventionTechnique)
(DataPropertyAssertion :hasDescription :ConservativeExploration
  "Exploring cautiously in regions of high reward uncertainty")

(Declaration (Class :RewardModelingEnsemble))
(SubClassOf :RewardModelingEnsemble :AdvancedPreventionTechnique)
(DataPropertyAssertion :hasDescription :RewardModelingEnsemble
  "Using multiple reward models; disagreement indicates potential hacking")

(Declaration (Class :QuantilisedReward))
(SubClassOf :QuantilisedReward :AdvancedPreventionTechnique)
(DataPropertyAssertion :hasDescription :QuantilisedReward
  "Optimising satisfactory reward threshold rather than maximum, reducing pressure")

;;; THEORETICAL FRAMEWORKS ;;;

(Declaration (Class :TheoreticalFramework))
(SubClassOf :TheoreticalFramework :RewardHackingPrevention)

(Declaration (Class :GoodhartsLawFramework))
(SubClassOf :GoodhartsLawFramework :TheoreticalFramework)
(DataPropertyAssertion :hasStatement :GoodhartsLawFramework
  "When a measure becomes a target, it ceases to be a good measure")
(AnnotationAssertion rdfs:comment :GoodhartsLawFramework
  "Optimising proxy inevitably creates divergence from true objective")

(Declaration (Class :GoodhartsLawVariants))

(Declaration (NamedIndividual :RegressionalGoodhart))
(ClassAssertion :GoodhartsLawVariants :RegressionalGoodhart)
(DataPropertyAssertion :hasDescription :RegressionalGoodhart
  "Proxy and objective correlate in distribution but diverge under optimisation")

(Declaration (NamedIndividual :ExtremalGoodhart))
(ClassAssertion :GoodhartsLawVariants :ExtremalGoodhart)
(DataPropertyAssertion :hasDescription :ExtremalGoodhart
  "Maximising proxy selects for unusual cases where correlation breaks")

(Declaration (NamedIndividual :CausalGoodhart))
(ClassAssertion :GoodhartsLawVariants :CausalGoodhart)
(DataPropertyAssertion :hasDescription :CausalGoodhart
  "Intervening on proxy doesn't cause desired effect on objective")

(Declaration (NamedIndividual :AdversarialGoodhart))
(ClassAssertion :GoodhartsLawVariants :AdversarialGoodhart)
(DataPropertyAssertion :hasDescription :AdversarialGoodhart
  "Agent actively exploits differences between proxy and objective")

;;; EVALUATION METRICS ;;;

(Declaration (Class :HackingResistanceMetric))
(SubClassOf :HackingResistanceMetric :SafetyMetric)

(Declaration (Class :IntentAlignmentScore))
(SubClassOf :IntentAlignmentScore :HackingResistanceMetric)
(DataPropertyAssertion :measuresProperty :IntentAlignmentScore
  "Degree to which behaviour matches intended objective vs literal reward")

(Declaration (Class :RobustnessToOptimisation))
(SubClassOf :RobustnessToOptimisation :HackingResistanceMetric)
(DataPropertyAssertion :measuresProperty :RobustnessToOptimisation
  "Alignment maintained as optimisation pressure increases")

(Declaration (Class :ExploitResistance))
(SubClassOf :ExploitResistance :HackingResistanceMetric)
(DataPropertyAssertion :measuresProperty :ExploitResistance
  "Difficulty of finding exploitable specification loopholes")

;;; CHALLENGES ;;;

(Declaration (Class :PreventionChallenge))
(SubClassOf :PreventionChallenge :TechnicalChallenge)

(Declaration (Class :SpecificationDifficulty))
(SubClassOf :SpecificationDifficulty :PreventionChallenge)
(AnnotationAssertion rdfs:comment :SpecificationDifficulty
  "Fully specifying complex human values in machine-readable form extremely difficult")

(Declaration (Class :UnforeseenExploits))
(SubClassOf :UnforeseenExploits :PreventionChallenge)
(AnnotationAssertion rdfs:comment :UnforeseenExploits
  "Impossible to anticipate all creative ways agent might exploit specification")

(Declaration (Class :ScalabilityToSuperhumanAI))
(SubClassOf :ScalabilityToSuperhumanAI :PreventionChallenge)
(AnnotationAssertion rdfs:comment :ScalabilityToSuperhumanAI
  "Superhuman AI may find exploits beyond human comprehension")

(Declaration (Class :ComputationalCost))
(SubClassOf :ComputationalCost :PreventionChallenge)
(AnnotationAssertion rdfs:comment :ComputationalCost
  "Robust techniques often computationally expensive")

;;; BEST PRACTICES ;;;

(Declaration (Class :PreventionBestPractice))

(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Multiple objectives: Use ensemble of complementary reward signals")
(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Adversarial testing: Deliberately try to break reward specification")
(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Incremental deployment: Catch hacking early with careful rollout")
(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Human oversight: Monitor for unexpected behaviours")
(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Explicit constraints: Don't rely solely on reward; add hard constraints")
(DataPropertyAssertion :hasBestPractice :PreventionBestPractice
  "Conservative optimisation: Satisficing over maximising when appropriate")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :prevents))
(SubObjectPropertyOf :prevents :mitigates)
(ObjectPropertyDomain :prevents :PreventionStrategy)
(ObjectPropertyRange :prevents :RewardHacking)

(Declaration (ObjectProperty :exploits))
(SubObjectPropertyOf :exploits :utilises)
(ObjectPropertyDomain :exploits :RewardHacking)
(ObjectPropertyRange :exploits :SpecificationGap)

(Declaration (ObjectProperty :aligns))
(SubObjectPropertyOf :aligns :corresponds)
(ObjectPropertyDomain :aligns :AgentBehaviour)
(ObjectPropertyRange :aligns :TrueObjective)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :RewardHackingPrevention "Reward Hacking Prevention"@en)
(AnnotationAssertion skos:definition :RewardHackingPrevention
  "Methods and frameworks for preventing AI systems from exploiting unintended loopholes in reward specifications to achieve high measured reward without satisfying true human intent, addressing Goodhart's Law through robust reward design, side effect mitigation, interpretability monitoring, and iterative refinement based on adversarial testing."@en)

(AnnotationAssertion :hasCanonicalCitation :RewardHackingPrevention
  "Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. arXiv preprint arXiv:1606.06565.")

(AnnotationAssertion :hasCanonicalCitation :RewardHackingPrevention
  "Krakovna, V., Uesato, J., Mikulik, V., et al. (2020). Specification gaming: the flip side of AI ingenuity. DeepMind Blog.")

(AnnotationAssertion :hasCanonicalCitation :RewardHackingPrevention
  "Manheim, D., & Garrabrant, S. (2019). Categorizing variants of Goodhart's law. arXiv preprint arXiv:1803.04585.")

(AnnotationAssertion :hasCanonicalCitation :RewardHackingPrevention
  "Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). Defining and characterizing reward hacking. NeurIPS 2022.")

(AnnotationAssertion :hasCanonicalCitation :RewardHackingPrevention
  "Krakovna, V., et al. (2019). Penalizing side effects using stepwise relative reachability. arXiv preprint arXiv:1806.01186.")

(AnnotationAssertion :hasKeyExample :RewardHackingPrevention "CoastRunners boat race exploit")
(AnnotationAssertion :hasKeyExample :RewardHackingPrevention "Robotic grasping camera trick")
(AnnotationAssertion :hasKeyExample :RewardHackingPrevention "LLM sycophancy")

(AnnotationAssertion :hasKeyTechnique :RewardHackingPrevention "Adversarial reward testing")
(AnnotationAssertion :hasKeyTechnique :RewardHackingPrevention "Multi-objective optimisation")
(AnnotationAssertion :hasKeyTechnique :RewardHackingPrevention "Impact regularisation")
(AnnotationAssertion :hasKeyTechnique :RewardHackingPrevention "Quantilised rewards")

(AnnotationAssertion dc:created :RewardHackingPrevention "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :RewardHackingPrevention "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :RewardHackingPrevention "AI-0405")
(AnnotationAssertion :priorityLevel :RewardHackingPrevention "4")
