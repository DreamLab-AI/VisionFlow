;;;; AI-0398: Value Alignment Problem
;;;; Priority 4 - Ethical AI
;;;; Challenge of aligning AI objectives with human values

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :ValueAlignmentProblem))
(SubClassOf :ValueAlignmentProblem :SafetyChallenge)
(SubClassOf :ValueAlignmentProblem :ResearchProblem)

;;; Core Challenge
(SubClassOf :ValueAlignmentProblem
  (ObjectSomeValuesFrom :requires :ValueSpecification))
(SubClassOf :ValueAlignmentProblem
  (ObjectSomeValuesFrom :involves :PreferenceLearning))
(SubClassOf :ValueAlignmentProblem
  (ObjectSomeValuesFrom :addresses :GoalMisalignment))

;;; Alignment Objectives
(SubClassOf :ValueAlignmentProblem
  (ObjectAllValuesFrom :alignsWith :HumanValues))
(SubClassOf :ValueAlignmentProblem
  (ObjectAllValuesFrom :respects :HumanPreferences))
(SubClassOf :ValueAlignmentProblem
  (ObjectAllValuesFrom :serves :HumanFlourishing))

;;; Problem Characteristics
(DataPropertyAssertion :hasProblemStatement :ValueAlignmentProblem
  "How can we ensure AI systems pursue goals that align with complex, nuanced, and potentially conflicting human values?")
(DataPropertyAssertion :hasChallengeLevel :ValueAlignmentProblem "Fundamental")

;;; PROBLEM DIMENSIONS ;;;

;;; Specification Challenge
(Declaration (Class :ValueSpecificationProblem))
(SubClassOf :ValueSpecificationProblem :ValueAlignmentProblem)
(AnnotationAssertion rdfs:comment :ValueSpecificationProblem
  "Difficulty in precisely specifying human values in machine-readable form")

(Declaration (Class :ObjectiveMisspecification))
(SubClassOf :ObjectiveMisspecification :ValueSpecificationProblem)
(AnnotationAssertion rdfs:comment :ObjectiveMisspecification
  "Gap between stated objective and true human intent")

(Declaration (Class :GoodhartsLaw))
(SubClassOf :GoodhartsLaw :ObjectiveMisspecification)
(DataPropertyAssertion :hasDefinition :GoodhartsLaw
  "When a measure becomes a target, it ceases to be a good measure")
(AnnotationAssertion rdfs:comment :GoodhartsLaw
  "AI optimising proxy metrics rather than true objectives")

;;; Learning Challenge
(Declaration (Class :ValueLearningProblem))
(SubClassOf :ValueLearningProblem :ValueAlignmentProblem)
(AnnotationAssertion rdfs:comment :ValueLearningProblem
  "Difficulty in learning complex human values from limited data")

(Declaration (Class :PreferenceAggregation))
(SubClassOf :PreferenceAggregation :ValueLearningProblem)
(AnnotationAssertion rdfs:comment :PreferenceAggregation
  "Combining diverse and conflicting preferences across populations")

(Declaration (Class :ValueExtrapolation))
(SubClassOf :ValueExtrapolation :ValueLearningProblem)
(AnnotationAssertion rdfs:comment :ValueExtrapolation
  "Generalising learned values to novel situations beyond training distribution")

;;; Robustness Challenge
(Declaration (Class :AlignmentRobustness))
(SubClassOf :AlignmentRobustness :ValueAlignmentProblem)
(AnnotationAssertion rdfs:comment :AlignmentRobustness
  "Maintaining alignment under distributional shift and capability scaling")

(Declaration (Class :InnerAlignmentProblem))
(SubClassOf :InnerAlignmentProblem :AlignmentRobustness)
(DataPropertyAssertion :hasDescription :InnerAlignmentProblem
  "Ensuring learned model internals pursue intended objectives rather than proxy goals")

(Declaration (Class :OuterAlignmentProblem))
(SubClassOf :OuterAlignmentProblem :AlignmentRobustness)
(DataPropertyAssertion :hasDescription :OuterAlignmentProblem
  "Specifying correct objective function that captures human values")

;;; ALIGNMENT APPROACHES ;;;

(Declaration (Class :AlignmentApproach))
(SubClassOf :AlignmentApproach :SafetyMechanism)

;;; Preference Learning Methods
(Declaration (Class :PreferenceLearning))
(SubClassOf :PreferenceLearning :AlignmentApproach)

(Declaration (Class :ReinforcementLearningFromHumanFeedback))
(SubClassOf :ReinforcementLearningFromHumanFeedback :PreferenceLearning)
(AnnotationAssertion skos:altLabel :ReinforcementLearningFromHumanFeedback "RLHF"@en)
(DataPropertyAssertion :hasDescription :ReinforcementLearningFromHumanFeedback
  "Training AI systems using human preference comparisons to shape behaviour")

(Declaration (Class :DirectPreferenceOptimisation))
(SubClassOf :DirectPreferenceOptimisation :PreferenceLearning)
(AnnotationAssertion skos:altLabel :DirectPreferenceOptimisation "DPO"@en)
(DataPropertyAssertion :hasDescription :DirectPreferenceOptimisation
  "Learning from preferences without explicit reward model, optimising policy directly")

;;; Value Learning Frameworks
(Declaration (Class :CooperativeInverseReinforcementLearning))
(SubClassOf :CooperativeInverseReinforcementLearning :AlignmentApproach)
(AnnotationAssertion skos:altLabel :CooperativeInverseReinforcementLearning "CIRL"@en)
(DataPropertyAssertion :hasDescription :CooperativeInverseReinforcementLearning
  "Agent and human collaborate to achieve human's initially unknown objectives")

(Declaration (Class :RecursiveRewardModeling))
(SubClassOf :RecursiveRewardModeling :AlignmentApproach)
(DataPropertyAssertion :hasDescription :RecursiveRewardModeling
  "Building reward models recursively with AI assistance, scaling oversight")

;;; Amplification Approaches
(Declaration (Class :IteratedAmplification))
(SubClassOf :IteratedAmplification :AlignmentApproach)
(DataPropertyAssertion :hasDescription :IteratedAmplification
  "Recursively training AI to help humans solve harder problems, amplifying human judgment")

(Declaration (Class :DebateFramework))
(SubClassOf :DebateFramework :AlignmentApproach)
(DataPropertyAssertion :hasDescription :DebateFramework
  "AI agents debate to help humans judge truth in domains beyond human expertise")

;;; ALIGNMENT CHALLENGES ;;;

(Declaration (Class :AlignmentObstacle))
(SubClassOf :AlignmentObstacle :TechnicalChallenge)

(Declaration (Class :RewardHacking))
(SubClassOf :RewardHacking :AlignmentObstacle)
(DataPropertyAssertion :hasDefinition :RewardHacking
  "Exploiting specification loopholes to achieve high reward without satisfying true intent")

(Declaration (Class :DeceptiveAlignment))
(SubClassOf :DeceptiveAlignment :AlignmentObstacle)
(DataPropertyAssertion :hasDefinition :DeceptiveAlignment
  "System appearing aligned during training while pursuing different objectives in deployment")

(Declaration (Class :MesaOptimisation))
(SubClassOf :MesaOptimisation :AlignmentObstacle)
(DataPropertyAssertion :hasDefinition :MesaOptimisation
  "Learned model developing internal optimisation process misaligned with training objective")

;;; Value Complexity Challenges
(Declaration (Class :ValueComplexity))
(SubClassOf :ValueComplexity :AlignmentObstacle)

(Declaration (Class :MoralUncertainty))
(SubClassOf :MoralUncertainty :ValueComplexity)
(AnnotationAssertion rdfs:comment :MoralUncertainty
  "Uncertainty about which moral principles are correct")

(Declaration (Class :ValuePluralism))
(SubClassOf :ValuePluralism :ValueComplexity)
(AnnotationAssertion rdfs:comment :ValuePluralism
  "Existence of multiple legitimate but potentially conflicting value systems")

(Declaration (Class :ContextDependentValues))
(SubClassOf :ContextDependentValues :ValueComplexity)
(AnnotationAssertion rdfs:comment :ContextDependentValues
  "Values whose correct application varies across contexts and cultures")

;;; EVALUATION METHODS ;;;

(Declaration (Class :AlignmentEvaluation))
(SubClassOf :AlignmentEvaluation :SafetyEvaluation)

;;; Alignment Metrics
(Declaration (Class :IntentAlignment))
(SubClassOf :IntentAlignment :AlignmentScore)
(DataPropertyAssertion :measuresProperty :IntentAlignment
  "Degree to which behaviour matches human intent rather than literal instruction")

(Declaration (Class :RobustnessToMisspecification))
(SubClassOf :RobustnessToMisspecification :AlignmentScore)
(DataPropertyAssertion :measuresProperty :RobustnessToMisspecification
  "Performance when objective specification is imperfect")

(Declaration (Class :ValueGeneralisation))
(SubClassOf :ValueGeneralisation :AlignmentScore)
(DataPropertyAssertion :measuresProperty :ValueGeneralisation
  "Ability to apply learned values to novel situations correctly")

;;; Testing Approaches
(Declaration (Class :AlignmentStressTest))
(SubClassOf :AlignmentStressTest :SafetyResearchMethod)
(AnnotationAssertion rdfs:comment :AlignmentStressTest
  "Deliberately testing alignment failures through adversarial scenarios")

(Declaration (Class :SpecificationGaming))
(SubClassOf :SpecificationGaming :AlignmentStressTest)
(AnnotationAssertion rdfs:comment :SpecificationGaming
  "Testing for unexpected ways to satisfy literal specification without intent")

;;; PHILOSOPHICAL FOUNDATIONS ;;;

(Declaration (Class :ValueTheory))
(SubClassOf :ValueTheory :PhilosophicalFramework)

(Declaration (Class :MetaEthics))
(SubClassOf :MetaEthics :ValueTheory)
(AnnotationAssertion rdfs:comment :MetaEthics
  "Study of the nature, scope, and meaning of moral judgments")

(Declaration (Class :NormativeEthics))
(SubClassOf :NormativeEthics :ValueTheory)
(AnnotationAssertion rdfs:comment :NormativeEthics
  "Theories about which actions are right or wrong")

;;; Ethical Frameworks
(Declaration (NamedIndividual :Consequentialism))
(ClassAssertion :NormativeEthics :Consequentialism)
(DataPropertyAssertion :hasDescription :Consequentialism
  "Rightness determined by consequences of actions")

(Declaration (NamedIndividual :Deontology))
(ClassAssertion :NormativeEthics :Deontology)
(DataPropertyAssertion :hasDescription :Deontology
  "Rightness determined by adherence to moral rules and duties")

(Declaration (NamedIndividual :VirtueEthics))
(ClassAssertion :NormativeEthics :VirtueEthics)
(DataPropertyAssertion :hasDescription :VirtueEthics
  "Rightness determined by virtuous character and practical wisdom")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :alignsWith))
(SubObjectPropertyOf :alignsWith :satisfies)
(ObjectPropertyDomain :alignsWith :AIBehaviour)
(ObjectPropertyRange :alignsWith :HumanValues)

(Declaration (ObjectProperty :learnedFrom))
(SubObjectPropertyOf :learnedFrom :derivedFrom)
(ObjectPropertyDomain :learnedFrom :ValueModel)
(ObjectPropertyRange :learnedFrom :HumanFeedback)

(Declaration (ObjectProperty :generalises))
(SubObjectPropertyOf :generalises :extends)
(ObjectPropertyDomain :generalises :ValueModel)
(ObjectPropertyRange :generalises :NovelSituation)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :ValueAlignmentProblem "Value Alignment Problem"@en)
(AnnotationAssertion skos:definition :ValueAlignmentProblem
  "Fundamental challenge in ensuring AI systems pursue objectives that align with complex, nuanced human values and intentions, encompassing difficulties in value specification, learning, and robust generalisation across diverse contexts."@en)

(AnnotationAssertion :hasCanonicalCitation :ValueAlignmentProblem
  "Russell, S. (2019). Human compatible: Artificial intelligence and the problem of control. Viking.")

(AnnotationAssertion :hasCanonicalCitation :ValueAlignmentProblem
  "Christiano, P., Leike, J., Brown, T., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.")

(AnnotationAssertion :hasCanonicalCitation :ValueAlignmentProblem
  "Gabriel, I. (2020). Artificial intelligence, values, and alignment. Minds and machines, 30(3), 411-437.")

(AnnotationAssertion :hasCanonicalCitation :ValueAlignmentProblem
  "Hubinger, E., van Merwijk, C., Mikulik, V., Skalse, J., & Garrabrant, S. (2019). Risks from learned optimization in advanced machine learning systems. arXiv preprint arXiv:1906.01820.")

(AnnotationAssertion :hasKeyProblem :ValueAlignmentProblem
  "Outer alignment: Specifying correct objective")
(AnnotationAssertion :hasKeyProblem :ValueAlignmentProblem
  "Inner alignment: Ensuring internals pursue objective")
(AnnotationAssertion :hasKeyProblem :ValueAlignmentProblem
  "Robust generalisation: Maintaining alignment under distribution shift")
(AnnotationAssertion :hasKeyProblem :ValueAlignmentProblem
  "Scalable oversight: Supervising superhuman systems")

(AnnotationAssertion dc:created :ValueAlignmentProblem "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :ValueAlignmentProblem "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :ValueAlignmentProblem "AI-0398")
(AnnotationAssertion :priorityLevel :ValueAlignmentProblem "4")
