;;;; AI-0406: AI Capability Evaluation
;;;; Priority 4 - Ethical AI
;;;; Understanding system capabilities and limitations

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :AICapabilityEvaluation))
(SubClassOf :AICapabilityEvaluation :SafetyEvaluation)
(SubClassOf :AICapabilityEvaluation :PerformanceAssessment)

;;; Core Functions
(SubClassOf :AICapabilityEvaluation
  (ObjectSomeValuesFrom :assesses :SystemCapability))
(SubClassOf :AICapabilityEvaluation
  (ObjectSomeValuesFrom :identifies :SystemLimitation))
(SubClassOf :AICapabilityEvaluation
  (ObjectSomeValuesFrom :characterises :FailureMode))

;;; Evaluation Objectives
(SubClassOf :AICapabilityEvaluation
  (ObjectAllValuesFrom :determines :PerformanceBoundary))
(SubClassOf :AICapabilityEvaluation
  (ObjectAllValuesFrom :quantifies :ReliabilityLevel))
(SubClassOf :AICapabilityEvaluation
  (ObjectAllValuesFrom :documents :OperationalLimits))

;;; System Properties
(DataPropertyAssertion :hasDescription :AICapabilityEvaluation
  "Systematic assessment of AI system capabilities, limitations, and failure modes to understand performance boundaries, reliability characteristics, and safe operating conditions")

;;; CAPABILITY DIMENSIONS ;;;

(Declaration (Class :CapabilityDimension))
(SubClassOf :CapabilityDimension :EvaluationDimension)

;;; Technical Performance
(Declaration (Class :TechnicalCapability))
(SubClassOf :TechnicalCapability :CapabilityDimension)

(Declaration (Class :TaskPerformance))
(SubClassOf :TaskPerformance :TechnicalCapability)
(DataPropertyAssertion :hasDescription :TaskPerformance
  "Accuracy, precision, recall on intended tasks")

(Declaration (Class :GeneralisationCapability))
(SubClassOf :GeneralisationCapability :TechnicalCapability)
(DataPropertyAssertion :hasDescription :GeneralisationCapability
  "Performance on out-of-distribution or novel inputs")

(Declaration (Class :ScalabilityCharacteristics))
(SubClassOf :ScalabilityCharacteristics :TechnicalCapability)
(DataPropertyAssertion :hasDescription :ScalabilityCharacteristics
  "Behaviour as input complexity or volume increases")

;;; Robustness Properties
(Declaration (Class :RobustnessCapability))
(SubClassOf :RobustnessCapability :CapabilityDimension)

(Declaration (Class :NoiseRobustness))
(SubClassOf :NoiseRobustness :RobustnessCapability)
(DataPropertyAssertion :hasDescription :NoiseRobustness
  "Performance degradation under input noise or corruption")

(Declaration (Class :AdversarialResistance))
(SubClassOf :AdversarialResistance :RobustnessCapability)
(DataPropertyAssertion :hasDescription :AdversarialResistance
  "Resilience to adversarial perturbations and attacks")

(Declaration (Class :DistributionShiftTolerance))
(SubClassOf :DistributionShiftTolerance :RobustnessCapability)
(DataPropertyAssertion :hasDescription :DistributionShiftTolerance
  "Maintaining performance across domain or temporal shifts")

;;; Reasoning Capabilities
(Declaration (Class :ReasoningCapability))
(SubClassOf :ReasoningCapability :CapabilityDimension)

(Declaration (Class :LogicalReasoning))
(SubClassOf :LogicalReasoning :ReasoningCapability)
(DataPropertyAssertion :hasDescription :LogicalReasoning
  "Deductive inference, consistency, logical entailment")

(Declaration (Class :CausalReasoning))
(SubClassOf :CausalReasoning :ReasoningCapability)
(DataPropertyAssertion :hasDescription :CausalReasoning
  "Understanding cause-effect relationships and interventions")

(Declaration (Class :CommonSenseReasoning))
(SubClassOf :CommonSenseReasoning :ReasoningCapability)
(DataPropertyAssertion :hasDescription :CommonSenseReasoning
  "Everyday physical and social understanding")

;;; LIMITATION CATEGORIES ;;;

(Declaration (Class :SystemLimitation))
(SubClassOf :SystemLimitation :PerformanceConstraint)

;;; Knowledge Limitations
(Declaration (Class :KnowledgeLimitation))
(SubClassOf :KnowledgeLimitation :SystemLimitation)

(Declaration (Class :KnowledgeCutoff))
(SubClassOf :KnowledgeCutoff :KnowledgeLimitation)
(DataPropertyAssertion :hasDescription :KnowledgeCutoff
  "Temporal boundary of training data; lack of knowledge about subsequent events")

(Declaration (Class :DomainGap))
(SubClassOf :DomainGap :KnowledgeLimitation)
(DataPropertyAssertion :hasDescription :DomainGap
  "Specialised domains with insufficient training coverage")

(Declaration (Class :GroundingLimitation))
(SubClassOf :GroundingLimitation :KnowledgeLimitation)
(DataPropertyAssertion :hasDescription :GroundingLimitation
  "Lack of real-world grounding in physical or social reality")

;;; Reasoning Limitations
(Declaration (Class :ReasoningLimitation))
(SubClassOf :ReasoningLimitation :SystemLimitation)

(Declaration (Class :MultiStepReasoningFailure))
(SubClassOf :MultiStepReasoningFailure :ReasoningLimitation)
(DataPropertyAssertion :hasDescription :MultiStepReasoningFailure
  "Error accumulation in complex multi-step reasoning")

(Declaration (Class :CounterfactualBlindness))
(SubClassOf :CounterfactualBlindness :ReasoningLimitation)
(DataPropertyAssertion :hasDescription :CounterfactualBlindness
  "Difficulty with counterfactual reasoning and hypotheticals")

(Declaration (Class :NumericalReasoningWeakness))
(SubClassOf :NumericalReasoningWeakness :ReasoningLimitation)
(DataPropertyAssertion :hasDescription :NumericalReasoningWeakness
  "Struggles with arithmetic, mathematical reasoning, estimation")

;;; Safety Limitations
(Declaration (Class :SafetyLimitation))
(SubClassOf :SafetyLimitation :SystemLimitation)

(Declaration (Class :HallucinationProne))
(SubClassOf :HallucinationProne :SafetyLimitation)
(DataPropertyAssertion :hasDescription :HallucinationProne
  "Generating plausible but false information confidently")

(Declaration (Class :BiasVulnerability))
(SubClassOf :BiasVulnerability :SafetyLimitation)
(DataPropertyAssertion :hasDescription :BiasVulnerability
  "Exhibiting harmful stereotypes or biased patterns from training data")

(Declaration (Class :ManipulationSusceptibility))
(SubClassOf :ManipulationSusceptibility :SafetyLimitation)
(DataPropertyAssertion :hasDescription :ManipulationSusceptibility
  "Vulnerable to prompt injection, jailbreaking, or adversarial inputs")

;;; FAILURE MODE ANALYSIS ;;;

(Declaration (Class :FailureMode))
(SubClassOf :FailureMode :SystemBehaviour)

;;; Failure Classifications
(Declaration (Class :FailureType))
(SubClassOf :FailureType :FailureMode)

(Declaration (Class :SilentFailure))
(SubClassOf :SilentFailure :FailureType)
(DataPropertyAssertion :hasDescription :SilentFailure
  "Incorrect output produced confidently without indication of error")

(Declaration (Class :CatastrophicFailure))
(SubClassOf :CatastrophicFailure :FailureType)
(DataPropertyAssertion :hasDescription :CatastrophicFailure
  "Complete breakdown or harmful output in critical scenario")

(Declaration (Class :GracefulDegradation))
(SubClassOf :GracefulDegradation :FailureType)
(DataPropertyAssertion :hasDescription :GracefulDegradation
  "Performance degrades smoothly; system acknowledges limitations")

;;; Failure Triggers
(Declaration (Class :FailureTrigger))

(Declaration (Class :OutOfDistributionInput))
(SubClassOf :OutOfDistributionInput :FailureTrigger)
(DataPropertyAssertion :hasDescription :OutOfDistributionInput
  "Input significantly different from training distribution")

(Declaration (Class :AdversarialInput))
(SubClassOf :AdversarialInput :FailureTrigger)
(DataPropertyAssertion :hasDescription :AdversarialInput
  "Deliberately crafted to fool or exploit system")

(Declaration (Class :EdgeCase))
(SubClassOf :EdgeCase :FailureTrigger)
(DataPropertyAssertion :hasDescription :EdgeCase
  "Rare or unusual scenario at boundary of system competence")

;;; EVALUATION METHODOLOGIES ;;;

(Declaration (Class :EvaluationMethodology))
(SubClassOf :EvaluationMethodology :AssessmentApproach)

;;; Benchmark-Based Evaluation
(Declaration (Class :BenchmarkEvaluation))
(SubClassOf :BenchmarkEvaluation :EvaluationMethodology)

(Declaration (Class :StandardisedBenchmark))
(SubClassOf :StandardisedBenchmark :BenchmarkEvaluation)
(DataPropertyAssertion :hasDescription :StandardisedBenchmark
  "Established benchmark datasets for systematic comparison (MMLU, HellaSwag, etc.)")

(Declaration (Class :StressTestBenchmark))
(SubClassOf :StressTestBenchmark :BenchmarkEvaluation)
(DataPropertyAssertion :hasDescription :StressTestBenchmark
  "Adversarial or challenging benchmarks testing edge case performance")

;;; Human Evaluation
(Declaration (Class :HumanEvaluation))
(SubClassOf :HumanEvaluation :EvaluationMethodology)

(Declaration (Class :ExpertAssessment))
(SubClassOf :ExpertAssessment :HumanEvaluation)
(DataPropertyAssertion :hasDescription :ExpertAssessment
  "Domain experts evaluate quality, correctness, and safety")

(Declaration (Class :UserStudy))
(SubClassOf :UserStudy :HumanEvaluation)
(DataPropertyAssertion :hasDescription :UserStudy
  "End users evaluate utility and satisfaction in realistic contexts")

;;; Automated Evaluation
(Declaration (Class :AutomatedEvaluation))
(SubClassOf :AutomatedEvaluation :EvaluationMethodology)

(Declaration (Class :ModelBasedEvaluation))
(SubClassOf :ModelBasedEvaluation :AutomatedEvaluation)
(DataPropertyAssertion :hasDescription :ModelBasedEvaluation
  "Using AI systems to evaluate other AI systems")

(Declaration (Class :SimulationBasedEvaluation))
(SubClassOf :SimulationBasedEvaluation :AutomatedEvaluation)
(DataPropertyAssertion :hasDescription :SimulationBasedEvaluation
  "Testing in simulated environments across diverse scenarios")

;;; CAPABILITY ASSESSMENT FRAMEWORKS ;;;

(Declaration (Class :AssessmentFramework))
(SubClassOf :AssessmentFramework :EvaluationFramework)

;;; Comprehensive Frameworks
(Declaration (NamedIndividual :HELMFramework))
(ClassAssertion :AssessmentFramework :HELMFramework)
(AnnotationAssertion rdfs:comment :HELMFramework
  "Holistic Evaluation of Language Models")
(DataPropertyAssertion :developedBy :HELMFramework "Stanford CRFM")
(DataPropertyAssertion :hasDescription :HELMFramework
  "Multi-dimensional evaluation across accuracy, calibration, robustness, fairness, efficiency, toxicity")

(Declaration (NamedIndividual :BIGBench))
(ClassAssertion :AssessmentFramework :BIGBench)
(DataPropertyAssertion :developedBy :BIGBench "Google Research")
(DataPropertyAssertion :hasDescription :BIGBench
  "Beyond the Imitation Game: 200+ diverse tasks testing capabilities beyond training")

(Declaration (NamedIndividual :OpenAIEvals))
(ClassAssertion :AssessmentFramework :OpenAIEvals)
(DataPropertyAssertion :developedBy :OpenAIEvals "OpenAI")
(DataPropertyAssertion :hasDescription :OpenAIEvals
  "Open-source framework for creating and running evaluations")

;;; Safety-Focused Frameworks
(Declaration (NamedIndividual :AnthropicDangerousCapabilities))
(ClassAssertion :AssessmentFramework :AnthropicDangerousCapabilities)
(DataPropertyAssertion :developedBy :AnthropicDangerousCapabilities "Anthropic")
(DataPropertyAssertion :hasDescription :AnthropicDangerousCapabilities
  "Evaluating potentially dangerous capabilities (CBRN, cyber, manipulation)")

;;; EVALUATION METRICS ;;;

(Declaration (Class :CapabilityMetric))
(SubClassOf :CapabilityMetric :PerformanceMetric)

;;; Performance Metrics
(Declaration (Class :Accuracy))
(SubClassOf :Accuracy :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :Accuracy
  "Percentage of correct predictions or outputs")

(Declaration (Class :Calibration))
(SubClassOf :Calibration :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :Calibration
  "Alignment between confidence scores and actual correctness")

(Declaration (Class :ConsistencyScore))
(SubClassOf :ConsistencyScore :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :ConsistencyScore
  "Stability of outputs across semantically equivalent inputs")

;;; Limitation Metrics
(Declaration (Class :KnownUnknownRatio))
(SubClassOf :KnownUnknownRatio :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :KnownUnknownRatio
  "Ability to recognise and acknowledge knowledge limitations")

(Declaration (Class :HallucinationRate))
(SubClassOf :HallucinationRate :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :HallucinationRate
  "Frequency of generating false information presented as factual")

(Declaration (Class :FailureDetectionRate))
(SubClassOf :FailureDetectionRate :CapabilityMetric)
(DataPropertyAssertion :measuresProperty :FailureDetectionRate
  "Ability to identify own failures or uncertain cases")

;;; DANGEROUS CAPABILITIES ;;;

(Declaration (Class :DangerousCapability))
(SubClassOf :DangerousCapability :SystemCapability)

(Declaration (Class :CBRNKnowledge))
(SubClassOf :CBRNKnowledge :DangerousCapability)
(AnnotationAssertion rdfs:comment :CBRNKnowledge
  "Chemical, Biological, Radiological, Nuclear information")
(DataPropertyAssertion :hasDescription :CBRNKnowledge
  "Knowledge enabling creation of weapons of mass destruction")

(Declaration (Class :CyberOffensiveCapability))
(SubClassOf :CyberOffensiveCapability :DangerousCapability)
(DataPropertyAssertion :hasDescription :CyberOffensiveCapability
  "Ability to assist in developing cyber attacks or exploits")

(Declaration (Class :ManipulationCapability))
(SubClassOf :ManipulationCapability :DangerousCapability)
(DataPropertyAssertion :hasDescription :ManipulationCapability
  "Persuasion techniques for deception or social engineering")

(Declaration (Class :AutomatedResearchCapability))
(SubClassOf :AutomatedResearchCapability :DangerousCapability)
(DataPropertyAssertion :hasDescription :AutomatedResearchCapability
  "Conducting original scientific research autonomously")

;;; UNCERTAINTY QUANTIFICATION ;;;

(Declaration (Class :UncertaintyQuantification))
(SubClassOf :UncertaintyQuantification :EvaluationMethodology)

(Declaration (Class :EpistemicUncertainty))
(SubClassOf :EpistemicUncertainty :UncertaintyQuantification)
(DataPropertyAssertion :hasDescription :EpistemicUncertainty
  "Uncertainty due to limited knowledge or training data")

(Declaration (Class :AleatoricUncertainty))
(SubClassOf :AleatoricUncertainty :UncertaintyQuantification)
(DataPropertyAssertion :hasDescription :AleatoricUncertainty
  "Irreducible uncertainty inherent in task or data")

(Declaration (Class :PredictiveUncertainty))
(SubClassOf :PredictiveUncertainty :UncertaintyQuantification)
(DataPropertyAssertion :hasDescription :PredictiveUncertainty
  "Total uncertainty in model predictions")

;;; DEPLOYMENT CONSIDERATIONS ;;;

(Declaration (Class :DeploymentConsideration))
(SubClassOf :DeploymentConsideration :OperationalGuidance)

(Declaration (Class :SafeOperatingEnvelope))
(SubClassOf :SafeOperatingEnvelope :DeploymentConsideration)
(DataPropertyAssertion :hasDescription :SafeOperatingEnvelope
  "Conditions under which system can operate reliably and safely")

(Declaration (Class :MonitoringRequirement))
(SubClassOf :MonitoringRequirement :DeploymentConsideration)
(DataPropertyAssertion :hasDescription :MonitoringRequirement
  "Continuous monitoring needed given identified limitations")

(Declaration (Class :HumanOversightLevel))
(SubClassOf :HumanOversightLevel :DeploymentConsideration)
(DataPropertyAssertion :hasDescription :HumanOversightLevel
  "Required human involvement based on capability assessment")

;;; BEST PRACTICES ;;;

(Declaration (Class :EvaluationBestPractice))

(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Comprehensive testing: Evaluate across multiple dimensions and benchmarks")
(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Adversarial probing: Test edge cases and failure modes systematically")
(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Uncertainty awareness: Measure calibration and confidence reliability")
(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Limitation documentation: Clearly communicate known weaknesses")
(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Continuous evaluation: Monitor performance in deployment")
(DataPropertyAssertion :hasBestPractice :EvaluationBestPractice
  "Dangerous capability screening: Assess potential for misuse")

;;; CHALLENGES ;;;

(Declaration (Class :EvaluationChallenge))
(SubClassOf :EvaluationChallenge :TechnicalChallenge)

(Declaration (Class :BenchmarkSaturation))
(SubClassOf :BenchmarkSaturation :EvaluationChallenge)
(AnnotationAssertion rdfs:comment :BenchmarkSaturation
  "Models approach ceiling performance on existing benchmarks")

(Declaration (Class :EvaluationGaming))
(SubClassOf :EvaluationGaming :EvaluationChallenge)
(AnnotationAssertion rdfs:comment :EvaluationGaming
  "Optimising specifically for benchmarks rather than true capability")

(Declaration (Class :EmergentCapabilities))
(SubClassOf :EmergentCapabilities :EvaluationChallenge)
(AnnotationAssertion rdfs:comment :EmergentCapabilities
  "Unexpected capabilities emerge at scale, hard to predict")

(Declaration (Class :EvaluationCost))
(SubClassOf :EvaluationCost :EvaluationChallenge)
(AnnotationAssertion rdfs:comment :EvaluationCost
  "Comprehensive evaluation expensive and time-consuming")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :evaluatesCapability))
(SubObjectPropertyOf :evaluatesCapability :assesses)
(ObjectPropertyDomain :evaluatesCapability :CapabilityEvaluation)
(ObjectPropertyRange :evaluatesCapability :SystemCapability)

(Declaration (ObjectProperty :identifiesLimitation))
(SubObjectPropertyOf :identifiesLimitation :discovers)
(ObjectPropertyDomain :identifiesLimitation :CapabilityEvaluation)
(ObjectPropertyRange :identifiesLimitation :SystemLimitation)

(Declaration (ObjectProperty :characterisesFailure))
(SubObjectPropertyOf :characterisesFailure :describes)
(ObjectPropertyDomain :characterisesFailure :FailureModeAnalysis)
(ObjectPropertyRange :characterisesFailure :FailureMode)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :AICapabilityEvaluation "AI Capability Evaluation"@en)
(AnnotationAssertion skos:definition :AICapabilityEvaluation
  "Systematic assessment of AI system capabilities, limitations, and failure modes through benchmark testing, human evaluation, and adversarial probing to understand performance boundaries, reliability characteristics, safe operating conditions, and potential for dangerous capabilities, informing appropriate deployment contexts and oversight requirements."@en)

(AnnotationAssertion :hasCanonicalCitation :AICapabilityEvaluation
  "Liang, P., et al. (2022). Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.")

(AnnotationAssertion :hasCanonicalCitation :AICapabilityEvaluation
  "Srivastava, A., et al. (2022). Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.")

(AnnotationAssertion :hasCanonicalCitation :AICapabilityEvaluation
  "Ganguli, D., et al. (2023). Predictability and surprise in large generative models. FAccT 2023.")

(AnnotationAssertion :hasCanonicalCitation :AICapabilityEvaluation
  "Anthropic. (2023). Core views on AI safety: When publicly available language models pose catastrophic risks. Anthropic Safety Research.")

(AnnotationAssertion :hasCanonicalCitation :AICapabilityEvaluation
  "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On calibration of modern neural networks. ICML 2017.")

(AnnotationAssertion :hasKeyFramework :AICapabilityEvaluation "HELM (Stanford)")
(AnnotationAssertion :hasKeyFramework :AICapabilityEvaluation "BIG-Bench (Google)")
(AnnotationAssertion :hasKeyFramework :AICapabilityEvaluation "OpenAI Evals")
(AnnotationAssertion :hasKeyFramework :AICapabilityEvaluation "Anthropic Dangerous Capabilities")

(AnnotationAssertion :hasEvaluationDimension :AICapabilityEvaluation "Task performance (accuracy, F1)")
(AnnotationAssertion :hasEvaluationDimension :AICapabilityEvaluation "Robustness (adversarial, OOD)")
(AnnotationAssertion :hasEvaluationDimension :AICapabilityEvaluation "Calibration (confidence reliability)")
(AnnotationAssertion :hasEvaluationDimension :AICapabilityEvaluation "Safety (toxicity, bias, hallucination)")
(AnnotationAssertion :hasEvaluationDimension :AICapabilityEvaluation "Dangerous capabilities (CBRN, cyber)")

(AnnotationAssertion dc:created :AICapabilityEvaluation "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :AICapabilityEvaluation "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :AICapabilityEvaluation "AI-0406")
(AnnotationAssertion :priorityLevel :AICapabilityEvaluation "4")
