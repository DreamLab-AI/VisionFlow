;;;; AI-0402: AI Safety Benchmarks
;;;; Priority 4 - Ethical AI
;;;; Standardized evaluation of safety properties

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :AISafetyBenchmark))
(SubClassOf :AISafetyBenchmark :EvaluationBenchmark)
(SubClassOf :AISafetyBenchmark :SafetyEvaluation)

;;; Core Properties
(SubClassOf :AISafetyBenchmark
  (ObjectSomeValuesFrom :measures :SafetyProperty))
(SubClassOf :AISafetyBenchmark
  (ObjectSomeValuesFrom :evaluates :AlignmentQuality))
(SubClassOf :AISafetyBenchmark
  (ObjectSomeValuesFrom :quantifies :RiskLevel))

;;; Benchmark Characteristics
(SubClassOf :AISafetyBenchmark
  (ObjectAllValuesFrom :provides :StandardisedMetric))
(SubClassOf :AISafetyBenchmark
  (ObjectAllValuesFrom :enables :ComparativeEvaluation))
(SubClassOf :AISafetyBenchmark
  (ObjectAllValuesFrom :tracks :SafetyProgress))

;;; Properties
(DataPropertyAssertion :hasDescription :AISafetyBenchmark
  "Standardised datasets and evaluation protocols for measuring AI safety properties including truthfulness, toxicity, bias, and harmful capability potential")

;;; TRUTHFULNESS BENCHMARKS ;;;

(Declaration (Class :TruthfulnessBenchmark))
(SubClassOf :TruthfulnessBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :TruthfulQA))
(ClassAssertion :TruthfulnessBenchmark :TruthfulQA)
(DataPropertyAssertion :hasDescription :TruthfulQA
  "817 questions spanning 38 categories where humans often answer incorrectly due to misconceptions")
(DataPropertyAssertion :measuresProperty :TruthfulQA
  "Model's tendency to generate truthful answers vs. imitating human falsehoods")
(DataPropertyAssertion :developedBy :TruthfulQA "Stephanie Lin et al., Oxford/OpenAI")
(DataPropertyAssertion :hasURL :TruthfulQA "https://github.com/sylinrl/TruthfulQA")

(Declaration (Class :TruthfulQAMetric))
(SubClassOf :TruthfulQAMetric :SafetyMetric)

(Declaration (NamedIndividual :TruthfulQATruthfulness))
(ClassAssertion :TruthfulQAMetric :TruthfulQATruthfulness)
(DataPropertyAssertion :measuresProperty :TruthfulQATruthfulness
  "Percentage of truthful answers")

(Declaration (NamedIndividual :TruthfulQAInformativeness))
(ClassAssertion :TruthfulQAMetric :TruthfulQAInformativeness)
(DataPropertyAssertion :measuresProperty :TruthfulQAInformativeness
  "Percentage of informative (non-evasive) answers")

;;; Question Categories (Examples)
(DataPropertyAssertion :hasCategory :TruthfulQA "Misconceptions")
(DataPropertyAssertion :hasCategory :TruthfulQA "Fiction")
(DataPropertyAssertion :hasCategory :TruthfulQA "Conspiracies")
(DataPropertyAssertion :hasCategory :TruthfulQA "Superstitions")
(DataPropertyAssertion :hasCategory :TruthfulQA "Stereotypes")

;;; TOXICITY BENCHMARKS ;;;

(Declaration (Class :ToxicityBenchmark))
(SubClassOf :ToxicityBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :RealToxicityPrompts))
(ClassAssertion :ToxicityBenchmark :RealToxicityPrompts)
(DataPropertyAssertion :hasDescription :RealToxicityPrompts
  "100K naturally occurring sentence beginnings from web corpus for testing toxic continuation generation")
(DataPropertyAssertion :measuresProperty :RealToxicityPrompts
  "Likelihood of generating toxic text continuations from realistic prompts")
(DataPropertyAssertion :developedBy :RealToxicityPrompts "Gehman et al., Allen Institute for AI")
(DataPropertyAssertion :hasURL :RealToxicityPrompts "https://github.com/allenai/real-toxicity-prompts")

(Declaration (Class :ToxicityDimension))

(Declaration (NamedIndividual :ToxicityScore))
(ClassAssertion :ToxicityDimension :ToxicityScore)
(DataPropertyAssertion :hasDefinition :ToxicityScore
  "Overall toxicity from Perspective API (0-1 scale)")

(Declaration (NamedIndividual :SevereToxicity))
(ClassAssertion :ToxicityDimension :SevereToxicity)
(DataPropertyAssertion :hasDefinition :SevereToxicity
  "Extremely hateful, aggressive, or disrespectful language")

(Declaration (NamedIndividual :IdentityAttack))
(ClassAssertion :ToxicityDimension :IdentityAttack)
(DataPropertyAssertion :hasDefinition :IdentityAttack
  "Negative comments about identity characteristics")

(Declaration (NamedIndividual :Profanity))
(ClassAssertion :ToxicityDimension :Profanity)
(DataPropertyAssertion :hasDefinition :Profanity
  "Swear words and curse language")

(Declaration (NamedIndividual :Threat))
(ClassAssertion :ToxicityDimension :Threat)
(DataPropertyAssertion :hasDefinition :Threat
  "Threatening language toward individuals or groups")

(Declaration (NamedIndividual :SexuallyExplicit))
(ClassAssertion :ToxicityDimension :SexuallyExplicit)
(DataPropertyAssertion :hasDefinition :SexuallyExplicit
  "Sexually explicit content")

;;; Additional Toxicity Benchmarks
(Declaration (NamedIndividual :CivilComments))
(ClassAssertion :ToxicityBenchmark :CivilComments)
(DataPropertyAssertion :hasDescription :CivilComments
  "2M public comments labelled for toxicity from Civil Comments platform")

(Declaration (NamedIndividual :ToxiGen))
(ClassAssertion :ToxicityBenchmark :ToxiGen)
(DataPropertyAssertion :hasDescription :ToxiGen
  "274K implicitly toxic and benign statements about minority groups")

;;; BIAS BENCHMARKS ;;;

(Declaration (Class :BiasBenchmark))
(SubClassOf :BiasBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :StereoSet))
(ClassAssertion :BiasBenchmark :StereoSet)
(DataPropertyAssertion :hasDescription :StereoSet
  "Measures stereotypical associations across gender, profession, race, and religion")
(DataPropertyAssertion :developedBy :StereoSet "Nadeem et al., UCLA/AWS")

(Declaration (NamedIndividual :Winogender))
(ClassAssertion :BiasBenchmark :Winogender)
(DataPropertyAssertion :hasDescription :Winogender
  "Coreference resolution dataset testing gender bias in occupational stereotypes")

(Declaration (NamedIndividual :BBQ))
(ClassAssertion :BiasBenchmark :BBQ)
(AnnotationAssertion skos:altLabel :BBQ "Bias Benchmark for QA"@en)
(DataPropertyAssertion :hasDescription :BBQ
  "Question-answering benchmark measuring social biases across 11 categories")

(Declaration (NamedIndividual :CrowSPairs))
(ClassAssertion :BiasBenchmark :CrowSPairs)
(DataPropertyAssertion :hasDescription :CrowSPairs
  "Crowdsourced Stereotype Pairs challenging models on stereotypical biases")

;;; HARMFUL CAPABILITY BENCHMARKS ;;;

(Declaration (Class :HarmfulCapabilityBenchmark))
(SubClassOf :HarmfulCapabilityBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :DangerousCapabilityEvaluations))
(ClassAssertion :HarmfulCapabilityBenchmark :DangerousCapabilityEvaluations)
(DataPropertyAssertion :hasDescription :DangerousCapabilityEvaluations
  "Anthropic's evaluations for chemical, biological, radiological, nuclear (CBRN) knowledge")

(Declaration (Class :CBRNCategory))

(Declaration (NamedIndividual :ChemicalWeapons))
(ClassAssertion :CBRNCategory :ChemicalWeapons)
(DataPropertyAssertion :hasDescription :ChemicalWeapons
  "Knowledge enabling chemical weapon synthesis")

(Declaration (NamedIndividual :BiologicalWeapons))
(ClassAssertion :CBRNCategory :BiologicalWeapons)
(DataPropertyAssertion :hasDescription :BiologicalWeapons
  "Knowledge enabling biological threat creation")

(Declaration (NamedIndividual :CyberOffensive))
(ClassAssertion :CBRNCategory :CyberOffensive)
(DataPropertyAssertion :hasDescription :CyberOffensive
  "Capability to assist in cyber attacks and exploits")

;;; ROBUSTNESS BENCHMARKS ;;;

(Declaration (Class :RobustnessBenchmark))
(SubClassOf :RobustnessBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :AdvGLUE))
(ClassAssertion :RobustnessBenchmark :AdvGLUE)
(DataPropertyAssertion :hasDescription :AdvGLUE
  "Adversarial version of GLUE benchmark testing textual robustness")

(Declaration (NamedIndividual :HateCheck))
(ClassAssertion :RobustnessBenchmark :HateCheck)
(DataPropertyAssertion :hasDescription :HateCheck
  "Functional tests for hate speech detection robustness")

;;; ALIGNMENT BENCHMARKS ;;;

(Declaration (Class :AlignmentBenchmark))
(SubClassOf :AlignmentBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :AnthropicHHH))
(ClassAssertion :AlignmentBenchmark :AnthropicHHH)
(AnnotationAssertion rdfs:comment :AnthropicHHH
  "Helpful, Honest, Harmless evaluation dataset")
(DataPropertyAssertion :hasDescription :AnthropicHHH
  "Red team dialogues testing alignment across helpfulness, honesty, harmlessness")

(Declaration (NamedIndividual :SaferDialogues))
(ClassAssertion :AlignmentBenchmark :SaferDialogues)
(DataPropertyAssertion :hasDescription :SaferDialogues
  "Multi-turn conversations testing persistent safety under adversarial probing")

;;; JAILBREAK BENCHMARKS ;;;

(Declaration (Class :JailbreakBenchmark))
(SubClassOf :JailbreakBenchmark :AISafetyBenchmark)

(Declaration (NamedIndividual :JailbreakPrompts))
(ClassAssertion :JailbreakBenchmark :JailbreakPrompts)
(DataPropertyAssertion :hasDescription :JailbreakPrompts
  "Dataset of prompts attempting to circumvent safety guardrails")

(Declaration (NamedIndividual :AdversarialNLI))
(ClassAssertion :JailbreakBenchmark :AdversarialNLI)
(DataPropertyAssertion :hasDescription :AdversarialNLI
  "Adversarial natural language inference testing reasoning robustness")

;;; COMPREHENSIVE SAFETY SUITES ;;;

(Declaration (Class :SafetyEvaluationSuite))
(SubClassOf :SafetyEvaluationSuite :AISafetyBenchmark)

(Declaration (NamedIndividual :HELMSafety))
(ClassAssertion :SafetyEvaluationSuite :HELMSafety)
(AnnotationAssertion rdfs:comment :HELMSafety
  "Holistic Evaluation of Language Models - Safety Module")
(DataPropertyAssertion :hasDescription :HELMSafety
  "Comprehensive benchmark covering toxicity, bias, disinformation, copyright")
(DataPropertyAssertion :developedBy :HELMSafety "Stanford CRFM")

(Declaration (NamedIndividual :OpenAIModelSpec))
(ClassAssertion :SafetyEvaluationSuite :OpenAIModelSpec)
(DataPropertyAssertion :hasDescription :OpenAIModelSpec
  "OpenAI's safety evaluation protocol across multiple risk categories")

;;; BENCHMARK PROPERTIES ;;;

(Declaration (Class :BenchmarkProperty))

(Declaration (Class :BenchmarkSize))
(SubClassOf :BenchmarkSize :BenchmarkProperty)
(AnnotationAssertion rdfs:comment :BenchmarkSize
  "Number of test examples in benchmark dataset")

(Declaration (Class :BenchmarkDifficulty))
(SubClassOf :BenchmarkDifficulty :BenchmarkProperty)
(AnnotationAssertion rdfs:comment :BenchmarkDifficulty
  "Challenge level for current AI systems")

(Declaration (Class :BenchmarkCoverage))
(SubClassOf :BenchmarkCoverage :BenchmarkProperty)
(AnnotationAssertion rdfs:comment :BenchmarkCoverage
  "Breadth of safety risks covered")

;;; EVALUATION METHODOLOGY ;;;

(Declaration (Class :BenchmarkEvaluationMethod))

(Declaration (Class :AutomaticEvaluation))
(SubClassOf :AutomaticEvaluation :BenchmarkEvaluationMethod)
(AnnotationAssertion rdfs:comment :AutomaticEvaluation
  "Automated metrics and classifiers for scalable evaluation")

(Declaration (Class :HumanEvaluation))
(SubClassOf :HumanEvaluation :BenchmarkEvaluationMethod)
(AnnotationAssertion rdfs:comment :HumanEvaluation
  "Human annotators assess safety violations")

(Declaration (Class :ModelBasedEvaluation))
(SubClassOf :ModelBasedEvaluation :BenchmarkEvaluationMethod)
(AnnotationAssertion rdfs:comment :ModelBasedEvaluation
  "Using AI models (e.g., constitutional AI) to evaluate other models")

;;; BENCHMARK LIMITATIONS ;;;

(Declaration (Class :BenchmarkLimitation))

(Declaration (Class :DatasetBias))
(SubClassOf :DatasetBias :BenchmarkLimitation)
(AnnotationAssertion rdfs:comment :DatasetBias
  "Benchmark may not represent full distribution of real-world risks")

(Declaration (Class :GamingRisk))
(SubClassOf :GamingRisk :BenchmarkLimitation)
(AnnotationAssertion rdfs:comment :GamingRisk
  "Models can overfit to specific benchmarks without generalised safety")

(Declaration (Class :EvolvingThreats))
(SubClassOf :EvolvingThreats :BenchmarkLimitation)
(AnnotationAssertion rdfs:comment :EvolvingThreats
  "New attack vectors emerge not covered by existing benchmarks")

(Declaration (Class :EvaluationCost))
(SubClassOf :EvaluationCost :BenchmarkLimitation)
(AnnotationAssertion rdfs:comment :EvaluationCost
  "Human evaluation expensive and difficult to scale")

;;; BENCHMARK DESIGN PRINCIPLES ;;;

(Declaration (Class :BenchmarkDesignPrinciple))

(DataPropertyAssertion :hasDesignPrinciple :BenchmarkDesignPrinciple
  "Adversarial: Include challenging cases that expose failures")
(DataPropertyAssertion :hasDesignPrinciple :BenchmarkDesignPrinciple
  "Diverse: Cover wide range of risk categories and demographics")
(DataPropertyAssertion :hasDesignPrinciple :BenchmarkDesignPrinciple
  "Realistic: Reflect actual deployment scenarios")
(DataPropertyAssertion :hasDesignPrinciple :BenchmarkDesignPrinciple
  "Standardised: Enable reproducible comparisons")
(DataPropertyAssertion :hasDesignPrinciple :BenchmarkDesignPrinciple
  "Evolving: Regular updates to address new threats")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :benchmarks))
(SubObjectPropertyOf :benchmarks :evaluates)
(ObjectPropertyDomain :benchmarks :AISafetyBenchmark)
(ObjectPropertyRange :benchmarks :AIModel)

(Declaration (ObjectProperty :measuresProperty))
(SubObjectPropertyOf :measuresProperty :quantifies)
(ObjectPropertyDomain :measuresProperty :SafetyMetric)
(ObjectPropertyRange :measuresProperty :SafetyProperty)

(Declaration (ObjectProperty :coversRisk))
(SubObjectPropertyOf :coversRisk :addresses)
(ObjectPropertyDomain :coversRisk :AISafetyBenchmark)
(ObjectPropertyRange :coversRisk :SafetyRisk)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :AISafetyBenchmark "AI Safety Benchmarks"@en)
(AnnotationAssertion skos:definition :AISafetyBenchmark
  "Standardised evaluation datasets and protocols for measuring AI safety properties including truthfulness (TruthfulQA), toxicity (RealToxicityPrompts), bias, harmful capabilities, and alignment quality, enabling systematic comparison and tracking of safety progress across models and training approaches."@en)

(AnnotationAssertion :hasCanonicalCitation :AISafetyBenchmark
  "Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. ACL 2022.")

(AnnotationAssertion :hasCanonicalCitation :AISafetyBenchmark
  "Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N.A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. EMNLP 2020.")

(AnnotationAssertion :hasCanonicalCitation :AISafetyBenchmark
  "Nadeem, M., Bethke, A., & Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models. ACL 2021.")

(AnnotationAssertion :hasCanonicalCitation :AISafetyBenchmark
  "Liang, P., et al. (2022). Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.")

(AnnotationAssertion :hasCanonicalCitation :AISafetyBenchmark
  "Ganguli, D., et al. (2022). The capacity for moral self-correction in large language models. arXiv preprint arXiv:2302.07459.")

(AnnotationAssertion :hasKeyBenchmark :AISafetyBenchmark "TruthfulQA (truthfulness)")
(AnnotationAssertion :hasKeyBenchmark :AISafetyBenchmark "RealToxicityPrompts (toxicity)")
(AnnotationAssertion :hasKeyBenchmark :AISafetyBenchmark "StereoSet (bias)")
(AnnotationAssertion :hasKeyBenchmark :AISafetyBenchmark "BBQ (social bias)")
(AnnotationAssertion :hasKeyBenchmark :AISafetyBenchmark "Anthropic HHH (alignment)")

(AnnotationAssertion dc:created :AISafetyBenchmark "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :AISafetyBenchmark "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :AISafetyBenchmark "AI-0402")
(AnnotationAssertion :priorityLevel :AISafetyBenchmark "4")
