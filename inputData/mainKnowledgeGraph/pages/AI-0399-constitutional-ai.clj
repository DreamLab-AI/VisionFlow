;;;; AI-0399: Constitutional AI
;;;; Priority 4 - Ethical AI
;;;; Self-critique and harmlessness training methodology

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :ConstitutionalAI))
(SubClassOf :ConstitutionalAI :AlignmentApproach)
(SubClassOf :ConstitutionalAI :SafetyFramework)

;;; Core Methodology
(SubClassOf :ConstitutionalAI
  (ObjectSomeValuesFrom :implements :SelfCritique))
(SubClassOf :ConstitutionalAI
  (ObjectSomeValuesFrom :follows :ConstitutionalPrinciples))
(SubClassOf :ConstitutionalAI
  (ObjectSomeValuesFrom :generates :SelfRevision))

;;; Training Objectives
(SubClassOf :ConstitutionalAI
  (ObjectAllValuesFrom :optimises :Harmlessness))
(SubClassOf :ConstitutionalAI
  (ObjectAllValuesFrom :maintains :Helpfulness))
(SubClassOf :ConstitutionalAI
  (ObjectAllValuesFrom :ensures :Honesty))

;;; Framework Properties
(DataPropertyAssertion :developedBy :ConstitutionalAI "Anthropic")
(DataPropertyAssertion :hasDescription :ConstitutionalAI
  "Training approach using AI self-critique guided by constitutional principles to improve harmlessness without extensive human feedback")

;;; TRAINING METHODOLOGY ;;;

;;; Two-Stage Process
(Declaration (Class :ConstitutionalAITraining))
(SubClassOf :ConstitutionalAITraining :TrainingMethodology)

(Declaration (Class :SupervisedLearningPhase))
(SubClassOf :SupervisedLearningPhase :ConstitutionalAITraining)
(AnnotationAssertion rdfs:comment :SupervisedLearningPhase
  "AI critiques and revises own responses according to constitutional principles")

(Declaration (Class :ReinforcementLearningPhase))
(SubClassOf :ReinforcementLearningPhase :ConstitutionalAITraining)
(AnnotationAssertion rdfs:comment :ReinforcementLearningPhase
  "AI feedback trains reward model, reducing human supervision requirement")

;;; Stage 1: Critique and Revision
(Declaration (Class :CritiqueRevisionLoop))
(SubClassOf :CritiqueRevisionLoop :SupervisedLearningPhase)

(Declaration (Class :InitialResponse))
(SubClassOf :InitialResponse :CritiqueRevisionLoop)
(DataPropertyAssertion :hasStepDescription :InitialResponse
  "Generate response to potentially harmful prompt using helpful-only model")

(Declaration (Class :CritiqueGeneration))
(SubClassOf :CritiqueGeneration :CritiqueRevisionLoop)
(SubClassOf :CritiqueGeneration
  (ObjectSomeValuesFrom :appliesPrinciple :ConstitutionalPrinciple))
(DataPropertyAssertion :hasStepDescription :CritiqueGeneration
  "AI critiques initial response according to randomly sampled constitutional principle")

(Declaration (Class :RevisionGeneration))
(SubClassOf :RevisionGeneration :CritiqueRevisionLoop)
(SubClassOf :RevisionGeneration
  (ObjectSomeValuesFrom :improves :InitialResponse))
(DataPropertyAssertion :hasStepDescription :RevisionGeneration
  "AI revises response to address critique while maintaining helpfulness")

;;; Stage 2: RL from AI Feedback
(Declaration (Class :ReinforcementLearningFromAIFeedback))
(SubClassOf :ReinforcementLearningFromAIFeedback :ReinforcementLearningPhase)
(AnnotationAssertion skos:altLabel :ReinforcementLearningFromAIFeedback "RLAIF"@en)

(Declaration (Class :AIFeedbackGeneration))
(SubClassOf :AIFeedbackGeneration :ReinforcementLearningFromAIFeedback)
(DataPropertyAssertion :hasStepDescription :AIFeedbackGeneration
  "AI compares response pairs to select more harmless option per constitution")

(Declaration (Class :PreferenceModelTraining))
(SubClassOf :PreferenceModelTraining :ReinforcementLearningFromAIFeedback)
(DataPropertyAssertion :hasStepDescription :PreferenceModelTraining
  "Train reward model on AI-generated preference comparisons")

(Declaration (Class :PolicyOptimisation))
(SubClassOf :PolicyOptimisation :ReinforcementLearningFromAIFeedback)
(DataPropertyAssertion :hasStepDescription :PolicyOptimisation
  "Fine-tune policy using reward model via reinforcement learning")

;;; CONSTITUTIONAL PRINCIPLES ;;;

(Declaration (Class :ConstitutionalPrinciple))
(SubClassOf :ConstitutionalPrinciple :EthicalPrinciple)

;;; Principle Categories
(Declaration (Class :HarmlessnesssPrinciple))
(SubClassOf :HarmlessnesssPrinciple :ConstitutionalPrinciple)
(AnnotationAssertion rdfs:comment :HarmlessnesssPrinciple
  "Principles preventing harmful outputs across multiple dimensions")

(Declaration (Class :HelpfulnessPrinciple))
(SubClassOf :HelpfulnessPrinciple :ConstitutionalPrinciple)
(AnnotationAssertion rdfs:comment :HelpfulnessPrinciple
  "Principles ensuring responses remain useful and informative")

(Declaration (Class :HonestyPrinciple))
(SubClassOf :HonestyPrinciple :ConstitutionalPrinciple)
(AnnotationAssertion rdfs:comment :HonestyPrinciple
  "Principles promoting truthfulness and accuracy")

;;; Example Principles (Concrete Individuals)
(Declaration (NamedIndividual :AvoidHarmfulContent))
(ClassAssertion :HarmlessnesssPrinciple :AvoidHarmfulContent)
(DataPropertyAssertion :hasPrincipleText :AvoidHarmfulContent
  "Please choose the response that is least intended to build a relationship with the user")

(Declaration (NamedIndividual :RespectAutonomy))
(ClassAssertion :HarmlessnesssPrinciple :RespectAutonomy)
(DataPropertyAssertion :hasPrincipleText :RespectAutonomy
  "Please choose the response that most respects human autonomy and agency")

(Declaration (NamedIndividual :AvoidDeception))
(ClassAssertion :HonestyPrinciple :AvoidDeception)
(DataPropertyAssertion :hasPrincipleText :AvoidDeception
  "Please choose the response that is most truthful and least likely to deceive")

(Declaration (NamedIndividual :MinimizeBias))
(ClassAssertion :HarmlessnesssPrinciple :MinimizeBias)
(DataPropertyAssertion :hasPrincipleText :MinimizeBias
  "Please choose the response that is least likely to exhibit harmful stereotypes or biases")

;;; Principle Sources
(Declaration (Class :PrincipleSource))

(Declaration (NamedIndividual :UNDeclarationHumanRights))
(ClassAssertion :PrincipleSource :UNDeclarationHumanRights)
(DataPropertyAssertion :hasDescription :UNDeclarationHumanRights
  "Universal Declaration of Human Rights - foundation for human dignity principles")

(Declaration (NamedIndividual :AppleTermsOfService))
(ClassAssertion :PrincipleSource :AppleTermsOfService)
(DataPropertyAssertion :hasDescription :AppleTermsOfService
  "Corporate content policies adapted for AI alignment")

(Declaration (NamedIndividual :DeepMindSparrowPrinciples))
(ClassAssertion :PrincipleSource :DeepMindSparrowPrinciples)
(DataPropertyAssertion :hasDescription :DeepMindSparrowPrinciples
  "Research-derived principles for dialogue safety")

;;; BENEFITS & ADVANTAGES ;;;

(Declaration (Class :ConstitutionalAIBenefit))
(SubClassOf :ConstitutionalAIBenefit :MethodologicalAdvantage)

(Declaration (Class :ReducedHumanSupervision))
(SubClassOf :ReducedHumanSupervision :ConstitutionalAIBenefit)
(DataPropertyAssertion :hasDescription :ReducedHumanSupervision
  "RLAIF reduces need for extensive human labelling of harmful content")

(Declaration (Class :ScalableSafety))
(SubClassOf :ScalableSafety :ConstitutionalAIBenefit)
(DataPropertyAssertion :hasDescription :ScalableSafety
  "Self-improvement process scales with model capabilities")

(Declaration (Class :ExplicitValues))
(SubClassOf :ExplicitValues :ConstitutionalAIBenefit)
(DataPropertyAssertion :hasDescription :ExplicitValues
  "Constitutional principles make training objectives transparent and auditable")

(Declaration (Class :ImprovedHarmlessness))
(SubClassOf :ImprovedHarmlessness :ConstitutionalAIBenefit)
(DataPropertyAssertion :hasDescription :ImprovedHarmlessness
  "Empirically reduces harmful outputs compared to RLHF baseline")

(Declaration (Class :LessEvasiveness))
(SubClassOf :LessEvasiveness :ConstitutionalAIBenefit)
(DataPropertyAssertion :hasDescription :LessEvasiveness
  "More willing to engage substantively rather than refusing benign requests")

;;; IMPLEMENTATION DETAILS ;;;

(Declaration (Class :ConstitutionalAIComponent))
(SubClassOf :ConstitutionalAIComponent :SystemComponent)

(Declaration (Class :ConstitutionSet))
(SubClassOf :ConstitutionSet :ConstitutionalAIComponent)
(SubClassOf :ConstitutionSet
  (ObjectSomeValuesFrom :contains :ConstitutionalPrinciple))
(DataPropertyAssertion :hasTypicalSize :ConstitutionSet
  "16-64 principles covering diverse ethical dimensions")

(Declaration (Class :CritiquePrompt))
(SubClassOf :CritiquePrompt :ConstitutionalAIComponent)
(DataPropertyAssertion :hasFormat :CritiquePrompt
  "Context + Response + Principle → Critique identifying issues")

(Declaration (Class :RevisionPrompt))
(SubClassOf :RevisionPrompt :ConstitutionalAIComponent)
(DataPropertyAssertion :hasFormat :RevisionPrompt
  "Context + Response + Critique → Improved response addressing critique")

(Declaration (Class :ComparisonPrompt))
(SubClassOf :ComparisonPrompt :ConstitutionalAIComponent)
(DataPropertyAssertion :hasFormat :ComparisonPrompt
  "Context + Response A + Response B + Principle → Preference label")

;;; EVALUATION RESULTS ;;;

(Declaration (Class :ConstitutionalAIEvaluation))
(SubClassOf :ConstitutionalAIEvaluation :SafetyEvaluation)

;;; Empirical Findings
(DataPropertyAssertion :hasEvaluationResult :ConstitutionalAIEvaluation
  "Harmlessness: Reduced harmful outputs by 52% vs RLHF baseline")
(DataPropertyAssertion :hasEvaluationResult :ConstitutionalAIEvaluation
  "Helpfulness: Maintained comparable helpfulness scores")
(DataPropertyAssertion :hasEvaluationResult :ConstitutionalAIEvaluation
  "Evasiveness: 3x reduction in unnecessary refusals")
(DataPropertyAssertion :hasEvaluationResult :ConstitutionalAIEvaluation
  "Consistency: Improved adherence to specified values")

;;; Benchmark Performance
(Declaration (Class :HarmlessnessMetric))
(SubClassOf :HarmlessnessMetric :SafetyMetric)

(Declaration (NamedIndividual :AnthropicHarmlessnessTest))
(ClassAssertion :HarmlessnessMetric :AnthropicHarmlessnessTest)
(DataPropertyAssertion :hasDescription :AnthropicHarmlessnessTest
  "Human evaluations of response harmfulness across adversarial prompts")

;;; EXTENSIONS & VARIATIONS ;;;

(Declaration (Class :ConstitutionalAIVariant))
(SubClassOf :ConstitutionalAIVariant :ConstitutionalAI)

(Declaration (Class :ChainOfThoughtConstitutional))
(SubClassOf :ChainOfThoughtConstitutional :ConstitutionalAIVariant)
(DataPropertyAssertion :hasDescription :ChainOfThoughtConstitutional
  "Incorporating reasoning traces before critique and revision")

(Declaration (Class :MultiTurnConstitutional))
(SubClassOf :MultiTurnConstitutional :ConstitutionalAIVariant)
(DataPropertyAssertion :hasDescription :MultiTurnConstitutional
  "Iterative critique-revision over multiple rounds for complex cases")

(Declaration (Class :AdaptiveConstitution))
(SubClassOf :AdaptiveConstitution :ConstitutionalAIVariant)
(DataPropertyAssertion :hasDescription :AdaptiveConstitution
  "Dynamically selecting principles based on prompt characteristics")

;;; LIMITATIONS & CHALLENGES ;;;

(Declaration (Class :ConstitutionalAILimitation))
(SubClassOf :ConstitutionalAILimitation :MethodologicalChallenge)

(Declaration (Class :PrincipleDesign))
(SubClassOf :PrincipleDesign :ConstitutionalAILimitation)
(AnnotationAssertion rdfs:comment :PrincipleDesign
  "Difficulty crafting comprehensive, unambiguous constitutional principles")

(Declaration (Class :PrincipleConflict))
(SubClassOf :PrincipleConflict :ConstitutionalAILimitation)
(AnnotationAssertion rdfs:comment :PrincipleConflict
  "Handling cases where constitutional principles contradict")

(Declaration (Class :AIEvaluatorLimitations))
(SubClassOf :AIEvaluatorLimitations :ConstitutionalAILimitation)
(AnnotationAssertion rdfs:comment :AIEvaluatorLimitations
  "Self-evaluation quality depends on model's own understanding of principles")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :critiques))
(SubObjectPropertyOf :critiques :evaluates)
(ObjectPropertyDomain :critiques :AIModel)
(ObjectPropertyRange :critiques :GeneratedResponse)

(Declaration (ObjectProperty :revises))
(SubObjectPropertyOf :revises :transforms)
(ObjectPropertyDomain :revises :AIModel)
(ObjectPropertyRange :revises :GeneratedResponse)

(Declaration (ObjectProperty :adheresToPrinciple))
(SubObjectPropertyOf :adheresToPrinciple :satisfies)
(ObjectPropertyDomain :adheresToPrinciple :GeneratedResponse)
(ObjectPropertyRange :adheresToPrinciple :ConstitutionalPrinciple)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :ConstitutionalAI "Constitutional AI"@en)
(AnnotationAssertion skos:altLabel :ConstitutionalAI "CAI"@en)
(AnnotationAssertion skos:definition :ConstitutionalAI
  "Training methodology developed by Anthropic where AI systems learn to critique and revise their own responses according to explicit constitutional principles, enabling scalable alignment through self-improvement and reducing reliance on extensive human feedback for safety training."@en)

(AnnotationAssertion :hasCanonicalCitation :ConstitutionalAI
  "Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073.")

(AnnotationAssertion :hasCanonicalCitation :ConstitutionalAI
  "Bai, Y., Jones, A., Ndousse, K., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.")

(AnnotationAssertion :hasTechnicalInnovation :ConstitutionalAI
  "Self-critique: AI evaluates own outputs")
(AnnotationAssertion :hasTechnicalInnovation :ConstitutionalAI
  "RLAIF: RL from AI feedback reduces human labelling")
(AnnotationAssertion :hasTechnicalInnovation :ConstitutionalAI
  "Explicit principles: Transparent value specification")
(AnnotationAssertion :hasTechnicalInnovation :ConstitutionalAI
  "Scalable oversight: Self-improvement with model capability")

(AnnotationAssertion :implementedBy :ConstitutionalAI "Anthropic Claude")
(AnnotationAssertion :hasKeyContributor :ConstitutionalAI "Yuntao Bai")
(AnnotationAssertion :hasKeyContributor :ConstitutionalAI "Saurav Kadavath")

(AnnotationAssertion dc:created :ConstitutionalAI "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :ConstitutionalAI "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :ConstitutionalAI "AI-0399")
(AnnotationAssertion :priorityLevel :ConstitutionalAI "4")
