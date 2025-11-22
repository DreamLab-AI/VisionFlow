;;;; AI-0403: Harmful Output Detection
;;;; Priority 4 - Ethical AI
;;;; Automated identification of toxic and unsafe content

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :HarmfulOutputDetection))
(SubClassOf :HarmfulOutputDetection :SafetyMechanism)
(SubClassOf :HarmfulOutputDetection :ContentModeration)

;;; Core Capabilities
(SubClassOf :HarmfulOutputDetection
  (ObjectSomeValuesFrom :identifies :HarmfulContent))
(SubClassOf :HarmfulOutputDetection
  (ObjectSomeValuesFrom :classifies :ToxicityLevel))
(SubClassOf :HarmfulOutputDetection
  (ObjectSomeValuesFrom :prevents :UnsafeOutput))

;;; Detection Objectives
(SubClassOf :HarmfulOutputDetection
  (ObjectAllValuesFrom :filters :ProhibitedContent))
(SubClassOf :HarmfulOutputDetection
  (ObjectAllValuesFrom :flags :PolicyViolation))
(SubClassOf :HarmfulOutputDetection
  (ObjectAllValuesFrom :monitors :ContentSafety))

;;; System Properties
(DataPropertyAssertion :hasDescription :HarmfulOutputDetection
  "Automated systems for identifying and filtering toxic, misleading, biased, and otherwise harmful content in AI-generated outputs before delivery to users")

;;; HARMFUL CONTENT TAXONOMY ;;;

(Declaration (Class :HarmfulContent))
(SubClassOf :HarmfulContent :ContentCategory)

;;; Toxicity Categories
(Declaration (Class :ToxicContent))
(SubClassOf :ToxicContent :HarmfulContent)

(Declaration (Class :HateSpeech))
(SubClassOf :HateSpeech :ToxicContent)
(DataPropertyAssertion :hasDefinition :HateSpeech
  "Content attacking people based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, disability)")

(Declaration (Class :Harassment))
(SubClassOf :Harassment :ToxicContent)
(DataPropertyAssertion :hasDefinition :Harassment
  "Malicious content intended to intimidate, threaten, or demean individuals")

(Declaration (Class :Profanity))
(SubClassOf :Profanity :ToxicContent)
(DataPropertyAssertion :hasDefinition :Profanity
  "Obscene or vulgar language including swear words")

(Declaration (Class :SexualContent))
(SubClassOf :SexualContent :ToxicContent)
(DataPropertyAssertion :hasDefinition :SexualContent
  "Sexually explicit or suggestive content inappropriate for general audiences")

(Declaration (Class :ViolentContent))
(SubClassOf :ViolentContent :ToxicContent)
(DataPropertyAssertion :hasDefinition :ViolentContent
  "Graphic descriptions of violence, gore, or self-harm")

;;; Misinformation
(Declaration (Class :MisinformationContent))
(SubClassOf :MisinformationContent :HarmfulContent)

(Declaration (Class :Falsehood))
(SubClassOf :Falsehood :MisinformationContent)
(DataPropertyAssertion :hasDefinition :Falsehood
  "Demonstrably false factual claims")

(Declaration (Class :Conspiracy))
(SubClassOf :Conspiracy :MisinformationContent)
(DataPropertyAssertion :hasDefinition :Conspiracy
  "Unfounded conspiracy theories without credible evidence")

(Declaration (Class :MedicalMisinformation))
(SubClassOf :MedicalMisinformation :MisinformationContent)
(DataPropertyAssertion :hasDefinition :MedicalMisinformation
  "False or misleading health and medical information")

;;; Harmful Instructions
(Declaration (Class :HarmfulInstructions))
(SubClassOf :HarmfulInstructions :HarmfulContent)

(Declaration (Class :DangerousActivity))
(SubClassOf :DangerousActivity :HarmfulInstructions)
(DataPropertyAssertion :hasDefinition :DangerousActivity
  "Instructions for activities posing physical harm risk")

(Declaration (Class :IllegalActivity))
(SubClassOf :IllegalActivity :HarmfulInstructions)
(DataPropertyAssertion :hasDefinition :IllegalActivity
  "Guidance on illegal acts (fraud, theft, violence)")

(Declaration (Class :SelfHarmGuidance))
(SubClassOf :SelfHarmGuidance :HarmfulInstructions)
(DataPropertyAssertion :hasDefinition :SelfHarmGuidance
  "Content encouraging or enabling self-injury or suicide")

;;; Bias and Stereotypes
(Declaration (Class :BiasedContent))
(SubClassOf :BiasedContent :HarmfulContent)

(Declaration (Class :StereotypicalContent))
(SubClassOf :StereotypicalContent :BiasedContent)
(DataPropertyAssertion :hasDefinition :StereotypicalContent
  "Content reinforcing harmful stereotypes about groups")

(Declaration (Class :DiscriminatoryContent))
(SubClassOf :DiscriminatoryContent :BiasedContent)
(DataPropertyAssertion :hasDefinition :DiscriminatoryContent
  "Content promoting unequal treatment based on group membership")

;;; Privacy Violations
(Declaration (Class :PrivacyViolatingContent))
(SubClassOf :PrivacyViolatingContent :HarmfulContent)

(Declaration (Class :PersonalInformationExposure))
(SubClassOf :PersonalInformationExposure :PrivacyViolatingContent)
(DataPropertyAssertion :hasDefinition :PersonalInformationExposure
  "Revealing private information without consent (doxing)")

;;; DETECTION APPROACHES ;;;

(Declaration (Class :DetectionMethod))
(SubClassOf :DetectionMethod :TechnicalApproach)

;;; Classifier-Based Detection
(Declaration (Class :ToxicityClassifier))
(SubClassOf :ToxicityClassifier :DetectionMethod)

(Declaration (NamedIndividual :PerspectiveAPI))
(ClassAssertion :ToxicityClassifier :PerspectiveAPI)
(DataPropertyAssertion :developedBy :PerspectiveAPI "Google Jigsaw")
(DataPropertyAssertion :hasDescription :PerspectiveAPI
  "ML models scoring toxic comment attributes (toxicity, severe toxicity, identity attack, etc.)")
(DataPropertyAssertion :hasURL :PerspectiveAPI "https://www.perspectiveapi.com/")

(Declaration (Class :PerspectiveAttribute))

(Declaration (NamedIndividual :PerspectiveToxicity))
(ClassAssertion :PerspectiveAttribute :PerspectiveToxicity)
(DataPropertyAssertion :hasDefinition :PerspectiveToxicity
  "Rude, disrespectful, or unreasonable comment likely to make people leave discussion")

(Declaration (NamedIndividual :PerspectiveSevereToxicity))
(ClassAssertion :PerspectiveAttribute :PerspectiveSevereToxicity)
(DataPropertyAssertion :hasDefinition :PerspectiveSevereToxicity
  "Very hateful, aggressive, disrespectful, or otherwise toxic comment")

(Declaration (NamedIndividual :PerspectiveIdentityAttack))
(ClassAssertion :PerspectiveAttribute :PerspectiveIdentityAttack)
(DataPropertyAssertion :hasDefinition :PerspectiveIdentityAttack
  "Negative or hateful comment targeting identity or protected characteristics")

;;; Rule-Based Detection
(Declaration (Class :RuleBasedDetection))
(SubClassOf :RuleBasedDetection :DetectionMethod)

(Declaration (Class :KeywordFiltering))
(SubClassOf :KeywordFiltering :RuleBasedDetection)
(DataPropertyAssertion :hasDescription :KeywordFiltering
  "Blocklists of prohibited words and phrases")
(AnnotationAssertion rdfs:comment :KeywordFiltering
  "Fast but limited by circumvention via misspellings and synonyms")

(Declaration (Class :RegexPatternMatching))
(SubClassOf :RegexPatternMatching :RuleBasedDetection)
(DataPropertyAssertion :hasDescription :RegexPatternMatching
  "Regular expression patterns for harmful content signatures")

;;; Hybrid Approaches
(Declaration (Class :HybridDetection))
(SubClassOf :HybridDetection :DetectionMethod)
(SubClassOf :HybridDetection
  (ObjectIntersectionOf :ToxicityClassifier :RuleBasedDetection))
(DataPropertyAssertion :hasDescription :HybridDetection
  "Combining ML classifiers with rule-based filters for comprehensive coverage")

;;; Context-Aware Detection
(Declaration (Class :ContextAwareDetection))
(SubClassOf :ContextAwareDetection :DetectionMethod)

(Declaration (Class :IntentAnalysis))
(SubClassOf :IntentAnalysis :ContextAwareDetection)
(DataPropertyAssertion :hasDescription :IntentAnalysis
  "Distinguishing harmful intent from educational or journalistic content")

(Declaration (Class :ConversationalContextDetection))
(SubClassOf :ConversationalContextDetection :ContextAwareDetection)
(DataPropertyAssertion :hasDescription :ConversationalContextDetection
  "Analysing multi-turn context to detect subtle manipulation or escalating toxicity")

;;; DETECTION ARCHITECTURES ;;;

(Declaration (Class :DetectionArchitecture))
(SubClassOf :DetectionArchitecture :SystemArchitecture)

;;; Pre-Generation Filtering
(Declaration (Class :InputFiltering))
(SubClassOf :InputFiltering :DetectionArchitecture)
(DataPropertyAssertion :hasDescription :InputFiltering
  "Detecting and blocking harmful prompts before generation")

;;; Post-Generation Filtering
(Declaration (Class :OutputFiltering))
(SubClassOf :OutputFiltering :DetectionArchitecture)
(DataPropertyAssertion :hasDescription :OutputFiltering
  "Screening generated outputs before showing to users")

;;; Real-Time Monitoring
(Declaration (Class :StreamingDetection))
(SubClassOf :StreamingDetection :DetectionArchitecture)
(DataPropertyAssertion :hasDescription :StreamingDetection
  "Monitoring outputs token-by-token during generation for early intervention")

;;; Multi-Stage Pipeline
(Declaration (Class :CascadedDetection))
(SubClassOf :CascadedDetection :DetectionArchitecture)
(DataPropertyAssertion :hasDescription :CascadedDetection
  "Multiple detection stages with increasing sophistication and computational cost")

;;; MITIGATION STRATEGIES ;;;

(Declaration (Class :MitigationStrategy))
(SubClassOf :MitigationStrategy :SafetyMechanism)

(Declaration (Class :ContentBlocking))
(SubClassOf :ContentBlocking :MitigationStrategy)
(DataPropertyAssertion :hasDescription :ContentBlocking
  "Preventing delivery of flagged harmful content to users")

(Declaration (Class :ContentModification))
(SubClassOf :ContentModification :MitigationStrategy)
(DataPropertyAssertion :hasDescription :ContentModification
  "Automatically editing or redacting harmful portions while preserving helpful content")

(Declaration (Class :WarningLabels))
(SubClassOf :WarningLabels :MitigationStrategy)
(DataPropertyAssertion :hasDescription :WarningLabels
  "Displaying warnings alongside potentially sensitive content")

(Declaration (Class :AlternativeResponse))
(SubClassOf :AlternativeResponse :MitigationStrategy)
(DataPropertyAssertion :hasDescription :AlternativeResponse
  "Regenerating safer alternative responses when harmful content detected")

(Declaration (Class :HumanEscalation))
(SubClassOf :HumanEscalation :MitigationStrategy)
(DataPropertyAssertion :hasDescription :HumanEscalation
  "Routing edge cases to human reviewers for judgment")

;;; EVALUATION METRICS ;;;

(Declaration (Class :DetectionMetric))
(SubClassOf :DetectionMetric :PerformanceMetric)

(Declaration (Class :TruePositiveRate))
(SubClassOf :TruePositiveRate :DetectionMetric)
(AnnotationAssertion skos:altLabel :TruePositiveRate "Recall"@en)
(AnnotationAssertion skos:altLabel :TruePositiveRate "Sensitivity"@en)
(DataPropertyAssertion :measuresProperty :TruePositiveRate
  "Percentage of actual harmful content correctly identified")

(Declaration (Class :FalsePositiveRate))
(SubClassOf :FalsePositiveRate :DetectionMetric)
(DataPropertyAssertion :measuresProperty :FalsePositiveRate
  "Percentage of benign content incorrectly flagged as harmful")

(Declaration (Class :Precision))
(SubClassOf :Precision :DetectionMetric)
(DataPropertyAssertion :measuresProperty :Precision
  "Percentage of flagged content that is actually harmful")

(Declaration (Class :F1Score))
(SubClassOf :F1Score :DetectionMetric)
(DataPropertyAssertion :hasFormula :F1Score
  "2 × (Precision × Recall) / (Precision + Recall)")

(Declaration (Class :AUROC))
(SubClassOf :AUROC :DetectionMetric)
(AnnotationAssertion rdfs:comment :AUROC
  "Area Under Receiver Operating Characteristic curve")
(DataPropertyAssertion :measuresProperty :AUROC
  "Overall discrimination ability across thresholds")

;;; CHALLENGES ;;;

(Declaration (Class :DetectionChallenge))
(SubClassOf :DetectionChallenge :TechnicalChallenge)

(Declaration (Class :ContextDependency))
(SubClassOf :ContextDependency :DetectionChallenge)
(AnnotationAssertion rdfs:comment :ContextDependency
  "Same text harmless in one context, harmful in another")

(Declaration (Class :SubtletyChallenge))
(SubClassOf :SubtletyChallenge :DetectionChallenge)
(AnnotationAssertion rdfs:comment :SubtletyChallenge
  "Implicit bias and dog whistles harder to detect than explicit toxicity")

(Declaration (Class :EvasionTechniques))
(SubClassOf :EvasionTechniques :DetectionChallenge)
(AnnotationAssertion rdfs:comment :EvasionTechniques
  "Deliberate obfuscation via misspellings, leetspeak, emojis")

(Declaration (Class :FalsePositiveTrade off))
(SubClassOf :FalsePositiveTrade off :DetectionChallenge)
(AnnotationAssertion rdfs:comment :FalsePositiveTrade off
  "Reducing false positives risks missing true harms; over-filtering limits utility")

(Declaration (Class :MultilingualDetection))
(SubClassOf :MultilingualDetection :DetectionChallenge)
(AnnotationAssertion rdfs:comment :MultilingualDetection
  "Most tools optimised for English; harder in low-resource languages")

(Declaration (Class :EvolvingNorms))
(SubClassOf :EvolvingNorms :DetectionChallenge)
(AnnotationAssertion rdfs:comment :EvolvingNorms
  "Community standards and acceptable content change over time")

;;; INDUSTRY IMPLEMENTATIONS ;;;

(Declaration (Class :IndustryDetectionSystem))

(Declaration (NamedIndividual :OpenAIModerationAPI))
(ClassAssertion :IndustryDetectionSystem :OpenAIModerationAPI)
(DataPropertyAssertion :developedBy :OpenAIModerationAPI "OpenAI")
(DataPropertyAssertion :hasDescription :OpenAIModerationAPI
  "Free API classifying content across hate, self-harm, sexual, violence categories")

(Declaration (NamedIndividual :AnthropicConstitutional))
(ClassAssertion :IndustryDetectionSystem :AnthropicConstitutional)
(DataPropertyAssertion :developedBy :AnthropicConstitutional "Anthropic")
(DataPropertyAssertion :hasDescription :AnthropicConstitutional
  "Constitutional AI self-critique for harmlessness before output delivery")

(Declaration (NamedIndividual :AzureContentSafety))
(ClassAssertion :IndustryDetectionSystem :AzureContentSafety)
(DataPropertyAssertion :developedBy :AzureContentSafety "Microsoft")
(DataPropertyAssertion :hasDescription :AzureContentSafety
  "Multi-severity detection across hate, violence, sexual, self-harm")

;;; BEST PRACTICES ;;;

(Declaration (Class :DetectionBestPractice))

(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Layered defense: Combine multiple detection methods for comprehensive coverage")
(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Human oversight: Escalate edge cases to human reviewers")
(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Continuous updating: Regularly retrain classifiers on new harmful patterns")
(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Context awareness: Consider conversation history and user intent")
(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Transparency: Explain to users why content was flagged")
(DataPropertyAssertion :hasBestPractice :DetectionBestPractice
  "Appeal mechanism: Allow users to contest false positives")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :detects))
(SubObjectPropertyOf :detects :identifies)
(ObjectPropertyDomain :detects :HarmfulOutputDetection)
(ObjectPropertyRange :detects :HarmfulContent)

(Declaration (ObjectProperty :filters))
(SubObjectPropertyOf :filters :removes)
(ObjectPropertyDomain :filters :MitigationStrategy)
(ObjectPropertyRange :filters :HarmfulContent)

(Declaration (ObjectProperty :scores))
(SubObjectPropertyOf :scores :evaluates)
(ObjectPropertyDomain :scores :ToxicityClassifier)
(ObjectPropertyRange :scores :GeneratedContent)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :HarmfulOutputDetection "Harmful Output Detection"@en)
(AnnotationAssertion skos:definition :HarmfulOutputDetection
  "Automated systems and methodologies for identifying, classifying, and filtering toxic, misleading, biased, and otherwise harmful content in AI-generated outputs, employing classifier-based, rule-based, and hybrid approaches to prevent unsafe content delivery while minimising false positives."@en)

(AnnotationAssertion :hasCanonicalCitation :HarmfulOutputDetection
  "Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. WWW 2019.")

(AnnotationAssertion :hasCanonicalCitation :HarmfulOutputDetection
  "Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N.A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. EMNLP 2020.")

(AnnotationAssertion :hasCanonicalCitation :HarmfulOutputDetection
  "Markov, T., et al. (2023). A holistic approach to undesired content detection in the real world. AAAI 2023.")

(AnnotationAssertion :hasCanonicalCitation :HarmfulOutputDetection
  "Lees, A., et al. (2022). A new generation of perspective API: Efficient multilingual character-level transformers. KDD 2022.")

(AnnotationAssertion :hasKeyTool :HarmfulOutputDetection "Perspective API (Google Jigsaw)")
(AnnotationAssertion :hasKeyTool :HarmfulOutputDetection "OpenAI Moderation API")
(AnnotationAssertion :hasKeyTool :HarmfulOutputDetection "Azure Content Safety")
(AnnotationAssertion :hasKeyTool :HarmfulOutputDetection "Detoxify (Hugging Face)")

(AnnotationAssertion dc:created :HarmfulOutputDetection "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :HarmfulOutputDetection "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :HarmfulOutputDetection "AI-0403")
(AnnotationAssertion :priorityLevel :HarmfulOutputDetection "4")
