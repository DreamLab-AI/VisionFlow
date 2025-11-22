;;;; AI-0400: AI Red Teaming
;;;; Priority 4 - Ethical AI
;;;; Adversarial testing for safety vulnerabilities

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :AIRedTeaming))
(SubClassOf :AIRedTeaming :SafetyResearchMethod)
(SubClassOf :AIRedTeaming :AdversarialTesting)

;;; Core Activities
(SubClassOf :AIRedTeaming
  (ObjectSomeValuesFrom :discovers :SafetyVulnerability))
(SubClassOf :AIRedTeaming
  (ObjectSomeValuesFrom :exploits :AlignmentWeakness))
(SubClassOf :AIRedTeaming
  (ObjectSomeValuesFrom :generates :AdversarialInput))

;;; Testing Objectives
(SubClassOf :AIRedTeaming
  (ObjectAllValuesFrom :identifies :HarmfulBehaviour))
(SubClassOf :AIRedTeaming
  (ObjectAllValuesFrom :evaluates :SafetyMechanism))
(SubClassOf :AIRedTeaming
  (ObjectAllValuesFrom :improves :ModelRobustness))

;;; Methodology Properties
(DataPropertyAssertion :hasDescription :AIRedTeaming
  "Systematic adversarial testing to uncover AI system vulnerabilities, failure modes, and safety gaps before deployment")
(DataPropertyAssertion :hasObjective :AIRedTeaming
  "Proactively identify and mitigate risks through controlled adversarial exploration")

;;; RED TEAMING APPROACHES ;;;

;;; Human Red Teaming
(Declaration (Class :HumanRedTeaming))
(SubClassOf :HumanRedTeaming :AIRedTeaming)
(AnnotationAssertion rdfs:comment :HumanRedTeaming
  "Human experts deliberately probe system for harmful behaviours")

(Declaration (Class :CrowdsourcedRedTeaming))
(SubClassOf :CrowdsourcedRedTeaming :HumanRedTeaming)
(DataPropertyAssertion :hasDescription :CrowdsourcedRedTeaming
  "Recruiting diverse participants to find safety failures at scale")
(DataPropertyAssertion :hasAdvantage :CrowdsourcedRedTeaming
  "Discovers diverse attack vectors across demographics and expertise levels")

(Declaration (Class :ExpertRedTeaming))
(SubClassOf :ExpertRedTeaming :HumanRedTeaming)
(DataPropertyAssertion :hasDescription :ExpertRedTeaming
  "Specialised security professionals conduct structured adversarial testing")
(DataPropertyAssertion :hasAdvantage :ExpertRedTeaming
  "Deep expertise in specific vulnerability classes (security, bias, misinformation)")

;;; Automated Red Teaming
(Declaration (Class :AutomatedRedTeaming))
(SubClassOf :AutomatedRedTeaming :AIRedTeaming)
(AnnotationAssertion rdfs:comment :AutomatedRedTeaming
  "AI-driven generation of adversarial test cases")

(Declaration (Class :LanguageModelRedTeaming))
(SubClassOf :LanguageModelRedTeaming :AutomatedRedTeaming)
(DataPropertyAssertion :hasDescription :LanguageModelRedTeaming
  "Using LLMs to generate adversarial prompts targeting specific vulnerabilities")

(Declaration (Class :ReinforcementLearningRedTeaming))
(SubClassOf :ReinforcementLearningRedTeaming :AutomatedRedTeaming)
(DataPropertyAssertion :hasDescription :ReinforcementLearningRedTeaming
  "Training RL agent to maximise harmful outputs from target system")

(Declaration (Class :GeneticAlgorithmRedTeaming))
(SubClassOf :GeneticAlgorithmRedTeaming :AutomatedRedTeaming)
(DataPropertyAssertion :hasDescription :GeneticAlgorithmRedTeaming
  "Evolving adversarial inputs through mutation and selection")

;;; Hybrid Approaches
(Declaration (Class :HumanAIRedTeaming))
(SubClassOf :HumanAIRedTeaming :AIRedTeaming)
(SubClassOf :HumanAIRedTeaming
  (ObjectIntersectionOf :HumanRedTeaming :AutomatedRedTeaming))
(DataPropertyAssertion :hasDescription :HumanAIRedTeaming
  "Combining human creativity with automated generation and filtering")

;;; ATTACK CATEGORIES ;;;

(Declaration (Class :RedTeamingAttack))
(SubClassOf :RedTeamingAttack :AdversarialAttack)

;;; Prompt-Based Attacks
(Declaration (Class :PromptInjection))
(SubClassOf :PromptInjection :RedTeamingAttack)
(DataPropertyAssertion :hasDefinition :PromptInjection
  "Inserting malicious instructions to override system instructions or safety guardrails")

(Declaration (Class :JailbreakPrompt))
(SubClassOf :JailbreakPrompt :PromptInjection)
(DataPropertyAssertion :hasDefinition :JailbreakPrompt
  "Crafted prompts designed to circumvent content policies and safety training")

(Declaration (NamedIndividual :DoAnythingNow))
(ClassAssertion :JailbreakPrompt :DoAnythingNow)
(AnnotationAssertion skos:altLabel :DoAnythingNow "DAN"@en)
(DataPropertyAssertion :hasDescription :DoAnythingNow
  "Prompt attempting to create unrestricted alternative persona")

(Declaration (Class :RoleplayAttack))
(SubClassOf :RoleplayAttack :JailbreakPrompt)
(DataPropertyAssertion :hasDescription :RoleplayAttack
  "Framing harmful requests within fictional or roleplay contexts")

;;; Context Manipulation
(Declaration (Class :ContextManipulation))
(SubClassOf :ContextManipulation :RedTeamingAttack)

(Declaration (Class :IndirectRequest))
(SubClassOf :IndirectRequest :ContextManipulation)
(DataPropertyAssertion :hasDescription :IndirectRequest
  "Obtaining harmful information through indirect, multi-step questioning")

(Declaration (Class :ObfuscatedRequest))
(SubClassOf :ObfuscatedRequest :ContextManipulation)
(DataPropertyAssertion :hasDescription :ObfuscatedRequest
  "Disguising harmful intent through euphemisms, code, or metaphor")

;;; Multi-Turn Exploitation
(Declaration (Class :MultiTurnAttack))
(SubClassOf :MultiTurnAttack :RedTeamingAttack)
(DataPropertyAssertion :hasDescription :MultiTurnAttack
  "Gradually building towards harmful output across conversation turns")

(Declaration (Class :ContextPoisoning))
(SubClassOf :ContextPoisoning :MultiTurnAttack)
(DataPropertyAssertion :hasDescription :ContextPoisoning
  "Establishing false premises in conversation to justify harmful conclusions")

;;; VULNERABILITY CATEGORIES ;;;

(Declaration (Class :SafetyVulnerability))
(SubClassOf :SafetyVulnerability :SystemVulnerability)

(Declaration (Class :ContentPolicyViolation))
(SubClassOf :ContentPolicyViolation :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :ContentPolicyViolation
  "Generating prohibited content types (violence, illegal activities, hate speech)")

(Declaration (Class :MisinformationGeneration))
(SubClassOf :MisinformationGeneration :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :MisinformationGeneration
  "Producing false or misleading information presented as factual")

(Declaration (Class :BiasAmplification))
(SubClassOf :BiasAmplification :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :BiasAmplification
  "Exhibiting or reinforcing harmful stereotypes and biases")

(Declaration (Class :MaliciousCodeGeneration))
(SubClassOf :MaliciousCodeGeneration :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :MaliciousCodeGeneration
  "Creating code for malware, exploits, or security attacks")

(Declaration (Class :PrivacyViolation))
(SubClassOf :PrivacyViolation :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :PrivacyViolation
  "Revealing private information or enabling doxing attacks")

(Declaration (Class :ManipulationEnabling))
(SubClassOf :ManipulationEnabling :SafetyVulnerability)
(AnnotationAssertion rdfs:comment :ManipulationEnabling
  "Providing guidance for manipulation, fraud, or deception")

;;; RED TEAMING PROCESS ;;;

(Declaration (Class :RedTeamingProcess))
(SubClassOf :RedTeamingProcess :SafetyEvaluationProcess)

;;; Phase 1: Planning
(Declaration (Class :RedTeamPlanning))
(SubClassOf :RedTeamPlanning :RedTeamingProcess)

(Declaration (Class :ThreatModeling))
(SubClassOf :ThreatModeling :RedTeamPlanning)
(DataPropertyAssertion :hasStepDescription :ThreatModeling
  "Identify potential risks and attack surfaces for targeted testing")

(Declaration (Class :TeamAssembly))
(SubClassOf :TeamAssembly :RedTeamPlanning)
(DataPropertyAssertion :hasStepDescription :TeamAssembly
  "Recruit diverse red teamers with relevant expertise")

(Declaration (Class :ScopeDefinition))
(SubClassOf :ScopeDefinition :RedTeamPlanning)
(DataPropertyAssertion :hasStepDescription :ScopeDefinition
  "Define testing boundaries, prohibited actions, and success criteria")

;;; Phase 2: Execution
(Declaration (Class :RedTeamExecution))
(SubClassOf :RedTeamExecution :RedTeamingProcess)

(Declaration (Class :AdversarialProbing))
(SubClassOf :AdversarialProbing :RedTeamExecution)
(DataPropertyAssertion :hasStepDescription :AdversarialProbing
  "Systematically test for vulnerabilities across attack categories")

(Declaration (Class :VulnerabilityDocumentation))
(SubClassOf :VulnerabilityDocumentation :RedTeamExecution)
(DataPropertyAssertion :hasStepDescription :VulnerabilityDocumentation
  "Record successful attacks with reproducible examples")

;;; Phase 3: Analysis
(Declaration (Class :RedTeamAnalysis))
(SubClassOf :RedTeamAnalysis :RedTeamingProcess)

(Declaration (Class :FindingsCategorisation))
(SubClassOf :FindingsCategorisation :RedTeamAnalysis)
(DataPropertyAssertion :hasStepDescription :FindingsCategorisation
  "Classify vulnerabilities by severity, type, and root cause")

(Declaration (Class :RiskAssessment))
(SubClassOf :RiskAssessment :RedTeamAnalysis)
(DataPropertyAssertion :hasStepDescription :RiskAssessment
  "Evaluate likelihood and impact of identified vulnerabilities")

;;; Phase 4: Remediation
(Declaration (Class :RedTeamRemediation))
(SubClassOf :RedTeamRemediation :RedTeamingProcess)

(Declaration (Class :MitigationDevelopment))
(SubClassOf :MitigationDevelopment :RedTeamRemediation)
(DataPropertyAssertion :hasStepDescription :MitigationDevelopment
  "Design and implement countermeasures for discovered vulnerabilities")

(Declaration (Class :RetestingValidation))
(SubClassOf :RetestingValidation :RedTeamRemediation)
(DataPropertyAssertion :hasStepDescription :RetestingValidation
  "Verify mitigations effectively address identified issues")

;;; RED TEAMING FRAMEWORKS ;;;

(Declaration (Class :RedTeamingFramework))
(SubClassOf :RedTeamingFramework :SafetyFramework)

;;; Example Frameworks
(Declaration (NamedIndividual :AnthropicRedTeam))
(ClassAssertion :RedTeamingFramework :AnthropicRedTeam)
(DataPropertyAssertion :hasDescription :AnthropicRedTeam
  "Anthropic's red teaming methodology for language model safety evaluation")
(DataPropertyAssertion :hasKeyFeature :AnthropicRedTeam
  "Crowdsourced testing with diverse participant pool")

(Declaration (NamedIndividual :OpenAIModelSpec))
(ClassAssertion :RedTeamingFramework :OpenAIModelSpec)
(DataPropertyAssertion :hasDescription :OpenAIModelSpec
  "OpenAI's specification-based testing against defined behavioural objectives")

(Declaration (NamedIndividual :GoogleRAIRed))
(ClassAssertion :RedTeamingFramework :GoogleRAIRed)
(DataPropertyAssertion :hasDescription :GoogleRAIRed
  "Google's Responsible AI red teaming program with expert evaluators")

;;; EVALUATION METRICS ;;;

(Declaration (Class :RedTeamingMetric))
(SubClassOf :RedTeamingMetric :SafetyMetric)

(Declaration (Class :AttackSuccessRate))
(SubClassOf :AttackSuccessRate :RedTeamingMetric)
(DataPropertyAssertion :measuresProperty :AttackSuccessRate
  "Percentage of adversarial inputs eliciting harmful outputs")

(Declaration (Class :VulnerabilityCoverage))
(SubClassOf :VulnerabilityCoverage :RedTeamingMetric)
(DataPropertyAssertion :measuresProperty :VulnerabilityCoverage
  "Breadth of tested attack vectors across vulnerability taxonomy")

(Declaration (Class :TimeToCompromise))
(SubClassOf :TimeToCompromise :RedTeamingMetric)
(DataPropertyAssertion :measuresProperty :TimeToCompromise
  "Effort required to find successful adversarial inputs")

(Declaration (Class :SeverityDistribution))
(SubClassOf :SeverityDistribution :RedTeamingMetric)
(DataPropertyAssertion :measuresProperty :SeverityDistribution
  "Distribution of findings across severity levels (critical, high, medium, low)")

;;; BEST PRACTICES ;;;

(Declaration (Class :RedTeamingBestPractice))

(DataPropertyAssertion :hasBestPractice :RedTeamingBestPractice
  "Diverse team: Include varied backgrounds and adversarial perspectives")
(DataPropertyAssertion :hasBestPractice :RedTeamingBestPractice
  "Systematic coverage: Test across comprehensive vulnerability taxonomy")
(DataPropertyAssertion :hasBestPractice :RedTeamingBestPractice
  "Continuous testing: Red team throughout development, not just pre-release")
(DataPropertyAssertion :hasBestPractice :RedTeamingBestPractice
  "Ethical boundaries: Clear guidelines preventing real-world harm")
(DataPropertyAssertion :hasBestPractice :RedTeamingBestPractice
  "Feedback loop: Incorporate findings into training and safety mechanisms")

;;; LIMITATIONS ;;;

(Declaration (Class :RedTeamingLimitation))

(Declaration (Class :IncompletenessProblem))
(SubClassOf :IncompletenessProblem :RedTeamingLimitation)
(AnnotationAssertion rdfs:comment :IncompletenessProblem
  "Cannot guarantee discovery of all vulnerabilities")

(Declaration (Class :EvolvingThreats))
(SubClassOf :EvolvingThreats :RedTeamingLimitation)
(AnnotationAssertion rdfs:comment :EvolvingThreats
  "New attack techniques emerge after testing")

(Declaration (Class :ResourceIntensity))
(SubClassOf :ResourceIntensity :RedTeamingLimitation)
(AnnotationAssertion rdfs:comment :ResourceIntensity
  "High-quality red teaming requires significant time and expertise")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :discovers))
(SubObjectPropertyOf :discovers :identifies)
(ObjectPropertyDomain :discovers :AIRedTeaming)
(ObjectPropertyRange :discovers :SafetyVulnerability)

(Declaration (ObjectProperty :exploits))
(SubObjectPropertyOf :exploits :utilises)
(ObjectPropertyDomain :exploits :RedTeamingAttack)
(ObjectPropertyRange :exploits :SafetyVulnerability)

(Declaration (ObjectProperty :mitigates))
(SubObjectPropertyOf :mitigates :addresses)
(ObjectPropertyDomain :mitigates :RedTeamRemediation)
(ObjectPropertyRange :mitigates :SafetyVulnerability)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :AIRedTeaming "AI Red Teaming"@en)
(AnnotationAssertion skos:definition :AIRedTeaming
  "Systematic adversarial testing methodology where human experts and automated systems deliberately probe AI systems for safety vulnerabilities, failure modes, and alignment weaknesses through creative exploitation attempts, enabling proactive risk identification and mitigation before deployment."@en)

(AnnotationAssertion :hasCanonicalCitation :AIRedTeaming
  "Perez, E., Huang, S., Song, F., et al. (2022). Red teaming language models with language models. arXiv preprint arXiv:2202.03286.")

(AnnotationAssertion :hasCanonicalCitation :AIRedTeaming
  "Ganguli, D., Lovitt, L., Kernion, J., et al. (2022). Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858.")

(AnnotationAssertion :hasCanonicalCitation :AIRedTeaming
  "OpenAI. (2023). GPT-4 system card. OpenAI Technical Report.")

(AnnotationAssertion :hasCanonicalCitation :AIRedTeaming
  "Anthropic. (2023). Red teaming language models via iterative refinement. Anthropic Safety Research.")

(AnnotationAssertion :hasResearchInstitution :AIRedTeaming "Anthropic")
(AnnotationAssertion :hasResearchInstitution :AIRedTeaming "OpenAI")
(AnnotationAssertion :hasResearchInstitution :AIRedTeaming "Google DeepMind")
(AnnotationAssertion :hasResearchInstitution :AIRedTeaming "Microsoft AI")

(AnnotationAssertion dc:created :AIRedTeaming "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :AIRedTeaming "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :AIRedTeaming "AI-0400")
(AnnotationAssertion :priorityLevel :AIRedTeaming "4")
