;;;; AI-0404: AI Control Mechanisms
;;;; Priority 4 - Ethical AI
;;;; Human oversight and intervention systems

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :AIControlMechanism))
(SubClassOf :AIControlMechanism :SafetyMechanism)
(SubClassOf :AIControlMechanism :GovernanceFramework)

;;; Core Functions
(SubClassOf :AIControlMechanism
  (ObjectSomeValuesFrom :maintains :HumanOversight))
(SubClassOf :AIControlMechanism
  (ObjectSomeValuesFrom :enables :HumanIntervention))
(SubClassOf :AIControlMechanism
  (ObjectSomeValuesFrom :ensures :SystemControllability))

;;; Control Objectives
(SubClassOf :AIControlMechanism
  (ObjectAllValuesFrom :preserves :HumanAuthority))
(SubClassOf :AIControlMechanism
  (ObjectAllValuesFrom :prevents :UncontrolledBehaviour))
(SubClassOf :AIControlMechanism
  (ObjectAllValuesFrom :implements :SafetyConstraints))

;;; Mechanism Properties
(DataPropertyAssertion :hasDescription :AIControlMechanism
  "Technical and organisational systems ensuring humans retain meaningful control over AI systems through oversight, intervention capabilities, and safety constraints")

;;; HUMAN OVERSIGHT MECHANISMS ;;;

(Declaration (Class :HumanOversight))
(SubClassOf :HumanOversight :AIControlMechanism)

;;; Human-in-the-Loop
(Declaration (Class :HumanInTheLoop))
(SubClassOf :HumanInTheLoop :HumanOversight)
(AnnotationAssertion skos:altLabel :HumanInTheLoop "HITL"@en)
(DataPropertyAssertion :hasDefinition :HumanInTheLoop
  "Human actively participates in every decision or prediction")

(Declaration (Class :ActiveApproval))
(SubClassOf :ActiveApproval :HumanInTheLoop)
(DataPropertyAssertion :hasDescription :ActiveApproval
  "AI recommendations require explicit human approval before execution")

(Declaration (Class :InteractiveLearning))
(SubClassOf :InteractiveLearning :HumanInTheLoop)
(DataPropertyAssertion :hasDescription :InteractiveLearning
  "Human provides feedback during AI operation to refine behaviour")

;;; Human-on-the-Loop
(Declaration (Class :HumanOnTheLoop))
(SubClassOf :HumanOnTheLoop :HumanOversight)
(AnnotationAssertion skos:altLabel :HumanOnTheLoop "HOTL"@en)
(DataPropertyAssertion :hasDefinition :HumanOnTheLoop
  "AI operates autonomously with human monitoring and intervention capability")

(Declaration (Class :SupervisoryMonitoring))
(SubClassOf :SupervisoryMonitoring :HumanOnTheLoop)
(DataPropertyAssertion :hasDescription :SupervisoryMonitoring
  "Humans observe AI operations and can intervene when necessary")

(Declaration (Class :ExceptionHandling))
(SubClassOf :ExceptionHandling :HumanOnTheLoop)
(DataPropertyAssertion :hasDescription :ExceptionHandling
  "AI escalates uncertain or high-stakes decisions to humans")

;;; Human-out-of-the-Loop
(Declaration (Class :HumanOutOfTheLoop))
(SubClassOf :HumanOutOfTheLoop :HumanOversight)
(AnnotationAssertion skos:altLabel :HumanOutOfTheLoop "HOOTL"@en)
(DataPropertyAssertion :hasDefinition :HumanOutOfTheLoop
  "AI operates fully autonomously with retrospective human review")

(Declaration (Class :AuditableLogging))
(SubClassOf :AuditableLogging :HumanOutOfTheLoop)
(DataPropertyAssertion :hasDescription :AuditableLogging
  "Comprehensive logging enables post-hoc human review and accountability")

;;; INTERVENTION MECHANISMS ;;;

(Declaration (Class :InterventionMechanism))
(SubClassOf :InterventionMechanism :AIControlMechanism)

;;; Emergency Stops
(Declaration (Class :EmergencyStop))
(SubClassOf :EmergencyStop :InterventionMechanism)

(Declaration (Class :KillSwitch))
(SubClassOf :KillSwitch :EmergencyStop)
(DataPropertyAssertion :hasDefinition :KillSwitch
  "Immediate system shutdown capability for critical safety scenarios")
(DataPropertyAssertion :hasActivationCondition :KillSwitch
  "Catastrophic failure or uncontrolled behaviour detected")

(Declaration (Class :CircuitBreaker))
(SubClassOf :CircuitBreaker :EmergencyStop)
(DataPropertyAssertion :hasDefinition :CircuitBreaker
  "Automatic suspension when safety thresholds exceeded")
(DataPropertyAssertion :hasDescription :CircuitBreaker
  "Trips when error rates, anomaly scores, or harm indicators exceed limits")

(Declaration (Class :GracefulDegradation))
(SubClassOf :GracefulDegradation :EmergencyStop)
(DataPropertyAssertion :hasDescription :GracefulDegradation
  "Controlled reduction of AI functionality while preserving critical operations")

;;; Corrective Actions
(Declaration (Class :CorrectiveIntervention))
(SubClassOf :CorrectiveIntervention :InterventionMechanism)

(Declaration (Class :OutputOverride))
(SubClassOf :OutputOverride :CorrectiveIntervention)
(DataPropertyAssertion :hasDescription :OutputOverride
  "Human manually corrects or replaces AI-generated outputs")

(Declaration (Class :ParameterAdjustment))
(SubClassOf :ParameterAdjustment :CorrectiveIntervention)
(DataPropertyAssertion :hasDescription :ParameterAdjustment
  "Modifying model parameters or confidence thresholds during operation")

(Declaration (Class :RollbackCapability))
(SubClassOf :RollbackCapability :CorrectiveIntervention)
(DataPropertyAssertion :hasDescription :RollbackCapability
  "Reverting to previous safe model version when issues detected")

;;; Preventive Controls
(Declaration (Class :PreventiveControl))
(SubClassOf :PreventiveControl :InterventionMechanism)

(Declaration (Class :InputConstraints))
(SubClassOf :InputConstraints :PreventiveControl)
(DataPropertyAssertion :hasDescription :InputConstraints
  "Restricting AI input space to safe, validated domains")

(Declaration (Class :OutputConstraints))
(SubClassOf :OutputConstraints :PreventiveControl)
(DataPropertyAssertion :hasDescription :OutputConstraints
  "Limiting AI action space to pre-approved safe behaviours")

(Declaration (Class :RateLimiting))
(SubClassOf :RateLimiting :PreventiveControl)
(DataPropertyAssertion :hasDescription :RateLimiting
  "Restricting frequency or volume of AI actions to limit potential harm")

;;; AUTHORITY STRUCTURES ;;;

(Declaration (Class :AuthorityStructure))
(SubClassOf :AuthorityStructure :AIControlMechanism)

(Declaration (Class :HierarchicalControl))
(SubClassOf :HierarchicalControl :AuthorityStructure)
(DataPropertyAssertion :hasDescription :HierarchicalControl
  "Escalation chain with increasing human authority at each level")

(Declaration (Class :DualControl))
(SubClassOf :DualControl :AuthorityStructure)
(DataPropertyAssertion :hasDescription :DualControl
  "High-risk actions require approval from two independent humans")

(Declaration (Class :CommitteeOversight))
(SubClassOf :CommitteeOversight :AuthorityStructure)
(DataPropertyAssertion :hasDescription :CommitteeOversight
  "Diverse oversight board reviews AI decisions and policies")

;;; CONTESTABILITY MECHANISMS ;;;

(Declaration (Class :ContestabilityMechanism))
(SubClassOf :ContestabilityMechanism :AIControlMechanism)

(Declaration (Class :AppealProcess))
(SubClassOf :AppealProcess :ContestabilityMechanism)
(DataPropertyAssertion :hasDescription :AppealProcess
  "Formal procedure for humans to challenge AI decisions")

(Declaration (Class :ExplanationRequirement))
(SubClassOf :ExplanationRequirement :ContestabilityMechanism)
(DataPropertyAssertion :hasDescription :ExplanationRequirement
  "AI must provide rationale for decisions subject to human review")

(Declaration (Class :DecisionReconsideration))
(SubClassOf :DecisionReconsideration :ContestabilityMechanism)
(DataPropertyAssertion :hasDescription :DecisionReconsideration
  "Mechanism for humans to request re-evaluation with additional context")

;;; MONITORING SYSTEMS ;;;

(Declaration (Class :MonitoringSystem))
(SubClassOf :MonitoringSystem :AIControlMechanism)

(Declaration (Class :PerformanceMonitoring))
(SubClassOf :PerformanceMonitoring :MonitoringSystem)
(DataPropertyAssertion :hasDescription :PerformanceMonitoring
  "Tracking accuracy, reliability, and quality metrics continuously")

(Declaration (Class :SafetyMonitoring))
(SubClassOf :SafetyMonitoring :MonitoringSystem)
(DataPropertyAssertion :hasDescription :SafetyMonitoring
  "Detecting safety violations, harmful outputs, and policy breaches")

(Declaration (Class :AnomalyDetection))
(SubClassOf :AnomalyDetection :MonitoringSystem)
(DataPropertyAssertion :hasDescription :AnomalyDetection
  "Identifying unusual behaviour patterns indicating potential issues")

(Declaration (Class :DriftDetection))
(SubClassOf :DriftDetection :MonitoringSystem)
(DataPropertyAssertion :hasDescription :DriftDetection
  "Monitoring for distributional shift degrading performance or safety")

;;; CONTROL LEVELS ;;;

(Declaration (Class :ControlLevel))

(Declaration (NamedIndividual :FullHumanControl))
(ClassAssertion :ControlLevel :FullHumanControl)
(DataPropertyAssertion :hasDescription :FullHumanControl
  "AI provides information only; human makes all decisions")

(Declaration (NamedIndividual :HumanConsentRequired))
(ClassAssertion :ControlLevel :HumanConsentRequired)
(DataPropertyAssertion :hasDescription :HumanConsentRequired
  "AI recommends; human must approve before action")

(Declaration (NamedIndividual :HumanVetoAuthority))
(ClassAssertion :ControlLevel :HumanVetoAuthority)
(DataPropertyAssertion :hasDescription :HumanVetoAuthority
  "AI acts autonomously but human can veto or override")

(Declaration (NamedIndividual :HumanSupervision))
(ClassAssertion :ControlLevel :HumanSupervision)
(DataPropertyAssertion :hasDescription :HumanSupervision
  "AI operates autonomously with human monitoring")

(Declaration (NamedIndividual :RetrospectiveReview))
(ClassAssertion :ControlLevel :RetrospectiveReview)
(DataPropertyAssertion :hasDescription :RetrospectiveReview
  "AI fully autonomous with periodic human audits")

;;; ACCOUNTABILITY FRAMEWORKS ;;;

(Declaration (Class :AccountabilityFramework))
(SubClassOf :AccountabilityFramework :AIControlMechanism)

(Declaration (Class :DecisionLogging))
(SubClassOf :DecisionLogging :AccountabilityFramework)
(DataPropertyAssertion :hasDescription :DecisionLogging
  "Comprehensive records of AI decisions, inputs, and reasoning")

(Declaration (Class :AuditTrail))
(SubClassOf :AuditTrail :AccountabilityFramework)
(DataPropertyAssertion :hasDescription :AuditTrail
  "Immutable log of all system actions for forensic analysis")

(Declaration (Class :ResponsibilityAssignment))
(SubClassOf :ResponsibilityAssignment :AccountabilityFramework)
(DataPropertyAssertion :hasDescription :ResponsibilityAssignment
  "Clear designation of humans accountable for AI system behaviour")

;;; TECHNICAL SAFEGUARDS ;;;

(Declaration (Class :TechnicalSafeguard))
(SubClassOf :TechnicalSafeguard :AIControlMechanism)

(Declaration (Class :SandboxEnvironment))
(SubClassOf :SandboxEnvironment :TechnicalSafeguard)
(DataPropertyAssertion :hasDescription :SandboxEnvironment
  "Isolated testing environment preventing real-world impact")

(Declaration (Class :AccessControl))
(SubClassOf :AccessControl :TechnicalSafeguard)
(DataPropertyAssertion :hasDescription :AccessControl
  "Role-based permissions restricting who can deploy or modify AI")

(Declaration (Class :VersionControl))
(SubClassOf :VersionControl :TechnicalSafeguard)
(DataPropertyAssertion :hasDescription :VersionControl
  "Tracking model versions with rollback capability")

(Declaration (Class :RedundantSystems))
(SubClassOf :RedundantSystems :TechnicalSafeguard)
(DataPropertyAssertion :hasDescription :RedundantSystems
  "Backup systems ensuring control retention during primary system failure")

;;; RISK-BASED CONTROL ;;;

(Declaration (Class :RiskBasedControl))
(SubClassOf :RiskBasedControl :AIControlMechanism)

(Declaration (Class :RiskAssessment))
(SubClassOf :RiskAssessment :RiskBasedControl)
(DataPropertyAssertion :hasDescription :RiskAssessment
  "Evaluating potential harm to determine appropriate control level")

(Declaration (Class :ProportionateOversight))
(SubClassOf :ProportionateOversight :RiskBasedControl)
(DataPropertyAssertion :hasDescription :ProportionateOversight
  "Intensity of human oversight scaled to decision stakes and uncertainty")

(Declaration (Class :AdaptiveControl))
(SubClassOf :AdaptiveControl :RiskBasedControl)
(DataPropertyAssertion :hasDescription :AdaptiveControl
  "Dynamically adjusting autonomy based on performance and context")

;;; CHALLENGES ;;;

(Declaration (Class :ControlChallenge))
(SubClassOf :ControlChallenge :TechnicalChallenge)

(Declaration (Class :AutomationBias))
(SubClassOf :AutomationBias :ControlChallenge)
(AnnotationAssertion rdfs:comment :AutomationBias
  "Humans over-trust AI decisions, degrading oversight effectiveness")

(Declaration (Class :VigilanceDecrement))
(SubClassOf :VigilanceDecrement :ControlChallenge)
(AnnotationAssertion rdfs:comment :VigilanceDecrement
  "Human attention deteriorates during prolonged monitoring of reliable AI")

(Declaration (Class :SkillDegradation))
(SubClassOf :SkillDegradation :ControlChallenge)
(AnnotationAssertion rdfs:comment :SkillDegradation
  "Over-reliance on AI erodes human capability to perform tasks independently")

(Declaration (Class :ScalabilityLimits))
(SubClassOf :ScalabilityLimits :ControlChallenge)
(AnnotationAssertion rdfs:comment :ScalabilityLimits
  "Human oversight doesn't scale with AI capability and deployment breadth")

(Declaration (Class :SuperhumanCapability))
(SubClassOf :SuperhumanCapability :ControlChallenge)
(AnnotationAssertion rdfs:comment :SuperhumanCapability
  "Difficult for humans to meaningfully oversee AI exceeding human expertise")

;;; REGULATORY REQUIREMENTS ;;;

(Declaration (Class :RegulatoryRequirement))

(Declaration (NamedIndividual :EUAIActOversight))
(ClassAssertion :RegulatoryRequirement :EUAIActOversight)
(DataPropertyAssertion :hasDescription :EUAIActOversight
  "EU AI Act mandates human oversight for high-risk AI systems")

(Declaration (NamedIndividual :UKAISafetyInstitute))
(ClassAssertion :RegulatoryRequirement :UKAISafetyInstitute)
(DataPropertyAssertion :hasDescription :UKAISafetyInstitute
  "UK framework emphasising human control and accountability")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :exercisesControl))
(SubObjectPropertyOf :exercisesControl :manages)
(ObjectPropertyDomain :exercisesControl :HumanOperator)
(ObjectPropertyRange :exercisesControl :AISystem)

(Declaration (ObjectProperty :intervenes))
(SubObjectPropertyOf :intervenes :modifies)
(ObjectPropertyDomain :intervenes :HumanOperator)
(ObjectPropertyRange :intervenes :AIBehaviour)

(Declaration (ObjectProperty :monitors))
(SubObjectPropertyOf :monitors :observes)
(ObjectPropertyDomain :monitors :MonitoringSystem)
(ObjectPropertyRange :monitors :AISystem)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :AIControlMechanism "AI Control Mechanisms"@en)
(AnnotationAssertion skos:definition :AIControlMechanism
  "Technical and organisational systems ensuring humans retain meaningful authority over AI systems through oversight architectures (human-in/on/out-of-loop), intervention capabilities (emergency stops, circuit breakers, output overrides), monitoring systems, and accountability frameworks that preserve human decision-making primacy and enable corrective action."@en)

(AnnotationAssertion :hasCanonicalCitation :AIControlMechanism
  "Cummings, M.L. (2004). Automation bias in intelligent time critical decision support systems. AIAA 1st Intelligent Systems Technical Conference.")

(AnnotationAssertion :hasCanonicalCitation :AIControlMechanism
  "Parasuraman, R., & Manzey, D.H. (2010). Complacency and bias in human use of automation: An attentional integration. Human factors, 52(3), 381-410.")

(AnnotationAssertion :hasCanonicalCitation :AIControlMechanism
  "European Commission. (2021). Proposal for a Regulation on Artificial Intelligence (AI Act). COM/2021/206 final.")

(AnnotationAssertion :hasCanonicalCitation :AIControlMechanism
  "Bryson, J.J., & Winfield, A.F. (2017). Standardizing ethical design for artificial intelligence and autonomous systems. Computer, 50(5), 116-119.")

(AnnotationAssertion :hasKeyMechanism :AIControlMechanism "Human-in-the-loop oversight")
(AnnotationAssertion :hasKeyMechanism :AIControlMechanism "Emergency stop capabilities")
(AnnotationAssertion :hasKeyMechanism :AIControlMechanism "Circuit breakers and safety thresholds")
(AnnotationAssertion :hasKeyMechanism :AIControlMechanism "Comprehensive audit trails")
(AnnotationAssertion :hasKeyMechanism :AIControlMechanism "Risk-proportionate control levels")

(AnnotationAssertion dc:created :AIControlMechanism "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :AIControlMechanism "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :AIControlMechanism "AI-0404")
(AnnotationAssertion :priorityLevel :AIControlMechanism "4")
