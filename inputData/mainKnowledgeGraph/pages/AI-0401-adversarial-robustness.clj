;;;; AI-0401: Adversarial Robustness
;;;; Priority 4 - Ethical AI
;;;; Defense against adversarial attacks and distribution shift

(in-package :metaverse-ontology)

;;; ONTOLOGICAL DEFINITION ;;;

(Declaration (Class :AdversarialRobustness))
(SubClassOf :AdversarialRobustness :ModelRobustness)
(SubClassOf :AdversarialRobustness :SafetyProperty)

;;; Core Properties
(SubClassOf :AdversarialRobustness
  (ObjectSomeValuesFrom :resistsAttack :AdversarialPerturbation))
(SubClassOf :AdversarialRobustness
  (ObjectSomeValuesFrom :maintainsPerformance :DistributionShift))
(SubClassOf :AdversarialRobustness
  (ObjectSomeValuesFrom :ensuresReliability :AdversarialConditions))

;;; Robustness Dimensions
(SubClassOf :AdversarialRobustness
  (ObjectAllValuesFrom :tolerates :InputNoise))
(SubClassOf :AdversarialRobustness
  (ObjectAllValuesFrom :withstands :OptimisedAttacks))
(SubClassOf :AdversarialRobustness
  (ObjectAllValuesFrom :generalises :OutOfDistribution))

;;; Property Characteristics
(DataPropertyAssertion :hasDescription :AdversarialRobustness
  "AI system's ability to maintain correct behaviour and performance when subjected to adversarial perturbations, distributional shifts, and maliciously crafted inputs")

;;; ADVERSARIAL ATTACK TYPES ;;;

(Declaration (Class :AdversarialAttack))
(SubClassOf :AdversarialAttack :SecurityThreat)

;;; Perturbation-Based Attacks (Computer Vision)
(Declaration (Class :AdversarialPerturbation))
(SubClassOf :AdversarialPerturbation :AdversarialAttack)

(Declaration (Class :WhiteBoxAttack))
(SubClassOf :WhiteBoxAttack :AdversarialPerturbation)
(DataPropertyAssertion :hasDefinition :WhiteBoxAttack
  "Attack with full knowledge of model architecture, parameters, and gradients")

(Declaration (Class :FastGradientSignMethod))
(SubClassOf :FastGradientSignMethod :WhiteBoxAttack)
(AnnotationAssertion skos:altLabel :FastGradientSignMethod "FGSM"@en)
(DataPropertyAssertion :hasFormula :FastGradientSignMethod
  "x_adv = x + ε · sign(∇_x L(θ, x, y))")
(AnnotationAssertion rdfs:comment :FastGradientSignMethod
  "Single-step gradient-based attack adding perturbation in direction of loss gradient")

(Declaration (Class :ProjectedGradientDescent))
(SubClassOf :ProjectedGradientDescent :WhiteBoxAttack)
(AnnotationAssertion skos:altLabel :ProjectedGradientDescent "PGD"@en)
(DataPropertyAssertion :hasDescription :ProjectedGradientDescent
  "Iterative FGSM with projection back to epsilon-ball around original input")

(Declaration (Class :CarliniWagnerAttack))
(SubClassOf :CarliniWagnerAttack :WhiteBoxAttack)
(AnnotationAssertion skos:altLabel :CarliniWagnerAttack "C&W"@en)
(DataPropertyAssertion :hasDescription :CarliniWagnerAttack
  "Optimisation-based attack minimising perturbation while ensuring misclassification")

(Declaration (Class :BlackBoxAttack))
(SubClassOf :BlackBoxAttack :AdversarialPerturbation)
(DataPropertyAssertion :hasDefinition :BlackBoxAttack
  "Attack with only query access to model outputs, no internal knowledge")

(Declaration (Class :TransferAttack))
(SubClassOf :TransferAttack :BlackBoxAttack)
(DataPropertyAssertion :hasDescription :TransferAttack
  "Adversarial examples crafted for surrogate model transfer to target model")

;;; Text-Based Attacks (NLP)
(Declaration (Class :TextualAdversarialAttack))
(SubClassOf :TextualAdversarialAttack :AdversarialAttack)

(Declaration (Class :CharacterLevelAttack))
(SubClassOf :CharacterLevelAttack :TextualAdversarialAttack)
(AnnotationAssertion rdfs:comment :CharacterLevelAttack
  "Substituting, inserting, or deleting characters to fool model")

(Declaration (Class :WordLevelAttack))
(SubClassOf :WordLevelAttack :TextualAdversarialAttack)
(AnnotationAssertion rdfs:comment :WordLevelAttack
  "Replacing words with synonyms or paraphrasing to alter predictions")

(Declaration (Class :SentenceLevelAttack))
(SubClassOf :SentenceLevelAttack :TextualAdversarialAttack)
(AnnotationAssertion rdfs:comment :SentenceLevelAttack
  "Adding or modifying sentences to change model interpretation")

;;; Physical-World Attacks
(Declaration (Class :PhysicalAdversarialAttack))
(SubClassOf :PhysicalAdversarialAttack :AdversarialAttack)

(Declaration (Class :AdversarialPatch))
(SubClassOf :AdversarialPatch :PhysicalAdversarialAttack)
(DataPropertyAssertion :hasDescription :AdversarialPatch
  "Printed physical patches causing misclassification when placed in scene")

(Declaration (Class :AdversarialObjects))
(SubClassOf :AdversarialObjects :PhysicalAdversarialAttack)
(DataPropertyAssertion :hasDescription :AdversarialObjects
  "3D-printed objects designed to be misclassified from multiple viewpoints")

;;; DEFENSE MECHANISMS ;;;

(Declaration (Class :AdversarialDefense))
(SubClassOf :AdversarialDefense :SafetyMechanism)

;;; Training-Based Defenses
(Declaration (Class :AdversarialTraining))
(SubClassOf :AdversarialTraining :AdversarialDefense)
(DataPropertyAssertion :hasDescription :AdversarialTraining
  "Training on mixture of clean and adversarial examples to improve robustness")

(Declaration (Class :StandardAdversarialTraining))
(SubClassOf :StandardAdversarialTraining :AdversarialTraining)
(DataPropertyAssertion :hasFormula :StandardAdversarialTraining
  "min_θ E[(x,y)~D] [max_{δ∈S} L(θ, x+δ, y)]")

(Declaration (Class :TRADESTraining))
(SubClassOf :TRADESTraining :AdversarialTraining)
(AnnotationAssertion rdfs:comment :TRADESTraining
  "TRadeoff-inspired Adversarial Defense via Surrogate-loss minimisation")
(DataPropertyAssertion :hasDescription :TRADESTraining
  "Balancing clean accuracy and adversarial robustness via surrogate loss")

(Declaration (Class :MARTTraining))
(SubClassOf :MARTTraining :AdversarialTraining)
(AnnotationAssertion rdfs:comment :MARTTraining
  "Misclassification Aware adveRsarial Training")
(DataPropertyAssertion :hasDescription :MARTTraining
  "Focusing on boundary examples most vulnerable to misclassification")

;;; Input Preprocessing Defenses
(Declaration (Class :InputTransformation))
(SubClassOf :InputTransformation :AdversarialDefense)

(Declaration (Class :InputDenoising))
(SubClassOf :InputDenoising :InputTransformation)
(DataPropertyAssertion :hasDescription :InputDenoising
  "Removing adversarial perturbations through denoising autoencoders or filtering")

(Declaration (Class :InputQuantisation))
(SubClassOf :InputQuantisation :InputTransformation)
(DataPropertyAssertion :hasDescription :InputQuantisation
  "Reducing input precision to destroy fine-grained adversarial signals")

(Declaration (Class :ImageCompression))
(SubClassOf :ImageCompression :InputTransformation)
(DataPropertyAssertion :hasDescription :ImageCompression
  "Applying lossy compression (JPEG) to remove high-frequency perturbations")

;;; Model-Based Defenses
(Declaration (Class :CertifiedDefense))
(SubClassOf :CertifiedDefense :AdversarialDefense)
(DataPropertyAssertion :hasDescription :CertifiedDefense
  "Providing provable guarantees on robustness within specified perturbation bounds")

(Declaration (Class :RandomisedSmoothing))
(SubClassOf :RandomisedSmoothing :CertifiedDefense)
(DataPropertyAssertion :hasDescription :RandomisedSmoothing
  "Adding random noise and averaging predictions for certified L2 robustness")

(Declaration (Class :IntervalBoundPropagation))
(SubClassOf :IntervalBoundPropagation :CertifiedDefense)
(AnnotationAssertion skos:altLabel :IntervalBoundPropagation "IBP"@en)
(DataPropertyAssertion :hasDescription :IntervalBoundPropagation
  "Computing provable bounds on output perturbations via interval arithmetic")

;;; Detection-Based Defenses
(Declaration (Class :AdversarialDetection))
(SubClassOf :AdversarialDetection :AdversarialDefense)

(Declaration (Class :StatisticalTesting))
(SubClassOf :StatisticalTesting :AdversarialDetection)
(DataPropertyAssertion :hasDescription :StatisticalTesting
  "Detecting distributional anomalies indicating adversarial manipulation")

(Declaration (Class :PredictionInconsistency))
(SubClassOf :PredictionInconsistency :AdversarialDetection)
(DataPropertyAssertion :hasDescription :PredictionInconsistency
  "Flagging inputs where model predictions are unusually sensitive")

;;; Ensemble Defenses
(Declaration (Class :EnsembleRobustness))
(SubClassOf :EnsembleRobustness :AdversarialDefense)

(Declaration (Class :DiverseEnsemble))
(SubClassOf :DiverseEnsemble :EnsembleRobustness)
(DataPropertyAssertion :hasDescription :DiverseEnsemble
  "Training multiple models with different architectures or training procedures")

(Declaration (Class :AdversarialEnsemble))
(SubClassOf :AdversarialEnsemble :EnsembleRobustness)
(DataPropertyAssertion :hasDescription :AdversarialEnsemble
  "Ensemble specifically optimised to resist transfer attacks")

;;; ROBUSTNESS EVALUATION ;;;

(Declaration (Class :RobustnessEvaluation))
(SubClassOf :RobustnessEvaluation :SafetyEvaluation)

;;; Evaluation Metrics
(Declaration (Class :RobustnessMetric))
(SubClassOf :RobustnessMetric :SafetyMetric)

(Declaration (Class :CleanAccuracy))
(SubClassOf :CleanAccuracy :RobustnessMetric)
(DataPropertyAssertion :measuresProperty :CleanAccuracy
  "Accuracy on unperturbed test examples")

(Declaration (Class :RobustAccuracy))
(SubClassOf :RobustAccuracy :RobustnessMetric)
(DataPropertyAssertion :measuresProperty :RobustAccuracy
  "Accuracy against adversarial examples within epsilon-ball")

(Declaration (Class :RobustnessGap))
(SubClassOf :RobustnessGap :RobustnessMetric)
(DataPropertyAssertion :measuresProperty :RobustnessGap
  "Difference between clean and robust accuracy")

(Declaration (Class :CertifiedAccuracy))
(SubClassOf :CertifiedAccuracy :RobustnessMetric)
(DataPropertyAssertion :measuresProperty :CertifiedAccuracy
  "Percentage of examples with provable robustness guarantees")

;;; Perturbation Norms
(Declaration (Class :PerturbationBound))

(Declaration (NamedIndividual :L2Norm))
(ClassAssertion :PerturbationBound :L2Norm)
(DataPropertyAssertion :hasDefinition :L2Norm
  "Euclidean distance: ||δ||₂ ≤ ε")

(Declaration (NamedIndividual :LInfNorm))
(ClassAssertion :PerturbationBound :LInfNorm)
(DataPropertyAssertion :hasDefinition :LInfNorm
  "Maximum absolute change: ||δ||_∞ ≤ ε")

(Declaration (NamedIndividual :L0Norm))
(ClassAssertion :PerturbationBound :L0Norm)
(DataPropertyAssertion :hasDefinition :L0Norm
  "Number of changed pixels: ||δ||₀ ≤ k")

;;; Evaluation Benchmarks
(Declaration (Class :RobustnessBenchmark))
(SubClassOf :RobustnessBenchmark :EvaluationBenchmark)

(Declaration (NamedIndividual :RobustBench))
(ClassAssertion :RobustnessBenchmark :RobustBench)
(DataPropertyAssertion :hasDescription :RobustBench
  "Standardised adversarial robustness benchmark across vision datasets")
(DataPropertyAssertion :hasURL :RobustBench "https://robustbench.github.io/")

(Declaration (NamedIndividual :AutoAttack))
(ClassAssertion :RobustnessBenchmark :AutoAttack)
(DataPropertyAssertion :hasDescription :AutoAttack
  "Ensemble of complementary attacks for reliable robustness evaluation")

;;; THEORETICAL FOUNDATIONS ;;;

(Declaration (Class :RobustnessTheory))
(SubClassOf :RobustnessTheory :TheoreticalFramework)

;;; Trade-offs
(Declaration (Class :AccuracyRobustnessTradeoff))
(SubClassOf :AccuracyRobustnessTradeoff :RobustnessTheory)
(AnnotationAssertion rdfs:comment :AccuracyRobustnessTradeoff
  "Inherent tension between clean accuracy and adversarial robustness")

(Declaration (Class :RobustnessGeneralisationTradeoff))
(SubClassOf :RobustnessGeneralisationTradeoff :RobustnessTheory)
(AnnotationAssertion rdfs:comment :RobustnessGeneralisationTradeoff
  "Robust models may overfit to specific perturbation types")

;;; Theoretical Limits
(Declaration (Class :IntrinsicRobustness))
(SubClassOf :IntrinsicRobustness :RobustnessTheory)
(AnnotationAssertion rdfs:comment :IntrinsicRobustness
  "Upper bound on achievable robustness for given task and perturbation budget")

(Declaration (Class :ConcentrationOfMeasure))
(SubClassOf :ConcentrationOfMeasure :RobustnessTheory)
(AnnotationAssertion rdfs:comment :ConcentrationOfMeasure
  "High-dimensional geometry makes adversarial examples abundant")

;;; ROBUSTNESS CHALLENGES ;;;

(Declaration (Class :RobustnessChallenge))
(SubClassOf :RobustnessChallenge :TechnicalChallenge)

(Declaration (Class :ScalabilityChallenge))
(SubClassOf :ScalabilityChallenge :RobustnessChallenge)
(AnnotationAssertion rdfs:comment :ScalabilityChallenge
  "Adversarial training computationally expensive, especially at scale")

(Declaration (Class :TransferabilityProblem))
(SubClassOf :TransferabilityProblem :RobustnessChallenge)
(AnnotationAssertion rdfs:comment :TransferabilityProblem
  "Adversarial examples transfer between models, complicating defense")

(Declaration (Class :AdaptiveAttacks))
(SubClassOf :AdaptiveAttacks :RobustnessChallenge)
(AnnotationAssertion rdfs:comment :AdaptiveAttacks
  "Attackers adapt to known defenses, requiring continuous arms race")

(Declaration (Class :EvaluationDifficulty))
(SubClassOf :EvaluationDifficulty :RobustnessChallenge)
(AnnotationAssertion rdfs:comment :EvaluationDifficulty
  "Difficult to comprehensively evaluate all possible attack vectors")

;;; DISTRIBUTION ROBUSTNESS ;;;

(Declaration (Class :DistributionRobustness))
(SubClassOf :DistributionRobustness :AdversarialRobustness)

(Declaration (Class :OutOfDistributionDetection))
(SubClassOf :OutOfDistributionDetection :DistributionRobustness)
(AnnotationAssertion rdfs:comment :OutOfDistributionDetection
  "Identifying inputs from distribution shift for selective prediction")

(Declaration (Class :DomainAdaptation))
(SubClassOf :DomainAdaptation :DistributionRobustness)
(AnnotationAssertion rdfs:comment :DomainAdaptation
  "Maintaining performance across domain shifts and covariate shifts")

(Declaration (Class :DistributionallyRobustOptimisation))
(SubClassOf :DistributionallyRobustOptimisation :DistributionRobustness)
(AnnotationAssertion skos:altLabel :DistributionallyRobustOptimisation "DRO"@en)
(DataPropertyAssertion :hasDescription :DistributionallyRobustOptimisation
  "Optimising worst-case performance over distributional uncertainty set")

;;; RELATIONSHIPS ;;;

(Declaration (ObjectProperty :resistsAttack))
(SubObjectPropertyOf :resistsAttack :withstands)
(ObjectPropertyDomain :resistsAttack :AdversarialRobustness)
(ObjectPropertyRange :resistsAttack :AdversarialAttack)

(Declaration (ObjectProperty :defends))
(SubObjectPropertyOf :defends :protects)
(ObjectPropertyDomain :defends :AdversarialDefense)
(ObjectPropertyRange :defends :AIModel)

(Declaration (ObjectProperty :certifies))
(SubObjectPropertyOf :certifies :guarantees)
(ObjectPropertyDomain :certifies :CertifiedDefense)
(ObjectPropertyRange :certifies :RobustnessProperty)

;;; METADATA & CITATIONS ;;;

(AnnotationAssertion rdfs:label :AdversarialRobustness "Adversarial Robustness"@en)
(AnnotationAssertion skos:definition :AdversarialRobustness
  "Property of AI systems to maintain reliable performance and correct behaviour when subjected to adversarially crafted inputs, distributional shifts, and deliberately optimised perturbations, encompassing both empirical defenses and provably certified guarantees against bounded attacks."@en)

(AnnotationAssertion :hasCanonicalCitation :AdversarialRobustness
  "Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.")

(AnnotationAssertion :hasCanonicalCitation :AdversarialRobustness
  "Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. ICLR 2018.")

(AnnotationAssertion :hasCanonicalCitation :AdversarialRobustness
  "Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified adversarial robustness via randomized smoothing. ICML 2019.")

(AnnotationAssertion :hasCanonicalCitation :AdversarialRobustness
  "Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. IEEE Symposium on Security and Privacy.")

(AnnotationAssertion :hasCanonicalCitation :AdversarialRobustness
  "Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. ICML 2020.")

(AnnotationAssertion :hasKeyChallenge :AdversarialRobustness
  "Accuracy-robustness tradeoff")
(AnnotationAssertion :hasKeyChallenge :AdversarialRobustness
  "Scalability of adversarial training")
(AnnotationAssertion :hasKeyChallenge :AdversarialRobustness
  "Adaptive attack circumvention")
(AnnotationAssertion :hasKeyChallenge :AdversarialRobustness
  "Comprehensive evaluation difficulty")

(AnnotationAssertion dc:created :AdversarialRobustness "2025-10-28"^^xsd:date)
(AnnotationAssertion dc:creator :AdversarialRobustness "AI Safety Research Specialist")
(AnnotationAssertion :termIdentifier :AdversarialRobustness "AI-0401")
(AnnotationAssertion :priorityLevel :AdversarialRobustness "4")
