- ### OntologyBlock
  id:: generative-ai-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: AI-0860
    - preferred-term:: Generative AI
    - source-domain:: ai
    - status:: complete
    - public-access:: true
    - version:: 2.0.0
    - last-updated:: 2025-01-15
    - quality-score:: 0.90

  - **Definition**
    - definition:: [[Generative AI]] encompasses [[Machine Learning]] systems capable of creating new content across modalities including text, images, audio, video, and code through [[Neural Networks]] trained on large datasets. These systems use [[Deep Learning]] architectures like [[Transformer]]s, [[Diffusion Models]], [[GANs]], and [[VAEs]] to learn data distributions and generate novel, coherent outputs. Generative AI represents a paradigm shift from discriminative models, enabling creative applications in [[Content Creation]], [[Design]], [[Art]], [[Music Generation]], and [[Code Synthesis]].
    - maturity:: mature
    - source:: [[OpenAI]], [[Stability AI]], [[Midjourney]], [[Anthropic]], [[Google DeepMind]], [[NIST AI Standards]]
    - authority-score:: 0.93

  - **Semantic Classification**
    - owl:class:: ai:GenerativeAI
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - owl:inferred-class:: ai:VirtualProcess
    - belongsToDomain:: [[AI-GroundedDomain]], [[CreativeMediaDomain]], [[ComputationAndIntelligenceDomain]]

  - #### OWL Restrictions
    

  - #### CrossDomainBridges
    - bridges-to:: [[MachineLearning]] via is-subclass-of
    - bridges-from:: [[BehaviouralFeedbackLoop]] via has-part

  - 
### Relationships
- is-subclass-of:: [[MachineLearning]]

