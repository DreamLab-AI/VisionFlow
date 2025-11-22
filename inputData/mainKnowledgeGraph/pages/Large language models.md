- ### OntologyBlock
  id:: large-language-models-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: AI-0850
    - preferred-term:: Large Language Models
    - source-domain:: ai
    - status:: complete
    - public-access:: true
    - version:: 2.0.0
    - last-updated:: 2025-01-15
    - quality-score:: 0.92

  - **Definition**
    - definition:: [[Large Language Models]] (LLMs) are [[Foundation Models]] with billions to trillions of parameters trained on massive text corpora using [[Transformer]] architectures and [[Self-Supervised Learning]], capable of performing diverse [[Natural Language Processing]] tasks through [[Few-Shot Learning]], [[Zero-Shot Learning]], and [[Prompt Engineering]]. LLMs represent a paradigm shift in [[Artificial Intelligence]], demonstrating emergent capabilities in reasoning, code generation, multilingual understanding, and complex task decomposition.
    - maturity:: mature
    - source:: [[OpenAI Research]], [[Google DeepMind]], [[Anthropic]], [[Meta AI Research]], [[NIST AI Standards]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: ai:LargeLanguageModel
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - owl:inferred-class:: ai:VirtualProcess
    - belongsToDomain:: [[AI-GroundedDomain]], [[ComputationAndIntelligenceDomain]], [[DataManagementDomain]]

  - #### OWL Restrictions
    

  - #### CrossDomainBridges
    - bridges-from:: [[VoiceInteraction]] via has-part
    - dt:enables:: [[Intelligent Npc]]
    - dt:enables:: [[DigitalAvatar]]

  -
### Relationships
- is-subclass-of:: [[ModelArchitecture]]

