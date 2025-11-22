# Large Language Models

- ### OntologyBlock
  id:: llm-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: AI-0850
    - preferred-term:: Large Language Models
    - alt-terms:: [[LLM]], [[Foundation Models]]
    - source-domain:: ai
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-21
    - quality-score:: 0.85

  - **Definition**
    - definition:: A Large Language Model (LLM) is an artificial intelligence system based on deep neural networks (typically [[Transformer]] architectures) trained on vast text corpora to understand and generate human-like text, demonstrating emergent capabilities across diverse tasks.
    - maturity:: mature
    - source:: [[OpenAI Research]], [[Stanford AI Index]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: ai:LargeLanguageModel
    - owl:physicality:: VirtualEntity
    - owl:role:: Process
    - belongsToDomain:: [[AI-GroundedDomain]]

  - #### Relationships
    id:: llm-relationships
    - is-subclass-of:: [[Artificial Intelligence]], [[Neural Network Architecture]]
    - requires:: [[Training Data]], [[Computational Resources]]
    - enables:: [[Few-Shot Learning]], [[Text Generation]]
