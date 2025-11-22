- ### OntologyBlock
  id:: network-latency-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0081

    - filename-history:: ["BC-0081-network-latency.md"]
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0081
    - preferred-term:: Network Latency
    - source-domain:: metaverse
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Communication delay within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:NetworkLatency
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - is-subclass-of:: [[Metaverse Infrastructure]]
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-to:: [[Networkcomponent]] via is-subclass-of
    - bridges-from:: [[VoiceInteraction]] via depends-on

  - 