- ### OntologyBlock
  id:: light-node-ontology
  collapsed:: true

  - **Identification**
    - domain-prefix:: BC
    - sequence-number:: 0074
    - filename-history:: ["BC-0074-light-node.md"]
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0074
    - preferred-term:: Light Node
    - source-domain:: metaverse
    - status:: complete
    - version:: 2.0.0
    - last-updated:: 2025-11-14

  - **Definition**
    - definition:: A blockchain network node that stores only block headers and essential transaction data rather than the complete blockchain ledger, utilizing Simplified Payment Verification (SPV) to validate transactions efficiently with minimal resource requirements.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR 8202]], [[Bitcoin Whitepaper]]
    - authority-score:: 0.98

  - **Semantic Classification**
    - owl:class:: bc:LightNode
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - is-subclass-of:: [[Blockchain]]
    - belongsToDomain:: [[CryptographicDomain]], [[NetworkArchitecture]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[Networkcomponent]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-from:: [[SceneGraph]] via has-part

  - 