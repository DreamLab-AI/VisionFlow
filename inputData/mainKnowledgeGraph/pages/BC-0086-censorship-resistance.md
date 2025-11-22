- ### OntologyBlock
  id:: censorship-resistance-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0086
    - preferred-term:: Censorship Resistance
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Transaction inclusion guarantee within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:CensorshipResistance
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain]]

  - #### CrossDomainBridges
    - bridges-to:: [[Networkcomponent]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-from:: [[DecentralizationLayer]] via enables
    - bridges-from:: [[Bitcoin]] via enables

  - 