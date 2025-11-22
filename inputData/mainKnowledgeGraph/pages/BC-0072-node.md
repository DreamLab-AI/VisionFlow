- ### OntologyBlock
  id:: node-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: BC-0072
    - preferred-term:: Node
    - source-domain:: blockchain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Network participant computer within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Node
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Blockchain Network]]

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-to:: [[Networkcomponent]] via is-subclass-of

  - 