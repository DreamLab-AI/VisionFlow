- ### OntologyBlock
  id:: hash-function-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0027
    - preferred-term:: Hash Function
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: One-way data transformation within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: bc:HashFunction
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Cryptography]]

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of

  - 