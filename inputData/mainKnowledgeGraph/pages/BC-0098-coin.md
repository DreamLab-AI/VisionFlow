- ### OntologyBlock
  id:: coin-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: BC-0098
    - preferred-term:: Coin
    - source-domain:: blockchain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Native blockchain asset within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Coin
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[TokenEconomicsDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Token]]

  - #### CrossDomainBridges
    - bridges-to:: [[Economicmechanism]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of

  - 