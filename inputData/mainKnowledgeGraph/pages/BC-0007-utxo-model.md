- ### OntologyBlock
  id:: utxo-model-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0007
    - preferred-term:: UTXO Model
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Unspent Transaction Output accounting model within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:UTXOModel
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[BlockchainDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Transaction]]

  - #### CrossDomainBridges
    - bridges-to:: [[Distributeddatastructure]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-from:: [[Bitcoin]] via implements

  - 