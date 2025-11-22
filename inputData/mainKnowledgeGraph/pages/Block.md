- ### OntologyBlock
  id:: block-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0003

    - filename-history:: ["BC-0003-block.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
    - term-id:: BC-0003
    - preferred-term:: Block
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Fundamental unit containing transactions within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: bc:Block
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[BlockchainDomain]]
    - owl:disjointWith:: [[Algorithm]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[Distributeddatastructure]] via is-subclass-of
    - bridges-from:: [[Bitcoin]] via has-part

  - 