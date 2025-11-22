- ### OntologyBlock
  id:: peer-to-peer-network-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0075

    - filename-history:: ["BC-0075-peer-to-peer-network.md"]
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0075
    - preferred-term:: Peer-to-Peer Network
    - source-domain:: metaverse
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Decentralized communication within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Peer-to-peerNetwork
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - is-subclass-of:: [[Blockchain]]
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[Networkcomponent]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-from:: [[DistributedLedgerTechnologyDlt]] via requires

  - 