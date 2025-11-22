- ### OntologyBlock
  id:: public-key-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0037

    - filename-history:: ["BC-0037-public-key.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
    - term-id:: BC-0037
    - preferred-term:: Public Key
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-11-13

  - **Definition**
    - definition:: Publicly-shared cryptographic key within blockchain systems, providing essential functionality for distributed ledger technology operations and properties. In public-key cryptography, a public key is distributed openly and used to encrypt messages or verify digital signatures, while the corresponding private key remains secret with its owner.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:PublicKey
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of

  - 