- ### OntologyBlock
  id:: script-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0023

    - filename-history:: ["BC-0023-script.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
    - term-id:: BC-0023
    - preferred-term:: Script
    - source-domain:: blockchain
    - status:: complete
    - version:: 2.0.0
    - last-updated:: 2025-11-14

  - **Definition**
    - definition:: Programming language implementations for transaction validation, smart contract execution, and cryptographic operations in blockchain systems, enabling deterministic computation and consensus enforcement across distributed networks.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[BIP 141 (Bitcoin)]], [[Ethereum Yellow Paper]]
    - authority-score:: 0.97
    - blockchainRelevance:: High
	- lastValidated:: 2025-11-14
    - qualityScore:: 0.91

  - **Semantic Classification**
    - owl:class:: bc:Script
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[BlockchainDomain]], [[SmartContractDomain]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-to:: [[Distributeddatastructure]] via is-subclass-of
    - bridges-from:: [[Bitcoin]] via has-part

  - 