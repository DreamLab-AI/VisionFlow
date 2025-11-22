- ### OntologyBlock
  id:: schnorr-signature-ontology
  collapsed:: true

  - **Identification**

    - domain-prefix:: BC

    - sequence-number:: 0041

    - filename-history:: ["BC-0041-schnorr-signature.md"]
    - public-access:: true
    - ontology:: true
    - is-subclass-of:: [[BlockchainTechnology]]
    - term-id:: BC-0041
    - preferred-term:: Schnorr Signature
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Cryptographic digital signature scheme based on discrete logarithm problem hardness, providing efficient, aggregatable, and provably secure signatures for Bitcoin transactions, enabling multi-signature constructions (MuSig2), threshold signatures (FROST), enhanced privacy through signature indistinguishability, and reduced blockchain footprint via signature aggregation in Taproot.
    - maturity:: production
    - source:: [[BIP 340 Schnorr Signatures]], [[BIP 341 Taproot]], [[MuSig2 Specification]], [[FROST Protocol]], [[Bitcoin Core 2025]]
    - authority-score:: 0.98

  - **Semantic Classification**
    - owl:class:: bc:SchnorrSignature
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### CrossDomainBridges
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of

  - 