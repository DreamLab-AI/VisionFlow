- ### OntologyBlock
  id:: digital-signature-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0030
    - preferred-term:: Digital Signature
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Cryptographic authentication mechanism within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:DigitalSignature
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
    - bridges-from:: [[Tokenization]] via depends-on
    - bridges-from:: [[VirtualPropertyRight]] via depends-on
    - bridges-from:: [[DigitalEvidenceChainOfCustody]] via has-part
    - bridges-from:: [[DataProvenance]] via requires
    - bridges-from:: [[DistributedLedgerTechnologyDlt]] via requires
    - bridges-from:: [[NftSwapping]] via requires
    - bridges-from:: [[VerifiableCredentialVc]] via requires
    - bridges-from:: [[CulturalProvenanceRecord]] via requires
    - bridges-from:: [[VirtualNotaryService]] via requires

  - 