- ### OntologyBlock
  id:: blockchain-network-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0071
    - preferred-term:: Blockchain Network
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Distributed node infrastructure within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:BlockchainNetwork
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Distributed Ledger]]

  - #### CrossDomainBridges
    - bridges-to:: [[Networkcomponent]] via is-subclass-of
    - bridges-to:: [[BlockchainEntity]] via is-subclass-of
    - bridges-from:: [[TokenCustodyService]] via depends-on
    - bridges-from:: [[CryptoToken]] via is-part-of
    - bridges-from:: [[NftSwapping]] via requires
    - bridges-from:: [[SmartRoyaltyContract]] via depends-on
    - bridges-from:: [[DigitalTaxComplianceNode]] via depends-on
    - bridges-from:: [[Tokenization]] via requires
    - bridges-from:: [[VirtualSecuritiesOfferingVso]] via depends-on
    - bridges-from:: [[DigitalRightsManagementExtended]] via depends-on
    - bridges-from:: [[NftRenting]] via depends-on
    - bridges-from:: [[SmartRoyaltiesLedger]] via depends-on

  - 