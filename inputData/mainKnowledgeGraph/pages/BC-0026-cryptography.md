- ### OntologyBlock
  id:: cryptography-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: BC-0026
    - preferred-term:: Cryptography
    - source-domain:: blockchain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Cryptography is the mathematical science of securing information through techniques that transform data into unintelligible forms for unauthorized parties whilst enabling authorized parties to reverse the transformation, providing confidentiality, integrity, authentication, and non-repudiation in blockchain systems.
    - maturity:: mature
    - source:: [[ISO/IEC 18033]], [[NIST FIPS]], [[NIST SP]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: bc:Cryptography
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[CryptographicDomain]]

  - #### OWL Restrictions
    
    
    

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[Security Technology]]

  - #### CrossDomainBridges
    - bridges-to:: [[MathematicalScience]] via is-subclass-of
    - bridges-to:: [[InformationProtection]] via is-subclass-of
    - bridges-to:: [[SecurityTechnology]] via is-subclass-of
    - bridges-from:: [[DistributedLedgerTechnologyDlt]] via depends-on

  - 