- ### OntologyBlock
  id:: blockchain-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: BC-0001
    - preferred-term:: Blockchain
    - source-domain:: blockchain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: A blockchain is a distributed, cryptographically-secured data structure consisting of an ordered chain of blocks, where each block contains a cryptographic hash of the previous block, a timestamp, and transaction data, maintained through a consensus mechanism across a peer-to-peer network without requiring a trusted central authority.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[ISO/IEC TR]], [[ISO/IEC 23455:2019]], [[ITU-T Y.4460]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: bc:Blockchain
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[BlockchainDomain]]
    - implementedInLayer:: [[ConceptualLayer]]

  - #### Relationships
    id:: blockchain-relationships
    - is-subclass-of:: [[Distributed Data Structure]], [[Distributed Ledger]], [[Cryptographic System]]

  - #### OWL Axioms
    id:: blockchain-owl-axioms
    collapsed:: true
    - ```clojure
      Prefix(:=<http://metaverse-ontology.org/blockchain#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
Prefix(xml:=<http://www.w3.org/XML/1998/namespace>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Prefix(dct:=<http://purl.org/dc/terms/>)

Ontology(<http://metaverse-ontology.org/blockchain/BC-0001>
  Import(<http://metaverse-ontology.org/blockchain/core>)

  ## Class Declaration
  Declaration(Class(:Blockchain))

  ## Subclass Relationships
  SubClassOf(:Blockchain :DistributedDataStructure)
  SubClassOf(:Blockchain :CryptographicSystem)
  SubClassOf(:Blockchain :DistributedLedger)

  ## Essential Properties
  SubClassOf(:Blockchain
    (ObjectExactCardinality 1 :hasGenesisBlock :Block))

  SubClassOf(:Blockchain
    (ObjectMinCardinality 1 :containsBlock :Block))

  SubClassOf(:Blockchain
    (ObjectExactCardinality 1 :usesConsensus :ConsensusMechanism))

  SubClassOf(:Blockchain
    (ObjectExactCardinality 1 :usesHashFunction :CryptographicHashFunction))

  SubClassOf(:Blockchain
    (ObjectMinCardinality 1 :maintainedBy :Node))

  SubClassOf(:Blockchain
    (DataHasValue :isImmutable "true"^^xsd:boolean))

  SubClassOf(:Blockchain
    (DataHasValue :isDistributed "true"^^xsd:boolean))

  SubClassOf(:Blockchain
    (DataHasValue :isDecentralized "true"^^xsd:boolean))

  ## Data Properties
  DataPropertyAssertion(:hasBlockHeight :Blockchain xsd:nonNegativeInteger)
  DataPropertyAssertion(:hasChainDifficulty :Blockchain xsd:decimal)
  DataPropertyAssertion(:hasBlockTime :Blockchain xsd:duration)
  DataPropertyAssertion(:hasCreationDate :Blockchain xsd:dateTime)

  ## Object Properties
  ObjectPropertyAssertion(:containsBlock :Blockchain :Block)
  ObjectPropertyAssertion(:maintainedBy :Blockchain :Node)
  ObjectPropertyAssertion(:executesTransaction :Blockchain :Transaction)
  ObjectPropertyAssertion(:implementsProtocol :Blockchain :BlockchainProtocol)

  ## Annotations
  AnnotationAssertion(rdfs:label :Blockchain "Blockchain"@en)
  AnnotationAssertion(rdfs:comment :Blockchain
    "A distributed, cryptographically-secured chain of blocks maintained through consensus"@en)
  AnnotationAssertion(dct:description :Blockchain
    "Core data structure combining cryptographic hashing, distributed consensus, and sequential block ordering"@en)
  AnnotationAssertion(:termID :Blockchain "BC-0001")
  AnnotationAssertion(:authorityScore :Blockchain "1.0"^^xsd:decimal)

  ## Disjoint Classes
  DisjointClasses(:Blockchain :TraditionalDatabase)
  DisjointClasses(:Blockchain :CentralizedLedger)
)
      ```
