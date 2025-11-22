- ### OntologyBlock
    - term-id:: BC-0011
    - preferred-term:: Block Height
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]
    - version:: 1.0.0

## Block Height

Block Height refers to the sequential position of a block within a blockchain, measured as the number of blocks between it and the genesis block (block 0). This fundamental metric serves as a universal coordinate system for blockchain state, enabling nodes to synchronize, clients to verify chain progression, and protocols to implement time-locked transactions or consensus checkpoints. Block height functions as both a chronological marker and a security indicator, with higher block depths providing exponentially greater immutability guarantees through accumulated proof-of-work or finality mechanisms.

In proof-of-work blockchains like Bitcoin (currently exceeding 870,000 blocks as of early 2025), block height correlates directly with network security through accumulated computational difficulty. Ethereum's transition to proof-of-stake maintains block height as a state progression metric while achieving finality through epoch checkpoints. Enterprise blockchains like Hyperledger Fabric use block height for ordering service coordination and channel state verification. Technical applications include: time-locked transactions using CheckLockTimeVerify (CLTV) opcodes, hard fork activation heights (e.g., Ethereum's Shanghai upgrade at block 15,537,393), and light client verification via block height proofs.

Current adoption encompasses all major blockchain protocols, with block height serving as the primary indexing mechanism for blockchain explorers (Etherscan, Blockchain.com), API services, and distributed applications. DeFi protocols utilize block height for oracle price feeds, governance proposal voting windows, and vesting schedule calculations. Layer-2 scaling solutions reference layer-1 block heights for state commitments and fraud proof challenges. Standards from IEEE 2418.1 specify block height encoding in blockchain interoperability protocols, while NIST guidelines reference block height in blockchain forensics and audit trail verification. The metric remains critical for consensus algorithm coordination, with proof-of-stake protocols using slot heights and finalized heights to distinguish proposed versus confirmed states.

## Technical Details

- **Id**: 66314bd7-86ef-4ca2-8f39-704e133ac0a3
- **Collapsed**: true
- **Domain Prefix**: BC
- **Sequence Number**: 0011
- **Filename History**: ["BC-0011-block-height.md"]
- **Public Access**: true
- **Source Domain**: blockchain
- **Status**: complete
- **Last Updated**: 2025-11-18
- **Maturity**: mature
- **Source**: [[IEEE 2418.1]], [[NIST NISTIR 8202]], [[Bitcoin Developer Reference]], [[Ethereum Yellow Paper]]
- **Authority Score**: 0.96
- **Owl:Class**: bc:BlockHeight
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Object
- **Owl:Inferred Class**: bc:VirtualObject
- **Belongstodomain**: [[BlockchainDomain]]
- **Implementedinlayer**: [[ConceptualLayer]]
- **Is Subclass Of**: [[Blockchain Entity]], [[DistributedDataStructure]]

## Metadata

- **Migration Status**: Ontology block enriched on 2025-11-12
- **Last Updated**: 2025-11-12
- **Review Status**: Comprehensive editorial review
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable
