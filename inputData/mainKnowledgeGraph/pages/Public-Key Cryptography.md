- ### OntologyBlock
    - term-id:: BC-0031
    - preferred-term:: Public-Key Cryptography
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]
    - version:: 1.0.0

## Public-Key Cryptography

Public-Key Cryptography refers to asymmetric cryptographic systems utilizing mathematically related key pairs for encryption, digital signatures, and secure communication in distributed networks. In blockchain technology, public-key cryptography forms the foundational security layer enabling trustless transactions, identity verification, and cryptographic ownership without centralized authorities. Each participant possesses a private key (secret) and corresponding public key (shared), where operations performed with one key can only be reversed or verified with its counterpart.

The mechanism underpins all major blockchain protocols through digital signature schemes (ECDSA, EdDSA, Schnorr), address generation systems, and cryptographic authentication protocols. Bitcoin employs secp256k1 elliptic curve cryptography for transaction signing, while Ethereum uses the same curve for account generation and smart contract interactions. Modern proof-of-stake networks like Ethereum 2.0 utilize BLS signatures for validator attestations, enabling signature aggregation and reducing bandwidth requirements. The cryptographic security relies on computational hardness assumptions such as discrete logarithm problem and elliptic curve discrete logarithm problem.

Current adoption spans all blockchain platforms, cryptocurrency wallets, decentralized identity systems (DIDs), and Web3 authentication protocols. Enterprise implementations include Hyperledger Fabric's MSP (Membership Service Provider) using X.509 certificates, and permissioned networks employing RSA or ECDSA for participant authentication. Technical standards from NIST FIPS 186-5 specify digital signature algorithms, while ISO/IEC 14888 defines signature schemes with message recovery. The transition to post-quantum cryptography is underway, with NIST selecting ML-DSA (Dilithium), SLH-DSA (SPHINCS+), and Falcon for standardization to protect against quantum computing threats to current elliptic curve and RSA-based systems.

## Technical Details

- **Id**: public-key-cryptography-standards
- **Collapsed**: true
- **Domain Prefix**: BC
- **Sequence Number**: 0031
- **Filename History**: ["BC-0031-public-key-cryptography.md"]
- **Public Access**: true
- **Source Domain**: blockchain
- **Status**: complete
- **Last Updated**: 2025-11-18
- **Maturity**: mature
- **Source**: [[NIST FIPS 186-5]], [[ISO/IEC 14888]], [[RFC 8017]], [[IEEE 2418.1]]
- **Authority Score**: 0.99
- **Owl:Class**: bc:Public-keyCryptography
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Object
- **Owl:Inferred Class**: bc:VirtualObject
- **Belongstodomain**: [[CryptographicDomain]]
- **Blockchainrelevance**: High
- **Lastvalidated**: 2025-11-14
- **Implementedinlayer**: [[SecurityLayer]]
- **Is Subclass Of**: [[Blockchain Entity]], [[CryptographicPrimitive]]
