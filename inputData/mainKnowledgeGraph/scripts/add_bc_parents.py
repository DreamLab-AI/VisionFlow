#!/usr/bin/env python3
"""
Add semantic parent relationships to BC domain orphan concepts.

Updated to use shared ontology_block_parser library for better compatibility.
"""

import os
import sys
import re
from pathlib import Path
from typing import Tuple, List

# Add shared library to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Ontology-Tools' / 'tools' / 'lib'))

from ontology_block_parser import OntologyBlockParser, OntologyBlock

# Default pages directory - can be overridden via command line
DEFAULT_PAGES_DIR = Path(__file__).parent.parent / 'pages'

# Semantic parent mapping for BC domain orphans
# Format: BC-ID -> parent concept name
PARENT_MAPPING = {
    # Blockchain Fundamentals (BC-0003 to BC-0025)
    "BC-0003": "Blockchain",  # Block
    "BC-0004": "Block",  # Block Header
    "BC-0005": "Block",  # Genesis Block
    "BC-0006": "Blockchain",  # Transaction
    "BC-0007": "Transaction",  # UTXO Model
    "BC-0008": "Transaction",  # Account Model
    "BC-0010": "Blockchain",  # Decentralization
    "BC-0011": "Block",  # Block Height
    "BC-0012": "Mining",  # Block Reward
    "BC-0013": "Block",  # Block Size
    "BC-0014": "Block",  # Block Time
    "BC-0015": "Blockchain",  # Chain Reorganization
    "BC-0016": "Block",  # Orphan Block
    "BC-0017": "Block",  # Timestamp
    "BC-0018": "Transaction",  # Transaction Pool
    "BC-0019": "Transaction Pool",  # Mempool
    "BC-0020": "Transaction",  # Transaction Confirmation
    "BC-0021": "Transaction",  # Input
    "BC-0022": "Transaction",  # Output
    "BC-0023": "Transaction",  # Script
    "BC-0024": "Script",  # Opcodes
    "BC-0025": "Blockchain",  # Chain State

    # Cryptography (BC-0026 to BC-0050)
    "BC-0026": "Security Technology",  # Cryptography
    "BC-0027": "Cryptography",  # Hash Function
    "BC-0028": "Hash Function",  # SHA-256
    "BC-0029": "Hash Function",  # Merkle Tree
    "BC-0030": "Cryptography",  # Digital Signature
    "BC-0031": "Cryptography",  # Public Key Cryptography
    "BC-0032": "Public Key Cryptography",  # Elliptic Curve Cryptography
    "BC-0033": "Cryptography",  # Zero Knowledge Proof
    "BC-0034": "Cryptography",  # Nonce
    "BC-0035": "Consensus Mechanism",  # Difficulty
    "BC-0036": "Public Key Cryptography",  # Private Key
    "BC-0038": "Public Key Cryptography",  # Address
    "BC-0039": "Digital Signature",  # Signature Scheme
    "BC-0040": "Signature Scheme",  # ECDSA
    "BC-0041": "Signature Scheme",  # Schnorr Signature
    "BC-0042": "Hash Function",  # Keccak-256
    "BC-0043": "Hash Function",  # BLAKE2
    "BC-0044": "Merkle Tree",  # Merkle Root
    "BC-0045": "Merkle Tree",  # Merkle Proof
    "BC-0046": "Hash Function",  # Hash Collision
    "BC-0047": "Hash Function",  # Preimage Resistance
    "BC-0048": "Hash Function",  # Collision Resistance
    "BC-0049": "Cryptography",  # Salt
    "BC-0050": "Cryptography",  # Cryptographic Commitment

    # Consensus and Mining (BC-0053 to BC-0070)
    "BC-0053": "Consensus Mechanism",  # Mining
    "BC-0054": "Mining",  # Miner
    "BC-0055": "Mining",  # Block Reward (duplicate ID)
    "BC-0056": "Mining",  # Difficulty Adjustment
    "BC-0057": "Blockchain Network",  # Network Synchronization
    "BC-0058": "Consensus Mechanism",  # Consensus Rule
    "BC-0059": "Consensus Rule",  # Longest Chain Rule
    "BC-0060": "Consensus Rule",  # Fork Choice Rule
    "BC-0061": "Consensus Mechanism",  # Nakamoto Consensus
    "BC-0062": "Blockchain Network",  # Block Propagation
    "BC-0063": "Mining",  # Network Hash Rate
    "BC-0064": "Mining",  # Mining Pool
    "BC-0065": "Mining",  # Solo Mining
    "BC-0066": "Mining Pool",  # Pool Share
    "BC-0067": "Mining",  # Difficulty Target
    "BC-0068": "Transaction",  # Coinbase Transaction
    "BC-0069": "Mining",  # Mining Reward
    "BC-0070": "Mining",  # Halving

    # Network and Nodes (BC-0071 to BC-0095)
    "BC-0071": "Distributed Ledger",  # Blockchain Network
    "BC-0072": "Blockchain Network",  # Node
    "BC-0073": "Node",  # Full Node
    "BC-0074": "Node",  # Light Node
    "BC-0075": "Blockchain Network",  # Peer-to-Peer Network
    "BC-0076": "Blockchain",  # Double Spending
    "BC-0077": "Blockchain",  # 51% Attack
    "BC-0078": "Blockchain Network",  # Sybil Attack
    "BC-0079": "Blockchain",  # Immutability
    "BC-0080": "Consensus Mechanism",  # Finality
    "BC-0081": "Blockchain Network",  # Network Latency
    "BC-0082": "Block Propagation",  # Block Propagation Time
    "BC-0083": "Blockchain Network",  # Eclipse Attack
    "BC-0084": "Blockchain Network",  # Partition Attack
    "BC-0085": "Mining",  # Selfish Mining
    "BC-0086": "Blockchain",  # Censorship Resistance
    "BC-0087": "Blockchain Network",  # Network Topology
    "BC-0088": "Blockchain Network",  # Gossip Protocol
    "BC-0089": "Blockchain Network",  # Peer Discovery
    "BC-0090": "Blockchain Network",  # Permissionless Network
    "BC-0091": "Blockchain Network",  # Permissioned Network
    "BC-0092": "Node",  # Validator Node
    "BC-0093": "Node",  # Archival Node
    "BC-0094": "Node",  # Pruned Node
    "BC-0095": "Node",  # Bootstrap Node

    # Tokens and Economics (BC-0096 to BC-0120)
    "BC-0098": "Token",  # Coin
    "BC-0099": "Token",  # Native Token
    "BC-0100": "Transaction Fee",  # Gas
    "BC-0101": "Token Economics",  # Transaction Fee
    "BC-0104": "Token Economics",  # Supply Cap
    "BC-0105": "Token Economics",  # Tokenomics
    "BC-0106": "Gas",  # Gas Price
    "BC-0108": "Transaction Fee",  # Base Fee
    "BC-0109": "Transaction Fee",  # Priority Fee
    "BC-0110": "Transaction Fee",  # Fee Market
    "BC-0111": "Token",  # Deflationary Token
    "BC-0112": "Token",  # Inflationary Token
    "BC-0113": "Token Economics",  # Emission Schedule
    "BC-0114": "Token Economics",  # Burning Mechanism
    "BC-0115": "Token Economics",  # Minting
    "BC-0116": "Token Economics",  # Total Supply
    "BC-0117": "Token Economics",  # Circulating Supply
    "BC-0118": "Token Economics",  # Market Capitalization
    "BC-0119": "Blockchain",  # Economic Security
    "BC-0120": "Token Economics",  # Incentive Alignment

    # Enterprise Blockchain (BC-0426 to BC-0440)
    "BC-0426": "Enterprise Blockchain",  # Hyperledger Fabric
    "BC-0427": "Enterprise Blockchain",  # Hyperledger Besu
    "BC-0428": "Blockchain",  # Enterprise Blockchain Architecture
    "BC-0429": "Blockchain",  # Permissioned Blockchain
    "BC-0430": "Permissioned Blockchain",  # Private Channels
    "BC-0431": "Blockchain",  # Privacy-Preserving Blockchain
    "BC-0432": "Permissioned Blockchain",  # Consortium Blockchain
    "BC-0433": "Enterprise Blockchain",  # Enterprise Token Standards
    "BC-0434": "Enterprise Blockchain",  # Blockchain as a Service
    "BC-0435": "Enterprise Blockchain",  # Hyperledger Indy
    "BC-0436": "Enterprise Blockchain",  # Hyperledger Iroha
    "BC-0437": "Enterprise Blockchain",  # R3 Corda
    "BC-0438": "Enterprise Blockchain",  # Quorum Blockchain
    "BC-0439": "Smart Contract",  # Enterprise Smart Contracts
    "BC-0440": "Blockchain",  # Blockchain Interoperability

    # Supply Chain Applications (BC-0441 to BC-0455)
    "BC-0441": "Supply Chain Blockchain",  # Provenance Tracking
    "BC-0442": "Supply Chain Blockchain",  # Pharmaceutical Traceability
    "BC-0443": "Supply Chain Blockchain",  # Food Safety Blockchain
    "BC-0444": "Supply Chain Blockchain",  # Luxury Goods Authentication
    "BC-0445": "Supply Chain Blockchain",  # Conflict Mineral Tracking
    "BC-0446": "Supply Chain Blockchain",  # Supply Chain Traceability
    "BC-0447": "Supply Chain Blockchain",  # Anti-Counterfeiting
    "BC-0448": "Supply Chain Blockchain",  # Cold Chain Monitoring
    "BC-0449": "Supply Chain Blockchain",  # Circular Economy
    "BC-0450": "Supply Chain Blockchain",  # Carbon Credit Tracking
    "BC-0451": "Supply Chain Blockchain",  # Logistics Optimization
    "BC-0452": "Supply Chain Blockchain",  # Customs Trade Facilitation
    "BC-0453": "Supply Chain Blockchain",  # Ethical Sourcing
    "BC-0454": "Supply Chain Blockchain",  # Waste Management
    "BC-0455": "Supply Chain Blockchain",  # Product Recall Management

    # Digital Identity (BC-0456 to BC-0460)
    "BC-0456": "Digital Identity",  # Self-Sovereign Identity
    "BC-0457": "Self-Sovereign Identity",  # Decentralized Identifiers
    "BC-0458": "Self-Sovereign Identity",  # Verifiable Credentials
    "BC-0459": "Self-Sovereign Identity",  # Digital Identity Wallet
    "BC-0460": "Self-Sovereign Identity",  # Identity Verification

    # DAO and Governance (BC-0461 to BC-0475)
    "BC-0461": "Smart Contract",  # Decentralized Autonomous Organization
    "BC-0462": "Decentralized Autonomous Organization",  # On-Chain Voting
    "BC-0463": "Decentralized Autonomous Organization",  # Governance Token
    "BC-0464": "Decentralized Autonomous Organization",  # Treasury Management
    "BC-0465": "Decentralized Autonomous Organization",  # Proposal System
    "BC-0466": "On-Chain Voting",  # Quadratic Voting
    "BC-0467": "On-Chain Voting",  # Conviction Voting
    "BC-0468": "Decentralized Autonomous Organization",  # Multi-Sig Governance
    "BC-0469": "On-Chain Voting",  # Snapshot Voting
    "BC-0470": "Decentralized Autonomous Organization",  # DAO Legal Structures
    "BC-0471": "Decentralized Autonomous Organization",  # Tokenomics Governance
    "BC-0472": "Decentralized Autonomous Organization",  # DAO Tooling
    "BC-0473": "On-Chain Voting",  # Delegate Democracy
    "BC-0474": "Decentralized Autonomous Organization",  # Grant Programs
    "BC-0475": "Decentralized Autonomous Organization",  # DAO Analytics

    # Compliance and Regulation (BC-0476 to BC-0490)
    "BC-0476": "Blockchain Compliance",  # AML/KYC Compliance
    "BC-0477": "Blockchain Compliance",  # Travel Rule
    "BC-0478": "Blockchain Compliance",  # Securities Regulation
    "BC-0479": "Blockchain Compliance",  # Stablecoin Regulation
    "BC-0480": "Blockchain Compliance",  # CBDC Frameworks
    "BC-0481": "Blockchain Compliance",  # FATF Recommendations
    "BC-0482": "Blockchain Compliance",  # EU MiCA Regulation
    "BC-0483": "Blockchain Compliance",  # US Regulatory Framework
    "BC-0484": "Blockchain Compliance",  # Asia-Pacific Regulation
    "BC-0485": "Blockchain Compliance",  # Tax Treatment Crypto
    "BC-0486": "Blockchain Compliance",  # Regulatory Reporting
    "BC-0487": "Blockchain Compliance",  # Compliance Monitoring
    "BC-0488": "Blockchain Compliance",  # Licensing Requirements
    "BC-0489": "Blockchain Compliance",  # Consumer Protection
    "BC-0490": "Blockchain Compliance",  # Cross-Border Compliance

    # Applications (BC-0491 to BC-0495)
    "BC-0491": "Blockchain Application",  # Healthcare Records
    "BC-0492": "Blockchain Application",  # Clinical Trials
    "BC-0493": "Blockchain Application",  # Real Estate Tokenization
    "BC-0494": "Blockchain Application",  # Property Registry
    "BC-0495": "Blockchain Application",  # Voting Systems

    # Sustainability (BC-0496 to BC-0505)
    "BC-0496": "Blockchain Sustainability",  # Energy Consumption Blockchain
    "BC-0497": "Blockchain Sustainability",  # Proof of Stake Sustainability
    "BC-0498": "Blockchain Sustainability",  # Carbon Footprint Measurement
    "BC-0499": "Blockchain Sustainability",  # Green Blockchain Initiatives
    "BC-0501": "Blockchain Sustainability",  # ESG Reporting
    "BC-0502": "Decentralized Autonomous Organization",  # Climate Action DAO
    "BC-0503": "Consensus Mechanism",  # Sustainable Consensus
    "BC-0504": "Blockchain Sustainability",  # Environmental Impact Assessment
    "BC-0505": "Blockchain Sustainability",  # Carbon Neutral Blockchain
}


def get_term_id(filename):
    """Extract term ID from filename like BC-0001-blockchain.md"""
    match = re.match(r'(BC-\d+)', filename)
    return match.group(1) if match else None


def add_relationship_section(content, parent):
    """Add is-subclass-of relationship to content."""

    # Check if file already has Relationships section with is-subclass-of
    if 'is-subclass-of::' in content:
        return None  # Already has parent

    # Pattern 1: File has CrossDomainBridges section
    if '#### CrossDomainBridges' in content:
        # Insert Relationships section before CrossDomainBridges
        pattern = r'(  - #### CrossDomainBridges)'
        replacement = f'''  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[{parent}]]

\\1'''
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            return new_content

    # Pattern 2: File has OWL Restrictions section
    if '#### OWL Restrictions' in content:
        # Insert after OWL Restrictions
        pattern = r'(  - #### OWL Restrictions\n(?:    [^\n]*\n)*)'
        def add_relationships(m):
            return m.group(1) + f'''
  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[{parent}]]
'''
        new_content = re.sub(pattern, add_relationships, content)
        if new_content != content:
            return new_content

    # Pattern 3: File has Semantic Classification section
    if '**Semantic Classification**' in content:
        # Find end of Semantic Classification and add after
        pattern = r'(  - \*\*Semantic Classification\*\*\n(?:    - [^\n]+\n)*)'
        def add_after_semantic(m):
            return m.group(1) + f'''
  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[{parent}]]
'''
        new_content = re.sub(pattern, add_after_semantic, content)
        if new_content != content:
            return new_content

    # Pattern 4: Simple file with just OntologyBlock header
    if '### OntologyBlock' in content:
        # Find a good insertion point - after belongsToDomain if present
        if 'belongsToDomain::' in content:
            lines = content.split('\n')
            new_lines = []
            inserted = False
            for i, line in enumerate(lines):
                new_lines.append(line)
                if not inserted and 'belongsToDomain::' in line:
                    # Insert after this line
                    new_lines.append('')
                    new_lines.append('  - #### Relationships')
                    new_lines.append('    id:: relationships')
                    new_lines.append(f'    - is-subclass-of:: [[{parent}]]')
                    inserted = True
            if inserted:
                return '\n'.join(new_lines)

        # Insert after OntologyBlock section markers
        if 'owl:inferred-class::' in content:
            pattern = r'(    - owl:inferred-class:: [^\n]+)'
            replacement = f'''\\1

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[{parent}]]'''
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                return new_content

    # Pattern 5: Very simple file - add after term-id
    if 'term-id::' in content:
        pattern = r'(    - term-id:: [^\n]+)'
        replacement = f'''\\1

  - #### Relationships
    id:: relationships
    - is-subclass-of:: [[{parent}]]'''
        new_content = re.sub(pattern, replacement, content, count=1)
        if new_content != content:
            return new_content

    return None


def process_files(pages_dir: Path, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """Process all BC orphan files and add parent relationships."""
    # Parse ontology blocks
    parser = OntologyBlockParser()
    print(f"Parsing ontology blocks from {pages_dir}...")
    blocks = parser.parse_directory(pages_dir)
    print(f"Parsed {len(blocks)} blocks")

    # Filter to BC-domain blocks that need parents
    bc_blocks = [
        b for b in blocks
        if b.term_id and b.term_id in PARENT_MAPPING
    ]
    print(f"Found {len(bc_blocks)} BC blocks with mappings\n")

    processed = 0
    skipped = 0
    errors = []

    for block in bc_blocks:
        term_id = block.term_id
        if not term_id:
            continue

        parent = PARENT_MAPPING[term_id]

        # Read file content
        try:
            with open(block.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            errors.append(f"{block.file_path.name} (read error)")
            print(f"ERROR reading {block.file_path.name}: {e}")
            continue

        # Skip if already has is-subclass-of
        if block.is_subclass_of:
            skipped += 1
            continue

        # Add relationship
        new_content = add_relationship_section(content, parent)

        if new_content:
            if not dry_run:
                # Write updated content
                with open(block.file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            processed += 1
            print(f"{'Would update' if dry_run else 'Updated'}: {block.file_path.name} -> parent: {parent}")
        else:
            errors.append(block.file_path.name)
            print(f"ERROR: Could not update {block.file_path.name}")

    print(f"\nSummary:")
    print(f"  {'Would process' if dry_run else 'Processed'}: {processed}")
    print(f"  Skipped (already has parent): {skipped}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print(f"\nFiles with errors:")
        for e in errors:
            print(f"  - {e}")

    return processed, skipped, errors


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Add semantic parent relationships to BC domain orphan concepts'
    )
    parser.add_argument(
        '--pages-dir',
        type=Path,
        default=DEFAULT_PAGES_DIR,
        help='Directory containing ontology markdown files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - No files will be modified\n")

    # Validate path
    if not args.pages_dir.exists():
        print(f"Error: Pages directory not found: {args.pages_dir}")
        sys.exit(1)

    process_files(args.pages_dir, dry_run=args.dry_run)
