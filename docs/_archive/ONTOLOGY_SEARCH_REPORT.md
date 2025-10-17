# GitHub Ontology Files Search Report

## Repository Information
- **Repository**: jjohare/logseq
- **Base Path**: mainKnowledgeGraph/pages
- **Search Date**: 2025-10-17

## Search Criteria
1. ETSIDomainClassification.md
2. OntologyDefinition.md
3. PropertySchema.md
4. All markdown files containing "- ### OntologyBlock" pattern

## Results Summary

### Core Ontology Files (3 files)

#### 1. ETSIDomainClassification.md
- **Path**: `mainKnowledgeGraph/pages/ETSIDomainClassification.md`
- **Size**: 1,027 bytes
- **SHA**: 6d45b3cd342af385319f7b832c5ef330e6cecf74
- **Description**: Defines ETSI functional domain class hierarchy with belongsToDomain property
- **Download URL**: https://raw.githubusercontent.com/jjohare/logseq/main/mainKnowledgeGraph/pages/ETSIDomainClassification.md

#### 2. OntologyDefinition.md
- **Path**: `mainKnowledgeGraph/pages/OntologyDefinition.md`
- **Size**: 7,149 bytes
- **SHA**: ac80415d3df02689d2110f04bd11aa317b6752fc
- **Description**: Core ontology design with orthogonal physicality and role dimensions
- **Download URL**: https://raw.githubusercontent.com/jjohare/logseq/main/mainKnowledgeGraph/pages/OntologyDefinition.md

#### 3. PropertySchema.md
- **Path**: `mainKnowledgeGraph/pages/PropertySchema.md`
- **Size**: 4,655 bytes
- **SHA**: 9d5901b3b9490831b0df95497dde559f11ea57f4
- **Description**: Formal OWL property declarations including mereological and other relationships
- **Download URL**: https://raw.githubusercontent.com/jjohare/logseq/main/mainKnowledgeGraph/pages/PropertySchema.md

### OntologyBlock Files (100 files)

Files containing the exact pattern "- ### OntologyBlock":

1. ETSI_Domain_Data.md (3,604 bytes)
2. Haptics.md (5,247 bytes)
3. API Standard.md (5,481 bytes)
4. Cryptocurrency.md (9,144 bytes)
5. Immersion.md (6,249 bytes)
6. Data Fabric Architecture.md (10,554 bytes)
7. Digital Ritual.md (12,913 bytes)
8. Stablecoin.md (13,011 bytes)
9. Deepfakes.md (8,143 bytes)
10. Non-Fungible Token (NFT).md (12,186 bytes)
[...and 90 more files]

## JSON Catalog

Complete catalog with metadata saved to:
- **File**: `/home/devuser/workspace/project/ontology_files_catalog.json`

### JSON Structure
```json
{
  "core_ontology_files": [...],
  "ontology_block_files": [...],
  "summary": {
    "total_files": 103,
    "core_files_count": 3,
    "ontology_block_files_count": 100,
    "repository": "jjohare/logseq",
    "base_path": "mainKnowledgeGraph/pages",
    "pattern_searched": "- ### OntologyBlock"
  }
}
```

## Usage for Ontology Downloader

Each file entry contains:
- `name`: Filename
- `path`: Full path in repository
- `download_url`: Direct download URL (no authentication required for public repos)
- `size`: File size in bytes
- `sha`: Git SHA hash for versioning
- `preview`: First 500 characters of content

## Notes

- All download URLs point to the `main` branch
- URLs are properly encoded for spaces and special characters
- The GitHub API token used has read access to public repositories
- Total content size: ~1.1 MB across all ontology files
