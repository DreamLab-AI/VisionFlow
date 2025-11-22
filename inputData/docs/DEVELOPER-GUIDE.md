# Developer Guide - Extending the Ontology Tooling

**Version**: 1.0.0
**Last Updated**: 2025-11-21
**Purpose**: Guide for developers adding new tools and extending existing ones

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Architecture Overview](#architecture-overview)
3. [Using Shared Libraries](#using-shared-libraries)
4. [Adding a New Converter](#adding-a-new-converter)
5. [Adding a New Validator](#adding-a-new-validator)
6. [Adding a New Generator](#adding-a-new-generator)
7. [Extending the Migration Pipeline](#extending-the-migration-pipeline)
8. [Extending WasmVOWL](#extending-wasmvowl)
9. [Testing Requirements](#testing-requirements)
10. [Code Standards](#code-standards)
11. [Documentation Standards](#documentation-standards)
12. [Contributing](#contributing)

---

## Getting Started

### Prerequisites

#### Python Development
```bash
# Python 3.8+
python3 --version

# Install development dependencies
pip install -r requirements-dev.txt

# Or install manually
pip install rdflib owlrl pyshacl pytest black mypy
```

#### JavaScript/Node.js Development
```bash
# Node.js v18+
node --version

# No external dependencies for migration tools
```

#### Rust/WASM Development
```bash
# Rust (latest stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Verify installation
rustc --version
wasm-pack --version
```

### Repository Setup

```bash
# Clone repository
git clone https://github.com/DreamLab-AI/knowledgeGraph.git
cd knowledgeGraph

# Create development branch
git checkout -b feature/my-new-tool

# Set up Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

---

## Architecture Overview

### Tool Organization

```
Ontology-Tools/
├── tools/
│   ├── lib/                    # Shared libraries (CORE)
│   │   ├── ontology_block_parser.py
│   │   └── ontology_loader.py
│   ├── converters/             # Format converters
│   │   ├── convert-to-*.py
│   │   └── generate_*.py
│   ├── validators/             # Validation tools
│   └── generators/             # Content generators

scripts/
├── ontology-migration/         # Batch processing pipeline
│   ├── cli.js
│   ├── scanner.js
│   ├── parser.js
│   ├── generator.js
│   └── updater.js
└── *.py                        # Utility scripts

publishing-tools/
└── WasmVOWL/
    ├── rust-wasm/              # WASM physics engine
    └── modern/                 # React frontend
```

### Design Principles

1. **Shared Libraries First**: Always use `ontology_loader` and `ontology_block_parser`
2. **Fail Gracefully**: Handle errors, don't crash
3. **Progress Reporting**: Show progress for long operations
4. **Consistent CLI**: Use argparse/commander with standard options
5. **Comprehensive Logging**: Log at INFO, WARNING, ERROR levels
6. **Type Safety**: Use type hints (Python), TypeScript, strict Rust

---

## Using Shared Libraries

### Python: Ontology Loader

**Always use `OntologyLoader` for loading markdown files:**

```python
#!/usr/bin/env python3
"""
My New Converter Tool
"""

from pathlib import Path
from ontology_loader import OntologyLoader
import argparse
import logging

def convert_to_my_format(blocks, output_path):
    """Convert ontology blocks to my format"""
    with open(output_path, 'w') as f:
        for block in blocks:
            # Access properties
            term_id = block.term_id
            label = block.preferred_term
            definition = block.definition
            iri = block.get_full_iri()
            domain = block.get_domain()

            # Write to your format
            f.write(f"{term_id}: {label}\n")
            f.write(f"  IRI: {iri}\n")
            f.write(f"  Domain: {domain}\n")
            f.write(f"  Definition: {definition}\n\n")

def main():
    parser = argparse.ArgumentParser(
        description='Convert ontology to my format'
    )
    parser.add_argument('--input', required=True, help='Input directory')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--domain', help='Filter by domain')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    # Load ontology blocks
    loader = OntologyLoader(cache_size=128)
    blocks = loader.load_directory(
        Path(args.input),
        domain=args.domain,
        progress=True  # Show progress bar
    )

    logging.info(f"Loaded {len(blocks)} ontology blocks")

    # Convert
    convert_to_my_format(blocks, args.output)

    # Get statistics
    stats = loader.get_statistics(blocks)
    logging.info(f"Statistics: {stats.to_dict()}")

if __name__ == '__main__':
    main()
```

### OntologyBlock Properties

```python
# Tier 1 (Required)
block.ontology              # bool
block.term_id               # str
block.preferred_term        # str
block.source_domain         # str
block.status                # str
block.public_access         # bool
block.last_updated          # str
block.definition            # str

# OWL Classification
block.owl_class_uri         # str
block.subclass_of           # List[str]
block.equivalent_class      # List[str]
block.disjoint_with         # List[str]

# Tier 2 (Detailed)
block.alternative_terms     # List[str]
block.dc_subject            # List[str]
block.maturity_level        # str
block.see_also              # List[str]
block.version_info          # str

# Tier 3 (Extended)
block.extension_properties  # Dict[str, Any]
block.use_cases             # List[str]
block.references            # List[str]
block.examples              # List[str]

# Helper methods
block.get_full_iri()        # Returns full IRI
block.get_domain()          # Returns domain code (ai, bc, rb, mv, tc, dt)
block.get_namespace()       # Returns namespace URI
block.validate()            # Returns validation results
```

### JavaScript: Parser

**For migration tools, use the JavaScript parser:**

```javascript
const parser = require('./parser');

// Parse file
const result = parser.parseFile('/path/to/file.md');

if (result.success) {
  console.log('Term ID:', result.block.termId);
  console.log('Label:', result.block.preferredTerm);
  console.log('Domain:', result.block.sourceDomain);
} else {
  console.error('Parse errors:', result.errors);
}
```

---

## Adding a New Converter

### Step 1: Create Converter File

```bash
cd /home/user/logseq/Ontology-Tools/tools/converters
touch convert-to-myformat.py
chmod +x convert-to-myformat.py
```

### Step 2: Implement Converter

```python
#!/usr/bin/env python3
"""
Convert Ontology to MyFormat

Usage:
    python convert-to-myformat.py --input DIR --output FILE
"""

from pathlib import Path
from ontology_loader import OntologyLoader
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MyFormatConverter:
    """Converts ontology blocks to MyFormat"""

    def __init__(self):
        self.loader = OntologyLoader(cache_size=128)

    def convert(self, input_path: Path, output_path: Path, domain: str = None):
        """
        Convert ontology files to MyFormat

        Args:
            input_path: Directory containing markdown files
            output_path: Output file path
            domain: Optional domain filter (ai, bc, rb, mv, tc, dt)
        """
        logger.info(f"Loading ontology from {input_path}")

        # Load blocks
        blocks = self.loader.load_directory(
            input_path,
            domain=domain,
            progress=True
        )

        if not blocks:
            logger.error("No ontology blocks found")
            return False

        logger.info(f"Converting {len(blocks)} blocks to MyFormat")

        # Convert to your format
        output_data = self._convert_blocks(blocks)

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_data)

        logger.info(f"Wrote output to {output_path}")

        # Print statistics
        stats = self.loader.get_statistics(blocks)
        logger.info(f"Conversion complete: {stats.to_dict()}")

        return True

    def _convert_blocks(self, blocks):
        """Convert blocks to MyFormat string"""
        lines = []
        lines.append("# MyFormat Ontology Export\n")
        lines.append(f"# Total Concepts: {len(blocks)}\n\n")

        for block in blocks:
            # Build your format
            lines.append(f"CONCEPT {block.term_id}\n")
            lines.append(f"  LABEL: {block.preferred_term}\n")
            lines.append(f"  IRI: {block.get_full_iri()}\n")
            lines.append(f"  DOMAIN: {block.get_domain()}\n")
            lines.append(f"  DEFINITION: {block.definition}\n")

            # Alternative terms
            if block.alternative_terms:
                lines.append(f"  SYNONYMS: {', '.join(block.alternative_terms)}\n")

            # Parent classes
            if block.subclass_of:
                for parent in block.subclass_of:
                    lines.append(f"  PARENT: {parent}\n")

            # Related concepts
            if block.see_also:
                for related in block.see_also:
                    lines.append(f"  RELATED: {related}\n")

            lines.append("\n")

        return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert ontology to MyFormat',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert all domains
  %(prog)s --input mainKnowledgeGraph/pages/ --output ontology.myformat

  # Convert AI domain only
  %(prog)s --input mainKnowledgeGraph/pages/ --output ai.myformat --domain ai
        '''
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory containing markdown files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output file path'
    )

    parser.add_argument(
        '--domain',
        choices=['ai', 'bc', 'rb', 'mv', 'tc', 'dt'],
        help='Filter by domain (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    # Convert
    converter = MyFormatConverter()
    success = converter.convert(args.input, args.output, args.domain)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

### Step 3: Test Your Converter

```bash
# Test on sample data
python convert-to-myformat.py \
  --input ../../mainKnowledgeGraph/pages/ \
  --output test-outputs/test.myformat \
  --verbose

# Verify output
cat test-outputs/test.myformat | head -50
```

### Step 4: Add Tests

```python
# tests/test_myformat_converter.py
import pytest
from pathlib import Path
from converters.convert_to_myformat import MyFormatConverter

def test_myformat_conversion():
    """Test MyFormat converter"""
    converter = MyFormatConverter()

    # Use test fixtures
    input_path = Path('tests/fixtures/sample-ontology/')
    output_path = Path('tests/outputs/test.myformat')

    success = converter.convert(input_path, output_path)

    assert success
    assert output_path.exists()

    # Verify content
    content = output_path.read_text()
    assert 'CONCEPT AI-0001' in content
    assert 'LABEL:' in content
    assert 'IRI:' in content
```

### Step 5: Document Your Converter

Add entry to `/home/user/logseq/Ontology-Tools/tools/README.md`:

```markdown
### MyFormat Converter (`convert-to-myformat.py`)
- **Purpose**: Export to MyFormat specification
- **Input**: Directory of markdown files
- **Output**: `.myformat` file
- **Usage**:
  ```bash
  python tools/converters/convert-to-myformat.py \
    --input mainKnowledgeGraph/pages/ \
    --output ontology.myformat
  ```
```

### Step 6: Update Tooling Overview

Add entry to `/home/user/logseq/docs/TOOLING-OVERVIEW.md` in the converters section.

---

## Adding a New Validator

### Validator Template

```python
#!/usr/bin/env python3
"""
My Custom Validator

Validates ontology for [specific criteria]
"""

from pathlib import Path
from ontology_loader import OntologyLoader
import logging

logger = logging.getLogger(__name__)


class MyValidator:
    """Validates ontology blocks for [criteria]"""

    def __init__(self):
        self.loader = OntologyLoader()
        self.errors = []
        self.warnings = []

    def validate(self, input_path: Path):
        """
        Validate ontology files

        Returns:
            dict: Validation results with score and issues
        """
        blocks = self.loader.load_directory(input_path, progress=True)

        for block in blocks:
            self._validate_block(block)

        # Calculate score
        total_checks = len(blocks) * 5  # 5 checks per block
        issues = len(self.errors) + len(self.warnings) * 0.5
        score = max(0, 100 - (issues / total_checks * 100))

        return {
            'score': round(score, 2),
            'blocks_validated': len(blocks),
            'errors': self.errors,
            'warnings': self.warnings,
            'passed': len(self.errors) == 0
        }

    def _validate_block(self, block):
        """Validate individual block"""
        # Check 1: Required fields
        if not block.term_id:
            self.errors.append({
                'file': str(block.file_path),
                'type': 'missing_term_id',
                'message': 'Term ID is required'
            })

        # Check 2: Definition length
        if block.definition and len(block.definition) < 20:
            self.warnings.append({
                'file': str(block.file_path),
                'type': 'short_definition',
                'message': 'Definition should be at least 20 characters'
            })

        # Check 3: Parent class exists
        if not block.subclass_of:
            self.warnings.append({
                'file': str(block.file_path),
                'type': 'missing_parent',
                'message': 'Consider adding parent class'
            })

        # Add more validation checks...


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate ontology')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    validator = MyValidator()
    results = validator.validate(args.input)

    # Print results
    print(f"Validation Score: {results['score']}/100")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error['file']}: {error['message']}")

    # Optionally save report
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
```

---

## Adding a New Generator

### Generator Template

```python
#!/usr/bin/env python3
"""
Generate [Something] from Ontology
"""

from pathlib import Path
from ontology_loader import OntologyLoader
import logging

logger = logging.getLogger(__name__)


class MyGenerator:
    """Generates [something] from ontology"""

    def __init__(self):
        self.loader = OntologyLoader()

    def generate(self, input_path: Path, output_path: Path):
        """Generate output from ontology"""
        # Load blocks
        blocks = self.loader.load_directory(input_path, progress=True)

        # Generate content
        content = self._generate_content(blocks)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding='utf-8')

        logger.info(f"Generated {output_path}")

    def _generate_content(self, blocks):
        """Generate content from blocks"""
        lines = []
        # Your generation logic here
        return '\n'.join(lines)
```

---

## Extending the Migration Pipeline

### Adding a New Transform

```javascript
// scripts/ontology-migration/transforms/my-transform.js

/**
 * My Custom Transform
 * Applies [specific transformation] to ontology blocks
 */

function applyTransform(block) {
  // Modify block in place
  if (block.someProperty) {
    block.someProperty = transformValue(block.someProperty);
  }

  return block;
}

function transformValue(value) {
  // Your transformation logic
  return value;
}

module.exports = {
  applyTransform
};
```

### Integrate Transform into Pipeline

```javascript
// scripts/ontology-migration/generator.js

const myTransform = require('./transforms/my-transform');

function generateCanonicalBlock(parsedBlock, options) {
  // Apply existing transforms
  let block = normalizeNamespaces(parsedBlock);
  block = normalizeCasing(block);

  // Apply your custom transform
  block = myTransform.applyTransform(block);

  // Continue with generation...
  return generateBlock(block);
}
```

---

## Extending WasmVOWL

### Adding a Rust Feature

```rust
// publishing-tools/WasmVOWL/rust-wasm/src/my_feature.rs

/// My new feature for ontology processing
pub struct MyFeature {
    // State
}

impl MyFeature {
    pub fn new() -> Self {
        MyFeature { /* ... */ }
    }

    pub fn process(&self, data: &str) -> Result<String, String> {
        // Your logic
        Ok(data.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_feature() {
        let feature = MyFeature::new();
        let result = feature.process("test");
        assert!(result.is_ok());
    }
}
```

### Export to JavaScript

```rust
// publishing-tools/WasmVOWL/rust-wasm/src/bindings/mod.rs

use wasm_bindgen::prelude::*;
use crate::my_feature::MyFeature;

#[wasm_bindgen]
impl WebVowl {
    /// My new JavaScript-callable method
    #[wasm_bindgen(js_name = myFeature)]
    pub fn my_feature(&self, input: String) -> Result<String, JsValue> {
        let feature = MyFeature::new();
        feature.process(&input)
            .map_err(|e| JsValue::from_str(&e))
    }
}
```

### Use in Frontend

```typescript
// publishing-tools/WasmVOWL/modern/src/hooks/useMyFeature.ts

import { useWasm } from './useWasm';

export function useMyFeature() {
  const { wasmInstance } = useWasm();

  const applyFeature = (input: string) => {
    if (!wasmInstance) return null;

    try {
      return wasmInstance.myFeature(input);
    } catch (error) {
      console.error('Feature error:', error);
      return null;
    }
  };

  return { applyFeature };
}
```

---

## Testing Requirements

### Python Testing

```bash
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=tools --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

### Test Structure

```
tests/
├── fixtures/                # Test data
│   └── sample-ontology/
│       ├── ai-0001.md
│       └── ai-0002.md
├── test_loader.py           # Loader tests
├── test_parser.py           # Parser tests
├── test_converters.py       # Converter tests
└── test_validators.py       # Validator tests
```

### JavaScript Testing

```bash
# Migration tools use manual testing
node cli.js test path/to/test-file.md
```

### Rust Testing

```bash
cd publishing-tools/WasmVOWL/rust-wasm

# Run all tests
cargo test

# With output
cargo test -- --nocapture

# Specific test
cargo test test_my_feature

# Benchmarks
cargo bench
```

---

## Code Standards

### Python Code Style

```python
# Follow PEP 8
# Use Black formatter
black tools/converters/my-converter.py

# Use type hints
def convert_data(input: Path, output: Path) -> bool:
    """Convert data from input to output"""
    pass

# Use docstrings
class MyConverter:
    """
    My converter description

    Attributes:
        loader: OntologyLoader instance
        options: Dict of options

    Example:
        >>> converter = MyConverter()
        >>> converter.convert(input_path, output_path)
    """
    pass
```

### JavaScript Code Style

```javascript
// Use JSDoc comments
/**
 * Parse ontology file
 * @param {string} filePath - Path to file
 * @returns {Object} Parsed block
 */
function parseFile(filePath) {
  // Use camelCase
  const fileContent = readFile(filePath);

  // Return objects explicitly
  return {
    success: true,
    block: parsedBlock
  };
}
```

### Rust Code Style

```rust
// Use rustfmt
cargo fmt

// Use clippy
cargo clippy -- -D warnings

// Document public APIs
/// Process ontology data
///
/// # Arguments
/// * `data` - Input data string
///
/// # Returns
/// Processed data or error
///
/// # Example
/// ```
/// let result = process_data("input");
/// assert!(result.is_ok());
/// ```
pub fn process_data(data: &str) -> Result<String, String> {
    // Implementation
}
```

---

## Documentation Standards

### Tool Documentation Template

Every tool should have:

1. **Purpose**: What it does
2. **Inputs**: What it accepts
3. **Outputs**: What it generates
4. **Dependencies**: What it requires
5. **Usage**: CLI examples
6. **Examples**: Real-world usage

### README Structure

```markdown
# Tool Name

Brief description

## Features
- Feature 1
- Feature 2

## Installation
```bash
# Installation commands
```

## Usage
```bash
# Usage examples
```

## Options
- `--input`: Description
- `--output`: Description

## Examples
### Example 1
```bash
# Command
```

Expected output...

## Troubleshooting
### Issue 1
Solution...

## Related Tools
- Tool A
- Tool B
```

---

## Contributing

### Contribution Workflow

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/knowledgeGraph.git
   cd knowledgeGraph
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/my-awesome-tool
   ```

3. **Develop**
   - Write code
   - Write tests
   - Write documentation

4. **Test**
   ```bash
   # Python
   pytest tests/

   # Rust
   cargo test

   # Integration
   python my-tool.py --input test-data/ --output test-output/
   ```

5. **Format Code**
   ```bash
   # Python
   black tools/

   # Rust
   cargo fmt
   ```

6. **Commit**
   ```bash
   git add .
   git commit -m "feat: add awesome new converter tool

   - Converts ontology to XYZ format
   - Includes validation and error handling
   - Comprehensive test coverage"
   ```

7. **Push and PR**
   ```bash
   git push origin feature/my-awesome-tool
   # Create PR on GitHub
   ```

### Commit Message Format

```
type(scope): brief description

Detailed explanation (optional)

- Change 1
- Change 2

Fixes #123
```

**Types**: feat, fix, docs, refactor, test, chore

---

## Related Documentation

- **Tooling Overview**: `/home/user/logseq/docs/TOOLING-OVERVIEW.md`
- **Workflows**: `/home/user/logseq/docs/TOOL-WORKFLOWS.md`
- **API Reference**: `/home/user/logseq/docs/API-REFERENCE.md`
- **User Guide**: `/home/user/logseq/docs/USER-GUIDE.md`

---

**Maintainer**: Claude Code Agent
**Last Updated**: 2025-11-21
**Version**: 1.0.0
