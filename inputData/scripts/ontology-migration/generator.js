#!/usr/bin/env node

/**
 * Block Generator - Ontology Block Migration Pipeline
 *
 * Generates new ontology blocks matching canonical format.
 * Preserves essential data, fixes issues, applies transformations.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const domainDetector = require('./domain-detector');
const domainConfig = require('./domain-config.json');
const iriRegistry = require('./iri-registry');

class OntologyGenerator {
  constructor() {
    this.currentDate = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
  }

  /**
   * Generate canonical ontology block from parsed data
   */
  generate(parsedData) {
    const props = parsedData.properties;
    const content = parsedData.content || '';
    const domain = this.detectDomain(parsedData, content);
    const namespace = this.fixNamespace(parsedData.namespace, domain);
    const className = this.fixClassName(parsedData.namespace?.className, props['preferred-term']);
    const domainConf = domainDetector.getDomainConfig(domain);

    // Generate and register IRI
    const conceptName = className || this.extractTitleFromFilename(parsedData.filename);
    const iri = iriRegistry.generateIRI(domain, conceptName);
    const termId = props['term-id'] || this.generateTermId(domain, parsedData.filename);

    try {
      iriRegistry.register(iri, domain, parsedData.filePath, conceptName, termId);
    } catch (error) {
      console.warn(`⚠️  IRI registration warning for ${parsedData.filename}: ${error.message}`);
    }

    const block = [];
    block.push('- ### OntologyBlock');
    block.push(`  id:: ${this.generateId(props, parsedData.filename)}`);
    block.push('  collapsed:: true');
    block.push('');

    // Identification Section
    block.push('  - **Identification**');
    block.push('    - ontology:: true');
    block.push(`    - term-id:: ${termId}`);
    block.push(`    - preferred-term:: ${props['preferred-term'] || this.extractTitleFromFilename(parsedData.filename)}`);

    if (props['alt-terms']) {
      block.push(`    - alt-terms:: ${props['alt-terms']}`);
    }

    block.push(`    - source-domain:: ${domain}`);
    block.push(`    - iri:: ${iri}`);
    block.push(`    - status:: ${this.normalizeStatus(props['status'])}`);

    // Handle public:: true property
    const publicAccess = this.determinePublicAccess(parsedData, props);
    block.push(`    - public-access:: ${publicAccess}`);

    block.push(`    - version:: ${props['version'] || '1.0.0'}`);
    block.push(`    - last-updated:: ${this.currentDate}`);

    if (props['quality-score']) {
      block.push(`    - quality-score:: ${props['quality-score']}`);
    }
    if (props['cross-domain-links']) {
      block.push(`    - cross-domain-links:: ${props['cross-domain-links']}`);
    }

    block.push('');

    // Definition Section
    block.push('  - **Definition**');
    block.push(`    - definition:: ${this.cleanDefinition(props['definition'])}`);
    block.push(`    - maturity:: ${this.normalizeMaturity(props['maturity'])}`);

    if (props['source']) {
      block.push(`    - source:: ${props['source']}`);
    }
    if (props['authority-score']) {
      block.push(`    - authority-score:: ${props['authority-score']}`);
    }

    block.push('');

    // Domain-Specific Extension Properties
    if (domainConf) {
      this.addDomainExtensionProperties(block, props, domainConf, content);
    }

    block.push('');

    // Semantic Classification Section
    block.push('  - **Semantic Classification**');
    block.push(`    - owl:class:: ${namespace}:${className}`);
    block.push(`    - owl:physicality:: ${props['owl:physicality'] || 'ConceptualEntity'}`);
    block.push(`    - owl:role:: ${props['owl:role'] || 'Concept'}`);

    if (props['owl:inferred-class']) {
      const inferredClass = this.fixNamespace(
        { prefix: namespace, className: props['owl:inferred-class'].split(':')[1] },
        domain
      );
      block.push(`    - owl:inferred-class:: ${inferredClass}:${this.fixClassName(props['owl:inferred-class'].split(':')[1])}`);
    }

    if (props['belongsToDomain']) {
      block.push(`    - belongsToDomain:: ${props['belongsToDomain']}`);
    }
    if (props['implementedInLayer']) {
      block.push(`    - implementedInLayer:: ${props['implementedInLayer']}`);
    }

    block.push('');

    // OWL Restrictions (only if present)
    if (parsedData.relationships.requires.length > 0 || props['owl-restrictions']) {
      block.push('  - #### OWL Restrictions');
      parsedData.relationships.requires.forEach(req => {
        block.push(`    - requires some ${req}`);
      });
      block.push('');
    }

    // Relationships Section
    block.push('  - #### Relationships');
    block.push(`    id:: ${this.generateId(props, parsedData.filename)}-relationships`);

    if (parsedData.relationships.isSubclassOf.length > 0) {
      block.push(`    - is-subclass-of:: ${parsedData.relationships.isSubclassOf.map(r => `[[${r}]]`).join(', ')}`);
    }
    if (parsedData.relationships.hasPart.length > 0) {
      block.push(`    - has-part:: ${parsedData.relationships.hasPart.map(r => `[[${r}]]`).join(', ')}`);
    }
    if (parsedData.relationships.requires.length > 0) {
      block.push(`    - requires:: ${parsedData.relationships.requires.map(r => `[[${r}]]`).join(', ')}`);
    }
    if (parsedData.relationships.enables.length > 0) {
      block.push(`    - enables:: ${parsedData.relationships.enables.map(r => `[[${r}]]`).join(', ')}`);
    }

    block.push('');

    // CrossDomainBridges (only if present)
    if (parsedData.relationships.bridgesTo.length > 0 || parsedData.relationships.bridgesFrom.length > 0) {
      block.push('  - #### CrossDomainBridges');
      parsedData.relationships.bridgesTo.forEach(bridge => {
        block.push(`    - bridges-to:: ${bridge}`);
      });
      parsedData.relationships.bridgesFrom.forEach(bridge => {
        block.push(`    - bridges-from:: ${bridge}`);
      });
      block.push('');
    }

    // OWL Axioms (preserve if present)
    if (parsedData.owlAxioms) {
      block.push('  - #### OWL Axioms');
      block.push(`    id:: ${this.generateId(props, parsedData.filename)}-owl-axioms`);
      block.push('    collapsed:: true');
      block.push('    - ```clojure');
      block.push(parsedData.owlAxioms);
      block.push('      ```');
      block.push('');
    }

    return block.join('\n');
  }

  /**
   * Detect domain from parsed data (now supports all 6 domains)
   */
  detectDomain(parsedData, content = '') {
    const props = parsedData.properties;

    // Priority 1: Check source-domain property
    if (props['source-domain'] && domainDetector.isValidDomain(props['source-domain'])) {
      return props['source-domain'];
    }

    // Priority 2: Use domain detector with content
    const filePath = parsedData.filePath || parsedData.filename;
    const detected = domainDetector.detect(filePath, content || JSON.stringify(props));

    return detected;
  }

  /**
   * Add domain-specific extension properties
   */
  addDomainExtensionProperties(block, props, domainConf, content) {
    block.push('  - **Domain Extensions**');

    // Add required domain properties
    for (const reqProp of domainConf.requiredProperties) {
      if (props[reqProp]) {
        block.push(`    - ${reqProp}:: ${props[reqProp]}`);
      } else {
        // Add placeholder for required property
        block.push(`    - ${reqProp}:: [TO BE DEFINED]`);
      }
    }

    // Add optional domain properties (only if present)
    for (const optProp of domainConf.optionalProperties) {
      if (props[optProp]) {
        block.push(`    - ${optProp}:: ${props[optProp]}`);
      }
    }

    // Detect and add sub-domain if applicable
    const subDomain = domainDetector.classifySubDomain(domainConf.namespace.replace(':', ''), content);
    if (subDomain) {
      block.push(`    - sub-domain:: ${subDomain}`);
    }
  }

  /**
   * Fix namespace (supports all 6 domains)
   */
  fixNamespace(namespace, domain) {
    if (!namespace) {
      return this.domainToNamespace(domain);
    }

    // Get expected namespace from domain
    const expectedNamespace = domainDetector.getNamespace(domain);
    if (expectedNamespace) {
      const expected = expectedNamespace.replace(':', '');

      // Fix namespace mismatch (e.g., robotics files using mv:)
      if (namespace.prefix !== expected) {
        return expected;
      }
    }

    return namespace.prefix;
  }

  /**
   * Convert domain to namespace prefix (supports all 6 domains)
   */
  domainToNamespace(domain) {
    const namespace = domainDetector.getNamespace(domain);
    return namespace ? namespace.replace(':', '') : 'mv';
  }

  /**
   * Fix class name to CamelCase
   */
  fixClassName(className, preferredTerm) {
    if (!className && preferredTerm) {
      return this.toCamelCase(preferredTerm);
    }

    if (!className) return 'UnknownClass';

    // If already in CamelCase, return as-is
    if (className[0] === className[0].toUpperCase() && !className.includes('-')) {
      return className;
    }

    // Convert to CamelCase
    return this.toCamelCase(className);
  }

  /**
   * Convert string to CamelCase (supports all 6 domain prefixes)
   */
  toCamelCase(str) {
    // Remove any namespace prefix (all 6 domains)
    str = str.replace(/^(ai|bc|rb|mv|tc|dt):/i, '');

    // Handle numbers followed by lowercase (e.g., rb0010aerialrobot)
    str = str.replace(/(\d)([a-z])/g, '$1 $2');

    // Split on spaces, hyphens, underscores
    return str
      .split(/[-_\s]+/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join('');
  }

  /**
   * Generate block ID from properties
   */
  generateId(props, filename) {
    if (props['id']) return props['id'];

    const term = props['preferred-term'] || filename.replace('.md', '');
    return term.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '') + '-ontology';
  }

  /**
   * Extract title from filename (supports all 6 domain prefixes)
   */
  extractTitleFromFilename(filename) {
    return filename
      .replace('.md', '')
      .replace(/^(AI-|BC-|RB-|MV-|TC-|DT-|ai-|bc-|rb-|mv-|tc-|dt-)\d+-/i, '')
      .replace(/-/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  /**
   * Clean definition text
   */
  cleanDefinition(definition) {
    if (!definition) return 'Definition to be added.';

    // Remove embedded markdown headings
    definition = definition.replace(/^###\s+/gm, '');

    // Clean up whitespace
    definition = definition.trim();

    return definition;
  }

  /**
   * Normalize status value
   */
  normalizeStatus(status) {
    if (!status) return 'draft';

    const normalized = status.toLowerCase();
    const validStatuses = ['draft', 'in-progress', 'complete', 'deprecated'];

    if (validStatuses.includes(normalized)) {
      return normalized;
    }

    // Map common variations
    if (normalized === 'approved') return 'complete';
    if (normalized === 'active') return 'complete';

    return 'draft';
  }

  /**
   * Normalize maturity value
   */
  normalizeMaturity(maturity) {
    if (!maturity) return 'draft';

    const normalized = maturity.toLowerCase();
    const validMaturities = ['draft', 'emerging', 'mature', 'established'];

    if (validMaturities.includes(normalized)) {
      return normalized;
    }

    return 'draft';
  }

  /**
   * Determine public access value
   * If file has public:: true property, preserve it in ontology block
   */
  determinePublicAccess(parsedData, props) {
    // Check if file has top-level public:: true
    if (parsedData.hasPublicProperty || parsedData.topLevelProperties?.public === 'true') {
      return 'true';
    }

    // Otherwise use existing value or default to true
    return props['public-access'] || 'true';
  }

  /**
   * Generate term-id from domain and filename
   */
  generateTermId(domain, filename) {
    const prefix = domainDetector.getPrefix(domain) || 'MV-';

    // Extract number from filename if present
    const numberMatch = filename.match(/\d{4}/);
    const number = numberMatch ? numberMatch[0] : '0001';

    return `${prefix}${number}`;
  }

  /**
   * Generate complete file content with new block at top
   */
  generateFullFile(parsedData, newBlock) {
    const content = parsedData.content || '';
    const lines = content.split('\n');
    const topProperties = [];
    const remainingContent = [];

    let inTopSection = true;
    let pastOntologyBlocks = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Preserve top-level property lines (but not public:: true)
      if (inTopSection && line.match(/^(title::|tags::|alias::|icon::)/)) {
        topProperties.push(line);
        continue;
      }

      // Skip public:: true at top level (it's now in ontology block)
      if (inTopSection && line.match(/^public::\s*true/i)) {
        continue;
      }

      // Skip all ontology blocks (we're replacing them)
      if (line.includes('### OntologyBlock')) {
        pastOntologyBlocks = true;
        inTopSection = false;

        // Skip until end of this ontology block
        while (i < lines.length) {
          i++;
          if (i >= lines.length) break;

          const nextLine = lines[i];
          // End of block detection
          if (nextLine.match(/^##[^#]/) || nextLine.match(/^---/) ||
              (nextLine.includes('### OntologyBlock'))) {
            i--; // Back up one line
            break;
          }
        }
        continue;
      }

      // Once we hit non-property, non-ontology content, add it
      if (line.trim() || pastOntologyBlocks) {
        inTopSection = false;
        remainingContent.push(line);
      }
    }

    // Construct final file:
    // 1. Top properties (title, tags, etc.)
    // 2. New ontology block
    // 3. Remaining content
    const result = [];

    if (topProperties.length > 0) {
      result.push(...topProperties);
      result.push(''); // Blank line after properties
    }

    result.push(newBlock);

    if (remainingContent.length > 0) {
      result.push('');
      result.push(...remainingContent);
    }

    return result.join('\n');
  }
}

// Run generator if executed directly
if (require.main === module) {
  const OntologyParser = require('./parser');
  const parser = new OntologyParser();
  const generator = new OntologyGenerator();

  const testFile = process.argv[2];

  if (!testFile) {
    console.error('Usage: node generator.js <file-path>');
    process.exit(1);
  }

  const parsed = parser.parseFile(testFile);
  const generated = generator.generate(parsed);

  console.log('Generated Ontology Block:');
  console.log('='.repeat(80));
  console.log(generated);
  console.log('='.repeat(80));
}

module.exports = OntologyGenerator;
