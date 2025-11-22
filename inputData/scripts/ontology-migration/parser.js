#!/usr/bin/env node

/**
 * Block Parser - Ontology Block Migration Pipeline
 *
 * Parses existing ontology blocks, extracts properties,
 * identifies namespace usage, and detects issues.
 */

const fs = require('fs');

class OntologyParser {
  constructor() {
    this.ontologyBlockRegex = /- ### OntologyBlock\s*\n([\s\S]*?)(?=\n##[^#]|\n---|\Z)/;
  }

  /**
   * Parse ontology block from file content
   */
  parseFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf-8');

    // Extract all ontology blocks
    const allBlocks = this.extractAllOntologyBlocks(content);
    const hasMultipleBlocks = allBlocks.length > 1;

    // Select the best block (most complete)
    const bestBlock = this.selectBestBlock(allBlocks);

    return {
      filePath,
      filename: require('path').basename(filePath),
      hasOntologyBlock: this.hasOntologyBlock(content),
      ontologyBlock: bestBlock || this.extractOntologyBlock(content),
      allBlocks: allBlocks,
      hasMultipleBlocks: hasMultipleBlocks,
      properties: this.extractProperties(bestBlock || content),
      relationships: this.extractRelationships(bestBlock || content),
      owlAxioms: this.extractOwlAxioms(bestBlock || content),
      namespace: this.detectNamespace(bestBlock || content),
      hasPublicProperty: /public::\s*true/i.test(content),
      topLevelProperties: this.extractTopLevelProperties(content),
      issues: this.detectIssues(filePath, content),
      contentBelowBlock: this.extractContentBelowBlock(content),
      content: content
    };
  }

  /**
   * Check if file has ontology block
   */
  hasOntologyBlock(content) {
    return /- ### OntologyBlock/.test(content) || /##OntologyBlock/.test(content);
  }

  /**
   * Extract entire ontology block
   */
  extractOntologyBlock(content) {
    const match = content.match(this.ontologyBlockRegex);
    return match ? match[0] : null;
  }

  /**
   * Extract ALL ontology blocks from content
   */
  extractAllOntologyBlocks(content) {
    const blocks = [];
    const blockRegex = /- ### OntologyBlock\s*\n([\s\S]*?)(?=\n- ### OntologyBlock|\n##[^#]|\n---|\Z)/g;
    let match;

    while ((match = blockRegex.exec(content)) !== null) {
      blocks.push(match[0]);
    }

    return blocks;
  }

  /**
   * Select the best block from multiple blocks
   * Criteria: most properties, most complete, newest
   */
  selectBestBlock(blocks) {
    if (blocks.length === 0) return null;
    if (blocks.length === 1) return blocks[0];

    let bestBlock = blocks[0];
    let bestScore = this.scoreBlock(blocks[0]);

    for (let i = 1; i < blocks.length; i++) {
      const score = this.scoreBlock(blocks[i]);
      if (score > bestScore) {
        bestScore = score;
        bestBlock = blocks[i];
      }
    }

    return bestBlock;
  }

  /**
   * Score a block based on completeness
   */
  scoreBlock(block) {
    let score = 0;

    // Count properties
    const propertyMatches = block.match(/^\s*-?\s*([a-zA-Z0-9_:-]+)::/gm);
    score += propertyMatches ? propertyMatches.length : 0;

    // Bonus for required fields
    const requiredFields = ['term-id', 'preferred-term', 'definition', 'owl:class'];
    requiredFields.forEach(field => {
      if (block.includes(`${field}::`)) score += 5;
    });

    // Bonus for sections
    if (block.includes('**Identification**')) score += 3;
    if (block.includes('**Definition**')) score += 3;
    if (block.includes('**Semantic Classification**')) score += 3;
    if (block.includes('#### Relationships')) score += 2;

    // Bonus for last-updated (prefer newer)
    const dateMatch = block.match(/last-updated::\s*(\d{4}-\d{2}-\d{2})/);
    if (dateMatch) {
      const date = new Date(dateMatch[1]);
      const now = new Date();
      const daysDiff = (now - date) / (1000 * 60 * 60 * 24);
      // Less penalty for newer dates
      score += Math.max(0, 10 - Math.floor(daysDiff / 30));
    }

    return score;
  }

  /**
   * Extract top-level properties (before ontology block)
   */
  extractTopLevelProperties(content) {
    const props = {};
    const lines = content.split('\n');

    for (const line of lines) {
      // Stop at first ontology block
      if (line.includes('### OntologyBlock')) break;

      // Match property:: value
      const match = line.match(/^([a-zA-Z0-9_-]+)::\s*(.+)$/);
      if (match) {
        props[match[1]] = match[2].trim();
      }
    }

    return props;
  }

  /**
   * Extract all properties from ontology block
   */
  extractProperties(content) {
    const properties = {};

    // Match all property:: value patterns
    const propertyRegex = /^\s*-?\s*([a-zA-Z0-9_:-]+)::\s*(.+?)$/gm;
    let match;

    while ((match = propertyRegex.exec(content)) !== null) {
      const key = match[1].trim();
      const value = match[2].trim();
      properties[key] = value;
    }

    return properties;
  }

  /**
   * Extract relationships section
   */
  extractRelationships(content) {
    const relationships = {
      isSubclassOf: [],
      hasPart: [],
      isPartOf: [],
      requires: [],
      dependsOn: [],
      enables: [],
      relatesTo: [],
      bridgesTo: [],
      bridgesFrom: []
    };

    // Extract is-subclass-of
    const subclassMatches = content.matchAll(/is-subclass-of::\s*(.+?)(?:\n|$)/gi);
    for (const match of subclassMatches) {
      relationships.isSubclassOf.push(...this.parseLinks(match[1]));
    }

    // Extract has-part
    const hasPartMatches = content.matchAll(/has-part::\s*(.+?)(?:\n|$)/gi);
    for (const match of hasPartMatches) {
      relationships.hasPart.push(...this.parseLinks(match[1]));
    }

    // Extract requires
    const requiresMatches = content.matchAll(/requires::\s*(.+?)(?:\n|$)/gi);
    for (const match of requiresMatches) {
      relationships.requires.push(...this.parseLinks(match[1]));
    }

    // Extract enables
    const enablesMatches = content.matchAll(/enables::\s*(.+?)(?:\n|$)/gi);
    for (const match of enablesMatches) {
      relationships.enables.push(...this.parseLinks(match[1]));
    }

    // Extract bridges-to
    const bridgesToMatches = content.matchAll(/bridges-to::\s*(.+?)(?:\n|$)/gi);
    for (const match of bridgesToMatches) {
      relationships.bridgesTo.push(match[1].trim());
    }

    return relationships;
  }

  /**
   * Parse wiki links from property value
   */
  parseLinks(text) {
    const links = [];
    const linkRegex = /\[\[([^\]]+)\]\]/g;
    let match;

    while ((match = linkRegex.exec(text)) !== null) {
      links.push(match[1]);
    }

    return links;
  }

  /**
   * Extract OWL Axioms section
   */
  extractOwlAxioms(content) {
    const axiomMatch = content.match(/#### OWL Axioms[\s\S]*?```clojure\s*([\s\S]*?)```/);
    return axiomMatch ? axiomMatch[1].trim() : null;
  }

  /**
   * Detect namespace from owl:class
   */
  detectNamespace(content) {
    const classMatch = content.match(/owl:class::\s*(\w+):(\w+)/);
    if (classMatch) {
      return {
        prefix: classMatch[1],
        className: classMatch[2]
      };
    }
    return null;
  }

  /**
   * Detect issues in ontology block
   */
  detectIssues(filePath, content) {
    const issues = [];
    const filename = require('path').basename(filePath);

    // Namespace issues
    if (filename.match(/^rb-/i)) {
      if (content.includes('mv:rb')) {
        issues.push({
          type: 'namespace-error',
          severity: 'critical',
          description: 'Robotics file using mv: namespace instead of rb:',
          fix: 'Replace mv: with rb:'
        });
      }
    }

    // Class naming issues
    const classMatch = content.match(/owl:class::\s*(\w+):(\w+)/);
    if (classMatch && classMatch[2]) {
      const className = classMatch[2];
      if (className[0] === className[0].toLowerCase()) {
        issues.push({
          type: 'naming-convention',
          severity: 'high',
          description: `Class name not in CamelCase: ${className}`,
          fix: `Convert to CamelCase: ${this.toCamelCase(className)}`
        });
      }
    }

    // Indentation issues (tabs vs spaces)
    if (content.match(/\t- ontology::/)) {
      issues.push({
        type: 'formatting',
        severity: 'low',
        description: 'Using tabs instead of spaces for indentation',
        fix: 'Convert tabs to 2 spaces'
      });
    }

    // Duplicate sections
    if (content.includes('## Technical Details') && content.includes('**Identification**')) {
      issues.push({
        type: 'duplicate-section',
        severity: 'medium',
        description: 'Duplicate Technical Details section found',
        fix: 'Remove duplicate Technical Details section'
      });
    }

    // Empty sections
    const emptyOWLRestrictions = content.match(/#### OWL Restrictions\s*\n\s*\n\s*-\s*####/);
    if (emptyOWLRestrictions) {
      issues.push({
        type: 'empty-section',
        severity: 'low',
        description: 'Empty OWL Restrictions section',
        fix: 'Remove empty section'
      });
    }

    return issues;
  }

  /**
   * Convert string to CamelCase
   */
  toCamelCase(str) {
    return str
      .replace(/[-_\s]+(.)?/g, (_, c) => c ? c.toUpperCase() : '')
      .replace(/^[a-z]/, c => c.toUpperCase());
  }

  /**
   * Extract content below ontology block
   */
  extractContentBelowBlock(content) {
    const blockMatch = content.match(/- ### OntologyBlock[\s\S]*?(?=\n##[^#]|\Z)/);
    if (!blockMatch) return content;

    const blockEnd = blockMatch.index + blockMatch[0].length;
    return content.substring(blockEnd);
  }

  /**
   * Parse batch of files
   */
  parseBatch(filePaths) {
    return filePaths.map(filePath => {
      try {
        return this.parseFile(filePath);
      } catch (error) {
        console.error(`Error parsing ${filePath}:`, error.message);
        return null;
      }
    }).filter(result => result !== null);
  }
}

// Run parser if executed directly
if (require.main === module) {
  const parser = new OntologyParser();
  const testFile = process.argv[2];

  if (!testFile) {
    console.error('Usage: node parser.js <file-path>');
    process.exit(1);
  }

  const result = parser.parseFile(testFile);
  console.log(JSON.stringify(result, null, 2));
}

module.exports = OntologyParser;
