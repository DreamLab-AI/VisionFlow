#!/usr/bin/env node

/**
 * IRI Registry - Ontology Block Migration Pipeline
 *
 * Manages IRI generation, registration, and validation.
 * Ensures uniqueness and proper formatting of IRIs.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const domainDetector = require('./domain-detector');

class IRIRegistry {
  constructor() {
    this.registryPath = config.iriRegistryPath;
    this.baseUrl = config.baseIriUrl;
    this.registry = this.loadRegistry();
  }

  /**
   * Load existing IRI registry
   */
  loadRegistry() {
    if (fs.existsSync(this.registryPath)) {
      try {
        return JSON.parse(fs.readFileSync(this.registryPath, 'utf-8'));
      } catch (error) {
        console.warn('⚠️  Failed to load IRI registry, creating new one');
        return this.createEmptyRegistry();
      }
    }
    return this.createEmptyRegistry();
  }

  /**
   * Create empty registry structure
   */
  createEmptyRegistry() {
    return {
      metadata: {
        version: '1.0.0',
        created: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        totalIRIs: 0
      },
      domains: {
        ai: {},
        mv: {},
        tc: {},
        rb: {},
        dt: {},
        bc: {}
      },
      index: {} // IRI -> file mapping
    };
  }

  /**
   * Generate IRI from domain and concept name
   * Format: http://ontology.logseq.io/{domain}#{ConceptName}
   */
  generateIRI(domain, conceptName) {
    // Clean and format concept name
    const cleanName = this.cleanConceptName(conceptName);

    // Generate IRI
    const iri = `${this.baseUrl}/${domain}#${cleanName}`;

    return iri;
  }

  /**
   * Clean concept name to CamelCase format
   */
  cleanConceptName(name) {
    // Remove any existing prefix
    name = name.replace(/^(ai|bc|rb|mv|tc|dt)[-:]?\d*/i, '');

    // Convert to CamelCase
    return name
      .split(/[-_\s]+/)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join('');
  }

  /**
   * Register IRI for a file
   */
  register(iri, domain, filePath, conceptName, termId) {
    // Check uniqueness
    if (this.registry.index[iri]) {
      const existing = this.registry.index[iri];
      if (existing.filePath !== filePath) {
        throw new Error(`IRI collision: ${iri} already registered for ${existing.filePath}`);
      }
      // Same file, update registration
      return { registered: true, updated: true, iri };
    }

    // Register IRI
    const registration = {
      iri,
      filePath,
      conceptName,
      termId,
      domain,
      registeredAt: new Date().toISOString()
    };

    // Add to domain index
    if (!this.registry.domains[domain]) {
      this.registry.domains[domain] = {};
    }
    this.registry.domains[domain][conceptName] = registration;

    // Add to main index
    this.registry.index[iri] = registration;

    // Update metadata
    this.registry.metadata.totalIRIs = Object.keys(this.registry.index).length;
    this.registry.metadata.lastUpdated = new Date().toISOString();

    return { registered: true, updated: false, iri };
  }

  /**
   * Validate IRI format
   */
  validateIRI(iri) {
    const validation = {
      valid: true,
      errors: []
    };

    // Check format
    const iriPattern = new RegExp(`^${this.baseUrl}/[a-z]{2}#[A-Z][a-zA-Z0-9]*$`);
    if (!iriPattern.test(iri)) {
      validation.valid = false;
      validation.errors.push(`IRI does not match expected format: ${this.baseUrl}/{domain}#{ConceptName}`);
    }

    // Extract domain and check validity
    const match = iri.match(/\/([a-z]{2})#/);
    if (match) {
      const domain = match[1];
      if (!domainDetector.isValidDomain(domain)) {
        validation.valid = false;
        validation.errors.push(`Invalid domain in IRI: ${domain}`);
      }
    } else {
      validation.valid = false;
      validation.errors.push('Could not extract domain from IRI');
    }

    // Check concept name format (must start with uppercase)
    const conceptMatch = iri.match(/#([A-Z][a-zA-Z0-9]*)$/);
    if (!conceptMatch) {
      validation.valid = false;
      validation.errors.push('Concept name must start with uppercase letter');
    }

    return validation;
  }

  /**
   * Check if IRI exists
   */
  exists(iri) {
    return !!this.registry.index[iri];
  }

  /**
   * Get IRI for file
   */
  getByFile(filePath) {
    return Object.values(this.registry.index).find(reg => reg.filePath === filePath);
  }

  /**
   * Get all IRIs for domain
   */
  getByDomain(domain) {
    return this.registry.domains[domain] || {};
  }

  /**
   * Save registry to disk
   */
  save() {
    // Ensure directory exists
    const dir = path.dirname(this.registryPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Write registry
    fs.writeFileSync(this.registryPath, JSON.stringify(this.registry, null, 2));
  }

  /**
   * Generate statistics
   */
  getStatistics() {
    const stats = {
      total: this.registry.metadata.totalIRIs,
      byDomain: {}
    };

    for (const domain of Object.keys(this.registry.domains)) {
      stats.byDomain[domain] = Object.keys(this.registry.domains[domain]).length;
    }

    return stats;
  }

  /**
   * Find potential collisions
   */
  findCollisions() {
    const collisions = [];
    const conceptNames = {};

    for (const [iri, reg] of Object.entries(this.registry.index)) {
      const key = `${reg.domain}:${reg.conceptName}`;
      if (conceptNames[key]) {
        collisions.push({
          conceptName: reg.conceptName,
          domain: reg.domain,
          files: [conceptNames[key].filePath, reg.filePath],
          iris: [conceptNames[key].iri, iri]
        });
      } else {
        conceptNames[key] = reg;
      }
    }

    return collisions;
  }
}

// Export singleton instance
module.exports = new IRIRegistry();

// CLI usage
if (require.main === module) {
  const registry = new IRIRegistry();
  const command = process.argv[2];

  if (command === 'stats') {
    const stats = registry.getStatistics();
    console.log('IRI Registry Statistics:');
    console.log(`Total IRIs: ${stats.total}`);
    console.log('\nBy Domain:');
    Object.entries(stats.byDomain).forEach(([domain, count]) => {
      console.log(`  ${domain}: ${count}`);
    });
  } else if (command === 'collisions') {
    const collisions = registry.findCollisions();
    if (collisions.length === 0) {
      console.log('✅ No collisions found');
    } else {
      console.log(`⚠️  Found ${collisions.length} collisions:`);
      collisions.forEach(col => {
        console.log(`\n${col.domain}:${col.conceptName}`);
        col.files.forEach((file, i) => {
          console.log(`  - ${file}`);
          console.log(`    IRI: ${col.iris[i]}`);
        });
      });
    }
  } else if (command === 'validate') {
    const iri = process.argv[3];
    if (!iri) {
      console.error('Usage: node iri-registry.js validate <iri>');
      process.exit(1);
    }
    const validation = registry.validateIRI(iri);
    if (validation.valid) {
      console.log('✅ Valid IRI');
    } else {
      console.log('❌ Invalid IRI:');
      validation.errors.forEach(err => console.log(`  - ${err}`));
    }
  } else {
    console.log('Usage:');
    console.log('  node iri-registry.js stats');
    console.log('  node iri-registry.js collisions');
    console.log('  node iri-registry.js validate <iri>');
  }
}
