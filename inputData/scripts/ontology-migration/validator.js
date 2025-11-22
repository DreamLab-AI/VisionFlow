#!/usr/bin/env node

/**
 * Validator - Ontology Block Migration Pipeline
 *
 * Validates ontology blocks against canonical format.
 * Checks required fields, OWL syntax, namespace correctness.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const OntologyParser = require('./parser');
const domainDetector = require('./domain-detector');
const domainConfig = require('./domain-config.json');

class OntologyValidator {
  constructor() {
    this.parser = new OntologyParser();
    this.results = {
      totalValidated: 0,
      passed: 0,
      failed: 0,
      warnings: 0,
      validationErrors: []
    };
  }

  /**
   * Validate a single file
   */
  validateFile(filePath) {
    const filename = path.basename(filePath);
    const validation = {
      file: filename,
      path: filePath,
      valid: true,
      errors: [],
      warnings: [],
      score: 0
    };

    try {
      const parsed = this.parser.parseFile(filePath);

      if (!parsed.hasOntologyBlock) {
        validation.warnings.push('No ontology block found');
        this.results.warnings++;
        return validation;
      }

      // Detect domain
      const domain = this.detectDomain(parsed);
      validation.domain = domain;

      // Validate required fields
      this.validateRequiredFields(parsed, validation);

      // Validate domain-specific properties
      this.validateDomainProperties(parsed, validation, domain);

      // Validate OWL properties
      this.validateOwlProperties(parsed, validation, domain);

      // Validate namespace
      this.validateNamespace(parsed, validation, domain);

      // Validate relationships
      this.validateRelationships(parsed, validation);

      // Validate format
      this.validateFormat(parsed, validation);

      // Calculate validation score (0-100)
      validation.score = this.calculateScore(validation);

      // Determine if valid
      validation.valid = validation.errors.length === 0;

      if (validation.valid) {
        this.results.passed++;
      } else {
        this.results.failed++;
      }

      this.results.totalValidated++;

    } catch (error) {
      validation.valid = false;
      validation.errors.push(`Validation error: ${error.message}`);
      this.results.failed++;
    }

    return validation;
  }

  /**
   * Validate required fields
   */
  validateRequiredFields(parsed, validation) {
    const props = parsed.properties;

    for (const field of config.requiredFields) {
      if (!props[field]) {
        validation.errors.push(`Missing required field: ${field}`);
      }
    }

    // Validate field values
    if (props['term-id']) {
      if (!props['term-id'].match(/^(AI-|BC-|RB-|)\d+$/)) {
        validation.warnings.push('term-id format may be non-standard');
      }
    }

    if (props['status']) {
      const validStatuses = ['draft', 'in-progress', 'complete', 'deprecated'];
      if (!validStatuses.includes(props['status'])) {
        validation.warnings.push(`Non-standard status value: ${props['status']}`);
      }
    }

    if (props['maturity']) {
      const validMaturities = ['draft', 'emerging', 'mature', 'established'];
      if (!validMaturities.includes(props['maturity'])) {
        validation.warnings.push(`Non-standard maturity value: ${props['maturity']}`);
      }
    }

    if (props['public-access']) {
      if (props['public-access'] !== 'true' && props['public-access'] !== 'false') {
        validation.errors.push('public-access must be true or false');
      }
    }
  }

  /**
   * Detect domain for validation
   */
  detectDomain(parsed) {
    const props = parsed.properties;
    const content = fs.readFileSync(parsed.filePath, 'utf-8');

    if (props['source-domain'] && domainDetector.isValidDomain(props['source-domain'])) {
      return props['source-domain'];
    }

    return domainDetector.detect(parsed.filePath, content);
  }

  /**
   * Validate domain-specific required properties
   */
  validateDomainProperties(parsed, validation, domain) {
    const props = parsed.properties;
    const domainConf = domainDetector.getDomainConfig(domain);

    if (!domainConf) {
      validation.warnings.push(`Unknown domain: ${domain}`);
      return;
    }

    // Check required domain properties
    for (const reqProp of domainConf.requiredProperties) {
      if (!props[reqProp]) {
        validation.errors.push(`Missing required domain property: ${reqProp} (${domain} domain)`);
      }
    }

    // Validate domain-specific property values if present
    for (const optProp of domainConf.optionalProperties) {
      if (props[optProp]) {
        // Property exists - could add value validation here if needed
      }
    }
  }

  /**
   * Validate OWL properties (domain-aware)
   */
  validateOwlProperties(parsed, validation, domain) {
    const props = parsed.properties;
    const domainConf = domainDetector.getDomainConfig(domain);

    // Validate owl:class
    if (props['owl:class']) {
      const classMatch = props['owl:class'].match(/^(\w+):(\w+)$/);
      if (!classMatch) {
        validation.errors.push('owl:class must be in format namespace:ClassName');
      } else {
        const [_, namespace, className] = classMatch;

        // Check namespace validity (all 6 domains)
        const validNamespaces = ['ai', 'bc', 'rb', 'mv', 'tc', 'dt'];
        if (!validNamespaces.includes(namespace)) {
          validation.warnings.push(`Non-standard namespace: ${namespace}`);
        }

        // Check if namespace matches domain
        if (domainConf) {
          const expectedNs = domainConf.namespace.replace(':', '');
          if (namespace !== expectedNs) {
            validation.errors.push(`Namespace ${namespace} does not match domain ${domain} (expected ${expectedNs})`);
          }
        }

        // Check CamelCase
        if (className[0] !== className[0].toUpperCase()) {
          validation.errors.push(`Class name must start with uppercase: ${className}`);
        }
      }
    }

    // Validate owl:physicality (domain-specific)
    if (props['owl:physicality']) {
      if (domainConf && domainConf.validPhysicalities) {
        if (!domainConf.validPhysicalities.includes(props['owl:physicality'])) {
          validation.warnings.push(`Physicality ${props['owl:physicality']} may not be appropriate for ${domain} domain`);
        }
      } else {
        // Fallback to universal validation
        const validPhysicalities = ['PhysicalEntity', 'VirtualEntity', 'ConceptualEntity', 'AbstractEntity', 'HybridEntity'];
        if (!validPhysicalities.includes(props['owl:physicality'])) {
          validation.warnings.push(`Non-standard physicality: ${props['owl:physicality']}`);
        }
      }
    }

    // Validate owl:role (domain-specific)
    if (props['owl:role']) {
      if (domainConf && domainConf.validRoles) {
        if (!domainConf.validRoles.includes(props['owl:role'])) {
          validation.warnings.push(`Role ${props['owl:role']} may not be appropriate for ${domain} domain`);
        }
      } else {
        // Fallback to universal validation
        const validRoles = ['Object', 'Process', 'Agent', 'Quality', 'Relation', 'Concept'];
        if (!validRoles.includes(props['owl:role'])) {
          validation.warnings.push(`Non-standard role: ${props['owl:role']}`);
        }
      }
    }
  }

  /**
   * Validate namespace correctness (multi-ontology aware)
   */
  validateNamespace(parsed, validation, domain) {
    const filename = path.basename(parsed.filePath);
    const props = parsed.properties;

    // Check if namespace matches domain for all 6 domains
    if (props['owl:class']) {
      const namespace = props['owl:class'].split(':')[0];
      const expectedNamespace = domainDetector.getNamespace(domain);

      if (expectedNamespace) {
        const expected = expectedNamespace.replace(':', '');
        if (namespace !== expected) {
          validation.errors.push(
            `Namespace mismatch: ${namespace}: should be ${expected}: for ${domain} domain`
          );
        }
      }
    }

    // Check filename prefix alignment with domain
    const detectedFromFilename = domainDetector.detectFromPath(parsed.filePath);
    if (detectedFromFilename !== domain) {
      validation.warnings.push(
        `Filename suggests ${detectedFromFilename} domain but source-domain is ${domain}`
      );
    }

    // Validate term-id prefix matches domain
    if (props['term-id']) {
      const detectedFromTermId = domainDetector.detectFromTermId(props['term-id']);
      if (detectedFromTermId && detectedFromTermId !== domain) {
        validation.errors.push(
          `term-id prefix suggests ${detectedFromTermId} domain but source-domain is ${domain}`
        );
      }
    }
  }

  /**
   * Validate relationships
   */
  validateRelationships(parsed, validation) {
    // Check if is-subclass-of exists
    if (parsed.relationships.isSubclassOf.length === 0) {
      validation.warnings.push('No is-subclass-of relationship defined');
    }

    // Validate link format
    const checkLinks = (links, propName) => {
      links.forEach(link => {
        if (!link.match(/^[A-Z]/)) {
          validation.warnings.push(`${propName} link should start with capital: ${link}`);
        }
      });
    };

    checkLinks(parsed.relationships.isSubclassOf, 'is-subclass-of');
    checkLinks(parsed.relationships.hasPart, 'has-part');
  }

  /**
   * Validate format
   */
  validateFormat(parsed, validation) {
    const content = fs.readFileSync(parsed.filePath, 'utf-8');

    // Check for tabs (should use spaces)
    if (content.match(/\t- ontology::/)) {
      validation.warnings.push('Using tabs instead of spaces for indentation');
    }

    // Check for collapsed state
    if (!content.includes('collapsed:: true')) {
      validation.warnings.push('OntologyBlock not marked as collapsed');
    }

    // Check for empty sections
    if (content.match(/#### OWL Restrictions\s*\n\s*\n\s*- ####/)) {
      validation.warnings.push('Empty OWL Restrictions section should be removed');
    }

    // Check for duplicate sections
    if (content.includes('## Technical Details') && content.includes('**Identification**')) {
      validation.warnings.push('Duplicate Technical Details section found');
    }
  }

  /**
   * Calculate validation score (0-100)
   */
  calculateScore(validation) {
    let score = 100;

    // Deduct for errors (10 points each, max 50 points)
    score -= Math.min(validation.errors.length * 10, 50);

    // Deduct for warnings (2 points each, max 20 points)
    score -= Math.min(validation.warnings.length * 2, 20);

    return Math.max(score, 0);
  }

  /**
   * Validate batch of files
   */
  validateBatch(filePaths) {
    console.log(`\n🔍 Validating ${filePaths.length} files...\n`);

    const validations = [];

    for (const filePath of filePaths) {
      const validation = this.validateFile(filePath);
      validations.push(validation);

      if (config.verboseLogging) {
        const status = validation.valid ? '✅' : '❌';
        const score = validation.score;
        console.log(`${status} ${validation.file} (score: ${score}/100)`);

        if (validation.errors.length > 0) {
          validation.errors.forEach(err => console.log(`   ❌ ${err}`));
        }
        if (validation.warnings.length > 0) {
          validation.warnings.forEach(warn => console.log(`   ⚠️  ${warn}`));
        }
      }
    }

    return validations;
  }

  /**
   * Generate validation report
   */
  generateReport(validations) {
    console.log('\n' + '='.repeat(80));
    console.log('📊 VALIDATION SUMMARY');
    console.log('='.repeat(80));
    console.log(`Total files validated: ${this.results.totalValidated}`);
    console.log(`Passed: ${this.results.passed} (${(this.results.passed/this.results.totalValidated*100).toFixed(1)}%)`);
    console.log(`Failed: ${this.results.failed} (${(this.results.failed/this.results.totalValidated*100).toFixed(1)}%)`);
    console.log(`Warnings: ${this.results.warnings}`);

    // Calculate average score
    const avgScore = validations.reduce((sum, v) => sum + v.score, 0) / validations.length;
    console.log(`Average validation score: ${avgScore.toFixed(1)}/100`);
    console.log('='.repeat(80));

    // Save report
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.results,
      averageScore: avgScore,
      validations: validations
    };

    const reportPath = path.join(config.reportsDirectory, 'validation-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n💾 Report saved to: ${reportPath}`);

    // List top issues
    const allErrors = {};
    const allWarnings = {};

    validations.forEach(v => {
      v.errors.forEach(err => {
        allErrors[err] = (allErrors[err] || 0) + 1;
      });
      v.warnings.forEach(warn => {
        allWarnings[warn] = (allWarnings[warn] || 0) + 1;
      });
    });

    console.log('\n📋 Top Errors:');
    Object.entries(allErrors)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .forEach(([err, count]) => {
        console.log(`   ${count}x ${err}`);
      });

    console.log('\n⚠️  Top Warnings:');
    Object.entries(allWarnings)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .forEach(([warn, count]) => {
        console.log(`   ${count}x ${warn}`);
      });
  }
}

// Run validator if executed directly
if (require.main === module) {
  const validator = new OntologyValidator();
  const filePath = process.argv[2];

  if (!filePath) {
    console.error('Usage: node validator.js <file-path>');
    process.exit(1);
  }

  const validation = validator.validateFile(filePath);
  console.log(JSON.stringify(validation, null, 2));
}

module.exports = OntologyValidator;
