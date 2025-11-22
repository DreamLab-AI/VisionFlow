#!/usr/bin/env node

/**
 * File Scanner - Ontology Block Migration Pipeline
 *
 * Scans all markdown files in mainKnowledgeGraph/pages directory,
 * identifies files with ontology blocks, classifies by pattern,
 * and generates inventory report.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const domainDetector = require('./domain-detector');

class OntologyScanner {
  constructor() {
    this.results = {
      totalFiles: 0,
      filesWithOntology: 0,
      filesWithoutOntology: 0,
      patternDistribution: {
        pattern1: 0,
        pattern2: 0,
        pattern3: 0,
        pattern4: 0,
        pattern5: 0,
        pattern6: 0,
        unknown: 0
      },
      domainDistribution: {
        ai: 0,
        mv: 0,
        tc: 0,
        rb: 0,
        dt: 0,
        bc: 0,
        unknown: 0
      },
      issues: {
        namespaceErrors: [],
        namingIssues: [],
        duplicateSections: [],
        missingFields: [],
        multipleBlocks: [],
        publicTrueFiles: [],
        blockNotAtTop: []
      },
      fileInventory: []
    };
  }

  /**
   * Scan all markdown files in source directory
   */
  async scan() {
    console.log('🔍 Starting ontology block scan...');
    console.log(`📂 Source directory: ${config.sourceDirectory}`);

    const files = this.getAllMarkdownFiles(config.sourceDirectory);
    this.results.totalFiles = files.length;

    console.log(`📊 Found ${files.length} markdown files`);

    for (const file of files) {
      await this.scanFile(file);
    }

    return this.results;
  }

  /**
   * Recursively get all markdown files
   */
  getAllMarkdownFiles(dir) {
    const files = [];

    const items = fs.readdirSync(dir, { withFileTypes: true });

    for (const item of items) {
      const fullPath = path.join(dir, item.name);

      // Skip .deleted directory
      if (item.name === '.deleted') continue;

      if (item.isDirectory()) {
        files.push(...this.getAllMarkdownFiles(fullPath));
      } else if (item.isFile() && item.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }

    return files;
  }

  /**
   * Scan individual file for ontology block
   */
  async scanFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const hasOntology = content.includes('### OntologyBlock') || content.includes('##OntologyBlock');

      if (!hasOntology) {
        this.results.filesWithoutOntology++;
        return;
      }

      this.results.filesWithOntology++;

      const domain = this.detectDomain(filePath, content);
      const subDomain = domainDetector.classifySubDomain(domain, content);

      const fileInfo = {
        path: filePath,
        filename: path.basename(filePath),
        pattern: this.detectPattern(content),
        domain: domain,
        subDomain: subDomain,
        metadata: this.extractMetadata(content),
        blockCount: this.countOntologyBlocks(content),
        hasPublicTrue: this.hasPublicProperty(content),
        blockAtTop: this.isBlockAtTop(content),
        issues: this.detectIssues(filePath, content, domain)
      };

      this.results.fileInventory.push(fileInfo);
      this.updateStatistics(fileInfo);

      if (config.verboseLogging) {
        console.log(`✓ ${fileInfo.filename} - ${fileInfo.pattern} - ${fileInfo.domain}`);
      }

    } catch (error) {
      console.error(`❌ Error scanning ${filePath}:`, error.message);
    }
  }

  /**
   * Detect ontology block pattern
   */
  detectPattern(content) {
    // Pattern 1: Comprehensive Structured Format
    if (content.includes('**Identification**') &&
        content.includes('**Definition**') &&
        content.includes('**Semantic Classification**')) {
      return 'pattern1';
    }

    // Pattern 2: Blockchain with OWL Axioms
    if (content.includes('#### OWL Axioms') &&
        content.includes('Prefix(:=<http://')) {
      return 'pattern2';
    }

    // Pattern 3: Robotics Simplified
    if (content.match(/\t- ontology::/)) {
      return 'pattern3';
    }

    // Pattern 4: Logseq Native Minimal
    if (!content.includes('**Identification**') &&
        content.includes('- ### OntologyBlock') &&
        content.includes('## Technical Details')) {
      return 'pattern4';
    }

    // Pattern 5: Metaverse Flat
    if (!content.includes('**Identification**') &&
        content.match(/^\t- ontology::/m) &&
        content.includes('#### Relationships')) {
      return 'pattern5';
    }

    // Pattern 6: Extended Metadata
    if (content.includes('domain-prefix::') &&
        content.includes('sequence-number::')) {
      return 'pattern6';
    }

    return 'unknown';
  }

  /**
   * Count ontology blocks in file
   */
  countOntologyBlocks(content) {
    const blockMatches = content.match(/- ### OntologyBlock/g);
    return blockMatches ? blockMatches.length : 0;
  }

  /**
   * Check if file has public:: true property
   */
  hasPublicProperty(content) {
    return /public::\s*true/i.test(content);
  }

  /**
   * Check if ontology block is at top of file
   */
  isBlockAtTop(content) {
    const lines = content.split('\n');
    let firstContentLine = 0;

    // Skip empty lines and property lines
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line && !line.match(/^(public::|title::|tags::|alias::|icon::)/)) {
        firstContentLine = i;
        break;
      }
    }

    // Check if ontology block starts within first 5 lines after properties
    const topSection = lines.slice(firstContentLine, firstContentLine + 5).join('\n');
    return topSection.includes('### OntologyBlock');
  }

  /**
   * Detect domain from filename and content (supports all 6 domains)
   */
  detectDomain(filePath, content) {
    return domainDetector.detect(filePath, content);
  }

  /**
   * Extract metadata from ontology block
   */
  extractMetadata(content) {
    const metadata = {};

    const fields = [
      'term-id', 'preferred-term', 'source-domain', 'status', 'maturity',
      'version', 'last-updated', 'authority-score', 'quality-score',
      'public-access', 'owl:class', 'owl:physicality', 'owl:role'
    ];

    for (const field of fields) {
      const regex = new RegExp(`${field.replace(':', '\\:')}::\\s*(.+?)(?:\\n|$)`, 'i');
      const match = content.match(regex);
      if (match) {
        metadata[field] = match[1].trim();
      }
    }

    return metadata;
  }

  /**
   * Detect issues in ontology block (multi-ontology aware)
   */
  detectIssues(filePath, content, domain) {
    const issues = [];
    const filename = path.basename(filePath);

    // Check namespace mismatch for all domains
    const expectedNamespace = domainDetector.getNamespace(domain);
    if (expectedNamespace) {
      const expected = expectedNamespace.replace(':', '');
      const classMatch = content.match(/owl:class::\s*(\w+):(\w+)/);
      if (classMatch && classMatch[1] !== expected) {
        issues.push(`namespace-error-${classMatch[1]}-should-be-${expected}`);
        this.results.issues.namespaceErrors.push(filePath);
      }
    }

    // Check for lowercase class names
    const classMatch = content.match(/owl:class::\s*(\w+):(\w+)/);
    if (classMatch && classMatch[2] && classMatch[2][0] === classMatch[2][0].toLowerCase()) {
      issues.push('class-naming-lowercase');
      this.results.issues.namingIssues.push(filePath);
    }

    // Check for duplicate Technical Details section
    if (content.includes('## Technical Details') &&
        content.includes('**Identification**')) {
      issues.push('duplicate-technical-details');
      this.results.issues.duplicateSections.push(filePath);
    }

    // Check for missing core required fields
    const requiredFields = ['term-id', 'preferred-term', 'definition', 'owl:class'];
    const missingFields = requiredFields.filter(field => {
      const regex = new RegExp(`${field.replace(':', '\\:')}::`, 'i');
      return !regex.test(content);
    });

    // Check for missing domain-specific required properties
    const domainConf = domainDetector.getDomainConfig(domain);
    if (domainConf) {
      for (const reqProp of domainConf.requiredProperties) {
        const regex = new RegExp(`${reqProp.replace(':', '\\:')}::`, 'i');
        if (!regex.test(content)) {
          missingFields.push(reqProp);
        }
      }
    }

    if (missingFields.length > 0) {
      issues.push(`missing-fields-${missingFields.join(',')}`);
      this.results.issues.missingFields.push({ file: filePath, missing: missingFields });
    }

    // Detect cross-domain links
    const crossDomainLinks = domainDetector.detectCrossDomainLinks(content, domain);
    if (crossDomainLinks.length > 0) {
      issues.push(`cross-domain-links-${crossDomainLinks.length}`);
    }

    // Check for multiple ontology blocks
    const blockCount = this.countOntologyBlocks(content);
    if (blockCount > 1) {
      issues.push(`multiple-ontology-blocks-${blockCount}`);
      this.results.issues.multipleBlocks.push(filePath);
    }

    // Check for public:: true
    if (this.hasPublicProperty(content) && !content.includes('### OntologyBlock')) {
      issues.push('has-public-true-no-ontology');
      this.results.issues.publicTrueFiles.push(filePath);
    }

    // Check if block is not at top
    if (!this.isBlockAtTop(content)) {
      issues.push('block-not-at-top');
      this.results.issues.blockNotAtTop.push(filePath);
    }

    return issues;
  }

  /**
   * Update statistics based on file info
   */
  updateStatistics(fileInfo) {
    this.results.patternDistribution[fileInfo.pattern]++;

    const domain = fileInfo.domain || 'unknown';
    if (this.results.domainDistribution[domain] !== undefined) {
      this.results.domainDistribution[domain]++;
    } else {
      this.results.domainDistribution.unknown++;
    }
  }

  /**
   * Generate and save report
   */
  async generateReport() {
    const reportPath = path.join(config.reportsDirectory, 'file-inventory.json');

    console.log('\n📊 Scan Results:');
    console.log(`   Total files: ${this.results.totalFiles}`);
    console.log(`   Files with ontology: ${this.results.filesWithOntology}`);
    console.log(`   Files without ontology: ${this.results.filesWithoutOntology}`);

    console.log('\n📈 Pattern Distribution:');
    Object.entries(this.results.patternDistribution).forEach(([pattern, count]) => {
      if (count > 0) {
        console.log(`   ${pattern}: ${count} (${(count/this.results.filesWithOntology*100).toFixed(1)}%)`);
      }
    });

    console.log('\n🌍 Domain Distribution:');
    Object.entries(this.results.domainDistribution).forEach(([domain, count]) => {
      if (count > 0) {
        console.log(`   ${domain}: ${count} (${(count/this.results.filesWithOntology*100).toFixed(1)}%)`);
      }
    });

    console.log('\n⚠️  Issues Found:');
    console.log(`   Namespace errors: ${this.results.issues.namespaceErrors.length}`);
    console.log(`   Naming issues: ${this.results.issues.namingIssues.length}`);
    console.log(`   Duplicate sections: ${this.results.issues.duplicateSections.length}`);
    console.log(`   Missing fields: ${this.results.issues.missingFields.length}`);
    console.log(`   Multiple blocks: ${this.results.issues.multipleBlocks.length}`);
    console.log(`   Public:: true files: ${this.results.issues.publicTrueFiles.length}`);
    console.log(`   Block not at top: ${this.results.issues.blockNotAtTop.length}`);

    // Save report
    fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2));
    console.log(`\n💾 Report saved to: ${reportPath}`);

    return reportPath;
  }
}

// Run scanner if executed directly
if (require.main === module) {
  const scanner = new OntologyScanner();
  scanner.scan()
    .then(() => scanner.generateReport())
    .then(() => console.log('✅ Scan complete'))
    .catch(err => {
      console.error('❌ Scan failed:', err);
      process.exit(1);
    });
}

module.exports = OntologyScanner;
