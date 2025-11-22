#!/usr/bin/env node

/**
 * CLI - Ontology Block Migration Pipeline
 *
 * Command-line interface for the migration pipeline.
 * Provides easy access to all pipeline tools.
 */

const OntologyScanner = require('./scanner');
const OntologyParser = require('./parser');
const OntologyGenerator = require('./generator');
const OntologyUpdater = require('./updater');
const OntologyValidator = require('./validator');
const BatchProcessor = require('./batch-process');
const config = require('./config.json');
const fs = require('fs');
const path = require('path');

const domainDetector = require('./domain-detector');

const COMMANDS = {
  scan: 'Scan all files and generate inventory',
  preview: 'Preview transformations (dry-run mode)',
  process: 'Process all files (use --live for actual updates)',
  validate: 'Validate all ontology blocks',
  test: 'Test transformation on a single file',
  domain: 'Process specific domain only',
  pattern: 'Process specific pattern only',
  domains: 'List all 6 domains with statistics',
  'domain-stats': 'Show statistics for a specific domain',
  'validate-domain': 'Validate specific domain files',
  'process-domain': 'Process specific domain files',
  'cross-domain-links': 'Analyze cross-domain references',
  'detect-domain': 'Detect domain for a specific file',
  'audit-blocks': 'Find files with multiple ontology blocks',
  'audit-public': 'Find files with public:: true property',
  'audit-position': 'Find files with blocks not at top',
  'fix-blocks': 'Fix files with multiple blocks',
  'fix-position': 'Move blocks to top of files',
  'iri-stats': 'Show IRI registry statistics',
  stats: 'Show current statistics',
  help: 'Show this help message'
};

class CLI {
  constructor() {
    this.args = process.argv.slice(2);
    this.command = this.args[0];
    this.options = this.parseOptions();
  }

  parseOptions() {
    const options = {
      live: this.args.includes('--live'),
      dryRun: !this.args.includes('--live'),
      batch: parseInt(this.args.find(arg => arg.startsWith('--batch='))?.split('=')[1]) || config.batchSize,
      validate: this.args.includes('--validate'),
      verbose: this.args.includes('--verbose') || this.args.includes('-v')
    };

    if (options.verbose) {
      config.verboseLogging = true;
    }

    return options;
  }

  async run() {
    console.log(`
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              🔧 ONTOLOGY BLOCK MIGRATION PIPELINE - CLI                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
`);

    if (!this.command || this.command === 'help') {
      this.showHelp();
      return;
    }

    try {
      switch (this.command) {
        case 'scan':
          await this.runScan();
          break;

        case 'preview':
          await this.runPreview();
          break;

        case 'process':
          await this.runProcess();
          break;

        case 'validate':
          await this.runValidate();
          break;

        case 'audit-blocks':
          await this.auditMultipleBlocks();
          break;

        case 'audit-public':
          await this.auditPublicProperties();
          break;

        case 'audit-position':
          await this.auditBlockPosition();
          break;

        case 'fix-blocks':
          await this.fixMultipleBlocks();
          break;

        case 'fix-position':
          await this.fixBlockPosition();
          break;

        case 'iri-stats':
          await this.showIRIStats();
          break;

        case 'test':
          await this.runTest();
          break;

        case 'domain':
          await this.runDomain();
          break;

        case 'pattern':
          await this.runPattern();
          break;

        case 'stats':
          await this.showStats();
          break;

        case 'domains':
          await this.listDomains();
          break;

        case 'domain-stats':
          await this.showDomainStats();
          break;

        case 'validate-domain':
          await this.validateDomain();
          break;

        case 'process-domain':
          await this.processDomain();
          break;

        case 'cross-domain-links':
          await this.analyzeCrossDomainLinks();
          break;

        case 'detect-domain':
          await this.detectFileDomain();
          break;

        default:
          console.error(`❌ Unknown command: ${this.command}`);
          this.showHelp();
          process.exit(1);
      }
    } catch (error) {
      console.error(`\n❌ Error: ${error.message}`);
      if (this.options.verbose) {
        console.error(error.stack);
      }
      process.exit(1);
    }
  }

  async runScan() {
    console.log('📊 Scanning files...\n');
    const scanner = new OntologyScanner();
    await scanner.scan();
    await scanner.generateReport();
  }

  async runPreview() {
    const limit = parseInt(this.args[1]) || 10;
    console.log(`🔍 Previewing first ${limit} transformations...\n`);

    const scanner = new OntologyScanner();
    await scanner.scan();

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));

    const filesToPreview = inventory.fileInventory.slice(0, limit);

    const updater = new OntologyUpdater({ dryRun: true });

    for (const file of filesToPreview) {
      await updater.updateFile(file.path);
    }
  }

  async runProcess() {
    if (!this.options.live) {
      console.log('⚠️  DRY RUN MODE - No files will be modified');
      console.log('   Add --live flag to perform actual updates\n');
    } else {
      console.log('⚠️  LIVE UPDATE MODE - Files will be modified!');
      console.log('   Git will track all changes (use git to revert if needed)\n');

      // Confirmation prompt
      const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
      });

      const answer = await new Promise(resolve => {
        readline.question('Continue? (yes/no): ', resolve);
      });
      readline.close();

      if (answer.toLowerCase() !== 'yes') {
        console.log('❌ Aborted');
        return;
      }
    }

    const processor = new BatchProcessor({
      dryRun: this.options.dryRun,
      batchSize: this.options.batch,
      validateAfter: this.options.validate
    });

    await processor.run();
  }

  async runValidate() {
    console.log('🔍 Validating ontology blocks...\n');

    // Load inventory
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');

    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const filesToValidate = inventory.fileInventory.map(item => item.path);

    const validator = new OntologyValidator();
    const validations = validator.validateBatch(filesToValidate);
    validator.generateReport(validations);
  }

  async auditMultipleBlocks() {
    console.log('🔍 Auditing files with multiple ontology blocks...\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const multiBlockFiles = inventory.fileInventory.filter(f => f.blockCount > 1);

    console.log(`Found ${multiBlockFiles.length} files with multiple blocks:\n`);
    multiBlockFiles.forEach(f => {
      console.log(`  ${f.filename} - ${f.blockCount} blocks`);
    });
  }

  async auditPublicProperties() {
    console.log('🔍 Auditing files with public:: true property...\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const publicFiles = inventory.fileInventory.filter(f => f.hasPublicTrue);

    console.log(`Found ${publicFiles.length} files with public:: true:\n`);
    publicFiles.forEach(f => {
      console.log(`  ${f.filename}`);
    });
  }

  async auditBlockPosition() {
    console.log('🔍 Auditing files with blocks not at top...\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const mispositionedFiles = inventory.fileInventory.filter(f => !f.blockAtTop);

    console.log(`Found ${mispositionedFiles.length} files with blocks not at top:\n`);
    mispositionedFiles.forEach(f => {
      console.log(`  ${f.filename}`);
    });
  }

  async fixMultipleBlocks() {
    console.log('🔧 Fixing files with multiple ontology blocks...\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const multiBlockFiles = inventory.fileInventory
      .filter(f => f.blockCount > 1)
      .map(f => f.path);

    if (multiBlockFiles.length === 0) {
      console.log('✅ No files with multiple blocks found');
      return;
    }

    console.log(`Processing ${multiBlockFiles.length} files...\n`);

    const updater = new OntologyUpdater({ dryRun: this.options.dryRun });
    await updater.updateBatch(multiBlockFiles, this.options.batch);
    updater.generateReport();
  }

  async fixBlockPosition() {
    console.log('🔧 Moving blocks to top of files...\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const mispositionedFiles = inventory.fileInventory
      .filter(f => !f.blockAtTop)
      .map(f => f.path);

    if (mispositionedFiles.length === 0) {
      console.log('✅ All blocks are already at top');
      return;
    }

    console.log(`Processing ${mispositionedFiles.length} files...\n`);

    const updater = new OntologyUpdater({ dryRun: this.options.dryRun });
    await updater.updateBatch(mispositionedFiles, this.options.batch);
    updater.generateReport();
  }

  async showIRIStats() {
    const iriRegistry = require('./iri-registry');
    const stats = iriRegistry.getStatistics();

    console.log('\n📊 IRI Registry Statistics\n');
    console.log('='.repeat(80));
    console.log(`Total IRIs registered: ${stats.total}`);
    console.log('\nBy Domain:');
    Object.entries(stats.byDomain).forEach(([domain, count]) => {
      console.log(`  ${domain.padEnd(5)} ${count} IRIs`);
    });

    // Check for collisions
    const collisions = iriRegistry.findCollisions();
    if (collisions.length > 0) {
      console.log(`\n⚠️  Found ${collisions.length} potential collisions`);
    } else {
      console.log('\n✅ No IRI collisions detected');
    }
    console.log('='.repeat(80));
  }

  async runTest() {
    const filePath = this.args[1];

    if (!filePath) {
      console.error('❌ Please provide a file path');
      console.error('Usage: node cli.js test <file-path>');
      process.exit(1);
    }

    console.log(`🧪 Testing transformation on: ${filePath}\n`);

    // Parse
    console.log('1️⃣  Parsing file...');
    const parser = new OntologyParser();
    const parsed = parser.parseFile(filePath);
    console.log(`   ✅ Parsed successfully`);
    console.log(`   Pattern: ${parsed.issues.length > 0 ? parsed.issues[0].type : 'none'}`);
    console.log(`   Issues: ${parsed.issues.length}`);

    // Generate
    console.log('\n2️⃣  Generating canonical block...');
    const generator = new OntologyGenerator();
    const generated = generator.generate(parsed);
    console.log('   ✅ Generated successfully');

    // Display
    console.log('\n3️⃣  Preview of generated block:');
    console.log('   ' + '─'.repeat(70));
    console.log(generated.split('\n').map(line => '   ' + line).join('\n'));
    console.log('   ' + '─'.repeat(70));

    // Validate
    console.log('\n4️⃣  Validating...');
    const validator = new OntologyValidator();
    const validation = validator.validateFile(filePath);
    console.log(`   ${validation.valid ? '✅' : '❌'} Validation ${validation.valid ? 'passed' : 'failed'}`);
    console.log(`   Score: ${validation.score}/100`);

    if (validation.errors.length > 0) {
      console.log('\n   Errors:');
      validation.errors.forEach(err => console.log(`   ❌ ${err}`));
    }

    if (validation.warnings.length > 0) {
      console.log('\n   Warnings:');
      validation.warnings.forEach(warn => console.log(`   ⚠️  ${warn}`));
    }
  }

  async runDomain() {
    const domain = this.args[1];

    if (!domain) {
      console.error('❌ Please specify a domain');
      console.error('Usage: node cli.js domain <ai|mv|tc|rb|dt|bc>');
      process.exit(1);
    }

    if (!domainDetector.isValidDomain(domain)) {
      console.error(`❌ Invalid domain: ${domain}`);
      console.error('Valid domains: ai, mv, tc, rb, dt, bc');
      process.exit(1);
    }

    const processor = new BatchProcessor({
      dryRun: this.options.dryRun,
      batchSize: this.options.batch
    });

    await processor.processDomain(domain);
  }

  async listDomains() {
    console.log('📚 All 6 Domains in Multi-Ontology Architecture\n');
    console.log('='.repeat(80));

    const domains = domainDetector.getAllDomains();
    for (const domainKey of domains) {
      const domain = domainDetector.getDomainConfig(domainKey);
      console.log(`\n${domain.namespace.padEnd(5)} ${domain.name}`);
      console.log(`      Prefix: ${domain.prefix}`);
      console.log(`      Required Props: ${domain.requiredProperties.join(', ')}`);
      console.log(`      Extensions: ${domain.extensions.slice(0, 3).join(', ')}...`);
    }

    console.log('\n' + '='.repeat(80));

    // Show statistics if inventory exists
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (fs.existsSync(inventoryPath)) {
      const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
      console.log('\n📊 Current Distribution:\n');

      for (const domainKey of domains) {
        const count = inventory.domainDistribution[domainKey] || 0;
        const percentage = inventory.filesWithOntology > 0
          ? ((count / inventory.filesWithOntology) * 100).toFixed(1)
          : '0.0';
        console.log(`   ${domainKey.padEnd(5)} ${count.toString().padStart(4)} files (${percentage}%)`);
      }
    }
  }

  async showDomainStats() {
    const domain = this.args[1];

    if (!domain) {
      console.error('❌ Please specify a domain');
      console.error('Usage: node cli.js domain-stats <ai|mv|tc|rb|dt|bc>');
      process.exit(1);
    }

    if (!domainDetector.isValidDomain(domain)) {
      console.error(`❌ Invalid domain: ${domain}`);
      process.exit(1);
    }

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const domainFiles = inventory.fileInventory.filter(f => f.domain === domain);
    const domainConf = domainDetector.getDomainConfig(domain);

    console.log(`\n📊 Statistics for ${domainConf.name} (${domain}:)\n`);
    console.log('='.repeat(80));
    console.log(`Total files: ${domainFiles.length}`);
    console.log(`Namespace: ${domainConf.namespace}`);
    console.log(`Prefix: ${domainConf.prefix}`);

    // Sub-domain distribution
    const subDomains = {};
    domainFiles.forEach(f => {
      if (f.subDomain) {
        subDomains[f.subDomain] = (subDomains[f.subDomain] || 0) + 1;
      }
    });

    if (Object.keys(subDomains).length > 0) {
      console.log('\n📈 Sub-domain Distribution:');
      Object.entries(subDomains)
        .sort((a, b) => b[1] - a[1])
        .forEach(([sub, count]) => {
          console.log(`   ${sub.padEnd(20)} ${count} files`);
        });
    }

    // Pattern distribution
    const patterns = {};
    domainFiles.forEach(f => {
      patterns[f.pattern] = (patterns[f.pattern] || 0) + 1;
    });

    console.log('\n📋 Pattern Distribution:');
    Object.entries(patterns)
      .sort((a, b) => b[1] - a[1])
      .forEach(([pattern, count]) => {
        console.log(`   ${pattern.padEnd(15)} ${count} files`);
      });

    // Issues
    const issueFiles = domainFiles.filter(f => f.issues && f.issues.length > 0);
    console.log(`\n⚠️  Files with issues: ${issueFiles.length}`);

    if (issueFiles.length > 0 && this.options.verbose) {
      console.log('\nTop Issues:');
      const issueTypes = {};
      issueFiles.forEach(f => {
        f.issues.forEach(issue => {
          issueTypes[issue] = (issueTypes[issue] || 0) + 1;
        });
      });
      Object.entries(issueTypes)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .forEach(([issue, count]) => {
          console.log(`   ${count}x ${issue}`);
        });
    }

    console.log('='.repeat(80));
  }

  async validateDomain() {
    const domain = this.args[1];

    if (!domain) {
      console.error('❌ Please specify a domain');
      console.error('Usage: node cli.js validate-domain <ai|mv|tc|rb|dt|bc>');
      process.exit(1);
    }

    if (!domainDetector.isValidDomain(domain)) {
      console.error(`❌ Invalid domain: ${domain}`);
      process.exit(1);
    }

    console.log(`🔍 Validating ${domain}: domain files...\n`);

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const domainFiles = inventory.fileInventory
      .filter(f => f.domain === domain)
      .map(f => f.path);

    const validator = new OntologyValidator();
    const validations = validator.validateBatch(domainFiles);
    validator.generateReport(validations);
  }

  async processDomain() {
    const domain = this.args[1];

    if (!domain) {
      console.error('❌ Please specify a domain');
      console.error('Usage: node cli.js process-domain <ai|mv|tc|rb|dt|bc> [--live]');
      process.exit(1);
    }

    if (!domainDetector.isValidDomain(domain)) {
      console.error(`❌ Invalid domain: ${domain}`);
      process.exit(1);
    }

    console.log(`⚙️  Processing ${domain}: domain files...\n`);

    if (this.options.live) {
      console.log('⚠️  LIVE MODE - Files will be modified!\n');
      const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
      });

      const answer = await new Promise(resolve => {
        readline.question('Continue? (yes/no): ', resolve);
      });
      readline.close();

      if (answer.toLowerCase() !== 'yes') {
        console.log('❌ Aborted');
        return;
      }
    }

    const processor = new BatchProcessor({
      dryRun: this.options.dryRun,
      batchSize: this.options.batch
    });

    await processor.processDomain(domain);
  }

  async analyzeCrossDomainLinks() {
    console.log('🔗 Analyzing Cross-Domain Links\n');

    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    if (!fs.existsSync(inventoryPath)) {
      console.log('⚠️  No inventory found. Running scan first...\n');
      await this.runScan();
    }

    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
    const crossLinks = {};
    let totalLinks = 0;

    for (const file of inventory.fileInventory) {
      const content = fs.readFileSync(file.path, 'utf-8');
      const links = domainDetector.detectCrossDomainLinks(content, file.domain);

      if (links.length > 0) {
        totalLinks += links.length;
        links.forEach(link => {
          const key = `${file.domain}-${link.targetDomain}`;
          crossLinks[key] = (crossLinks[key] || 0) + 1;
        });
      }
    }

    console.log('='.repeat(80));
    console.log(`Total cross-domain links found: ${totalLinks}\n`);

    if (Object.keys(crossLinks).length > 0) {
      console.log('Distribution by domain pair:\n');
      Object.entries(crossLinks)
        .sort((a, b) => b[1] - a[1])
        .forEach(([pair, count]) => {
          const [from, to] = pair.split('-');
          console.log(`   ${from.padEnd(5)} → ${to.padEnd(5)} ${count} links`);

          // Show recommended bridges
          const bridges = domainDetector.getRecommendedBridges(from, to);
          if (bridges.length > 0) {
            console.log(`      Recommended bridges: ${bridges.slice(0, 3).join(', ')}`);
          }
        });
    } else {
      console.log('No cross-domain links detected.');
    }

    console.log('='.repeat(80));
  }

  async detectFileDomain() {
    const filePath = this.args[1];

    if (!filePath) {
      console.error('❌ Please provide a file path');
      console.error('Usage: node cli.js detect-domain <file-path>');
      process.exit(1);
    }

    if (!fs.existsSync(filePath)) {
      console.error(`❌ File not found: ${filePath}`);
      process.exit(1);
    }

    console.log(`🔍 Detecting domain for: ${path.basename(filePath)}\n`);

    const content = fs.readFileSync(filePath, 'utf-8');
    const domain = domainDetector.detect(filePath, content);
    const domainConf = domainDetector.getDomainConfig(domain);
    const subDomain = domainDetector.classifySubDomain(domain, content);

    console.log('='.repeat(80));
    console.log(`Domain: ${domain}`);
    console.log(`Full Name: ${domainConf.name}`);
    console.log(`Namespace: ${domainConf.namespace}`);
    console.log(`Prefix: ${domainConf.prefix}`);

    if (subDomain) {
      console.log(`Sub-domain: ${subDomain}`);
    }

    console.log(`\nRequired Properties:`);
    domainConf.requiredProperties.forEach(prop => {
      console.log(`   - ${prop}`);
    });

    const crossLinks = domainDetector.detectCrossDomainLinks(content, domain);
    if (crossLinks.length > 0) {
      console.log(`\nCross-domain links: ${crossLinks.length}`);
      crossLinks.slice(0, 5).forEach(link => {
        console.log(`   → ${link.target} (${link.targetDomain})`);
      });
    }

    console.log('='.repeat(80));
  }

  async runPattern() {
    const pattern = this.args[1];

    if (!pattern) {
      console.error('❌ Please specify a pattern');
      console.error('Usage: node cli.js pattern <pattern1|pattern2|pattern3|pattern4|pattern5|pattern6>');
      process.exit(1);
    }

    const processor = new BatchProcessor({
      dryRun: this.options.dryRun,
      batchSize: this.options.batch
    });

    await processor.processPattern(pattern);
  }

  async showStats() {
    console.log('📊 Current Statistics\n');

    // Check if reports exist
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const validationPath = path.join(config.reportsDirectory, 'validation-report.json');
    const updatePath = path.join(config.reportsDirectory, 'update-report.json');

    if (fs.existsSync(inventoryPath)) {
      const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));
      console.log('📁 File Inventory:');
      console.log(`   Total files: ${inventory.totalFiles}`);
      console.log(`   Files with ontology: ${inventory.filesWithOntology}`);
      console.log(`   Files without ontology: ${inventory.filesWithoutOntology}`);

      console.log('\n📈 Pattern Distribution:');
      Object.entries(inventory.patternDistribution).forEach(([pattern, count]) => {
        if (count > 0) {
          console.log(`   ${pattern}: ${count}`);
        }
      });

      console.log('\n🌍 Domain Distribution:');
      Object.entries(inventory.domainDistribution).forEach(([domain, count]) => {
        if (count > 0) {
          console.log(`   ${domain}: ${count}`);
        }
      });

      console.log('\n⚠️  Issues:');
      console.log(`   Namespace errors: ${inventory.issues.namespaceErrors.length}`);
      console.log(`   Naming issues: ${inventory.issues.namingIssues.length}`);
      console.log(`   Duplicate sections: ${inventory.issues.duplicateSections.length}`);
    }

    if (fs.existsSync(validationPath)) {
      const validation = JSON.parse(fs.readFileSync(validationPath, 'utf-8'));
      console.log('\n🔍 Validation Results:');
      console.log(`   Passed: ${validation.summary.passed}`);
      console.log(`   Failed: ${validation.summary.failed}`);
      console.log(`   Average score: ${validation.averageScore.toFixed(1)}/100`);
    }

    if (fs.existsSync(updatePath)) {
      const update = JSON.parse(fs.readFileSync(updatePath, 'utf-8'));
      console.log('\n✏️  Update Results:');
      console.log(`   Processed: ${update.processed}`);
      console.log(`   Updated: ${update.updated}`);
      console.log(`   Skipped: ${update.skipped}`);
      console.log(`   Errors: ${update.errors}`);
    }
  }

  showHelp() {
    console.log('Available Commands:\n');

    Object.entries(COMMANDS).forEach(([cmd, desc]) => {
      console.log(`  ${cmd.padEnd(15)} ${desc}`);
    });

    console.log('\nOptions:\n');
    console.log('  --live         Perform actual updates (default: dry-run)');
    console.log('  --batch=N      Set batch size (default: 100)');
    console.log('  --validate     Run validation after processing');
    console.log('  --verbose, -v  Enable verbose logging');

    console.log('\nExamples:\n');
    console.log('  node cli.js scan');
    console.log('  node cli.js preview 10');
    console.log('  node cli.js audit-blocks         # Find multiple blocks');
    console.log('  node cli.js audit-public         # Find public:: true files');
    console.log('  node cli.js audit-position       # Find blocks not at top');
    console.log('  node cli.js fix-blocks --live    # Fix multiple blocks');
    console.log('  node cli.js fix-position --live  # Move blocks to top');
    console.log('  node cli.js process --live --batch=50');
    console.log('  node cli.js validate');
    console.log('  node cli.js test path/to/file.md');
    console.log('  node cli.js domain rb --live');
    console.log('  node cli.js iri-stats            # IRI registry statistics');
    console.log('  node cli.js domains');
    console.log('  node cli.js domain-stats ai');
    console.log('  node cli.js validate-domain tc');
    console.log('  node cli.js process-domain bc --live');
    console.log('  node cli.js cross-domain-links');
    console.log('  node cli.js detect-domain path/to/file.md');

    console.log('\nSafety Features:\n');
    console.log('  ✅ Dry-run mode by default');
    console.log('  ✅ Git-based version control (no backup files)');
    console.log('  ✅ Progress checkpointing');
    console.log('  ✅ IRI uniqueness validation');
    console.log('  ✅ Validation checks');
    console.log('  ✅ Batch processing with error handling');
    console.log('  ✅ Single block enforcement');
    console.log('  ✅ Automatic block positioning');

    console.log('\nWorkflow:\n');
    console.log('  1. node cli.js scan              # Scan and inventory files');
    console.log('  2. node cli.js audit-blocks      # Audit for issues');
    console.log('  3. node cli.js preview 10        # Preview transformations');
    console.log('  4. node cli.js test file.md      # Test single file');
    console.log('  5. node cli.js process --live    # Run full migration');
    console.log('  6. node cli.js validate          # Validate results');
    console.log('  7. node cli.js iri-stats         # Check IRI registry');
    console.log('  8. node cli.js stats             # Check statistics');
  }
}

// Run CLI
const cli = new CLI();
cli.run().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
