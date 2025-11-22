#!/usr/bin/env node

/**
 * Batch Processor - Ontology Block Migration Pipeline
 *
 * Orchestrates the complete migration pipeline:
 * scan → parse → generate → update → validate
 *
 * Processes files in batches with error handling,
 * progress tracking, and rollback capability.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const OntologyScanner = require('./scanner');
const OntologyParser = require('./parser');
const OntologyUpdater = require('./updater');
const OntologyValidator = require('./validator');

class BatchProcessor {
  constructor(options = {}) {
    this.scanner = new OntologyScanner();
    this.parser = new OntologyParser();
    this.updater = new OntologyUpdater({
      dryRun: options.dryRun !== undefined ? options.dryRun : config.dryRun
    });
    this.validator = new OntologyValidator();

    this.options = {
      batchSize: options.batchSize || config.batchSize,
      validateBefore: options.validateBefore !== undefined ? options.validateBefore : false,
      validateAfter: options.validateAfter !== undefined ? options.validateAfter : config.validateAfterUpdate,
      ...options
    };

    this.state = {
      phase: 'idle',
      totalFiles: 0,
      processedFiles: 0,
      successCount: 0,
      errorCount: 0,
      startTime: null,
      currentBatch: 0,
      totalBatches: 0
    };
  }

  /**
   * Run complete migration pipeline
   */
  async run() {
    console.log('\n' + '='.repeat(80));
    console.log('🚀 ONTOLOGY BLOCK MIGRATION PIPELINE');
    console.log('='.repeat(80));
    console.log(`Mode: ${this.updater.dryRun ? '🔍 DRY RUN' : '✍️  LIVE UPDATE'}`);
    console.log(`Batch size: ${this.options.batchSize}`);
    console.log(`Version control: Git-based (no backup files)`);
    console.log(`Validation: ${this.options.validateAfter ? '✅ ENABLED' : '❌ DISABLED'}`);
    console.log('='.repeat(80));

    this.state.startTime = Date.now();

    try {
      // Phase 1: Scan
      await this.runScanPhase();

      // Phase 2: Process
      await this.runProcessPhase();

      // Phase 3: Validate (optional)
      if (this.options.validateAfter) {
        await this.runValidationPhase();
      }

      // Phase 4: Report
      await this.generateFinalReport();

      console.log('\n✅ Pipeline completed successfully!');

    } catch (error) {
      console.error('\n❌ Pipeline failed:', error.message);
      console.error(error.stack);
      process.exit(1);
    }
  }

  /**
   * Phase 1: Scan files
   */
  async runScanPhase() {
    this.state.phase = 'scanning';
    console.log('\n' + '─'.repeat(80));
    console.log('📊 PHASE 1: SCANNING FILES');
    console.log('─'.repeat(80));

    const scanResults = await this.scanner.scan();
    await this.scanner.generateReport();

    this.state.totalFiles = scanResults.filesWithOntology;
    this.state.totalBatches = Math.ceil(this.state.totalFiles / this.options.batchSize);

    console.log(`\n✅ Scan complete: ${this.state.totalFiles} files with ontology blocks`);
  }

  /**
   * Phase 2: Process files
   */
  async runProcessPhase() {
    this.state.phase = 'processing';
    console.log('\n' + '─'.repeat(80));
    console.log('⚙️  PHASE 2: PROCESSING FILES');
    console.log('─'.repeat(80));

    // Load inventory
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));

    const filesToProcess = inventory.fileInventory.map(item => item.path);

    // Split into batches
    const batches = [];
    for (let i = 0; i < filesToProcess.length; i += this.options.batchSize) {
      batches.push(filesToProcess.slice(i, i + this.options.batchSize));
    }

    // Process each batch
    for (let i = 0; i < batches.length; i++) {
      this.state.currentBatch = i + 1;
      console.log(`\n${'='.repeat(80)}`);
      console.log(`📦 BATCH ${i + 1}/${batches.length}`);
      console.log(`Files: ${batches[i].length}`);
      console.log('='.repeat(80));

      for (const filePath of batches[i]) {
        const result = await this.updater.updateFile(filePath);

        if (result.success && !result.skipped) {
          this.state.successCount++;
        } else if (!result.success) {
          this.state.errorCount++;
        }

        this.state.processedFiles++;

        // Progress indicator
        if (this.state.processedFiles % 10 === 0) {
          const progress = (this.state.processedFiles / this.state.totalFiles * 100).toFixed(1);
          console.log(`\n📈 Progress: ${this.state.processedFiles}/${this.state.totalFiles} (${progress}%)`);
        }
      }

      // Save checkpoint after each batch
      this.saveCheckpoint();

      // Optional: pause between batches
      if (this.options.pauseBetweenBatches && i < batches.length - 1) {
        await this.pause(this.options.pauseDuration || 1000);
      }
    }

    console.log('\n✅ Processing complete');
    this.updater.generateReport();
  }

  /**
   * Phase 3: Validation
   */
  async runValidationPhase() {
    this.state.phase = 'validating';
    console.log('\n' + '─'.repeat(80));
    console.log('🔍 PHASE 3: VALIDATION');
    console.log('─'.repeat(80));

    // Load inventory
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));

    const filesToValidate = inventory.fileInventory.map(item => item.path);

    const validations = this.validator.validateBatch(filesToValidate);
    this.validator.generateReport(validations);

    console.log('\n✅ Validation complete');
  }

  /**
   * Generate final report
   */
  async generateFinalReport() {
    const duration = Date.now() - this.state.startTime;
    const minutes = Math.floor(duration / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);

    console.log('\n' + '='.repeat(80));
    console.log('📊 FINAL REPORT');
    console.log('='.repeat(80));
    console.log(`Total files processed: ${this.state.processedFiles}`);
    console.log(`Successful updates: ${this.state.successCount}`);
    console.log(`Errors: ${this.state.errorCount}`);
    console.log(`Success rate: ${(this.state.successCount/this.state.processedFiles*100).toFixed(1)}%`);
    console.log(`Duration: ${minutes}m ${seconds}s`);
    console.log(`Average speed: ${(this.state.processedFiles / (duration / 1000)).toFixed(1)} files/sec`);
    console.log('='.repeat(80));

    // Save final report
    const report = {
      timestamp: new Date().toISOString(),
      mode: this.updater.dryRun ? 'dry-run' : 'live',
      duration: {
        milliseconds: duration,
        formatted: `${minutes}m ${seconds}s`
      },
      statistics: {
        totalFiles: this.state.totalFiles,
        processedFiles: this.state.processedFiles,
        successCount: this.state.successCount,
        errorCount: this.state.errorCount,
        successRate: (this.state.successCount/this.state.processedFiles*100).toFixed(1) + '%'
      },
      configuration: {
        batchSize: this.options.batchSize,
        backupsEnabled: this.updater.createBackups,
        validationEnabled: this.options.validateAfter
      }
    };

    const reportPath = path.join(config.reportsDirectory, 'final-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n💾 Final report saved to: ${reportPath}`);
  }

  /**
   * Save checkpoint
   */
  saveCheckpoint() {
    const checkpoint = {
      timestamp: new Date().toISOString(),
      phase: this.state.phase,
      currentBatch: this.state.currentBatch,
      totalBatches: this.state.totalBatches,
      processedFiles: this.state.processedFiles,
      totalFiles: this.state.totalFiles,
      progress: (this.state.processedFiles / this.state.totalFiles * 100).toFixed(1) + '%',
      successCount: this.state.successCount,
      errorCount: this.state.errorCount
    };

    const checkpointPath = path.join(config.reportsDirectory, 'pipeline-checkpoint.json');
    fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2));

    if (config.verboseLogging) {
      console.log(`   💾 Checkpoint saved`);
    }
  }

  /**
   * Pause execution
   */
  async pause(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Resume from checkpoint
   */
  async resume() {
    const checkpointPath = path.join(config.reportsDirectory, 'pipeline-checkpoint.json');

    if (!fs.existsSync(checkpointPath)) {
      console.error('❌ No checkpoint found');
      return;
    }

    const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8'));

    console.log('\n🔄 Resuming from checkpoint:');
    console.log(`   Phase: ${checkpoint.phase}`);
    console.log(`   Progress: ${checkpoint.progress}`);
    console.log(`   Files processed: ${checkpoint.processedFiles}/${checkpoint.totalFiles}`);

    // Resume logic would go here
    // For now, just run full pipeline
    await this.run();
  }

  /**
   * Process specific domain only
   */
  async processDomain(domain) {
    console.log(`\n🎯 Processing ${domain} domain only`);

    // Load inventory and filter by domain
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));

    const domainFiles = inventory.fileInventory
      .filter(item => item.domain === domain)
      .map(item => item.path);

    console.log(`Found ${domainFiles.length} files in ${domain} domain`);

    // Process filtered files
    await this.updater.updateBatch(domainFiles, this.options.batchSize);
    this.updater.generateReport();
  }

  /**
   * Process specific pattern only
   */
  async processPattern(pattern) {
    console.log(`\n🎯 Processing ${pattern} only`);

    // Load inventory and filter by pattern
    const inventoryPath = path.join(config.reportsDirectory, 'file-inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf-8'));

    const patternFiles = inventory.fileInventory
      .filter(item => item.pattern === pattern)
      .map(item => item.path);

    console.log(`Found ${patternFiles.length} files with ${pattern}`);

    // Process filtered files
    await this.updater.updateBatch(patternFiles, this.options.batchSize);
    this.updater.generateReport();
  }
}

// Run processor if executed directly
if (require.main === module) {
  const args = process.argv.slice(2);

  const options = {
    dryRun: !args.includes('--live'),
    batchSize: parseInt(args.find(arg => arg.startsWith('--batch='))?.split('=')[1]) || config.batchSize,
    validateAfter: args.includes('--validate')
  };

  const processor = new BatchProcessor(options);

  if (args.includes('--resume')) {
    processor.resume();
  } else if (args.includes('--domain')) {
    const domainIndex = args.indexOf('--domain');
    const domain = args[domainIndex + 1];
    processor.processDomain(domain);
  } else if (args.includes('--pattern')) {
    const patternIndex = args.indexOf('--pattern');
    const pattern = args[patternIndex + 1];
    processor.processPattern(pattern);
  } else {
    processor.run();
  }
}

module.exports = BatchProcessor;
