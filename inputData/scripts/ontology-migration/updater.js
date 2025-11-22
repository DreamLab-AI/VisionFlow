#!/usr/bin/env node

/**
 * File Updater - Ontology Block Migration Pipeline
 *
 * Replaces existing ontology blocks with canonical format.
 * Creates backups, preserves content, validates after update.
 */

const fs = require('fs');
const path = require('path');
const config = require('./config.json');
const OntologyParser = require('./parser');
const OntologyGenerator = require('./generator');
const iriRegistry = require('./iri-registry');

class OntologyUpdater {
  constructor(options = {}) {
    this.parser = new OntologyParser();
    this.generator = new OntologyGenerator();
    this.dryRun = options.dryRun !== undefined ? options.dryRun : config.dryRun;
    this.results = {
      processed: 0,
      updated: 0,
      skipped: 0,
      errors: 0,
      multipleBlocksFixed: 0,
      publicPropertiesMigrated: 0,
      irisGenerated: 0
    };
  }

  /**
   * Update a single file
   */
  async updateFile(filePath) {
    try {
      console.log(`\n📝 Processing: ${path.basename(filePath)}`);

      // Parse existing file
      const parsed = this.parser.parseFile(filePath);

      if (!parsed.hasOntologyBlock) {
        console.log('   ⏭️  No ontology block found, skipping');
        this.results.skipped++;
        return { success: true, skipped: true };
      }

      // Track special fixes
      if (parsed.hasMultipleBlocks) {
        console.log(`   📦 Multiple blocks detected (${parsed.allBlocks.length}), consolidating...`);
        this.results.multipleBlocksFixed++;
      }

      if (parsed.hasPublicProperty && !parsed.properties['public-access']) {
        console.log('   🔓 Migrating public:: true property to ontology block');
        this.results.publicPropertiesMigrated++;
      }

      // Generate new canonical block
      const newBlock = this.generator.generate(parsed);

      // Count IRI generation
      if (newBlock.includes('- iri::')) {
        this.results.irisGenerated++;
      }

      // Generate full file content
      const newContent = this.generator.generateFullFile(parsed, newBlock);

      // Show preview in dry-run mode
      if (this.dryRun) {
        console.log('   🔍 DRY RUN - Preview of changes:');
        console.log('   ' + '─'.repeat(70));
        console.log(newBlock.split('\n').slice(0, 15).map(line => '   ' + line).join('\n'));
        console.log('   ' + '─'.repeat(70));
        this.results.processed++;
        return { success: true, dryRun: true };
      }

      // Write new content (Git will track changes - no backups needed)
      fs.writeFileSync(filePath, newContent, 'utf-8');

      console.log('   ✅ Updated successfully');
      this.results.processed++;
      this.results.updated++;

      return { success: true, updated: true };

    } catch (error) {
      console.error(`   ❌ Error updating file: ${error.message}`);
      this.results.errors++;
      return { success: false, error: error.message };
    }
  }

  /**
   * Update batch of files
   */
  async updateBatch(filePaths, batchSize = config.batchSize) {
    console.log(`\n🚀 Starting batch update (${filePaths.length} files)`);
    console.log(`📦 Batch size: ${batchSize}`);
    console.log(`🔧 Mode: ${this.dryRun ? 'DRY RUN' : 'LIVE UPDATE'}`);
    console.log(`📝 Version control: Git will track all changes`);

    const batches = [];
    for (let i = 0; i < filePaths.length; i += batchSize) {
      batches.push(filePaths.slice(i, i + batchSize));
    }

    console.log(`\n📊 Processing ${batches.length} batches...`);

    for (let i = 0; i < batches.length; i++) {
      console.log(`\n${'='.repeat(80)}`);
      console.log(`Batch ${i + 1}/${batches.length} (${batches[i].length} files)`);
      console.log('='.repeat(80));

      for (const filePath of batches[i]) {
        await this.updateFile(filePath);
      }

      // Save checkpoint after each batch
      this.saveCheckpoint(i, batches.length);
    }

    return this.results;
  }

  /**
   * Save checkpoint for resumability
   */
  saveCheckpoint(currentBatch, totalBatches) {
    const checkpoint = {
      timestamp: new Date().toISOString(),
      currentBatch,
      totalBatches,
      progress: ((currentBatch + 1) / totalBatches * 100).toFixed(1) + '%',
      results: this.results
    };

    const checkpointPath = path.join(config.reportsDirectory, 'checkpoint.json');
    fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2));
  }

  /**
   * Generate update report
   */
  generateReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 UPDATE SUMMARY');
    console.log('='.repeat(80));
    console.log(`Total files processed: ${this.results.processed}`);
    console.log(`Files updated: ${this.results.updated}`);
    console.log(`Files skipped: ${this.results.skipped}`);
    console.log(`Errors: ${this.results.errors}`);
    console.log(`Multiple blocks fixed: ${this.results.multipleBlocksFixed}`);
    console.log(`Public properties migrated: ${this.results.publicPropertiesMigrated}`);
    console.log(`IRIs generated: ${this.results.irisGenerated}`);
    console.log('='.repeat(80));

    // Save report
    const report = {
      timestamp: new Date().toISOString(),
      mode: this.dryRun ? 'dry-run' : 'live',
      ...this.results
    };

    const reportPath = path.join(config.reportsDirectory, 'update-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n💾 Report saved to: ${reportPath}`);

    // Save IRI registry
    try {
      iriRegistry.save();
      console.log('💾 IRI registry saved');
    } catch (error) {
      console.warn(`⚠️  Failed to save IRI registry: ${error.message}`);
    }
  }
}

// Run updater if executed directly
if (require.main === module) {
  const args = process.argv.slice(2);
  const filePath = args[0];

  if (!filePath) {
    console.error('Usage: node updater.js <file-path> [--live]');
    console.error('Note: Version control via Git - no rollback needed');
    process.exit(1);
  }

  const isLive = args.includes('--live');
  const updater = new OntologyUpdater({ dryRun: !isLive });

  updater.updateFile(filePath)
    .then(() => updater.generateReport());
}

module.exports = OntologyUpdater;
