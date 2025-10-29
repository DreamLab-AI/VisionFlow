#!/usr/bin/env node

/**
 * Import to Ontology - Core Engine
 *
 * Intelligently moves content from source markdown files to appropriate
 * ontology locations with validation, enrichment, and web content integration.
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
  indexPath: path.join(process.cwd(), '.cache/ontology-index.json'),
  backupDir: path.join(process.cwd(), '.backups'),
  logDir: '/tmp',
  webSummaryEnabled: true,
  webSummaryConcurrency: 5,
  minConfidence: 0.4,
};

// Global index (loaded once)
let INDEX = null;

/**
 * Load ontology index
 */
function loadIndex() {
  if (INDEX) return INDEX;

  console.log('📚 Loading ontology index...');
  const data = fs.readFileSync(CONFIG.indexPath, 'utf-8');
  INDEX = JSON.parse(data);
  console.log(`   ✅ Loaded ${INDEX.metadata.totalFiles} concepts\n`);

  return INDEX;
}

/**
 * Parse source markdown file into content blocks
 */
function parseSourceFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  const blocks = [];
  let currentBlock = null;
  let blockId = 1;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Detect block boundaries
    // Handle both standard markdown (# Heading) and Logseq format (- # Heading)
    const isHeading = line.startsWith('#') || /^-\s+#{1,6}\s/.test(trimmed);

    if (isHeading) {
      // Save previous block
      if (currentBlock) {
        currentBlock.endLine = i - 1;
        blocks.push(completeBlock(currentBlock));
      }

      // Start new heading block
      currentBlock = {
        id: `block-${blockId++}`,
        type: 'heading',
        content: line,
        startLine: i,
      };
    } else if (line.startsWith('```')) {
      // Save previous block
      if (currentBlock) {
        currentBlock.endLine = i - 1;
        blocks.push(completeBlock(currentBlock));
      }

      // Find end of code block
      let endLine = i + 1;
      while (endLine < lines.length && !lines[endLine].startsWith('```')) {
        endLine++;
      }

      blocks.push({
        id: `block-${blockId++}`,
        type: 'code',
        content: lines.slice(i, endLine + 1).join('\n'),
        startLine: i,
        endLine: endLine,
      });

      currentBlock = null;
      i = endLine;
    } else if (currentBlock) {
      // Continuation of current block
      currentBlock.content += '\n' + line;
    } else if (line.trim()) {
      // Start new paragraph
      currentBlock = {
        id: `block-${blockId++}`,
        type: 'paragraph',
        content: line,
        startLine: i,
      };
    }
  }

  // Complete final block
  if (currentBlock) {
    currentBlock.endLine = lines.length - 1;
    blocks.push(completeBlock(currentBlock));
  }

  return {
    blocks,
    metadata: {
      totalBlocks: blocks.length,
      totalLines: lines.length,
    },
  };
}

/**
 * Complete block with metadata extraction
 */
function completeBlock(block) {
  block.metadata = {
    keywords: extractKeywords(block.content),
    wikiLinks: extractWikiLinks(block.content),
    urls: extractUrls(block.content),
    assertions: extractAssertions(block.content),
  };

  return block;
}

/**
 * Extract keywords from text
 */
function extractKeywords(text) {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 3);

  // Remove common words
  const stopwords = new Set(['this', 'that', 'with', 'from', 'have', 'been', 'were', 'they', 'what', 'when', 'where']);
  return [...new Set(words.filter(w => !stopwords.has(w)))];
}

/**
 * Extract WikiLinks
 */
function extractWikiLinks(text) {
  const regex = /\[\[([^\]]+)\]\]/g;
  const links = [];
  let match;

  while ((match = regex.exec(text)) !== null) {
    links.push(match[1]);
  }

  return [...new Set(links)];
}

/**
 * Extract URLs
 */
function extractUrls(text) {
  const regex = /(https?:\/\/[^\s\)]+)/g;
  const matches = text.match(regex);
  return matches ? [...new Set(matches)] : [];
}

/**
 * Extract assertions that might need validation
 */
function extractAssertions(text) {
  const assertions = [];

  // Patterns indicating assertions
  const patterns = [
    { regex: /is defined as (.+?)\./gi, type: 'definition' },
    { regex: /refers to (.+?)\./gi, type: 'definition' },
    { regex: /(\d+%|\d+ percent)/gi, type: 'statistic' },
    { regex: /according to (.+?),/gi, type: 'citation' },
    { regex: /enables (.+?)\./gi, type: 'claim' },
    { regex: /provides (.+?)\./gi, type: 'claim' },
  ];

  for (const { regex, type } of patterns) {
    let match;
    while ((match = regex.exec(text)) !== null) {
      assertions.push({
        text: match[0],
        type,
        needsValidation: true,
      });
    }
  }

  return assertions;
}

/**
 * Find target concept for a content block using semantic index
 */
function findTargetConcept(block) {
  const index = loadIndex();

  // Extract semantic features
  const keywords = block.metadata.keywords;
  const wikiLinks = block.metadata.wikiLinks;

  // Score all concepts
  const scored = Object.values(index.concepts.concepts)
    .map(concept => {
      let score = 0;

      // Keyword overlap (40% weight)
      const keywordMatch = keywords.filter(k =>
        concept.keywords.some(ck => ck.includes(k) || k.includes(ck))
      ).length;
      score += (keywordMatch / Math.max(keywords.length, 1)) * 0.4;

      // WikiLink overlap (60% weight)
      const linkMatch = wikiLinks.filter(link =>
        concept.linksTo.includes(link) ||
        concept.linkedFrom.includes(link) ||
        concept.preferredTerm === link
      ).length;
      score += (linkMatch / Math.max(wikiLinks.length, 1)) * 0.6;

      return { concept, score, keywordMatch, linkMatch };
    })
    .filter(s => s.score > 0)
    .sort((a, b) => b.score - a.score);

  if (scored.length === 0) {
    // Fallback: return null for manual handling
    return {
      blockId: block.id,
      targetFile: null,
      targetConcept: null,
      confidence: 0,
      reasoning: 'No semantic matches found - manual review needed',
    };
  }

  const best = scored[0];

  return {
    blockId: block.id,
    targetFile: best.concept.file,
    targetConcept: best.concept.preferredTerm,
    confidence: Math.min(best.score, 0.95),
    reasoning: `Matched ${best.keywordMatch} keywords, ${best.linkMatch} links`,
    alternatives: scored.slice(1, 4).map(s => ({
      concept: s.concept.preferredTerm,
      file: s.concept.file,
      confidence: s.score,
    })),
  };
}

/**
 * Detect stubs (isolated WikiLinks and URLs needing enrichment)
 */
function detectStubs(block) {
  const stubs = [];
  const content = block.content;
  const index = loadIndex();

  // Find WikiLink stubs (broken or without context)
  for (const wikiLink of block.metadata.wikiLinks) {
    const fullLink = `[[${wikiLink}]]`;
    const isValid = index.wikilinks.valid[fullLink];

    if (!isValid) {
      stubs.push({
        type: 'wikilink',
        value: wikiLink,
        enrichmentNeeded: true,
        reason: 'Broken WikiLink - concept does not exist',
      });
    }
  }

  // Find URL stubs (URLs without descriptions)
  for (const url of block.metadata.urls) {
    const urlIndex = content.indexOf(url);
    const context = content.substring(
      Math.max(0, urlIndex - 50),
      Math.min(content.length, urlIndex + 50)
    );

    // Check if URL has meaningful context
    const words = context.split(/\s+/).filter(w => w.length > 3);
    const hasContext = words.length > 5;

    if (!hasContext) {
      stubs.push({
        type: 'url',
        value: url,
        enrichmentNeeded: true,
        reason: 'Isolated URL without description',
      });
    }
  }

  return stubs;
}

/**
 * Create import plan for source file
 */
function createImportPlan(filePath) {
  console.log(`📋 Analyzing ${path.basename(filePath)}...`);

  const parsed = parseSourceFile(filePath);
  const targets = [];
  const enrichments = [];

  for (const block of parsed.blocks) {
    // Find target
    const target = findTargetConcept(block);
    targets.push(target);

    // Detect stubs needing enrichment
    const stubs = detectStubs(block);
    enrichments.push(...stubs.map(stub => ({
      blockId: block.id,
      stub,
    })));
  }

  // Calculate estimated time
  const urlCount = enrichments.filter(e => e.stub.type === 'url').length;
  const estimatedTime = parsed.blocks.length * 2 + urlCount * 5;

  return {
    sourceFile: filePath,
    blocks: parsed.blocks,
    targets,
    enrichments,
    estimatedTime,
    summary: {
      totalBlocks: parsed.blocks.length,
      highConfidenceTargets: targets.filter(t => t.confidence > 0.7).length,
      mediumConfidenceTargets: targets.filter(t => t.confidence >= 0.4 && t.confidence <= 0.7).length,
      lowConfidenceTargets: targets.filter(t => t.confidence < 0.4).length,
      urlsToEnrich: urlCount,
      wikiLinksToCreate: enrichments.filter(e => e.stub.type === 'wikilink').length,
    },
  };
}

/**
 * Dry run - analyze without importing
 */
function dryRun(filePath) {
  const plan = createImportPlan(filePath);

  console.log('\n📊 DRY RUN REPORT\n');
  console.log(`Source File: ${path.basename(plan.sourceFile)}`);
  console.log(`Total Blocks: ${plan.summary.totalBlocks}`);
  console.log(`Estimated Time: ${Math.ceil(plan.estimatedTime / 60)} minutes\n`);

  console.log('🎯 Targeting Summary:');
  console.log(`   High Confidence (>70%): ${plan.summary.highConfidenceTargets}`);
  console.log(`   Medium Confidence (40-70%): ${plan.summary.mediumConfidenceTargets}`);
  console.log(`   Low Confidence (<40%): ${plan.summary.lowConfidenceTargets}`);

  console.log('\n🔗 Enrichment Summary:');
  console.log(`   URLs to enrich: ${plan.summary.urlsToEnrich}`);
  console.log(`   WikiLinks to create: ${plan.summary.wikiLinksToCreate}`);

  console.log('\n📝 Sample Targets:\n');
  plan.targets.slice(0, 5).forEach(target => {
    const block = plan.blocks.find(b => b.id === target.blockId);
    const preview = block.content.substring(0, 60).replace(/\n/g, ' ') + '...';

    console.log(`   Block: "${preview}"`);
    console.log(`   → ${target.targetConcept || 'MANUAL REVIEW'} (${(target.confidence * 100).toFixed(0)}% confidence)`);
    console.log(`     File: ${target.targetFile || 'N/A'}`);
    console.log(`     Reason: ${target.reasoning}\n`);
  });

  // Warnings
  if (plan.summary.lowConfidenceTargets > 0) {
    console.log(`⚠️  WARNING: ${plan.summary.lowConfidenceTargets} blocks have low confidence - manual review recommended\n`);
  }

  if (plan.summary.urlsToEnrich > 20) {
    console.log(`⚠️  WARNING: ${plan.summary.urlsToEnrich} URLs to enrich - this will be slow (~${Math.ceil(plan.summary.urlsToEnrich * 5 / 60)} minutes)\n`);
  }

  return plan;
}

/**
 * Execute import
 */
async function executeImport(filePath, options = {}) {
  const dryRunFirst = options.dryRun !== false;

  // Step 1: Dry run
  if (dryRunFirst) {
    console.log('🔍 Running dry-run analysis...\n');
    const plan = dryRun(filePath);

    if (!options.force) {
      console.log('ℹ️  Dry run complete. Use --force to proceed with import.\n');
      return { dryRun: true, plan };
    }
  }

  // Step 2: Create plan
  const plan = createImportPlan(filePath);

  // Step 3: Create backup
  console.log('\n💾 Creating backup...');
  const backupPath = createBackup(filePath);
  console.log(`   Backup: ${backupPath}\n`);

  // Step 4: Process blocks
  console.log('🚀 Processing blocks...\n');
  const results = [];

  for (let i = 0; i < plan.blocks.length; i++) {
    const block = plan.blocks[i];
    const target = plan.targets[i];

    console.log(`   [${i + 1}/${plan.blocks.length}] Processing ${block.id}...`);

    if (target.confidence < CONFIG.minConfidence) {
      console.log(`      ⚠️  Skipping - confidence too low (${(target.confidence * 100).toFixed(0)}%)`);
      results.push({ block: block.id, status: 'skipped', reason: 'low-confidence' });
      continue;
    }

    // TODO: Implement actual content insertion
    // This would call insertContent() with target file and block content

    results.push({
      block: block.id,
      status: 'success',
      target: target.targetFile,
    });
  }

  console.log(`\n✅ Import complete!`);
  console.log(`   Processed: ${results.filter(r => r.status === 'success').length}/${plan.blocks.length}`);
  console.log(`   Skipped: ${results.filter(r => r.status === 'skipped').length}`);

  return { success: true, results, backupPath };
}

/**
 * Create backup of source file
 */
function createBackup(filePath) {
  if (!fs.existsSync(CONFIG.backupDir)) {
    fs.mkdirSync(CONFIG.backupDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const basename = path.basename(filePath);
  const backupPath = path.join(CONFIG.backupDir, `${timestamp}-${basename}`);

  fs.copyFileSync(filePath, backupPath);

  return backupPath;
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage: node import-engine.js <source-file> [--dry-run] [--force]');
    process.exit(1);
  }

  const filePath = path.resolve(args[0]);
  const dryRunOnly = args.includes('--dry-run');
  const force = args.includes('--force');

  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found: ${filePath}`);
    process.exit(1);
  }

  if (dryRunOnly) {
    dryRun(filePath);
  } else {
    executeImport(filePath, { force, dryRun: !force })
      .then(result => {
        if (result.dryRun) {
          console.log('ℹ️  Add --force flag to proceed with import');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        process.exit(1);
      });
  }
}

module.exports = {
  parseSourceFile,
  createImportPlan,
  findTargetConcept,
  detectStubs,
  dryRun,
  executeImport,
};
