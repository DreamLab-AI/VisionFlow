#!/usr/bin/env node

/**
 * Markdown Link Validator
 *
 * Validates all markdown links in documentation:
 * - Internal links (to other docs)
 * - Anchors within documents
 * - Relative paths
 * - Image references
 * - Code file references
 */

const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = '/home/devuser/workspace/project';
const DOCS_DIR = path.join(PROJECT_ROOT, 'docs');

// Results tracking
const results = {
  validLinks: [],
  brokenLinks: [],
  fixesApplied: [],
  totalLinks: 0,
  filesScanned: 0
};

/**
 * Find all markdown files
 */
function findMarkdownFiles(dir) {
  const files = [];

  function walk(currentDir) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      // Skip node_modules and hidden directories
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) {
        continue;
      }

      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }
  }

  walk(dir);
  return files;
}

/**
 * Extract all links from markdown content
 */
function extractLinks(content, filePath) {
  const links = [];

  // Match markdown links: [text](url) or [text](url#anchor)
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let match;

  let lineNumber = 1;
  const lines = content.split('\n');

  for (const line of lines) {
    let lineMatch;
    const lineRegex = /\[([^\]]+)\]\(([^)]+)\)/g;

    while ((lineMatch = lineRegex.exec(line)) !== null) {
      const [fullMatch, text, url] = lineMatch;

      // Skip external URLs
      if (url.startsWith('http://') || url.startsWith('https://')) {
        continue;
      }

      links.push({
        text,
        url,
        line: lineNumber,
        fullMatch,
        file: filePath
      });

      results.totalLinks++;
    }

    lineNumber++;
  }

  return links;
}

/**
 * Resolve a link relative to the file it's in
 */
function resolveLink(link, baseFile) {
  const baseDir = path.dirname(baseFile);

  // Split URL and anchor
  const [urlPath, anchor] = link.url.split('#');

  if (!urlPath) {
    // Just an anchor - check in current file
    return {
      resolved: baseFile,
      anchor: anchor || null,
      isAnchorOnly: true
    };
  }

  // Resolve relative path
  const resolved = path.resolve(baseDir, urlPath);

  return {
    resolved,
    anchor: anchor || null,
    isAnchorOnly: false
  };
}

/**
 * Check if a file exists
 */
function fileExists(filePath) {
  try {
    return fs.existsSync(filePath);
  } catch (error) {
    return false;
  }
}

/**
 * Extract anchors from markdown content
 */
function extractAnchors(content) {
  const anchors = new Set();

  // Match headers: # Header, ## Header, etc.
  const headerRegex = /^#+\s+(.+)$/gm;
  let match;

  while ((match = headerRegex.exec(content)) !== null) {
    const headerText = match[1];

    // Convert to anchor format (GitHub-style)
    const anchor = headerText
      .toLowerCase()
      .replace(/[^\w\s-]/g, '') // Remove special chars
      .replace(/\s+/g, '-')      // Spaces to hyphens
      .replace(/-+/g, '-')       // Multiple hyphens to single
      .replace(/^-|-$/g, '');    // Remove leading/trailing hyphens

    anchors.add(anchor);
  }

  return anchors;
}

/**
 * Validate a single link
 */
function validateLink(link, baseFile) {
  const { resolved, anchor, isAnchorOnly } = resolveLink(link, baseFile);

  // Check if target file exists
  if (!fileExists(resolved)) {
    results.brokenLinks.push({
      file: link.file,
      line: link.line,
      target: link.url,
      reason: 'FILE NOT FOUND',
      resolvedPath: resolved
    });
    return false;
  }

  // If there's an anchor, check if it exists
  if (anchor) {
    const content = fs.readFileSync(resolved, 'utf-8');
    const anchors = extractAnchors(content);

    if (!anchors.has(anchor)) {
      results.brokenLinks.push({
        file: link.file,
        line: link.line,
        target: link.url,
        reason: 'ANCHOR NOT FOUND',
        resolvedPath: resolved,
        anchor
      });
      return false;
    }
  }

  results.validLinks.push({
    file: link.file,
    target: link.url
  });

  return true;
}

/**
 * Generate report
 */
function generateReport() {
  console.log('## Link Validation Report\n');

  // Statistics
  console.log('### 📊 Statistics:');
  console.log(`- Files scanned: ${results.filesScanned}`);
  console.log(`- Total links: ${results.totalLinks}`);
  console.log(`- Valid: ${results.validLinks.length} (${Math.round(results.validLinks.length / results.totalLinks * 100)}%)`);
  console.log(`- Broken: ${results.brokenLinks.length} (${Math.round(results.brokenLinks.length / results.totalLinks * 100)}%)`);
  console.log(`- Fixed: ${results.fixesApplied.length}\n`);

  // Valid links summary
  if (results.validLinks.length > 0) {
    console.log(`### ✅ Valid Links (Total: ${results.validLinks.length})\n`);

    // Group by file
    const byFile = {};
    results.validLinks.forEach(link => {
      const relPath = path.relative(PROJECT_ROOT, link.file);
      if (!byFile[relPath]) byFile[relPath] = [];
      byFile[relPath].push(link.target);
    });

    Object.keys(byFile).sort().forEach(file => {
      console.log(`- **${file}**: ${byFile[file].length} valid links`);
    });
    console.log();
  }

  // Broken links
  if (results.brokenLinks.length > 0) {
    console.log(`### ❌ Broken Links (Total: ${results.brokenLinks.length})\n`);

    results.brokenLinks.forEach(broken => {
      const relPath = path.relative(PROJECT_ROOT, broken.file);
      const reason = broken.reason === 'ANCHOR NOT FOUND'
        ? `${broken.reason} (#${broken.anchor})`
        : broken.reason;

      console.log(`- **${relPath}:${broken.line}** → \`${broken.target}\` (${reason})`);
    });
    console.log();
  }

  // Fixes applied
  if (results.fixesApplied.length > 0) {
    console.log(`### 🔧 Fixes Applied:\n`);

    results.fixesApplied.forEach(fix => {
      console.log(`- Updated \`${fix.old}\` to \`${fix.new}\` in ${path.relative(PROJECT_ROOT, fix.file)}`);
    });
    console.log();
  }

  // Exit code
  if (results.brokenLinks.length > 0) {
    console.log('⚠️  Validation failed: broken links found');
    process.exit(1);
  } else {
    console.log('✅ All links valid!');
    process.exit(0);
  }
}

/**
 * Main execution
 */
function main() {
  console.log('🔍 Scanning for markdown files...\n');

  // Find all markdown files
  const docsFiles = findMarkdownFiles(DOCS_DIR);
  const rootFiles = fs.readdirSync(PROJECT_ROOT)
    .filter(f => f.endsWith('.md'))
    .map(f => path.join(PROJECT_ROOT, f));

  const allFiles = [...docsFiles, ...rootFiles];

  console.log(`Found ${allFiles.length} markdown files\n`);
  console.log('📝 Validating links...\n');

  // Process each file
  allFiles.forEach(file => {
    results.filesScanned++;

    const content = fs.readFileSync(file, 'utf-8');
    const links = extractLinks(content, file);

    links.forEach(link => {
      validateLink(link, file);
    });
  });

  // Generate report
  generateReport();
}

// Run
main();
