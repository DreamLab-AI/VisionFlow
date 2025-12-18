#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const linkGraphPath = '/home/devuser/workspace/project/docs/working/complete-link-graph.json';
const outputPath = '/home/devuser/workspace/project/docs/working/link-fix-suggestions.md';

console.log('Loading link graph...');
const linkGraph = JSON.parse(fs.readFileSync(linkGraphPath, 'utf8'));

const fixes = [];
const patterns = new Map();

// Analyze broken link patterns
console.log('Analyzing broken link patterns...');
linkGraph.files.forEach(fileData => {
  fileData.links.broken.forEach(brokenLink => {
    const targetName = path.basename(brokenLink.target);

    // Track pattern frequency
    if (!patterns.has(targetName)) {
      patterns.set(targetName, {
        target: targetName,
        sources: [],
        count: 0
      });
    }

    const pattern = patterns.get(targetName);
    pattern.sources.push({
      file: fileData.file,
      line: brokenLink.line,
      fullTarget: brokenLink.target,
      attempted: brokenLink.attempted
    });
    pattern.count++;
  });
});

// Sort patterns by frequency
const sortedPatterns = Array.from(patterns.values())
  .sort((a, b) => b.count - a.count);

// Generate fix suggestions
const report = [];
report.push('# Link Fix Suggestions\n');
report.push(`Generated: ${new Date().toISOString()}\n\n`);
report.push('## Overview\n\n');
report.push(`Total broken links: ${linkGraph.files.reduce((sum, f) => sum + f.links.broken.length, 0)}\n`);
report.push(`Unique missing targets: ${sortedPatterns.length}\n\n`);

// Most common broken links
report.push('## Most Common Broken Links\n\n');
sortedPatterns.slice(0, 20).forEach((pattern, idx) => {
  report.push(`### ${idx + 1}. \`${pattern.target}\` (${pattern.count} occurrences)\n\n`);

  // Search for potential matches in existing files
  const potentialMatches = findPotentialMatches(pattern.target, linkGraph.files);

  if (potentialMatches.length > 0) {
    report.push('**Potential fixes:**\n\n');
    potentialMatches.forEach(match => {
      report.push(`- Replace with: \`${match.file}\` (similarity: ${match.score.toFixed(2)})\n`);
    });
    report.push('\n');
  } else {
    report.push('**Recommendation:** Create missing file or remove references\n\n');
  }

  report.push('**Referenced by:**\n\n');
  pattern.sources.slice(0, 5).forEach(src => {
    report.push(`- \`${src.file}\` (line ${src.line})\n`);
  });
  if (pattern.sources.length > 5) {
    report.push(`- ... and ${pattern.sources.length - 5} more\n`);
  }
  report.push('\n');
});

// Missing directories
report.push('## Missing Directory Structures\n\n');
const missingDirs = new Set();
linkGraph.files.forEach(fileData => {
  fileData.links.broken.forEach(brokenLink => {
    const dir = path.dirname(brokenLink.attempted);
    missingDirs.add(dir);
  });
});

Array.from(missingDirs).slice(0, 10).forEach(dir => {
  const filesExpected = linkGraph.files.reduce((acc, fileData) => {
    const broken = fileData.links.broken.filter(b =>
      path.dirname(b.attempted) === dir
    );
    return acc + broken.length;
  }, 0);

  report.push(`- \`${dir}/\` (${filesExpected} expected files)\n`);
});
report.push('\n');

// Orphaned files that could be linked
report.push('## Orphaned Files - Link Suggestions\n\n');
report.push('These files exist but have no inbound links:\n\n');
linkGraph.graph.orphaned.slice(0, 30).forEach(orphan => {
  const fileData = linkGraph.files.find(f => f.file === orphan.file);
  if (!fileData) return;

  // Suggest files that could link to this
  const basename = path.basename(orphan.file, '.md');
  const dirname = path.dirname(orphan.file);

  report.push(`### \`${orphan.file}\`\n\n`);
  report.push('**Could be linked from:**\n');

  // Find related index files
  const potentialLinkers = linkGraph.files.filter(f => {
    const sameDirIndex = path.dirname(f.file) === dirname &&
                        (f.file.endsWith('index.md') || f.file.endsWith('readme.md'));
    const parentDirIndex = path.dirname(dirname) === path.dirname(f.file) &&
                          (f.file.endsWith('index.md') || f.file.endsWith('readme.md'));
    return sameDirIndex || parentDirIndex;
  });

  if (potentialLinkers.length > 0) {
    potentialLinkers.forEach(linker => {
      report.push(`- \`${linker.file}\` (navigation/index file)\n`);
    });
  } else {
    report.push('- No obvious parent index file found\n');
  }
  report.push('\n');
});

// Automated fix scripts
report.push('## Automated Fix Scripts\n\n');
report.push('### Create Missing Directories\n\n');
report.push('```bash\n');
report.push('cd /home/devuser/workspace/project/docs\n');
Array.from(missingDirs).slice(0, 10).forEach(dir => {
  report.push(`mkdir -p "${dir}"\n`);
});
report.push('```\n\n');

// Find-replace suggestions
report.push('### Global Find-Replace Suggestions\n\n');
const replacePatterns = [];

sortedPatterns.slice(0, 10).forEach(pattern => {
  const matches = findPotentialMatches(pattern.target, linkGraph.files);
  if (matches.length > 0 && matches[0].score > 0.7) {
    replacePatterns.push({
      from: pattern.target,
      to: matches[0].file,
      confidence: matches[0].score,
      occurrences: pattern.count
    });
  }
});

if (replacePatterns.length > 0) {
  report.push('High-confidence replacements:\n\n');
  replacePatterns.forEach(rp => {
    report.push(`- Replace \`${rp.from}\` with \`${rp.to}\` (${rp.occurrences} files, confidence: ${(rp.confidence * 100).toFixed(0)}%)\n`);
  });
  report.push('\n');
}

// Invalid anchors
report.push('## Invalid Anchor Links\n\n');
const invalidAnchors = linkGraph.anchorValidation.filter(v => !v.valid);
const anchorsByFile = new Map();

invalidAnchors.forEach(anchor => {
  if (!anchorsByFile.has(anchor.target)) {
    anchorsByFile.set(anchor.target, []);
  }
  anchorsByFile.get(anchor.target).push(anchor);
});

Array.from(anchorsByFile.entries()).forEach(([targetFile, anchors]) => {
  report.push(`### \`${targetFile}\`\n\n`);
  report.push('Missing anchors:\n\n');
  anchors.forEach(a => {
    report.push(`- \`#${a.anchor}\` (referenced from \`${a.source}\` line ${a.line})\n`);
  });
  report.push('\n');
});

fs.writeFileSync(outputPath, report.join(''));
console.log(`\n✓ Fix suggestions generated: ${outputPath}`);

// Helper function to find potential matches
function findPotentialMatches(targetName, allFiles) {
  const matches = [];
  const targetBase = path.basename(targetName, '.md').toLowerCase();

  allFiles.forEach(fileData => {
    const fileBase = path.basename(fileData.file, '.md').toLowerCase();
    const score = calculateSimilarity(targetBase, fileBase);

    if (score > 0.5) {
      matches.push({
        file: fileData.file,
        score
      });
    }
  });

  return matches.sort((a, b) => b.score - a.score).slice(0, 3);
}

// Simple string similarity (Levenshtein-based)
function calculateSimilarity(str1, str2) {
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;

  if (longer.length === 0) return 1.0;

  // Exact match bonus
  if (str1 === str2) return 1.0;

  // Contains bonus
  if (longer.includes(shorter)) return 0.8;

  // Levenshtein distance
  const distance = levenshteinDistance(str1, str2);
  return (longer.length - distance) / longer.length;
}

function levenshteinDistance(str1, str2) {
  const matrix = [];

  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[str2.length][str1.length];
}
