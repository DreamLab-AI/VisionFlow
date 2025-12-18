#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const docsDir = '/home/devuser/workspace/project/docs';
const outputDir = '/home/devuser/workspace/project/docs/working';

// Regex patterns for different link types
const patterns = {
  markdownLink: /\[([^\]]*)\]\(([^)]+)\)/g,
  urlLink: /(https?:\/\/[^\s)]+)/g,
  anchorLink: /\[([^\]]*)\]\(([^)]*#[^)]+)\)/g,
  wikiLink: /\[\[([^\]]+)\]\]/g,
  headingAnchor: /^#+\s+(.+)$/gm
};

function findAllMarkdownFiles(dir) {
  let results = [];
  const items = fs.readdirSync(dir);

  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      results = results.concat(findAllMarkdownFiles(fullPath));
    } else if (item.endsWith('.md')) {
      results.push(fullPath);
    }
  }

  return results;
}

function extractHeadingAnchors(content) {
  const anchors = [];
  let match;
  const headingRegex = /^#+\s+(.+)$/gm;

  while ((match = headingRegex.exec(content)) !== null) {
    const heading = match[1].trim();
    // Convert heading to anchor format (lowercase, hyphens, no special chars)
    const anchor = heading
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-');
    anchors.push({ heading, anchor });
  }

  return anchors;
}

function analyzeLinks(filePath, content, allFiles) {
  const relPath = path.relative(docsDir, filePath);
  const links = {
    internal: [],
    external: [],
    anchors: [],
    broken: [],
    wiki: []
  };

  const fileDir = path.dirname(filePath);
  const headingAnchors = extractHeadingAnchors(content);

  // Extract all markdown links
  let match;
  const markdownLinkRegex = /\[([^\]]*)\]\(([^)]+)\)/g;

  while ((match = markdownLinkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const linkTarget = match[2];
    const lineNumber = content.substring(0, match.index).split('\n').length;
    const context = getContext(content, match.index);

    // Check if external URL
    if (linkTarget.startsWith('http://') || linkTarget.startsWith('https://')) {
      links.external.push({
        text: linkText,
        url: linkTarget,
        line: lineNumber,
        context
      });
      continue;
    }

    // Check if anchor link
    if (linkTarget.includes('#')) {
      const [targetFile, anchor] = linkTarget.split('#');
      const resolvedPath = targetFile
        ? path.resolve(fileDir, targetFile)
        : filePath;

      links.anchors.push({
        text: linkText,
        targetFile: targetFile || relPath,
        anchor: anchor,
        resolved: path.relative(docsDir, resolvedPath),
        line: lineNumber,
        context
      });
      continue;
    }

    // Internal file link
    const targetPath = path.resolve(fileDir, linkTarget);
    const exists = fs.existsSync(targetPath);

    if (exists) {
      links.internal.push({
        text: linkText,
        target: linkTarget,
        resolved: path.relative(docsDir, targetPath),
        line: lineNumber,
        context,
        status: 'valid'
      });
    } else {
      links.broken.push({
        text: linkText,
        target: linkTarget,
        attempted: path.relative(docsDir, targetPath),
        line: lineNumber,
        context,
        reason: 'file_not_found'
      });
    }
  }

  // Extract wiki-style links
  const wikiLinkRegex = /\[\[([^\]]+)\]\]/g;
  while ((match = wikiLinkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const lineNumber = content.substring(0, match.index).split('\n').length;
    const context = getContext(content, match.index);

    links.wiki.push({
      text: linkText,
      line: lineNumber,
      context
    });
  }

  return {
    file: relPath,
    absolutePath: filePath,
    headingAnchors,
    links,
    stats: {
      internal: links.internal.length,
      external: links.external.length,
      anchors: links.anchors.length,
      broken: links.broken.length,
      wiki: links.wiki.length,
      total: links.internal.length + links.external.length + links.anchors.length + links.broken.length + links.wiki.length
    }
  };
}

function getContext(content, index, contextLength = 100) {
  const start = Math.max(0, index - contextLength);
  const end = Math.min(content.length, index + contextLength);
  return content.substring(start, end).replace(/\n/g, ' ').trim();
}

function buildLinkGraph(analyses) {
  const graph = {
    nodes: [],
    edges: [],
    orphaned: [],
    isolated: [],
    anchorTargets: new Map()
  };

  const fileSet = new Set(analyses.map(a => a.file));
  const inboundLinks = new Map();
  const outboundLinks = new Map();

  // Initialize counts
  analyses.forEach(analysis => {
    inboundLinks.set(analysis.file, 0);
    outboundLinks.set(analysis.file, analysis.stats.total);
    graph.nodes.push({
      file: analysis.file,
      anchors: analysis.headingAnchors.map(h => h.anchor)
    });
  });

  // Build edges and count inbound links
  analyses.forEach(analysis => {
    const source = analysis.file;

    // Process internal links
    analysis.links.internal.forEach(link => {
      graph.edges.push({
        source,
        target: link.resolved,
        type: 'internal',
        text: link.text,
        line: link.line
      });
      inboundLinks.set(link.resolved, (inboundLinks.get(link.resolved) || 0) + 1);
    });

    // Process anchor links
    analysis.links.anchors.forEach(link => {
      graph.edges.push({
        source,
        target: link.resolved,
        anchor: link.anchor,
        type: 'anchor',
        text: link.text,
        line: link.line
      });

      if (link.resolved !== source) {
        inboundLinks.set(link.resolved, (inboundLinks.get(link.resolved) || 0) + 1);
      }
    });
  });

  // Find orphaned (no inbound links) and isolated (no outbound links)
  analyses.forEach(analysis => {
    const file = analysis.file;
    if (inboundLinks.get(file) === 0) {
      graph.orphaned.push({
        file,
        outbound: outboundLinks.get(file)
      });
    }
    if (outboundLinks.get(file) === 0) {
      graph.isolated.push({
        file,
        inbound: inboundLinks.get(file)
      });
    }
  });

  return { graph, inboundLinks, outboundLinks };
}

function validateAnchors(analyses) {
  const anchorValidation = [];
  const fileAnchors = new Map();

  // Build map of available anchors per file
  analyses.forEach(analysis => {
    fileAnchors.set(analysis.file, new Set(analysis.headingAnchors.map(h => h.anchor)));
  });

  // Validate anchor links
  analyses.forEach(analysis => {
    analysis.links.anchors.forEach(link => {
      const targetAnchors = fileAnchors.get(link.resolved);
      const valid = targetAnchors && targetAnchors.has(link.anchor);

      anchorValidation.push({
        source: analysis.file,
        target: link.resolved,
        anchor: link.anchor,
        valid,
        line: link.line,
        text: link.text
      });
    });
  });

  return anchorValidation;
}

function generateReport(analyses, graph, inboundLinks, outboundLinks, anchorValidation) {
  const report = [];

  report.push('# Complete Link Validation Report\n');
  report.push(`Generated: ${new Date().toISOString()}\n\n`);

  // Summary statistics
  const totalFiles = analyses.length;
  const totalLinks = analyses.reduce((sum, a) => sum + a.stats.total, 0);
  const totalBroken = analyses.reduce((sum, a) => sum + a.stats.broken, 0);
  const totalExternal = analyses.reduce((sum, a) => sum + a.stats.external, 0);
  const totalInternal = analyses.reduce((sum, a) => sum + a.stats.internal, 0);
  const totalAnchors = analyses.reduce((sum, a) => sum + a.stats.anchors, 0);
  const invalidAnchors = anchorValidation.filter(v => !v.valid).length;

  report.push('## Summary Statistics\n\n');
  report.push(`- **Total Files Analyzed**: ${totalFiles}\n`);
  report.push(`- **Total Links Found**: ${totalLinks}\n`);
  report.push(`  - Internal Links: ${totalInternal}\n`);
  report.push(`  - Anchor Links: ${totalAnchors}\n`);
  report.push(`  - External URLs: ${totalExternal}\n`);
  report.push(`  - Broken Links: ${totalBroken}\n`);
  report.push(`  - Invalid Anchors: ${invalidAnchors}\n`);
  report.push(`- **Orphaned Files** (no inbound links): ${graph.orphaned.length}\n`);
  report.push(`- **Isolated Files** (no outbound links): ${graph.isolated.length}\n\n`);

  // Broken links
  if (totalBroken > 0) {
    report.push('## Broken Links\n\n');
    analyses.forEach(analysis => {
      if (analysis.links.broken.length > 0) {
        report.push(`### ${analysis.file}\n\n`);
        analysis.links.broken.forEach(link => {
          report.push(`- **Line ${link.line}**: \`[${link.text}](${link.target})\`\n`);
          report.push(`  - Attempted: \`${link.attempted}\`\n`);
          report.push(`  - Reason: ${link.reason}\n`);
        });
        report.push('\n');
      }
    });
  }

  // Invalid anchor links
  const invalidAnchorLinks = anchorValidation.filter(v => !v.valid);
  if (invalidAnchorLinks.length > 0) {
    report.push('## Invalid Anchor Links\n\n');
    invalidAnchorLinks.forEach(v => {
      report.push(`- **${v.source}** (line ${v.line})\n`);
      report.push(`  - Target: \`${v.target}#${v.anchor}\`\n`);
      report.push(`  - Link text: "${v.text}"\n`);
      report.push(`  - Issue: Anchor not found in target file\n\n`);
    });
  }

  // Orphaned files
  if (graph.orphaned.length > 0) {
    report.push('## Orphaned Files (No Inbound Links)\n\n');
    report.push('These files have no other files linking to them:\n\n');
    graph.orphaned.forEach(item => {
      report.push(`- \`${item.file}\` (${item.outbound} outbound links)\n`);
    });
    report.push('\n');
  }

  // Isolated files
  if (graph.isolated.length > 0) {
    report.push('## Isolated Files (No Outbound Links)\n\n');
    report.push('These files don\'t link to any other files:\n\n');
    graph.isolated.forEach(item => {
      report.push(`- \`${item.file}\` (${item.inbound} inbound links)\n`);
    });
    report.push('\n');
  }

  // Link density analysis
  report.push('## Link Density Analysis\n\n');
  report.push('Files with most inbound links (top 20):\n\n');
  const sortedInbound = Array.from(inboundLinks.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  sortedInbound.forEach(([file, count]) => {
    report.push(`- \`${file}\`: ${count} inbound links\n`);
  });
  report.push('\n');

  report.push('Files with most outbound links (top 20):\n\n');
  const sortedOutbound = Array.from(outboundLinks.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20);
  sortedOutbound.forEach(([file, count]) => {
    report.push(`- \`${file}\`: ${count} outbound links\n`);
  });
  report.push('\n');

  // External URLs analysis
  const externalUrls = new Map();
  analyses.forEach(analysis => {
    analysis.links.external.forEach(link => {
      const domain = new URL(link.url).hostname;
      if (!externalUrls.has(domain)) {
        externalUrls.set(domain, []);
      }
      externalUrls.get(domain).push({
        file: analysis.file,
        url: link.url,
        line: link.line
      });
    });
  });

  if (externalUrls.size > 0) {
    report.push('## External URL Domains\n\n');
    const sortedDomains = Array.from(externalUrls.entries())
      .sort((a, b) => b[1].length - a[1].length);

    sortedDomains.forEach(([domain, links]) => {
      report.push(`### ${domain} (${links.length} links)\n\n`);
      links.slice(0, 10).forEach(link => {
        report.push(`- \`${link.file}\` (line ${link.line}): ${link.url}\n`);
      });
      if (links.length > 10) {
        report.push(`- ... and ${links.length - 10} more\n`);
      }
      report.push('\n');
    });
  }

  // Bidirectional link opportunities
  report.push('## Bidirectional Link Opportunities\n\n');
  report.push('Files that link to each other (strong relationships):\n\n');
  const bidirectional = new Set();
  graph.edges.forEach(edge => {
    const reverse = graph.edges.find(e =>
      e.source === edge.target && e.target === edge.source
    );
    if (reverse) {
      const pair = [edge.source, edge.target].sort().join(' <-> ');
      bidirectional.add(pair);
    }
  });

  Array.from(bidirectional).slice(0, 30).forEach(pair => {
    report.push(`- ${pair}\n`);
  });
  if (bidirectional.size > 30) {
    report.push(`- ... and ${bidirectional.size - 30} more\n`);
  }
  report.push('\n');

  return report.join('');
}

// Main execution
console.log('Finding markdown files...');
const files = findAllMarkdownFiles(docsDir);
console.log(`Found ${files.length} markdown files`);

console.log('Analyzing links...');
const analyses = files.map(file => {
  const content = fs.readFileSync(file, 'utf8');
  return analyzeLinks(file, content, files);
});

console.log('Building link graph...');
const { graph, inboundLinks, outboundLinks } = buildLinkGraph(analyses);

console.log('Validating anchors...');
const anchorValidation = validateAnchors(analyses);

console.log('Generating reports...');
const report = generateReport(analyses, graph, inboundLinks, outboundLinks, anchorValidation);

// Write JSON output
const jsonOutput = {
  metadata: {
    generated: new Date().toISOString(),
    totalFiles: files.length,
    totalLinks: analyses.reduce((sum, a) => sum + a.stats.total, 0)
  },
  files: analyses,
  graph: {
    nodes: graph.nodes,
    edges: graph.edges,
    orphaned: graph.orphaned,
    isolated: graph.isolated
  },
  anchorValidation,
  statistics: {
    inboundLinks: Object.fromEntries(inboundLinks),
    outboundLinks: Object.fromEntries(outboundLinks)
  }
};

fs.writeFileSync(
  path.join(outputDir, 'complete-link-graph.json'),
  JSON.stringify(jsonOutput, null, 2)
);

fs.writeFileSync(
  path.join(outputDir, 'link-validation-report.md'),
  report
);

console.log('\n✓ Analysis complete!');
console.log(`  JSON output: ${path.join(outputDir, 'complete-link-graph.json')}`);
console.log(`  Report: ${path.join(outputDir, 'link-validation-report.md')}`);
console.log(`\nSummary:`);
console.log(`  Files analyzed: ${files.length}`);
console.log(`  Total links: ${analyses.reduce((sum, a) => sum + a.stats.total, 0)}`);
console.log(`  Broken links: ${analyses.reduce((sum, a) => sum + a.stats.broken, 0)}`);
console.log(`  Orphaned files: ${graph.orphaned.length}`);
console.log(`  Isolated files: ${graph.isolated.length}`);
