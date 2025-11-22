#!/usr/bin/env node

/**
 * Domain Detector - Multi-Ontology Domain Detection Utilities
 *
 * Provides utilities for detecting and classifying ontology domains
 * across the 6-domain federated architecture:
 * - ai: Artificial Intelligence
 * - mv: Metaverse
 * - tc: Telecollaboration
 * - rb: Robotics
 * - dt: Disruptive Technologies
 * - bc: Blockchain
 */

const path = require('path');
const domainConfig = require('./domain-config.json');

class DomainDetector {
  constructor() {
    this.domains = domainConfig.domains;
    this.domainKeys = Object.keys(this.domains);
  }

  /**
   * Detect domain from file path
   */
  detectFromPath(filePath) {
    const filename = path.basename(filePath);

    // Check each domain prefix
    for (const [key, domain] of Object.entries(this.domains)) {
      const prefix = domain.prefix.toLowerCase();
      const namespace = domain.namespace.replace(':', '').toLowerCase();

      if (filename.toLowerCase().startsWith(prefix) ||
          filename.toLowerCase().startsWith(namespace + '-')) {
        return key;
      }
    }

    // Special case: files containing domain keywords
    const filenameLower = filename.toLowerCase();
    if (filenameLower.includes('robot') || filenameLower.includes('autonomous')) return 'rb';
    if (filenameLower.includes('blockchain') || filenameLower.includes('crypto')) return 'bc';
    if (filenameLower.includes('metaverse') || filenameLower.includes('virtual-world')) return 'mv';
    if (filenameLower.includes('collaboration') || filenameLower.includes('remote-work')) return 'tc';
    if (filenameLower.includes('disruptive') || filenameLower.includes('emerging-tech')) return 'dt';
    if (filenameLower.includes('machine-learning') || filenameLower.includes('neural')) return 'ai';

    return 'mv'; // Default to metaverse for backward compatibility
  }

  /**
   * Detect domain from term-id prefix
   */
  detectFromTermId(termId) {
    if (!termId) return null;

    for (const [key, domain] of Object.entries(this.domains)) {
      if (termId.startsWith(domain.prefix)) {
        return key;
      }
    }

    return null;
  }

  /**
   * Detect domain from namespace
   */
  detectFromNamespace(namespace) {
    if (!namespace) return null;

    // Remove trailing colon if present
    const ns = namespace.replace(':', '');

    for (const [key, domain] of Object.entries(this.domains)) {
      if (domain.namespace.replace(':', '') === ns) {
        return key;
      }
    }

    return null;
  }

  /**
   * Detect domain from content analysis
   */
  detectFromContent(content) {
    const domainScores = {};

    for (const key of this.domainKeys) {
      domainScores[key] = 0;
    }

    // Check for term-id
    const termIdMatch = content.match(/term-id::\s*(AI-|BC-|RB-|MV-|TC-|DT-)/i);
    if (termIdMatch) {
      const detected = this.detectFromTermId(termIdMatch[1]);
      if (detected) return detected;
    }

    // Check for source-domain
    const sourceDomainMatch = content.match(/source-domain::\s*(\w+)/);
    if (sourceDomainMatch) {
      const domain = sourceDomainMatch[1].toLowerCase();
      if (this.domainKeys.includes(domain)) {
        return domain;
      }
    }

    // Check for namespace in owl:class
    const owlClassMatch = content.match(/owl:class::\s*(\w+):/);
    if (owlClassMatch) {
      const detected = this.detectFromNamespace(owlClassMatch[1]);
      if (detected) return detected;
    }

    // Keyword-based scoring
    const keywords = {
      ai: ['algorithm', 'neural', 'learning', 'intelligence', 'model', 'training', 'inference'],
      mv: ['metaverse', 'virtual', 'immersive', 'avatar', 'world', 'xr', 'spatial'],
      tc: ['collaboration', 'remote', 'distributed', 'telepresence', 'communication', 'coordination'],
      rb: ['robot', 'autonomous', 'sensor', 'actuator', 'navigation', 'manipulation', 'control'],
      dt: ['disruptive', 'emerging', 'transformative', 'innovation', 'exponential', 'breakthrough'],
      bc: ['blockchain', 'crypto', 'consensus', 'decentralized', 'ledger', 'smart contract', 'token']
    };

    const contentLower = content.toLowerCase();
    for (const [domain, words] of Object.entries(keywords)) {
      for (const word of words) {
        if (contentLower.includes(word)) {
          domainScores[domain]++;
        }
      }
    }

    // Return domain with highest score (if score > 0)
    const maxScore = Math.max(...Object.values(domainScores));
    if (maxScore > 0) {
      return Object.entries(domainScores)
        .find(([_, score]) => score === maxScore)[0];
    }

    return null;
  }

  /**
   * Comprehensive domain detection (uses all methods)
   */
  detect(filePath, content = null) {
    // Priority 1: Detect from content (most reliable)
    if (content) {
      const fromContent = this.detectFromContent(content);
      if (fromContent) return fromContent;
    }

    // Priority 2: Detect from filename
    const fromPath = this.detectFromPath(filePath);
    if (fromPath !== 'mv') return fromPath; // If not default, trust it

    // Priority 3: Default to metaverse
    return 'mv';
  }

  /**
   * Classify sub-domain from content
   */
  classifySubDomain(domain, content) {
    if (!domain || !content) return null;

    const subDomains = domainConfig.subDomainPatterns[domain];
    if (!subDomains) return null;

    const contentLower = content.toLowerCase();
    const scores = {};

    for (const [subDomain, keywords] of Object.entries(subDomains)) {
      scores[subDomain] = 0;
      for (const keyword of keywords) {
        if (contentLower.includes(keyword)) {
          scores[subDomain]++;
        }
      }
    }

    const maxScore = Math.max(...Object.values(scores));
    if (maxScore > 0) {
      return Object.entries(scores)
        .find(([_, score]) => score === maxScore)[0];
    }

    return null;
  }

  /**
   * Get domain configuration
   */
  getDomainConfig(domain) {
    return this.domains[domain] || null;
  }

  /**
   * Get namespace for domain
   */
  getNamespace(domain) {
    const config = this.getDomainConfig(domain);
    return config ? config.namespace : null;
  }

  /**
   * Get prefix for domain
   */
  getPrefix(domain) {
    const config = this.getDomainConfig(domain);
    return config ? config.prefix : null;
  }

  /**
   * Validate domain key
   */
  isValidDomain(domain) {
    return this.domainKeys.includes(domain);
  }

  /**
   * Get all domain keys
   */
  getAllDomains() {
    return this.domainKeys;
  }

  /**
   * Get domain display name
   */
  getDomainName(domain) {
    const config = this.getDomainConfig(domain);
    return config ? config.name : null;
  }

  /**
   * Check if namespace matches domain
   */
  namespaceMatchesDomain(namespace, domain) {
    const expectedNamespace = this.getNamespace(domain);
    if (!expectedNamespace) return false;

    const ns = namespace.replace(':', '');
    const expected = expectedNamespace.replace(':', '');

    return ns === expected;
  }

  /**
   * Detect cross-domain references
   */
  detectCrossDomainLinks(content, sourceDomain) {
    const links = [];

    // Find all wiki-style links
    const linkMatches = content.matchAll(/\[\[([^\]]+)\]\]/g);

    for (const match of linkMatches) {
      const linkText = match[1];

      // Try to determine domain of linked term
      for (const [domain, config] of Object.entries(this.domains)) {
        if (domain === sourceDomain) continue; // Skip same domain

        if (linkText.startsWith(config.prefix) ||
            linkText.toLowerCase().includes(domain)) {
          links.push({
            target: linkText,
            targetDomain: domain,
            type: 'cross-domain'
          });
        }
      }
    }

    return links;
  }

  /**
   * Get recommended cross-domain bridges
   */
  getRecommendedBridges(domain1, domain2) {
    const key1 = `${domain1}-${domain2}`;
    const key2 = `${domain2}-${domain1}`;

    return domainConfig.crossDomainBridges[key1] ||
           domainConfig.crossDomainBridges[key2] ||
           [];
  }
}

// Export singleton instance
module.exports = new DomainDetector();

// CLI usage
if (require.main === module) {
  const detector = new DomainDetector();
  const command = process.argv[2];

  if (command === 'list') {
    console.log('Available domains:');
    detector.getAllDomains().forEach(domain => {
      const config = detector.getDomainConfig(domain);
      console.log(`  ${domain.padEnd(5)} ${config.namespace.padEnd(5)} ${config.name}`);
    });
  } else if (command === 'detect') {
    const filePath = process.argv[3];
    if (!filePath) {
      console.error('Usage: node domain-detector.js detect <file-path>');
      process.exit(1);
    }

    const fs = require('fs');
    const content = fs.existsSync(filePath) ? fs.readFileSync(filePath, 'utf-8') : null;
    const domain = detector.detect(filePath, content);

    console.log(`Detected domain: ${domain}`);
    console.log(`Namespace: ${detector.getNamespace(domain)}`);
    console.log(`Prefix: ${detector.getPrefix(domain)}`);

    if (content) {
      const subDomain = detector.classifySubDomain(domain, content);
      if (subDomain) {
        console.log(`Sub-domain: ${subDomain}`);
      }
    }
  } else {
    console.log('Usage:');
    console.log('  node domain-detector.js list');
    console.log('  node domain-detector.js detect <file-path>');
  }
}
