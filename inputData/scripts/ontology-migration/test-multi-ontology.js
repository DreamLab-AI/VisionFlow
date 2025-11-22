#!/usr/bin/env node

/**
 * Test Suite for Multi-Ontology Processing
 *
 * Tests the 6-domain federated ontology architecture:
 * - Domain detection
 * - Extension property generation
 * - Namespace validation
 * - Sub-domain classification
 * - Cross-domain link analysis
 */

const assert = require('assert');
const domainDetector = require('./domain-detector');
const OntologyGenerator = require('./generator');
const OntologyValidator = require('./validator');

class MultiOntologyTests {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.tests = [];
  }

  test(name, fn) {
    this.tests.push({ name, fn });
  }

  async run() {
    console.log('\n🧪 Multi-Ontology Test Suite\n');
    console.log('='.repeat(80));

    for (const { name, fn } of this.tests) {
      try {
        await fn();
        console.log(`✅ ${name}`);
        this.passed++;
      } catch (error) {
        console.log(`❌ ${name}`);
        console.log(`   Error: ${error.message}`);
        this.failed++;
      }
    }

    console.log('='.repeat(80));
    console.log(`\nResults: ${this.passed} passed, ${this.failed} failed`);
    console.log(`Total: ${this.passed + this.failed} tests\n`);

    return this.failed === 0;
  }
}

// Initialize tests
const tests = new MultiOntologyTests();

// Test 1: Domain detector recognizes all 6 domains
tests.test('Domain detector recognizes all 6 domains', () => {
  const domains = domainDetector.getAllDomains();
  assert.strictEqual(domains.length, 6, 'Should have 6 domains');
  assert(domains.includes('ai'), 'Should include ai');
  assert(domains.includes('mv'), 'Should include mv');
  assert(domains.includes('tc'), 'Should include tc');
  assert(domains.includes('rb'), 'Should include rb');
  assert(domains.includes('dt'), 'Should include dt');
  assert(domains.includes('bc'), 'Should include bc');
});

// Test 2: Domain detection from filename
tests.test('Detect domain from filename', () => {
  assert.strictEqual(domainDetector.detectFromPath('AI-001-machine-learning.md'), 'ai');
  assert.strictEqual(domainDetector.detectFromPath('BC-042-smart-contracts.md'), 'bc');
  assert.strictEqual(domainDetector.detectFromPath('RB-010-autonomous-robot.md'), 'rb');
  assert.strictEqual(domainDetector.detectFromPath('MV-100-virtual-world.md'), 'mv');
  assert.strictEqual(domainDetector.detectFromPath('TC-005-remote-collaboration.md'), 'tc');
  assert.strictEqual(domainDetector.detectFromPath('DT-030-emerging-tech.md'), 'dt');
});

// Test 3: Domain detection from term-id
tests.test('Detect domain from term-id', () => {
  assert.strictEqual(domainDetector.detectFromTermId('AI-001'), 'ai');
  assert.strictEqual(domainDetector.detectFromTermId('BC-042'), 'bc');
  assert.strictEqual(domainDetector.detectFromTermId('RB-010'), 'rb');
  assert.strictEqual(domainDetector.detectFromTermId('MV-100'), 'mv');
  assert.strictEqual(domainDetector.detectFromTermId('TC-005'), 'tc');
  assert.strictEqual(domainDetector.detectFromTermId('DT-030'), 'dt');
});

// Test 4: Domain detection from namespace
tests.test('Detect domain from namespace', () => {
  assert.strictEqual(domainDetector.detectFromNamespace('ai:'), 'ai');
  assert.strictEqual(domainDetector.detectFromNamespace('bc:'), 'bc');
  assert.strictEqual(domainDetector.detectFromNamespace('rb:'), 'rb');
  assert.strictEqual(domainDetector.detectFromNamespace('mv:'), 'mv');
  assert.strictEqual(domainDetector.detectFromNamespace('tc:'), 'tc');
  assert.strictEqual(domainDetector.detectFromNamespace('dt:'), 'dt');
});

// Test 5: Get domain configuration
tests.test('Get domain configuration for all domains', () => {
  const domains = ['ai', 'mv', 'tc', 'rb', 'dt', 'bc'];

  domains.forEach(domain => {
    const config = domainDetector.getDomainConfig(domain);
    assert(config, `Should have config for ${domain}`);
    assert(config.namespace, `${domain} should have namespace`);
    assert(config.prefix, `${domain} should have prefix`);
    assert(config.requiredProperties, `${domain} should have required properties`);
    assert(config.optionalProperties, `${domain} should have optional properties`);
  });
});

// Test 6: Namespace matches domain
tests.test('Namespace matches domain correctly', () => {
  assert(domainDetector.namespaceMatchesDomain('ai:', 'ai'));
  assert(domainDetector.namespaceMatchesDomain('bc:', 'bc'));
  assert(domainDetector.namespaceMatchesDomain('rb:', 'rb'));
  assert(domainDetector.namespaceMatchesDomain('mv:', 'mv'));
  assert(domainDetector.namespaceMatchesDomain('tc:', 'tc'));
  assert(domainDetector.namespaceMatchesDomain('dt:', 'dt'));

  // Should fail for mismatches
  assert(!domainDetector.namespaceMatchesDomain('mv:', 'rb'));
  assert(!domainDetector.namespaceMatchesDomain('ai:', 'bc'));
});

// Test 7: Domain-specific required properties
tests.test('Each domain has specific required properties', () => {
  const aiConfig = domainDetector.getDomainConfig('ai');
  assert(aiConfig.requiredProperties.includes('algorithm-type'));
  assert(aiConfig.requiredProperties.includes('computational-complexity'));

  const mvConfig = domainDetector.getDomainConfig('mv');
  assert(mvConfig.requiredProperties.includes('immersion-level'));
  assert(mvConfig.requiredProperties.includes('interaction-mode'));

  const tcConfig = domainDetector.getDomainConfig('tc');
  assert(tcConfig.requiredProperties.includes('collaboration-type'));
  assert(tcConfig.requiredProperties.includes('communication-mode'));

  const rbConfig = domainDetector.getDomainConfig('rb');
  assert(rbConfig.requiredProperties.includes('physicality'));
  assert(rbConfig.requiredProperties.includes('autonomy-level'));

  const dtConfig = domainDetector.getDomainConfig('dt');
  assert(dtConfig.requiredProperties.includes('disruption-level'));
  assert(dtConfig.requiredProperties.includes('maturity-stage'));

  const bcConfig = domainDetector.getDomainConfig('bc');
  assert(bcConfig.requiredProperties.includes('consensus-mechanism'));
  assert(bcConfig.requiredProperties.includes('decentralization-level'));
});

// Test 8: Sub-domain classification
tests.test('Sub-domain classification works', () => {
  const aiContent = 'This is about supervised machine learning with neural networks';
  const subDomain = domainDetector.classifySubDomain('ai', aiContent);
  assert(subDomain, 'Should detect sub-domain');

  const mvContent = 'Virtual reality immersive experience in VR headset';
  const mvSub = domainDetector.classifySubDomain('mv', mvContent);
  assert(mvSub, 'Should detect MV sub-domain');
});

// Test 9: Cross-domain bridge recommendations
tests.test('Cross-domain bridges are defined', () => {
  const aiMvBridges = domainDetector.getRecommendedBridges('ai', 'mv');
  assert(aiMvBridges.length > 0, 'Should have AI-MV bridges');

  const aiRbBridges = domainDetector.getRecommendedBridges('ai', 'rb');
  assert(aiRbBridges.length > 0, 'Should have AI-RB bridges');

  const mvTcBridges = domainDetector.getRecommendedBridges('mv', 'tc');
  assert(mvTcBridges.length > 0, 'Should have MV-TC bridges');
});

// Test 10: Generator creates domain-specific blocks
tests.test('Generator adds domain extension properties', () => {
  const generator = new OntologyGenerator();

  const aiParsedData = {
    filename: 'AI-001-test.md',
    filePath: 'AI-001-test.md',
    properties: {
      'term-id': 'AI-001',
      'preferred-term': 'Test AI Term',
      'definition': 'Test definition',
      'source-domain': 'ai',
      'status': 'draft',
      'algorithm-type': 'supervised-learning',
      'computational-complexity': 'O(n^2)'
    },
    namespace: { prefix: 'ai', className: 'TestAiTerm' },
    relationships: {
      isSubclassOf: [],
      hasPart: [],
      requires: [],
      enables: [],
      bridgesTo: [],
      bridgesFrom: []
    },
    content: 'machine learning algorithm'
  };

  const generated = generator.generate(aiParsedData);
  assert(generated.includes('**Domain Extensions**'), 'Should include Domain Extensions section');
  assert(generated.includes('algorithm-type::'), 'Should include AI-specific property');
  assert(generated.includes('computational-complexity::'), 'Should include AI-specific property');
});

// Test 11: Validator checks domain-specific properties
tests.test('Validator checks domain-specific required properties', () => {
  // This test verifies the validation logic structure
  const validator = new OntologyValidator();
  assert(typeof validator.validateDomainProperties === 'function',
    'Validator should have validateDomainProperties method');
});

// Test 12: Content-based domain detection
tests.test('Detect domain from content keywords', () => {
  const aiContent = 'This article discusses neural networks and machine learning algorithms';
  const aiDomain = domainDetector.detectFromContent(aiContent);
  assert.strictEqual(aiDomain, 'ai', 'Should detect AI domain from content');

  const bcContent = 'This explores blockchain technology and cryptocurrency consensus mechanisms';
  const bcDomain = domainDetector.detectFromContent(bcContent);
  assert.strictEqual(bcDomain, 'bc', 'Should detect BC domain from content');

  const rbContent = 'Autonomous robot with sensors and actuators for navigation';
  const rbDomain = domainDetector.detectFromContent(rbContent);
  assert.strictEqual(rbDomain, 'rb', 'Should detect RB domain from content');
});

// Test 13: Validate domain keys
tests.test('Validate domain keys', () => {
  assert(domainDetector.isValidDomain('ai'));
  assert(domainDetector.isValidDomain('mv'));
  assert(domainDetector.isValidDomain('tc'));
  assert(domainDetector.isValidDomain('rb'));
  assert(domainDetector.isValidDomain('dt'));
  assert(domainDetector.isValidDomain('bc'));

  assert(!domainDetector.isValidDomain('invalid'));
  assert(!domainDetector.isValidDomain('xyz'));
});

// Test 14: Get domain display names
tests.test('Get domain display names', () => {
  assert.strictEqual(domainDetector.getDomainName('ai'), 'Artificial Intelligence');
  assert.strictEqual(domainDetector.getDomainName('mv'), 'Metaverse');
  assert.strictEqual(domainDetector.getDomainName('tc'), 'Telecollaboration');
  assert.strictEqual(domainDetector.getDomainName('rb'), 'Robotics');
  assert.strictEqual(domainDetector.getDomainName('dt'), 'Disruptive Technologies');
  assert.strictEqual(domainDetector.getDomainName('bc'), 'Blockchain');
});

// Test 15: Domain-specific valid physicalities
tests.test('Domain-specific valid physicalities', () => {
  const aiConfig = domainDetector.getDomainConfig('ai');
  assert(aiConfig.validPhysicalities.includes('ConceptualEntity'));
  assert(aiConfig.validPhysicalities.includes('AbstractEntity'));

  const rbConfig = domainDetector.getDomainConfig('rb');
  assert(rbConfig.validPhysicalities.includes('PhysicalEntity'));
  assert(rbConfig.validPhysicalities.includes('HybridEntity'));
});

// Test 16: Domain-specific valid roles
tests.test('Domain-specific valid roles', () => {
  const aiConfig = domainDetector.getDomainConfig('ai');
  assert(aiConfig.validRoles.includes('Process'));
  assert(aiConfig.validRoles.includes('Agent'));

  const mvConfig = domainDetector.getDomainConfig('mv');
  assert(mvConfig.validRoles.includes('Object'));
  assert(mvConfig.validRoles.includes('Concept'));
});

// Run all tests
if (require.main === module) {
  tests.run().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test suite failed:', error);
    process.exit(1);
  });
}

module.exports = MultiOntologyTests;
