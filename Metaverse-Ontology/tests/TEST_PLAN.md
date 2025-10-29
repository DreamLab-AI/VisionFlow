# Comprehensive Testing Strategy
## Ontology Storage Architecture Validation

**Version**: 1.0.0
**Last Updated**: 2025-10-29
**Owner**: QA Engineering Team

---

## Executive Summary

This test plan validates the complete data flow: **GitHub Markdown → Database → OWL Extraction → Reasoning**

### Success Criteria
- ✅ All 1,297 ObjectSomeValuesFrom restrictions preserved
- ✅ Zero semantic loss through the pipeline
- ✅ 15x speed improvement on re-sync operations
- ✅ < 135s full ontology build time
- ✅ 100% Actor system coordination reliability

---

## 1. Happy Path Test Suite

### 1.1 End-to-End Data Flow Test

**Test ID**: `E2E-HAPPY-001`
**Priority**: CRITICAL
**Estimated Duration**: 2-3 minutes

#### Test Specification

```gherkin
Feature: Complete Ontology Pipeline Validation
  As a system architect
  I want to validate the entire data flow
  So that I can ensure zero semantic loss

Scenario: Process complete ontology from GitHub to reasoning
  Given a fresh database with schema migrations applied
  And GitHub repository contains 988 markdown files with OWL blocks
  And OwlValidatorService is initialized

  When the GitHubSyncActor processes all markdown files
  Then all 988 files should be stored in `ontology_classes` table
  And each record should have populated markdown_content
  And each record should have valid sha1_hash
  And content_hash should differ when content changes

  When OntologyExtractorActor extracts OWL from database
  Then 1,297 ObjectSomeValuesFrom restrictions should be extracted
  And extracted OWL should parse without errors
  And all class hierarchies should be preserved

  When whelk-rs reasoning engine processes the ontology
  Then classification should complete successfully
  And all inferred relationships should be valid
  And no inconsistencies should be detected
```

#### Code Implementation

```typescript
// tests/e2e/happy-path.test.ts

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { Database } from '../../src/database/Database';
import { GitHubSyncService } from '../../src/services/GitHubSyncService';
import { OntologyExtractorService } from '../../src/services/OntologyExtractorService';
import { OwlValidatorService } from '../../src/services/OwlValidatorService';
import { WhelkReasoningService } from '../../src/services/WhelkReasoningService';

describe('E2E-HAPPY-001: Complete Ontology Pipeline', () => {
  let db: Database;
  let githubSync: GitHubSyncService;
  let extractor: OntologyExtractorService;
  let validator: OwlValidatorService;
  let reasoner: WhelkReasoningService;

  const EXPECTED_CLASSES = 988;
  const EXPECTED_RESTRICTIONS = 1297;
  const MAX_PIPELINE_TIME = 135000; // 135 seconds

  beforeAll(async () => {
    // Initialize services
    db = new Database({ connectionString: process.env.TEST_DB_URL });
    await db.migrate();

    githubSync = new GitHubSyncService(db, {
      owner: 'YOUR_ORG',
      repo: 'Metaverse-Ontology',
      token: process.env.GITHUB_TOKEN
    });

    extractor = new OntologyExtractorService(db);
    validator = new OwlValidatorService();
    reasoner = new WhelkReasoningService();
  });

  afterAll(async () => {
    await db.close();
  });

  it('should process complete pipeline within time limit', async () => {
    const startTime = Date.now();

    // Step 1: GitHub Sync
    console.log('Starting GitHub sync...');
    const syncResult = await githubSync.syncAllMarkdownFiles();

    expect(syncResult.filesProcessed).toBe(EXPECTED_CLASSES);
    expect(syncResult.errors).toHaveLength(0);

    // Validate database storage
    const storedClasses = await db.query(
      'SELECT class_name, markdown_content, sha1_hash, content_hash FROM ontology_classes'
    );

    expect(storedClasses.rows).toHaveLength(EXPECTED_CLASSES);

    // Verify all required fields populated
    for (const row of storedClasses.rows) {
      expect(row.markdown_content).toBeTruthy();
      expect(row.sha1_hash).toMatch(/^[a-f0-9]{40}$/);
      expect(row.content_hash).toMatch(/^[a-f0-9]{64}$/);
    }

    // Step 2: OWL Extraction
    console.log('Extracting OWL from database...');
    const extractionResult = await extractor.extractAllOntologies();

    expect(extractionResult.classesExtracted).toBe(EXPECTED_CLASSES);
    expect(extractionResult.restrictionsFound).toBe(EXPECTED_RESTRICTIONS);

    // Validate OWL syntax
    const owlContent = extractionResult.combinedOntology;
    const parseResult = validator.parseOwlFunctional(owlContent);

    expect(parseResult.valid).toBe(true);
    expect(parseResult.errors).toHaveLength(0);

    // Step 3: Semantic Validation
    console.log('Validating semantic preservation...');
    const originalRestrictions = await extractOriginalRestrictions(db);
    const extractedRestrictions = parseResult.restrictions;

    expect(extractedRestrictions).toHaveLength(EXPECTED_RESTRICTIONS);

    // Compare restriction by restriction
    const semanticLoss = compareRestrictions(originalRestrictions, extractedRestrictions);
    expect(semanticLoss.missingRestrictions).toHaveLength(0);
    expect(semanticLoss.modifiedRestrictions).toHaveLength(0);

    // Step 4: Reasoning
    console.log('Running reasoning engine...');
    const reasoningResult = await reasoner.classify(owlContent);

    expect(reasoningResult.consistent).toBe(true);
    expect(reasoningResult.inferredAxioms).toBeGreaterThan(0);
    expect(reasoningResult.inconsistentClasses).toHaveLength(0);

    // Timing validation
    const totalTime = Date.now() - startTime;
    console.log(`Total pipeline time: ${totalTime}ms`);
    expect(totalTime).toBeLessThan(MAX_PIPELINE_TIME);
  }, 180000); // 3 minute timeout

  it('should preserve all ObjectSomeValuesFrom restrictions', async () => {
    const query = `
      SELECT class_name, COUNT(*) as restriction_count
      FROM ontology_restrictions
      WHERE restriction_type = 'ObjectSomeValuesFrom'
      GROUP BY class_name
    `;

    const result = await db.query(query);
    const totalRestrictions = result.rows.reduce(
      (sum, row) => sum + parseInt(row.restriction_count),
      0
    );

    expect(totalRestrictions).toBe(EXPECTED_RESTRICTIONS);
  });
});

// Helper functions
async function extractOriginalRestrictions(db: Database) {
  const result = await db.query(`
    SELECT class_name, property, filler
    FROM ontology_restrictions
    WHERE restriction_type = 'ObjectSomeValuesFrom'
    ORDER BY class_name, property
  `);
  return result.rows;
}

function compareRestrictions(original: any[], extracted: any[]) {
  const originalSet = new Set(original.map(r => `${r.class_name}|${r.property}|${r.filler}`));
  const extractedSet = new Set(extracted.map(r => `${r.class_name}|${r.property}|${r.filler}`));

  const missingRestrictions = [...originalSet].filter(x => !extractedSet.has(x));
  const modifiedRestrictions = [...extractedSet].filter(x => !originalSet.has(x));

  return { missingRestrictions, modifiedRestrictions };
}
```

---

## 2. Change Detection Test Suite

### 2.1 Initial Sync Performance Test

**Test ID**: `PERF-CHANGE-001`
**Priority**: HIGH
**Expected Duration**: 125 seconds

```typescript
// tests/performance/change-detection.test.ts

import { describe, it, expect, beforeEach } from '@jest/globals';
import { GitHubSyncService } from '../../src/services/GitHubSyncService';
import { Database } from '../../src/database/Database';

describe('PERF-CHANGE-001: Change Detection Performance', () => {
  let db: Database;
  let syncService: GitHubSyncService;

  beforeEach(async () => {
    db = new Database({ connectionString: process.env.TEST_DB_URL });
    await db.query('TRUNCATE TABLE ontology_classes CASCADE');
    syncService = new GitHubSyncService(db);
  });

  it('should complete initial sync within 125 seconds', async () => {
    const startTime = Date.now();

    const result = await syncService.syncAllMarkdownFiles();

    const duration = Date.now() - startTime;
    console.log(`Initial sync completed in ${duration}ms`);

    expect(duration).toBeLessThan(125000); // 125 seconds
    expect(result.filesProcessed).toBe(988);
    expect(result.newFiles).toBe(988);
    expect(result.updatedFiles).toBe(0);
    expect(result.unchangedFiles).toBe(0);
  });

  it('should complete re-sync with no changes within 8 seconds', async () => {
    // First sync
    await syncService.syncAllMarkdownFiles();

    // Second sync (no changes)
    const startTime = Date.now();
    const result = await syncService.syncAllMarkdownFiles();
    const duration = Date.now() - startTime;

    console.log(`Re-sync with no changes completed in ${duration}ms`);

    expect(duration).toBeLessThan(8000); // 8 seconds (15x faster)
    expect(result.filesProcessed).toBe(988);
    expect(result.newFiles).toBe(0);
    expect(result.updatedFiles).toBe(0);
    expect(result.unchangedFiles).toBe(988);
  });

  it('should detect and process 10 changed files within 12 seconds', async () => {
    // Initial sync
    await syncService.syncAllMarkdownFiles();

    // Simulate 10 file changes in database
    const filesToModify = await db.query(
      'SELECT class_name FROM ontology_classes LIMIT 10'
    );

    for (const file of filesToModify.rows) {
      // Simulate content change by updating markdown_content
      await db.query(
        `UPDATE ontology_classes
         SET markdown_content = markdown_content || '\n<!-- Modified -->'
         WHERE class_name = $1`,
        [file.class_name]
      );
    }

    // Re-sync with changes
    const startTime = Date.now();
    const result = await syncService.syncAllMarkdownFiles();
    const duration = Date.now() - startTime;

    console.log(`Re-sync with 10 changes completed in ${duration}ms`);

    expect(duration).toBeLessThan(12000); // 12 seconds
    expect(result.filesProcessed).toBe(988);
    expect(result.updatedFiles).toBe(10);
    expect(result.unchangedFiles).toBe(978);
  });

  it('should correctly compute SHA1 and content hashes', async () => {
    const testContent = 'Test markdown content with OWL blocks';

    const result = await syncService.processMarkdownFile({
      name: 'TestClass.md',
      path: 'Classes/TestClass.md',
      content: testContent,
      sha: 'abc123' // GitHub SHA1
    });

    // Verify SHA1 matches GitHub
    expect(result.sha1_hash).toBe('abc123');

    // Verify content hash is SHA-256 of content
    const expectedContentHash = await computeSHA256(testContent);
    expect(result.content_hash).toBe(expectedContentHash);

    // Verify change detection works
    const secondResult = await syncService.processMarkdownFile({
      name: 'TestClass.md',
      path: 'Classes/TestClass.md',
      content: testContent, // Same content
      sha: 'abc123'
    });

    expect(secondResult.changed).toBe(false);
  });
});

async function computeSHA256(content: string): Promise<string> {
  const crypto = await import('crypto');
  return crypto.createHash('sha256').update(content).digest('hex');
}
```

### 2.2 SHA1 Hash Collision Test

**Test ID**: `EDGE-HASH-001`
**Priority**: MEDIUM

```typescript
// tests/unit/hash-collision.test.ts

describe('EDGE-HASH-001: Hash Collision Handling', () => {
  it('should handle theoretical SHA1 collision gracefully', async () => {
    const db = new Database({ connectionString: process.env.TEST_DB_URL });

    // Insert first file with specific SHA1
    await db.query(
      `INSERT INTO ontology_classes (class_name, sha1_hash, content_hash, markdown_content)
       VALUES ($1, $2, $3, $4)`,
      ['ClassA', 'collision_sha1', 'hash_a', 'Content A']
    );

    // Attempt to insert different file with same SHA1
    const result = await db.query(
      `INSERT INTO ontology_classes (class_name, sha1_hash, content_hash, markdown_content)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (class_name) DO UPDATE
       SET sha1_hash = EXCLUDED.sha1_hash,
           content_hash = EXCLUDED.content_hash,
           markdown_content = EXCLUDED.markdown_content
       RETURNING *`,
      ['ClassB', 'collision_sha1', 'hash_b', 'Content B']
    );

    expect(result.rows).toHaveLength(1);

    // Verify both entries exist with different content hashes
    const allEntries = await db.query(
      'SELECT * FROM ontology_classes WHERE sha1_hash = $1',
      ['collision_sha1']
    );

    expect(allEntries.rows).toHaveLength(2);
    expect(allEntries.rows[0].content_hash).not.toBe(allEntries.rows[1].content_hash);
  });
});
```

---

## 3. Edge Case Test Suite

### 3.1 Malformed Data Handling

**Test ID**: `EDGE-DATA-001` to `EDGE-DATA-004`
**Priority**: HIGH

```typescript
// tests/unit/edge-cases.test.ts

import { describe, it, expect } from '@jest/globals';
import { OntologyExtractorService } from '../../src/services/OntologyExtractorService';
import { OwlParser } from '../../src/parsers/OwlParser';

describe('Edge Case Test Suite', () => {

  describe('EDGE-DATA-001: Markdown with no OWL blocks', () => {
    it('should handle markdown without OWL gracefully', async () => {
      const markdown = `
# Regular Markdown File

This file contains no OWL blocks, just regular content.

- List item 1
- List item 2
      `.trim();

      const extractor = new OntologyExtractorService();
      const result = await extractor.extractOwlFromMarkdown(markdown);

      expect(result.owlBlocks).toHaveLength(0);
      expect(result.hasOwl).toBe(false);
      expect(result.error).toBeUndefined();
    });
  });

  describe('EDGE-DATA-002: Malformed OWL Functional Syntax', () => {
    it('should detect and report syntax errors', async () => {
      const malformedOwl = `
\`\`\`owl
Class: InvalidClass
  SubClassOf: MissingColon
  ObjectSomeValuesFrom(property value
\`\`\`
      `.trim();

      const parser = new OwlParser();
      const result = parser.parseOwlFunctional(malformedOwl);

      expect(result.valid).toBe(false);
      expect(result.errors).toContainEqual(
        expect.objectContaining({
          type: 'SYNTAX_ERROR',
          message: expect.stringContaining('Unclosed parenthesis')
        })
      );
    });

    it('should provide line numbers for errors', () => {
      const malformedOwl = `
Class: ValidClass
  SubClassOf: ParentClass

Class: InvalidClass
  ObjectSomeValuesFrom(property
      `.trim();

      const parser = new OwlParser();
      const result = parser.parseOwlFunctional(malformedOwl);

      expect(result.errors[0].line).toBe(5);
      expect(result.errors[0].column).toBeDefined();
    });
  });

  describe('EDGE-DATA-003: Missing markdown_content in database', () => {
    it('should handle NULL markdown_content gracefully', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });

      // Insert record with NULL markdown_content
      await db.query(
        `INSERT INTO ontology_classes (class_name, sha1_hash, markdown_content)
         VALUES ($1, $2, NULL)`,
        ['TestClass', 'abc123']
      );

      const extractor = new OntologyExtractorService(db);
      const result = await extractor.extractOwlFromClass('TestClass');

      expect(result.success).toBe(false);
      expect(result.error).toMatch(/markdown_content is NULL/i);
    });

    it('should handle empty markdown_content', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });

      await db.query(
        `INSERT INTO ontology_classes (class_name, sha1_hash, markdown_content)
         VALUES ($1, $2, $3)`,
        ['EmptyClass', 'def456', '']
      );

      const extractor = new OntologyExtractorService(db);
      const result = await extractor.extractOwlFromClass('EmptyClass');

      expect(result.owlBlocks).toHaveLength(0);
      expect(result.hasOwl).toBe(false);
    });
  });

  describe('EDGE-DATA-004: Unicode and special characters', () => {
    it('should handle Unicode in OWL content', async () => {
      const unicodeOwl = `
\`\`\`owl
Class: 🌐Metaverse_Class
  Annotations: rdfs:label "クラス"@ja
  SubClassOf: 虚拟现实Entity
\`\`\`
      `.trim();

      const parser = new OwlParser();
      const result = parser.parseOwlFunctional(unicodeOwl);

      // Should either parse correctly or provide clear error
      if (result.valid) {
        expect(result.classes[0].name).toContain('Metaverse_Class');
      } else {
        expect(result.errors[0].message).toContain('Unicode');
      }
    });
  });
});
```

---

## 4. Performance Benchmark Suite

### 4.1 Component-Level Performance Tests

**Test ID**: `PERF-COMP-001` to `PERF-COMP-003`
**Priority**: CRITICAL

```typescript
// tests/performance/component-benchmarks.test.ts

import { describe, it, expect } from '@jest/globals';
import { performance } from 'perf_hooks';

describe('Performance Benchmark Suite', () => {

  describe('PERF-COMP-001: Single class extraction', () => {
    it('should extract OWL from single class under 130ms', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });
      const extractor = new OntologyExtractorService(db);

      const iterations = 100;
      const durations: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await extractor.extractOwlFromClass('VirtualWorld');
        const duration = performance.now() - start;
        durations.push(duration);
      }

      const avgDuration = durations.reduce((a, b) => a + b) / iterations;
      const p95Duration = durations.sort()[Math.floor(iterations * 0.95)];

      console.log(`Average extraction time: ${avgDuration.toFixed(2)}ms`);
      console.log(`P95 extraction time: ${p95Duration.toFixed(2)}ms`);

      expect(avgDuration).toBeLessThan(130);
      expect(p95Duration).toBeLessThan(200);
    });
  });

  describe('PERF-COMP-002: Full ontology build', () => {
    it('should build complete ontology under 135 seconds', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });
      const builder = new OntologyBuilderService(db);

      const start = performance.now();
      const result = await builder.buildCompleteOntology();
      const duration = performance.now() - start;

      console.log(`Full ontology build time: ${(duration / 1000).toFixed(2)}s`);
      console.log(`Classes processed: ${result.classCount}`);
      console.log(`Restrictions extracted: ${result.restrictionCount}`);

      expect(duration).toBeLessThan(135000); // 135 seconds
      expect(result.classCount).toBe(988);
      expect(result.restrictionCount).toBe(1297);
    });
  });

  describe('PERF-COMP-003: Database query performance', () => {
    it('should query markdown_content efficiently', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });

      const queries = [
        'SELECT markdown_content FROM ontology_classes WHERE class_name = $1',
        'SELECT markdown_content FROM ontology_classes WHERE sha1_hash = $1',
        'SELECT class_name, markdown_content FROM ontology_classes LIMIT 100'
      ];

      for (const query of queries) {
        const start = performance.now();
        await db.query(query, ['VirtualWorld']);
        const duration = performance.now() - start;

        console.log(`Query time: ${duration.toFixed(2)}ms - ${query.substring(0, 50)}...`);
        expect(duration).toBeLessThan(50); // 50ms max per query
      }
    });

    it('should use indexes effectively', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });

      // Explain query plan
      const explainResult = await db.query(`
        EXPLAIN ANALYZE
        SELECT markdown_content
        FROM ontology_classes
        WHERE sha1_hash = 'test_hash'
      `);

      const queryPlan = explainResult.rows.map(r => r['QUERY PLAN']).join('\n');

      // Should use index, not sequential scan
      expect(queryPlan).toContain('Index Scan');
      expect(queryPlan).not.toContain('Seq Scan');
    });
  });
});
```

### 4.2 Memory Usage Benchmarks

```typescript
// tests/performance/memory-benchmarks.test.ts

describe('PERF-MEM-001: Memory efficiency tests', () => {
  it('should not exceed 500MB during full extraction', async () => {
    const initialMemory = process.memoryUsage().heapUsed;

    const extractor = new OntologyExtractorService(db);
    await extractor.extractAllOntologies();

    const peakMemory = process.memoryUsage().heapUsed;
    const memoryIncrease = (peakMemory - initialMemory) / 1024 / 1024;

    console.log(`Memory increase: ${memoryIncrease.toFixed(2)}MB`);
    expect(memoryIncrease).toBeLessThan(500);
  });

  it('should release memory after processing', async () => {
    const extractor = new OntologyExtractorService(db);

    await extractor.extractAllOntologies();
    const afterExtraction = process.memoryUsage().heapUsed;

    // Force garbage collection
    if (global.gc) global.gc();
    await new Promise(resolve => setTimeout(resolve, 1000));

    const afterGC = process.memoryUsage().heapUsed;
    const memoryReleased = (afterExtraction - afterGC) / 1024 / 1024;

    console.log(`Memory released: ${memoryReleased.toFixed(2)}MB`);
    expect(memoryReleased).toBeGreaterThan(100); // At least 100MB released
  });
});
```

---

## 5. Integration Test Suite

### 5.1 Service Coordination Tests

**Test ID**: `INT-COORD-001` to `INT-COORD-003`
**Priority**: HIGH

```typescript
// tests/integration/service-coordination.test.ts

import { describe, it, expect, beforeAll } from '@jest/globals';
import { ActorSystem } from '../../src/actors/ActorSystem';
import { GitHubSyncActor } from '../../src/actors/GitHubSyncActor';
import { OntologyExtractorActor } from '../../src/actors/OntologyExtractorActor';
import { OwlValidatorActor } from '../../src/actors/OwlValidatorActor';

describe('Integration: Service Coordination', () => {
  let actorSystem: ActorSystem;

  beforeAll(async () => {
    actorSystem = new ActorSystem();
    await actorSystem.initialize();
  });

  describe('INT-COORD-001: Actor message passing', () => {
    it('should coordinate GitHub sync → OWL extraction', async () => {
      const githubActor = actorSystem.spawn('github-sync', GitHubSyncActor);
      const extractorActor = actorSystem.spawn('extractor', OntologyExtractorActor);

      // Send sync command
      const syncPromise = githubActor.tell({ type: 'SYNC_ALL' });

      // Extractor should receive EXTRACTION_READY message
      const extractionPromise = new Promise((resolve) => {
        extractorActor.on('EXTRACTION_READY', (msg) => {
          expect(msg.classCount).toBe(988);
          resolve(true);
        });
      });

      await Promise.all([syncPromise, extractionPromise]);
    });
  });

  describe('INT-COORD-002: OwlValidatorService integration', () => {
    it('should validate extracted ontologies from database', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });
      const extractor = new OntologyExtractorService(db);
      const validator = new OwlValidatorService();

      // Extract OWL
      const extraction = await extractor.extractOwlFromClass('VirtualWorld');
      expect(extraction.hasOwl).toBe(true);

      // Validate with OwlValidatorService
      const validation = await validator.validate(extraction.owlContent);

      expect(validation.valid).toBe(true);
      expect(validation.classCount).toBeGreaterThan(0);
      expect(validation.errors).toHaveLength(0);
    });
  });

  describe('INT-COORD-003: whelk-rs reasoning integration', () => {
    it('should classify ontology extracted from database', async () => {
      const db = new Database({ connectionString: process.env.TEST_DB_URL });
      const builder = new OntologyBuilderService(db);
      const reasoner = new WhelkReasoningService();

      // Build complete ontology from database
      const ontology = await builder.buildCompleteOntology();

      // Run whelk-rs classification
      const reasoningResult = await reasoner.classify(ontology.owlContent);

      expect(reasoningResult.consistent).toBe(true);
      expect(reasoningResult.classCount).toBe(988);
      expect(reasoningResult.inferredSubclasses).toBeGreaterThan(0);
    });
  });
});
```

### 5.2 Error Recovery Tests

```typescript
// tests/integration/error-recovery.test.ts

describe('INT-ERROR-001: Database connection failure recovery', () => {
  it('should retry on transient database errors', async () => {
    const mockDb = new MockDatabase();
    mockDb.setFailurePattern([true, true, false]); // Fail twice, then succeed

    const extractor = new OntologyExtractorService(mockDb, {
      maxRetries: 3,
      retryDelay: 100
    });

    const result = await extractor.extractOwlFromClass('VirtualWorld');

    expect(result.success).toBe(true);
    expect(mockDb.attemptCount).toBe(3);
  });

  it('should fail gracefully after max retries', async () => {
    const mockDb = new MockDatabase();
    mockDb.setAlwaysFail(true);

    const extractor = new OntologyExtractorService(mockDb, {
      maxRetries: 3,
      retryDelay: 10
    });

    await expect(
      extractor.extractOwlFromClass('VirtualWorld')
    ).rejects.toThrow('Max retries exceeded');
  });
});

describe('INT-ERROR-002: Partial failure handling', () => {
  it('should continue processing after individual class failures', async () => {
    const db = new Database({ connectionString: process.env.TEST_DB_URL });

    // Insert one malformed class
    await db.query(`
      INSERT INTO ontology_classes (class_name, markdown_content, sha1_hash)
      VALUES ('MalformedClass', 'Invalid OWL content', 'bad_hash')
    `);

    const builder = new OntologyBuilderService(db);
    const result = await builder.buildCompleteOntology({
      continueOnError: true
    });

    expect(result.classCount).toBe(987); // 988 - 1 failed
    expect(result.failedClasses).toContain('MalformedClass');
    expect(result.errors).toHaveLength(1);
  });
});
```

---

## 6. Regression Test Suite

### 6.1 Test Suite Structure

```
tests/
├── regression/
│   ├── baseline/
│   │   ├── ontology-snapshot-v1.0.0.json
│   │   ├── restrictions-snapshot-v1.0.0.json
│   │   └── performance-baseline-v1.0.0.json
│   ├── semantic-preservation.test.ts
│   ├── performance-regression.test.ts
│   └── api-compatibility.test.ts
```

### 6.2 Semantic Preservation Regression

```typescript
// tests/regression/semantic-preservation.test.ts

import { readFileSync } from 'fs';
import { join } from 'path';

describe('REGRESSION: Semantic Preservation', () => {
  const BASELINE_PATH = join(__dirname, 'baseline', 'restrictions-snapshot-v1.0.0.json');
  const baseline = JSON.parse(readFileSync(BASELINE_PATH, 'utf-8'));

  it('should maintain all baseline restrictions', async () => {
    const db = new Database({ connectionString: process.env.TEST_DB_URL });
    const extractor = new OntologyExtractorService(db);

    const currentRestrictions = await extractor.extractAllRestrictions();

    // Compare with baseline
    const diff = compareRestrictionSets(baseline.restrictions, currentRestrictions);

    expect(diff.missing).toHaveLength(0);
    expect(diff.modified).toHaveLength(0);

    // New restrictions are allowed, but document them
    if (diff.added.length > 0) {
      console.log(`New restrictions added: ${diff.added.length}`);
      console.log(JSON.stringify(diff.added, null, 2));
    }
  });

  it('should preserve class hierarchies', async () => {
    const db = new Database({ connectionString: process.env.TEST_DB_URL });
    const builder = new OntologyBuilderService(db);

    const ontology = await builder.buildCompleteOntology();
    const hierarchies = extractClassHierarchies(ontology);

    // Validate against baseline
    for (const [className, parents] of Object.entries(baseline.hierarchies)) {
      expect(hierarchies[className]).toEqual(
        expect.arrayContaining(parents as string[])
      );
    }
  });
});

function compareRestrictionSets(baseline: any[], current: any[]) {
  const baselineSet = new Set(baseline.map(JSON.stringify));
  const currentSet = new Set(current.map(JSON.stringify));

  const missing = [...baselineSet].filter(x => !currentSet.has(x)).map(JSON.parse);
  const added = [...currentSet].filter(x => !baselineSet.has(x)).map(JSON.parse);
  const modified: any[] = []; // Could add fuzzy matching here

  return { missing, added, modified };
}
```

### 6.3 Performance Regression Tests

```typescript
// tests/regression/performance-regression.test.ts

describe('REGRESSION: Performance', () => {
  const BASELINE_PATH = join(__dirname, 'baseline', 'performance-baseline-v1.0.0.json');
  const baseline = JSON.parse(readFileSync(BASELINE_PATH, 'utf-8'));

  it('should not regress on full ontology build time', async () => {
    const db = new Database({ connectionString: process.env.TEST_DB_URL });
    const builder = new OntologyBuilderService(db);

    const iterations = 5;
    const durations: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await builder.buildCompleteOntology();
      durations.push(performance.now() - start);
    }

    const avgDuration = durations.reduce((a, b) => a + b) / iterations;
    const baselineAvg = baseline.fullBuildTime.average;

    console.log(`Current average: ${avgDuration.toFixed(2)}ms`);
    console.log(`Baseline average: ${baselineAvg.toFixed(2)}ms`);

    // Allow 10% variance
    const maxAllowed = baselineAvg * 1.10;
    expect(avgDuration).toBeLessThan(maxAllowed);
  });

  it('should maintain or improve change detection speed', async () => {
    const syncService = new GitHubSyncService(db);

    // Warm-up sync
    await syncService.syncAllMarkdownFiles();

    const start = performance.now();
    const result = await syncService.syncAllMarkdownFiles();
    const duration = performance.now() - start;

    console.log(`Change detection time: ${duration.toFixed(2)}ms`);
    console.log(`Baseline time: ${baseline.changeDetection.average.toFixed(2)}ms`);

    expect(duration).toBeLessThan(baseline.changeDetection.average * 1.10);
  });
});
```

---

## 7. CI/CD Integration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/ontology-tests.yml

name: Ontology Storage Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: ontology_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run database migrations
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/ontology_test
        run: npm run db:migrate

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/ontology_test
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: npm run test:integration

  performance-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: ontology_test

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run performance benchmarks
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/ontology_test
        run: npm run test:performance

      - name: Check performance regression
        run: |
          npm run test:regression-check

      - name: Upload performance report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: tests/reports/performance-*.json

  e2e-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: ontology_test

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run E2E tests
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/ontology_test
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: npm run test:e2e
        timeout-minutes: 10
```

### 7.2 Test Commands in package.json

```json
{
  "scripts": {
    "test": "jest --coverage",
    "test:unit": "jest tests/unit --coverage",
    "test:integration": "jest tests/integration --runInBand",
    "test:performance": "jest tests/performance --runInBand --detectOpenHandles",
    "test:e2e": "jest tests/e2e --runInBand --forceExit",
    "test:regression": "jest tests/regression --runInBand",
    "test:regression-check": "node scripts/check-performance-regression.js",
    "test:watch": "jest --watch",
    "test:debug": "node --inspect-brk node_modules/.bin/jest --runInBand"
  }
}
```

---

## 8. Performance Acceptance Criteria

### 8.1 Timing Requirements

| Operation | Target | Maximum | Baseline |
|-----------|--------|---------|----------|
| Single class extraction | < 130ms | 200ms | 125ms |
| Full ontology build | < 135s | 150s | 125s |
| Initial GitHub sync | < 125s | 135s | 120s |
| Re-sync (no changes) | < 8s | 10s | 8.3s |
| Re-sync (10 changes) | < 12s | 15s | 11.7s |
| Database query (single) | < 50ms | 100ms | 45ms |

### 8.2 Semantic Preservation Requirements

| Metric | Requirement |
|--------|-------------|
| Restriction preservation | 100% (all 1,297) |
| Class hierarchy preservation | 100% |
| Property preservation | 100% |
| Annotation preservation | 95%+ |
| Namespace preservation | 100% |

### 8.3 Resource Requirements

| Resource | Target | Maximum |
|----------|--------|---------|
| Memory usage (peak) | < 400MB | 500MB |
| Database size | ~50MB | 100MB |
| Disk I/O | < 100 ops/s | 200 ops/s |
| CPU usage | < 60% | 80% |

---

## 9. Test Execution Schedule

### 9.1 Continuous Integration

- **On every PR**: Unit tests + integration tests
- **On merge to main**: Full test suite
- **Nightly**: Performance tests + regression tests
- **Weekly**: Baseline updates

### 9.2 Release Testing

- **Pre-release**: Full test suite + manual verification
- **Post-release**: Smoke tests + monitoring
- **Quarterly**: Performance baseline updates

---

## 10. Monitoring and Alerting

### 10.1 Performance Monitoring

```typescript
// tests/monitoring/performance-monitor.ts

export class PerformanceMonitor {
  async trackOntologyBuild() {
    const metrics = {
      timestamp: Date.now(),
      duration: 0,
      classCount: 0,
      restrictionCount: 0,
      memoryUsage: process.memoryUsage()
    };

    const start = performance.now();
    const result = await buildCompleteOntology();
    metrics.duration = performance.now() - start;
    metrics.classCount = result.classCount;
    metrics.restrictionCount = result.restrictionCount;

    // Send to monitoring service
    await this.sendToDatadog(metrics);

    // Alert if exceeding thresholds
    if (metrics.duration > 135000) {
      await this.alertSlack('Ontology build time exceeded 135s threshold');
    }
  }
}
```

### 10.2 Test Quality Metrics

- Test coverage: > 80%
- Test execution time: < 10 minutes (full suite)
- Test flakiness: < 1%
- Mean time to detect (MTTD): < 1 hour

---

## 11. Summary

This comprehensive testing strategy ensures:

✅ **Zero semantic loss** through complete data flow validation
✅ **15x performance improvement** with change detection
✅ **Robust edge case handling** for production reliability
✅ **Continuous performance monitoring** to prevent regressions
✅ **Automated CI/CD integration** for rapid feedback

### Test Execution Order

1. **Unit tests** (5 minutes) - Fast feedback on components
2. **Integration tests** (10 minutes) - Service coordination validation
3. **Performance tests** (15 minutes) - Benchmark validation
4. **E2E tests** (20 minutes) - Complete pipeline validation
5. **Regression tests** (30 minutes) - Baseline comparison

**Total execution time**: ~80 minutes for complete suite
**PR execution time**: ~15 minutes (unit + integration only)
