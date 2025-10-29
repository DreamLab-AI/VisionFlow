/**
 * E2E Happy Path Test - Complete Ontology Pipeline
 *
 * Test ID: E2E-HAPPY-001
 * Priority: CRITICAL
 * Duration: ~2-3 minutes
 *
 * This test validates the complete data flow:
 * GitHub Markdown → Database → OWL Extraction → Reasoning
 *
 * Success Criteria:
 * - All 988 files processed
 * - All 1,297 ObjectSomeValuesFrom restrictions preserved
 * - Zero semantic loss
 * - Complete pipeline < 135 seconds
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { Database } from '../../src/database/Database';
import { GitHubSyncService } from '../../src/services/GitHubSyncService';
import { OntologyExtractorService } from '../../src/services/OntologyExtractorService';
import { OwlValidatorService } from '../../src/services/OwlValidatorService';
import { WhelkReasoningService } from '../../src/services/WhelkReasoningService';

describe('E2E-HAPPY-001: Complete Ontology Pipeline Validation', () => {
  let db: Database;
  let githubSync: GitHubSyncService;
  let extractor: OntologyExtractorService;
  let validator: OwlValidatorService;
  let reasoner: WhelkReasoningService;

  // Expected values from ontology
  const EXPECTED_CLASSES = 988;
  const EXPECTED_RESTRICTIONS = 1297;
  const MAX_PIPELINE_TIME = 135000; // 135 seconds

  beforeAll(async () => {
    console.log('🚀 Initializing test environment...');

    // Initialize database connection
    db = new Database({
      connectionString: process.env.TEST_DATABASE_URL ||
        'postgresql://postgres:test@localhost:5432/ontology_test'
    });

    // Run migrations if needed
    await db.migrate();

    // Initialize services
    githubSync = new GitHubSyncService(db, {
      owner: process.env.GITHUB_OWNER || 'YOUR_ORG',
      repo: process.env.GITHUB_REPO || 'Metaverse-Ontology',
      token: process.env.GITHUB_TOKEN
    });

    extractor = new OntologyExtractorService(db);
    validator = new OwlValidatorService();
    reasoner = new WhelkReasoningService();

    console.log('✅ Test environment ready');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up test environment...');
    await db.close();
  });

  describe('Complete Pipeline Flow', () => {
    it('should process complete pipeline within time limit', async () => {
      const startTime = Date.now();
      console.log('\n📊 Starting complete pipeline test...\n');

      // ============================================
      // STEP 1: GitHub Sync
      // ============================================
      console.log('📥 Step 1: Syncing from GitHub...');
      const syncStartTime = Date.now();

      const syncResult = await githubSync.syncAllMarkdownFiles();

      const syncDuration = Date.now() - syncStartTime;
      console.log(`   ✅ GitHub sync completed in ${(syncDuration / 1000).toFixed(2)}s`);
      console.log(`   📁 Files processed: ${syncResult.filesProcessed}`);
      console.log(`   ➕ New files: ${syncResult.newFiles}`);
      console.log(`   🔄 Updated files: ${syncResult.updatedFiles}`);
      console.log(`   ✓ Unchanged files: ${syncResult.unchangedFiles}`);

      // Validate sync results
      expect(syncResult.filesProcessed).toBe(EXPECTED_CLASSES);
      expect(syncResult.errors).toHaveLength(0);

      // ============================================
      // STEP 2: Database Validation
      // ============================================
      console.log('\n💾 Step 2: Validating database storage...');

      const storedClasses = await db.query(`
        SELECT
          class_name,
          markdown_content,
          sha1_hash,
          content_hash,
          LENGTH(markdown_content) as content_length
        FROM ontology_classes
        ORDER BY class_name
      `);

      console.log(`   ✅ Stored classes: ${storedClasses.rows.length}`);

      // Validate all classes are stored
      expect(storedClasses.rows).toHaveLength(EXPECTED_CLASSES);

      // Validate required fields are populated
      let nullContentCount = 0;
      let emptyContentCount = 0;
      let invalidHashCount = 0;

      for (const row of storedClasses.rows) {
        // Check markdown_content
        if (row.markdown_content === null) {
          nullContentCount++;
        } else if (row.markdown_content.trim() === '') {
          emptyContentCount++;
        }

        // Check SHA1 hash format (40 hex characters)
        if (!row.sha1_hash || !/^[a-f0-9]{40}$/.test(row.sha1_hash)) {
          invalidHashCount++;
        }

        // Check content hash format (64 hex characters for SHA-256)
        if (!row.content_hash || !/^[a-f0-9]{64}$/.test(row.content_hash)) {
          invalidHashCount++;
        }
      }

      console.log(`   📝 Null content: ${nullContentCount}`);
      console.log(`   📝 Empty content: ${emptyContentCount}`);
      console.log(`   🔐 Invalid hashes: ${invalidHashCount}`);

      expect(nullContentCount).toBe(0);
      expect(emptyContentCount).toBe(0);
      expect(invalidHashCount).toBe(0);

      // Sample validation
      const sampleClass = storedClasses.rows[0];
      expect(sampleClass.markdown_content).toBeTruthy();
      expect(sampleClass.sha1_hash).toMatch(/^[a-f0-9]{40}$/);
      expect(sampleClass.content_hash).toMatch(/^[a-f0-9]{64}$/);
      expect(sampleClass.content_length).toBeGreaterThan(0);

      // ============================================
      // STEP 3: OWL Extraction
      // ============================================
      console.log('\n🦉 Step 3: Extracting OWL from database...');
      const extractStartTime = Date.now();

      const extractionResult = await extractor.extractAllOntologies();

      const extractDuration = Date.now() - extractStartTime;
      console.log(`   ✅ OWL extraction completed in ${(extractDuration / 1000).toFixed(2)}s`);
      console.log(`   🏛️  Classes extracted: ${extractionResult.classesExtracted}`);
      console.log(`   🔗 Restrictions found: ${extractionResult.restrictionsFound}`);
      console.log(`   ⚠️  Errors: ${extractionResult.errors?.length || 0}`);

      // Validate extraction results
      expect(extractionResult.classesExtracted).toBe(EXPECTED_CLASSES);
      expect(extractionResult.restrictionsFound).toBe(EXPECTED_RESTRICTIONS);
      expect(extractionResult.errors || []).toHaveLength(0);

      // ============================================
      // STEP 4: OWL Validation
      // ============================================
      console.log('\n✅ Step 4: Validating OWL syntax...');

      const owlContent = extractionResult.combinedOntology;
      const parseResult = validator.parseOwlFunctional(owlContent);

      console.log(`   📋 Validation status: ${parseResult.valid ? '✅ Valid' : '❌ Invalid'}`);
      console.log(`   🏛️  Classes found: ${parseResult.classes?.length || 0}`);
      console.log(`   🔗 Restrictions: ${parseResult.restrictions?.length || 0}`);
      console.log(`   ⚠️  Errors: ${parseResult.errors?.length || 0}`);

      // Validate OWL syntax
      expect(parseResult.valid).toBe(true);
      expect(parseResult.errors || []).toHaveLength(0);

      if (parseResult.errors && parseResult.errors.length > 0) {
        console.error('   ❌ Validation errors:');
        parseResult.errors.forEach(error => {
          console.error(`      - Line ${error.line}: ${error.message}`);
        });
      }

      // ============================================
      // STEP 5: Semantic Preservation Check
      // ============================================
      console.log('\n🔍 Step 5: Validating semantic preservation...');

      // Extract restrictions from database
      const originalRestrictions = await db.query(`
        SELECT
          class_name,
          property,
          filler,
          restriction_type
        FROM ontology_restrictions
        WHERE restriction_type = 'ObjectSomeValuesFrom'
        ORDER BY class_name, property
      `);

      // Compare with extracted restrictions
      const extractedRestrictions = parseResult.restrictions || [];

      console.log(`   📊 Original restrictions: ${originalRestrictions.rows.length}`);
      console.log(`   📊 Extracted restrictions: ${extractedRestrictions.length}`);

      // Create comparison sets
      const originalSet = new Set(
        originalRestrictions.rows.map(r =>
          `${r.class_name}|${r.property}|${r.filler}`
        )
      );

      const extractedSet = new Set(
        extractedRestrictions.map(r =>
          `${r.className}|${r.property}|${r.filler}`
        )
      );

      // Find differences
      const missingRestrictions = [...originalSet].filter(x => !extractedSet.has(x));
      const extraRestrictions = [...extractedSet].filter(x => !originalSet.has(x));

      console.log(`   ❌ Missing restrictions: ${missingRestrictions.length}`);
      console.log(`   ➕ Extra restrictions: ${extraRestrictions.length}`);

      if (missingRestrictions.length > 0) {
        console.error('   ⚠️  Missing restrictions (first 10):');
        missingRestrictions.slice(0, 10).forEach(r => {
          console.error(`      - ${r}`);
        });
      }

      // Validate zero semantic loss
      expect(missingRestrictions).toHaveLength(0);
      expect(extractedRestrictions).toHaveLength(EXPECTED_RESTRICTIONS);

      // ============================================
      // STEP 6: Reasoning
      // ============================================
      console.log('\n🧠 Step 6: Running reasoning engine...');
      const reasonStartTime = Date.now();

      const reasoningResult = await reasoner.classify(owlContent);

      const reasonDuration = Date.now() - reasonStartTime;
      console.log(`   ✅ Reasoning completed in ${(reasonDuration / 1000).toFixed(2)}s`);
      console.log(`   🔍 Consistent: ${reasoningResult.consistent ? '✅ Yes' : '❌ No'}`);
      console.log(`   🏛️  Classes: ${reasoningResult.classCount}`);
      console.log(`   💡 Inferred axioms: ${reasoningResult.inferredAxioms}`);
      console.log(`   ⚠️  Inconsistent classes: ${reasoningResult.inconsistentClasses?.length || 0}`);

      // Validate reasoning results
      expect(reasoningResult.consistent).toBe(true);
      expect(reasoningResult.inferredAxioms).toBeGreaterThan(0);
      expect(reasoningResult.inconsistentClasses || []).toHaveLength(0);

      if (reasoningResult.inconsistentClasses && reasoningResult.inconsistentClasses.length > 0) {
        console.error('   ❌ Inconsistent classes detected:');
        reasoningResult.inconsistentClasses.forEach(cls => {
          console.error(`      - ${cls}`);
        });
      }

      // ============================================
      // FINAL: Timing Validation
      // ============================================
      const totalTime = Date.now() - startTime;
      console.log(`\n⏱️  Total pipeline time: ${(totalTime / 1000).toFixed(2)}s`);
      console.log(`   📊 Target: < ${MAX_PIPELINE_TIME / 1000}s`);
      console.log(`   ${totalTime < MAX_PIPELINE_TIME ? '✅ PASS' : '❌ FAIL'}\n`);

      // Validate total execution time
      expect(totalTime).toBeLessThan(MAX_PIPELINE_TIME);

      // ============================================
      // Summary Report
      // ============================================
      console.log('📈 PIPELINE SUMMARY:');
      console.log('   ✅ GitHub sync: PASS');
      console.log('   ✅ Database storage: PASS');
      console.log('   ✅ OWL extraction: PASS');
      console.log('   ✅ OWL validation: PASS');
      console.log('   ✅ Semantic preservation: PASS');
      console.log('   ✅ Reasoning: PASS');
      console.log('   ✅ Performance: PASS');
      console.log('\n🎉 ALL TESTS PASSED!\n');

    }, 180000); // 3 minute timeout

    it('should preserve all ObjectSomeValuesFrom restrictions', async () => {
      const query = `
        SELECT
          class_name,
          COUNT(*) as restriction_count
        FROM ontology_restrictions
        WHERE restriction_type = 'ObjectSomeValuesFrom'
        GROUP BY class_name
        ORDER BY restriction_count DESC
      `;

      const result = await db.query(query);

      const totalRestrictions = result.rows.reduce(
        (sum, row) => sum + parseInt(row.restriction_count),
        0
      );

      console.log(`\n🔗 Restriction Analysis:`);
      console.log(`   Total restrictions: ${totalRestrictions}`);
      console.log(`   Classes with restrictions: ${result.rows.length}`);
      console.log(`   Expected restrictions: ${EXPECTED_RESTRICTIONS}`);

      // Show top classes by restriction count
      console.log(`\n   Top 10 classes by restriction count:`);
      result.rows.slice(0, 10).forEach((row, idx) => {
        console.log(`      ${idx + 1}. ${row.class_name}: ${row.restriction_count} restrictions`);
      });

      expect(totalRestrictions).toBe(EXPECTED_RESTRICTIONS);
    });

    it('should have valid class hierarchies', async () => {
      const hierarchyQuery = `
        SELECT
          c.class_name,
          c.parent_classes,
          COUNT(r.id) as restriction_count
        FROM ontology_classes c
        LEFT JOIN ontology_restrictions r ON c.class_name = r.class_name
        WHERE c.parent_classes IS NOT NULL
        GROUP BY c.class_name, c.parent_classes
        ORDER BY c.class_name
      `;

      const result = await db.query(hierarchyQuery);

      console.log(`\n🏛️  Class Hierarchy Analysis:`);
      console.log(`   Classes with parent classes: ${result.rows.length}`);

      // Validate that classes have valid parent relationships
      expect(result.rows.length).toBeGreaterThan(0);

      // Sample validation
      const sampledHierarchy = result.rows[0];
      expect(sampledHierarchy.class_name).toBeTruthy();
      expect(sampledHierarchy.parent_classes).toBeTruthy();
    });
  });

  describe('Performance Characteristics', () => {
    it('should extract single class under 130ms', async () => {
      const iterations = 100;
      const durations: number[] = [];

      console.log(`\n⏱️  Running ${iterations} iterations of single class extraction...`);

      for (let i = 0; i < iterations; i++) {
        const start = Date.now();
        await extractor.extractOwlFromClass('VirtualWorld');
        const duration = Date.now() - start;
        durations.push(duration);
      }

      const avgDuration = durations.reduce((a, b) => a + b) / iterations;
      const p50Duration = durations.sort()[Math.floor(iterations * 0.50)];
      const p95Duration = durations.sort()[Math.floor(iterations * 0.95)];
      const p99Duration = durations.sort()[Math.floor(iterations * 0.99)];

      console.log(`   Average: ${avgDuration.toFixed(2)}ms`);
      console.log(`   P50: ${p50Duration.toFixed(2)}ms`);
      console.log(`   P95: ${p95Duration.toFixed(2)}ms`);
      console.log(`   P99: ${p99Duration.toFixed(2)}ms`);
      console.log(`   Target: < 130ms`);

      expect(avgDuration).toBeLessThan(130);
      expect(p95Duration).toBeLessThan(200);
    });
  });
});
