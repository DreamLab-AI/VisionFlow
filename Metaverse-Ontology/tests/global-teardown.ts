// Global teardown - runs once after all tests
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export default async function globalTeardown() {
  console.log('\n🧹 Cleaning up test environment...\n');

  try {
    // Drop test database
    console.log('💾 Dropping test database...');
    await execAsync(`dropdb -U postgres ontology_test --if-exists`);

    console.log('✅ Test environment cleaned up!\n');
  } catch (error) {
    console.error('❌ Failed to clean up test environment:', error);
    // Don't throw - cleanup failures shouldn't fail the test run
  }
}
