// Global setup - runs once before all tests
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export default async function globalSetup() {
  console.log('\n🚀 Setting up test environment...\n');

  // Set up test database
  const databaseUrl = process.env.TEST_DATABASE_URL ||
    'postgresql://postgres:test@localhost:5432/ontology_test';

  try {
    // Create test database if it doesn't exist
    console.log('📊 Creating test database...');
    await execAsync(`createdb -U postgres ontology_test || true`);

    // Run migrations
    console.log('🔄 Running database migrations...');
    process.env.DATABASE_URL = databaseUrl;
    await execAsync('npm run db:migrate');

    // Seed test data
    console.log('🌱 Seeding test data...');
    await execAsync('npm run db:seed:test');

    console.log('✅ Test environment ready!\n');
  } catch (error) {
    console.error('❌ Failed to set up test environment:', error);
    throw error;
  }
}
