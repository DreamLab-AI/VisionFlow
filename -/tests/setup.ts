// Global test setup
import { config } from 'dotenv';

// Load test environment variables
config({ path: '.env.test' });

// Set test-specific environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error'; // Reduce noise in test output

// Global test utilities
global.testUtils = {
  createMockDatabase: () => {
    return {
      query: jest.fn(),
      close: jest.fn(),
      transaction: jest.fn()
    };
  },

  createTestData: {
    markdownWithOwl: `
# Test Class

\`\`\`owl
Class: TestClass
  SubClassOf: ParentClass
  SubClassOf: ObjectSomeValuesFrom(hasProperty SomeValue)
\`\`\`
    `,

    markdownWithoutOwl: `
# Regular Markdown

This is just regular content without any OWL.
    `,

    malformedOwl: `
\`\`\`owl
Class: InvalidClass
  ObjectSomeValuesFrom(unclosed
\`\`\`
    `,

    validOwlClass: {
      class_name: 'VirtualWorld',
      sha1_hash: 'abc123def456',
      content_hash: '789ghi012jkl',
      markdown_content: `
# Virtual World

\`\`\`owl
Class: VirtualWorld
  SubClassOf: DigitalEnvironment
  SubClassOf: ObjectSomeValuesFrom(hasUser User)
  SubClassOf: ObjectSomeValuesFrom(hasSpace Space3D)
\`\`\`
      `
    }
  }
};

// Configure Jest matchers
expect.extend({
  toBeWithinRange(received: number, floor: number, ceiling: number) {
    const pass = received >= floor && received <= ceiling;
    if (pass) {
      return {
        message: () => `expected ${received} not to be within range ${floor} - ${ceiling}`,
        pass: true
      };
    } else {
      return {
        message: () => `expected ${received} to be within range ${floor} - ${ceiling}`,
        pass: false
      };
    }
  }
});

// Clean up after each test
afterEach(() => {
  jest.clearAllMocks();
});
