---
title: Testing Guide
description: Comprehensive guide to testing strategies and practices in Turbo Flow Claude
category: developer-guide
tags: [testing, tdd, quality, automation]
difficulty: intermediate
last_updated: 2025-10-27
related:
  - /docs/guides/developer/01-development-setup.md
  - /docs/guides/developer/adding-a-feature.md
  - /docs/reference/sparc-methodology.md
---

# Testing Guide

> **⚠️ Work in Progress**: This guide is currently under development. Content will be expanded in future updates.

## Overview

This guide covers testing strategies, best practices, and tools for ensuring code quality in Turbo Flow Claude. We follow Test-Driven Development (TDD) principles and use the SPARC methodology for systematic testing.

## Testing Philosophy

### Test-Driven Development (TDD)

**The TDD cycle:**
1. **Red**: Write a failing test
2. **Green**: Write minimal code to pass
3. **Refactor**: Improve code while keeping tests green

**Benefits:**
- Better design through testability
- Faster debugging cycles
- Living documentation
- Confidence in refactoring

### Testing Pyramid

```
        /\
       /  \  E2E (10%)
      /____\
     /      \
    / Integ. \ (30%)
   /__________\
  /            \
 /     Unit     \ (60%)
/________________\
```

**Distribution:**
- **Unit tests**: 60% - Fast, isolated, focused
- **Integration tests**: 30% - Component interactions
- **E2E tests**: 10% - Full system workflows

## Test Types

### Unit Tests

**Purpose:** Test individual functions and classes in isolation.

**Example:**
```typescript
// src/services/user.service.ts
export class UserService {
  calculateDiscount(user: User): number {
    if (user.purchases >= 10) return 0.1;
    if (user.purchases >= 5) return 0.05;
    return 0;
  }
}

// tests/unit/services/user.service.test.ts
import { UserService } from '../../../src/services/user.service';

describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    service = new UserService();
  });

  describe('calculateDiscount', () => {
    it('should return 10% discount for 10+ purchases', () => {
      const user = { purchases: 10 };
      expect(service.calculateDiscount(user)).toBe(0.1);
    });

    it('should return 5% discount for 5-9 purchases', () => {
      const user = { purchases: 5 };
      expect(service.calculateDiscount(user)).toBe(0.05);
    });

    it('should return 0% discount for < 5 purchases', () => {
      const user = { purchases: 3 };
      expect(service.calculateDiscount(user)).toBe(0);
    });
  });
});
```

**Best practices:**
- Test one thing per test
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies

### Integration Tests

**Purpose:** Test interactions between components.

**Example:**
```typescript
// tests/integration/api/auth.test.ts
import { app } from '../../../src/app';
import request from 'supertest';

describe('Authentication API', () => {
  beforeEach(async () => {
    await setupTestDatabase();
  });

  afterEach(async () => {
    await cleanupTestDatabase();
  });

  describe('POST /auth/login', () => {
    it('should login with valid credentials', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'password123'
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('token');
    });

    it('should reject invalid credentials', async () => {
      const response = await request(app)
        .post('/auth/login')
        .send({
          email: 'test@example.com',
          password: 'wrongpassword'
        });

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error');
    });
  });
});
```

**Best practices:**
- Test component boundaries
- Use test databases/services
- Clean up after each test
- Test happy and error paths

### End-to-End (E2E) Tests

**Purpose:** Test complete user workflows.

**Example:**
```typescript
// tests/e2e/user-registration.test.ts
import { browser } from '@playwright/test';

describe('User Registration Flow', () => {
  it('should complete full registration process', async () => {
    const page = await browser.newPage();

    // Navigate to registration
    await page.goto('/register');

    // Fill form
    await page.fill('#email', 'newuser@example.com');
    await page.fill('#password', 'SecurePass123!');
    await page.fill('#confirmPassword', 'SecurePass123!');

    // Submit
    await page.click('button[type="submit"]');

    // Verify redirect to dashboard
    await page.waitForURL('/dashboard');
    expect(await page.textContent('h1')).toBe('Welcome');

    await page.close();
  });
});
```

**Best practices:**
- Test critical user journeys
- Keep tests independent
- Use page object pattern
- Run in CI/CD pipeline

## SPARC TDD Workflow

### Using SPARC for Test-Driven Development

**Run complete TDD workflow:**
```bash
npx claude-flow sparc tdd "feature description"
```

**This executes:**
1. **Specification**: Define test requirements
2. **Pseudocode**: Design test cases
3. **Architecture**: Plan test structure
4. **Refinement**: Implement tests first, then code
5. **Completion**: Verify integration

### SPARC TDD Example

```bash
# Run TDD workflow for authentication
npx claude-flow sparc tdd "user authentication with JWT tokens"
```

**Output:**
- Test specifications
- Test case designs
- Test implementation
- Code implementation
- Integration verification

## Agent-Driven Testing

### Spawning Test Agents

**Use Claude Code's Task tool for concurrent testing:**

```javascript
// Comprehensive testing with multiple agents
Task("Unit Test Agent", "Create comprehensive unit tests for user service", "tester")
Task("Integration Test Agent", "Build API integration test suite", "tester")
Task("E2E Test Agent", "Implement critical user journey tests", "tester")
Task("Performance Test Agent", "Create load and stress tests", "perf-analyzer")
Task("Security Test Agent", "Implement security vulnerability tests", "reviewer")
```

### TDD London Swarm

**Use specialized TDD agent:**
```bash
# Load TDD London Swarm agent
cat /home/devuser/agents/tdd-london-swarm.md

# Execute with SPARC
npx claude-flow sparc tdd "your feature"
```

**London School TDD principles:**
- Mock external dependencies
- Test behavior, not implementation
- Outside-in development
- Design through mocking

## Testing Tools & Setup

### Test Framework Configuration

**Jest configuration (`jest.config.js`):**
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: ['**/*.test.ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/index.ts'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

### Running Tests

**Basic commands:**
```bash
# Run all tests
npm run test

# Run specific test file
npm run test -- tests/unit/services/user.service.test.ts

# Run with coverage
npm run test -- --coverage

# Watch mode
npm run test -- --watch

# Verbose output
npm run test -- --verbose
```

**Coverage reports:**
```bash
# Generate coverage report
npm run test -- --coverage

# View HTML report
open coverage/lcov-report/index.html
```

### Test Organization

**Directory structure:**
```
tests/
├── unit/               # Unit tests
│   ├── services/
│   ├── utils/
│   └── models/
├── integration/        # Integration tests
│   ├── api/
│   ├── database/
│   └── services/
├── e2e/               # End-to-end tests
│   ├── user-flows/
│   └── critical-paths/
├── fixtures/          # Test data
│   ├── users.json
│   └── products.json
└── helpers/           # Test utilities
    ├── setup.ts
    └── mocks.ts
```

## Testing Best Practices

### Writing Effective Tests

**1. Descriptive Test Names**
```typescript
// ❌ Bad
it('works', () => { ... });

// ✅ Good
it('should return 401 when user provides invalid credentials', () => { ... });
```

**2. AAA Pattern**
```typescript
it('should calculate total with tax', () => {
  // Arrange
  const cart = new ShoppingCart();
  cart.addItem({ price: 100, quantity: 2 });

  // Act
  const total = cart.calculateTotal(0.1); // 10% tax

  // Assert
  expect(total).toBe(220);
});
```

**3. One Assertion Per Test (when possible)**
```typescript
// ✅ Focused tests
it('should set user name correctly', () => {
  const user = new User('John');
  expect(user.name).toBe('John');
});

it('should set user active status by default', () => {
  const user = new User('John');
  expect(user.isActive).toBe(true);
});
```

**4. Test Edge Cases**
```typescript
describe('divideNumbers', () => {
  it('should divide positive numbers', () => {
    expect(divideNumbers(10, 2)).toBe(5);
  });

  it('should handle division by zero', () => {
    expect(() => divideNumbers(10, 0)).toThrow('Division by zero');
  });

  it('should handle negative numbers', () => {
    expect(divideNumbers(-10, 2)).toBe(-5);
  });

  it('should handle floating point division', () => {
    expect(divideNumbers(10, 3)).toBeCloseTo(3.333, 2);
  });
});
```

### Mocking & Stubbing

**Using Jest mocks:**
```typescript
// Mock external service
jest.mock('../../../src/services/email.service');

describe('UserService', () => {
  it('should send welcome email on registration', async () => {
    const emailService = require('../../../src/services/email.service');
    emailService.sendEmail = jest.fn().mockResolvedValue(true);

    const userService = new UserService(emailService);
    await userService.registerUser({ email: 'test@example.com' });

    expect(emailService.sendEmail).toHaveBeenCalledWith(
      'test@example.com',
      'Welcome!'
    );
  });
});
```

**Dependency injection for testability:**
```typescript
// ✅ Good - Testable
class UserService {
  constructor(
    private emailService: EmailService,
    private database: Database
  ) {}
}

// ❌ Bad - Hard to test
class UserService {
  private emailService = new EmailService();
  private database = new Database();
}
```

### Test Data Management

**Using fixtures:**
```typescript
// tests/fixtures/users.ts
export const mockUsers = {
  validUser: {
    email: 'valid@example.com',
    password: 'SecurePass123!',
    name: 'Valid User'
  },
  adminUser: {
    email: 'admin@example.com',
    password: 'AdminPass123!',
    name: 'Admin User',
    role: 'admin'
  }
};

// tests/integration/api/users.test.ts
import { mockUsers } from '../../fixtures/users';

it('should create user with valid data', async () => {
  const response = await request(app)
    .post('/users')
    .send(mockUsers.validUser);

  expect(response.status).toBe(201);
});
```

**Factory functions:**
```typescript
// tests/helpers/factories.ts
export function createMockUser(overrides = {}) {
  return {
    id: Math.random().toString(36),
    email: 'default@example.com',
    name: 'Default User',
    isActive: true,
    ...overrides
  };
}

// Usage
const user = createMockUser({ email: 'custom@example.com' });
```

## Coverage Requirements

### Coverage Targets

**Minimum thresholds:**
- Statements: 80%
- Branches: 80%
- Functions: 80%
- Lines: 80%

**Priority areas (aim for 90%+):**
- Core business logic
- Authentication/authorization
- Payment processing
- Data validation

### Measuring Coverage

**Generate coverage report:**
```bash
npm run test -- --coverage
```

**Review uncovered code:**
```bash
# View coverage summary
npm run test -- --coverage --coverageReporters=text

# Generate detailed HTML report
npm run test -- --coverage --coverageReporters=html
open coverage/index.html
```

## Performance Testing

### Load Testing

**Using artillery or k6:**
```yaml
# load-test.yml
config:
  target: 'http://localhost:3000'
  phases:
    - duration: 60
      arrivalRate: 10
scenarios:
  - flow:
      - get:
          url: '/api/users'
      - post:
          url: '/api/auth/login'
          json:
            email: 'test@example.com'
            password: 'password123'
```

**Run load tests:**
```bash
artillery run load-test.yml
```

### Performance Benchmarks

**Using performance-benchmarker agent:**
```javascript
Task("Performance Benchmarker",
  "Benchmark API response times and identify bottlenecks",
  "performance-benchmarker")
```

## Continuous Integration

### CI/CD Pipeline Testing

**GitHub Actions example:**
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm install
      - run: npm run lint
      - run: npm run typecheck
      - run: npm run test -- --coverage
      - uses: codecov/codecov-action@v2
```

### Pre-commit Testing

**Using husky hooks:**
```bash
# .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

npm run lint
npm run typecheck
npm run test -- --bail --findRelatedTests
```

## Troubleshooting

### Common Testing Issues

**Tests timing out:**
```typescript
// Increase timeout for slow tests
it('should handle long operation', async () => {
  // Test code
}, 10000); // 10 second timeout
```

**Flaky tests:**
```typescript
// Fix race conditions
await waitFor(() => {
  expect(element).toBeVisible();
});

// Clear state between tests
afterEach(() => {
  jest.clearAllMocks();
  jest.restoreAllMocks();
});
```

**Memory leaks in tests:**
```typescript
// Clean up properly
afterEach(() => {
  // Close connections
  // Clear timers
  // Remove event listeners
});
```

## Best Practices Summary

### DO:
- ✅ Write tests before code (TDD)
- ✅ Use descriptive test names
- ✅ Test edge cases and error conditions
- ✅ Keep tests isolated and independent
- ✅ Mock external dependencies
- ✅ Aim for >80% coverage
- ✅ Run tests in CI/CD
- ✅ Use agent coordination for comprehensive testing

### DON'T:
- ❌ Test implementation details
- ❌ Write tests that depend on execution order
- ❌ Share state between tests
- ❌ Ignore flaky tests
- ❌ Skip edge cases
- ❌ Commit failing tests
- ❌ Test third-party libraries

## Related Resources

- [Development Setup](/docs/guides/developer/01-development-setup.md)
- [Adding a Feature](/docs/guides/developer/adding-a-feature.md)
- [SPARC Methodology](/docs/reference/sparc-methodology.md)
- [Code Quality Standards](/docs/reference/code-standards.md)

## Support

- GitHub Issues: https://github.com/ruvnet/claude-flow/issues
- Documentation: https://github.com/ruvnet/claude-flow
- Testing Best Practices: https://jestjs.io/docs/getting-started
