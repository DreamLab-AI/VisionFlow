// ESLint 9.0.0+ Flat Config for VisionFlow Client
// Handles TypeScript and React with built-in ESLint capabilities

export default [
  // Ignore patterns
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'build/**',
      '.vite/**',
      'coverage/**',
      '**/*.d.ts',
      'src/types/generated/**',
      'scripts/**',
      '__mocks__/**',
      '.claude-flow/**',
      '.next/**',
    ],
  },

  // CommonJS files
  {
    files: ['**/*.{cjs,mjs}'],
    languageOptions: {
      sourceType: 'module',
      ecmaVersion: 2020,
    },
  },

  // JavaScript files
  {
    files: ['src/**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        React: 'readonly',
        JSX: 'readonly',
      },
    },
    rules: {
      'no-console': [
        'warn',
        {
          allow: ['warn', 'error', 'debug'],
        },
      ],
      'no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      'no-undef': 'warn',
      'no-empty': 'warn',
      'no-trailing-spaces': 'warn',
    },
  },

  // TypeScript and TSX files (basic checking without @typescript-eslint)
  // Note: Full TypeScript support requires @typescript-eslint/parser
  // For now, we use a permissive approach
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        React: 'readonly',
        JSX: 'readonly',
      },
    },
    rules: {
      // Disable rules that require TypeScript parser
      'no-undef': 'off',
      'no-console': [
        'warn',
        {
          allow: ['warn', 'error', 'debug'],
        },
      ],
      'no-empty': 'warn',
      'no-trailing-spaces': 'warn',
    },
  },

  // Test files
  {
    files: ['**/*.{test,spec}.{ts,tsx,js,jsx}', '**/__tests__/**/*.{ts,tsx,js,jsx}'],
    languageOptions: {
      globals: {
        describe: 'readonly',
        it: 'readonly',
        test: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
        vi: 'readonly',
        jest: 'readonly',
      },
    },
  },
];
