import js from '@eslint/js';
import simpleImportSort from 'eslint-plugin-simple-import-sort';
import globals from 'globals';

export default [
  js.configs.recommended,
  {
    languageOptions: {
      globals: {
        ...globals.node
      }
    },
    plugins: {
      'simple-import-sort': simpleImportSort
    },
    rules: {
      semi: ['error', 'always'],
      indent: ['error', 2],
      'max-len': ['error', { code: 100, tabWidth: 2, ignoreUrls: true }],
      'quotes': ['error', 'single', { avoidEscape: true }],
      'object-curly-spacing': ['error', 'always'],
      'space-before-function-paren': ['error', 'never'],
      'array-bracket-spacing': ['error', 'never'],
      'eol-last': ['error', 'always'],
      'no-trailing-spaces': 'error',
      'linebreak-style': ['error', 'unix'],
      'comma-dangle': ['error', 'never'],
      'no-duplicate-imports': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always'],
      'curly': ['error', 'all'],
      'yoda': ['error', 'never'],
      'simple-import-sort/imports': 'error',
      'simple-import-sort/exports': 'error'
    }
  }
];
