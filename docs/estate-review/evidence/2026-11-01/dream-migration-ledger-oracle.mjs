// Executes the original, pinned TypeScript validator using Node type stripping.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const { validateLedger } = await import(pathToFileURL(process.argv[2]).href);
const cases = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(({ id, row }) => ({
  id, rules: validateLedger(row, '2026-09-06').map(v => v.rule),
})), null, 2));
