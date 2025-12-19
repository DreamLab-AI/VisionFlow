const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// US to UK spelling map
const spellingMap = {
  'color': 'colour',
  'colors': 'colours',
  'colored': 'coloured',
  'coloring': 'colouring',
  'favor': 'favour',
  'favors': 'favours',
  'favored': 'favoured',
  'favoring': 'favouring',
  'honor': 'honour',
  'honors': 'honours',
  'honored': 'honoured',
  'honoring': 'honouring',
  'behavior': 'behaviour',
  'behaviors': 'behaviours',
  'behavioral': 'behavioural',
  'neighbor': 'neighbour',
  'neighbors': 'neighbours',
  'neighboring': 'neighbouring',
  'organize': 'organise',
  'organizes': 'organises',
  'organized': 'organised',
  'organizing': 'organising',
  'organization': 'organisation',
  'organizations': 'organisations',
  'realize': 'realise',
  'realizes': 'realises',
  'realized': 'realised',
  'realizing': 'realising',
  'realization': 'realisation',
  'recognize': 'recognise',
  'recognizes': 'recognises',
  'recognized': 'recognised',
  'recognizing': 'recognising',
  'optimize': 'optimise',
  'optimizes': 'optimises',
  'optimized': 'optimised',
  'optimizing': 'optimising',
  'optimization': 'optimisation',
  'optimizations': 'optimisations',
  'analyze': 'analyse',
  'analyzes': 'analyses',
  'analyzed': 'analysed',
  'analyzing': 'analysing',
  'analyzer': 'analyser',
  'analyzers': 'analysers',
  'paralyze': 'paralyse',
  'paralyzes': 'paralyses',
  'paralyzed': 'paralysed',
  'paralyzing': 'paralysing',
  'center': 'centre',
  'centers': 'centres',
  'centered': 'centred',
  'centering': 'centring',
  'meter': 'metre',
  'meters': 'metres',
  'theater': 'theatre',
  'theaters': 'theatres',
  'fiber': 'fibre',
  'fibers': 'fibres',
  'defense': 'defence',
  'defenses': 'defences',
  'offense': 'offence',
  'offenses': 'offences',
  'license': 'licence',
  'licenses': 'licences',
  'catalog': 'catalogue',
  'catalogs': 'catalogues',
  'cataloged': 'catalogued',
  'cataloging': 'cataloguing',
  'dialog': 'dialogue',
  'dialogs': 'dialogues',
  'traveled': 'travelled',
  'traveler': 'traveller',
  'travelers': 'travellers',
  'traveling': 'travelling',
  'canceled': 'cancelled',
  'canceling': 'cancelling',
  'labeled': 'labelled',
  'labeling': 'labelling',
  'modeling': 'modelling',
  'modeled': 'modelled',
  'signaling': 'signalling',
  'signaled': 'signalled'
};

// Get all markdown and text files
const files = execSync('find /home/devuser/workspace/project/docs -type f \\( -name "*.md" -o -name "*.txt" \\)', { encoding: 'utf8' })
  .trim()
  .split('\n')
  .filter(Boolean);

const violations = [];
const violationsByWord = {};
const filesWithViolations = new Set();

// Scan each file
files.forEach(file => {
  try {
    const content = fs.readFileSync(file, 'utf8');
    const lines = content.split('\n');

    let inCodeBlock = false;

    lines.forEach((line, lineNum) => {
      // Track code blocks
      if (line.trim().startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        return;
      }

      // Skip code blocks
      if (inCodeBlock) return;

      // Remove inline code
      const cleanLine = line.replace(/`[^`]+`/g, '');

      // Check for US spellings
      Object.entries(spellingMap).forEach(([usWord, ukWord]) => {
        const regex = new RegExp('\\b' + usWord + '\\b', 'gi');
        const matches = cleanLine.match(regex);

        if (matches) {
          matches.forEach(match => {
            const violation = {
              file: file.replace('/home/devuser/workspace/project/docs/', ''),
              line: lineNum + 1,
              us_spelling: match,
              uk_spelling: match.toLowerCase() === usWord ? ukWord : ukWord.charAt(0).toUpperCase() + ukWord.slice(1),
              context: line.trim().substring(0, 100)
            };

            violations.push(violation);
            filesWithViolations.add(file);

            const key = usWord.toLowerCase();
            if (!violationsByWord[key]) {
              violationsByWord[key] = [];
            }
            violationsByWord[key].push(violation);
          });
        }
      });
    });
  } catch (err) {
    console.error(`Error processing ${file}: ${err.message}`);
  }
});

const totalFiles = files.length;
const cleanFiles = totalFiles - filesWithViolations.size;
const compliancePercentage = ((cleanFiles / totalFiles) * 100).toFixed(2);

const report = {
  violations: violations,
  violations_by_word: violationsByWord,
  files_with_violations: Array.from(filesWithViolations).map(f => f.replace('/home/devuser/workspace/project/docs/', '')),
  clean_files_count: cleanFiles,
  total_files: totalFiles,
  total_violations: violations.length,
  uk_compliance_percentage: compliancePercentage
};

// Write JSON report
fs.writeFileSync('/home/devuser/workspace/project/docs/working/hive-spelling-audit.json', JSON.stringify(report, null, 2));

console.log('Spelling audit complete');
console.log('Total files: ' + totalFiles);
console.log('Files with violations: ' + filesWithViolations.size);
console.log('Total violations: ' + violations.length);
console.log('UK compliance: ' + compliancePercentage + '%');
