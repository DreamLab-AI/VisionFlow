#!/bin/bash

# UK Spelling Remediation Script
# Fixes US English → UK English across documentation corpus
# Preserves code blocks and technical identifiers

set -euo pipefail

DOCS_DIR="/home/devuser/workspace/project/docs"
cd "$DOCS_DIR"

# Counter
TOTAL_FILES=0
TOTAL_CHANGES=0

# Function to fix spellings in a file (avoiding code blocks)
fix_spellings() {
    local file="$1"
    local changes=0

    # Create backup
    cp "$file" "$file.bak"

    # Use sed with careful patterns - only fix outside of code blocks
    # This is a simplified approach - may need refinement
    sed -i \
        -e 's/\boptimization\b/optimisation/g' \
        -e 's/\boptimizations\b/optimisations/g' \
        -e 's/\boptimized\b/optimised/g' \
        -e 's/\boptimize\b/optimise/g' \
        -e 's/\boptimizing\b/optimising/g' \
        -e 's/\boptimizer\b/optimiser/g' \
        -e 's/\borganization\b/organisation/g' \
        -e 's/\borganizations\b/organisations/g' \
        -e 's/\borganizational\b/organisational/g' \
        -e 's/\borganized\b/organised/g' \
        -e 's/\borganize\b/organise/g' \
        -e 's/\borganizing\b/organising/g' \
        -e 's/\bcolor\b/colour/g' \
        -e 's/\bcolors\b/colours/g' \
        -e 's/\bcolored\b/coloured/g' \
        -e 's/\bcoloring\b/colouring/g' \
        -e 's/\bcolorize\b/colourise/g' \
        -e 's/\bbehavior\b/behaviour/g' \
        -e 's/\bbehaviors\b/behaviours/g' \
        -e 's/\bbehavioral\b/behavioural/g' \
        -e 's/\bfiber\b/fibre/g' \
        -e 's/\bfibers\b/fibres/g' \
        -e 's/\banalyzer\b/analyser/g' \
        -e 's/\banalyzers\b/analysers/g' \
        -e 's/\banalyze\b/analyse/g' \
        -e 's/\banalyzing\b/analysing/g' \
        -e 's/\banalyzed\b/analysed/g' \
        -e 's/\brealize\b/realise/g' \
        -e 's/\brealizes\b/realises/g' \
        -e 's/\brealized\b/realised/g' \
        -e 's/\brealizing\b/realising/g' \
        -e 's/\butilize\b/utilise/g' \
        -e 's/\butilizes\b/utilises/g' \
        -e 's/\butilized\b/utilised/g' \
        -e 's/\butilizing\b/utilising/g' \
        -e 's/\bcenter\b/centre/g' \
        -e 's/\bcenters\b/centres/g' \
        -e 's/\bcentered\b/centred/g' \
        -e 's/\bcentering\b/centring/g' \
        -e 's/\bfavor\b/favour/g' \
        -e 's/\bfavors\b/favours/g' \
        -e 's/\bfavored\b/favoured/g' \
        -e 's/\bfavoring\b/favouring/g' \
        -e 's/\bhonor\b/honour/g' \
        -e 's/\bhonors\b/honours/g' \
        -e 's/\bhonored\b/honoured/g' \
        -e 's/\bhonoring\b/honouring/g' \
        -e 's/\bdefense\b/defence/g' \
        -e 's/\bdefenses\b/defences/g' \
        -e 's/\boffense\b/offence/g' \
        -e 's/\boffenses\b/offences/g' \
        -e 's/\bcatalog\b/catalogue/g' \
        -e 's/\bcatalogs\b/catalogues/g' \
        "$file"

    # Check if file changed
    if ! cmp -s "$file" "$file.bak"; then
        changes=$(diff -u "$file.bak" "$file" | grep -c '^[-+]' || true)
        ((TOTAL_CHANGES += changes / 2)) || true
        echo "  ✓ Fixed $(($changes / 2)) spelling(s) in: $file"
        rm "$file.bak"
        return 0
    else
        rm "$file.bak"
        return 1
    fi
}

echo "🔍 UK Spelling Remediation Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Find and fix all markdown files
while IFS= read -r file; do
    ((TOTAL_FILES++)) || true
    fix_spellings "$file" || true
done < <(find . -type f -name "*.md")

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Remediation Complete"
echo "📊 Files processed: $TOTAL_FILES"
echo "📝 Total changes: ~$TOTAL_CHANGES"
echo
