#!/bin/bash
# US to UK English Conversion Script
# Usage: ./convert-to-uk.sh [file_or_directory]

set -e

# Function to convert a single file
convert_file() {
    local file="$1"
    echo "Converting: $file"
    
    # Create backup
    cp "$file" "$file.backup"
    
    # Apply conversions using sed
    sed -i \
        -e 's/\b[Oo]rganize\b/&|TEMP|/g; s/organize|TEMP|/organise/g; s/Organize|TEMP|/Organise/g' \
        -e 's/\b[Oo]rganization\b/&|TEMP|/g; s/organization|TEMP|/organisation/g; s/Organization|TEMP|/Organisation/g' \
        -e 's/\b[Oo]rganizing\b/&|TEMP|/g; s/organizing|TEMP|/organising/g; s/Organizing|TEMP|/Organising/g' \
        -e 's/\b[Oo]rganized\b/&|TEMP|/g; s/organized|TEMP|/organised/g; s/Organized|TEMP|/Organised/g' \
        -e 's/\b[Rr]ealize\b/&|TEMP|/g; s/realize|TEMP|/realise/g; s/Realize|TEMP|/Realise/g' \
        -e 's/\b[Rr]ealization\b/&|TEMP|/g; s/realization|TEMP|/realisation/g; s/Realization|TEMP|/Realisation/g' \
        -e 's/\b[Rr]ealizing\b/&|TEMP|/g; s/realizing|TEMP|/realising/g; s/Realizing|TEMP|/Realising/g' \
        -e 's/\b[Rr]ealized\b/&|TEMP|/g; s/realized|TEMP|/realised/g; s/Realized|TEMP|/Realised/g' \
        -e 's/\b[Vv]isualize\b/&|TEMP|/g; s/visualize|TEMP|/visualise/g; s/Visualize|TEMP|/Visualise/g' \
        -e 's/\b[Vv]isualization\b/&|TEMP|/g; s/visualization|TEMP|/visualisation/g; s/Visualization|TEMP|/Visualisation/g' \
        -e 's/\b[Vv]isualizing\b/&|TEMP|/g; s/visualizing|TEMP|/visualising/g; s/Visualizing|TEMP|/Visualising/g' \
        -e 's/\b[Vv]isualized\b/&|TEMP|/g; s/visualized|TEMP|/visualised/g; s/Visualized|TEMP|/Visualised/g' \
        -e 's/\b[Oo]ptimize\b/&|TEMP|/g; s/optimize|TEMP|/optimise/g; s/Optimize|TEMP|/Optimise/g' \
        -e 's/\b[Oo]ptimization\b/&|TEMP|/g; s/optimization|TEMP|/optimisation/g; s/Optimization|TEMP|/Optimisation/g' \
        -e 's/\b[Oo]ptimizing\b/&|TEMP|/g; s/optimizing|TEMP|/optimising/g; s/Optimizing|TEMP|/Optimising/g' \
        -e 's/\b[Oo]ptimized\b/&|TEMP|/g; s/optimized|TEMP|/optimised/g; s/Optimized|TEMP|/Optimised/g' \
        -e 's/\b[Mm]aximize\b/&|TEMP|/g; s/maximize|TEMP|/maximise/g; s/Maximize|TEMP|/Maximise/g' \
        -e 's/\b[Mm]inimize\b/&|TEMP|/g; s/minimize|TEMP|/minimise/g; s/Minimize|TEMP|/Minimise/g' \
        -e 's/\b[Ss]tandardize\b/&|TEMP|/g; s/standardize|TEMP|/standardise/g; s/Standardize|TEMP|/Standardise/g' \
        -e 's/\b[Ss]tandardization\b/&|TEMP|/g; s/standardization|TEMP|/standardisation/g; s/Standardization|TEMP|/Standardisation/g' \
        -e 's/\b[Ss]ynchronize\b/&|TEMP|/g; s/synchronize|TEMP|/synchronise/g; s/Synchronize|TEMP|/Synchronise/g' \
        -e 's/\b[Ss]ynchronization\b/&|TEMP|/g; s/synchronization|TEMP|/synchronisation/g; s/Synchronization|TEMP|/Synchronisation/g' \
        -e 's/\b[Cc]entralize\b/&|TEMP|/g; s/centralize|TEMP|/centralise/g; s/Centralize|TEMP|/Centralise/g' \
        -e 's/\b[Cc]entralization\b/&|TEMP|/g; s/centralization|TEMP|/centralisation/g; s/Centralization|TEMP|/Centralisation/g' \
        -e 's/\b[Cc]haracterize\b/&|TEMP|/g; s/characterize|TEMP|/characterise/g; s/Characterize|TEMP|/Characterise/g' \
        -e 's/\b[Cc]haracterization\b/&|TEMP|/g; s/characterization|TEMP|/characterisation/g; s/Characterization|TEMP|/Characterisation/g' \
        -e 's/\b[Ss]pecialized\b/&|TEMP|/g; s/specialized|TEMP|/specialised/g; s/Specialized|TEMP|/Specialised/g' \
        -e 's/\b[Rr]ecognize\b/&|TEMP|/g; s/recognize|TEMP|/recognise/g; s/Recognize|TEMP|/Recognise/g' \
        -e 's/\b[Rr]ecognition\b/&|TEMP|/g; s/recognition|TEMP|/recognition/g; s/Recognition|TEMP|/Recognition/g' \
        -e 's/\b[Uu]tilize\b/&|TEMP|/g; s/utilize|TEMP|/utilise/g; s/Utilize|TEMP|/Utilise/g' \
        -e 's/\b[Cc]ustomize\b/&|TEMP|/g; s/customize|TEMP|/customise/g; s/Customize|TEMP|/Customise/g' \
        -e 's/\b[Cc]ustomization\b/&|TEMP|/g; s/customization|TEMP|/customisation/g; s/Customization|TEMP|/Customisation/g' \
        -e 's/\b[Pp]rioritize\b/&|TEMP|/g; s/prioritize|TEMP|/prioritise/g; s/Prioritize|TEMP|/Prioritise/g' \
        -e 's/\b[Nn]ormalize\b/&|TEMP|/g; s/normalize|TEMP|/normalise/g; s/Normalize|TEMP|/Normalise/g' \
        -e 's/\b[Ee]mphasize\b/&|TEMP|/g; s/emphasize|TEMP|/emphasise/g; s/Emphasize|TEMP|/Emphasise/g' \
        \
        -e 's/\b[Bb]ehavior\b/&|TEMP|/g; s/behavior|TEMP|/behaviour/g; s/Behavior|TEMP|/Behaviour/g' \
        -e 's/\b[Cc]olor\b/&|TEMP|/g; s/color|TEMP|/colour/g; s/Color|TEMP|/Colour/g' \
        -e 's/\b[Hh]onor\b/&|TEMP|/g; s/honor|TEMP|/honour/g; s/Honor|TEMP|/Honour/g' \
        -e 's/\b[Ll]abor\b/&|TEMP|/g; s/labor|TEMP|/labour/g; s/Labor|TEMP|/Labour/g' \
        -e 's/\b[Nn]eighbor\b/&|TEMP|/g; s/neighbor|TEMP|/neighbour/g; s/Neighbor|TEMP|/Neighbour/g' \
        -e 's/\b[Ff]avor\b/&|TEMP|/g; s/favor|TEMP|/favour/g; s/Favor|TEMP|/Favour/g' \
        -e 's/\b[Hh]arbor\b/&|TEMP|/g; s/harbor|TEMP|/harbour/g; s/Harbor|TEMP|/Harbour/g' \
        -e 's/\b[Ss]avor\b/&|TEMP|/g; s/savor|TEMP|/savour/g; s/Savor|TEMP|/Savour/g' \
        -e 's/\b[Vv]apor\b/&|TEMP|/g; s/vapor|TEMP|/vapour/g; s/Vapor|TEMP|/Vapour/g' \
        \
        -e 's/\b[Cc]enter\b/&|TEMP|/g; s/center|TEMP|/centre/g; s/Center|TEMP|/Centre/g' \
        -e 's/\b[Ff]iber\b/&|TEMP|/g; s/fiber|TEMP|/fibre/g; s/Fiber|TEMP|/Fibre/g' \
        -e 's/\b[Ll]iter\b/&|TEMP|/g; s/liter|TEMP|/litre/g; s/Liter|TEMP|/Litre/g' \
        -e 's/\b[Mm]eter\b/&|TEMP|/g; s/meter|TEMP|/metre/g; s/Meter|TEMP|/Metre/g' \
        -e 's/\b[Tt]heater\b/&|TEMP|/g; s/theater|TEMP|/theatre/g; s/Theater|TEMP|/Theatre/g' \
        \
        -e 's/\b[Dd]efense\b/&|TEMP|/g; s/defense|TEMP|/defence/g; s/Defense|TEMP|/Defence/g' \
        -e 's/\b[Ll]icense\b/&|TEMP|/g; s/license|TEMP|/licence/g; s/License|TEMP|/Licence/g' \
        -e 's/\b[Oo]ffense\b/&|TEMP|/g; s/offense|TEMP|/offence/g; s/Offense|TEMP|/Offence/g' \
        \
        -e 's/\b[Aa]nalog\b/&|TEMP|/g; s/analog|TEMP|/analogue/g; s/Analog|TEMP|/Analogue/g' \
        -e 's/\b[Cc]atalog\b/&|TEMP|/g; s/catalog|TEMP|/catalogue/g; s/Catalog|TEMP|/Catalogue/g' \
        -e 's/\b[Dd]ialog\b/&|TEMP|/g; s/dialog|TEMP|/dialogue/g; s/Dialog|TEMP|/Dialogue/g' \
        \
        -e 's/\bcolored\b/coloured/g' \
        -e 's/\bColored\b/Coloured/g' \
        -e 's/\bflavored\b/flavoured/g' \
        -e 's/\bFlavored\b/Flavoured/g' \
        -e 's/\bhonored\b/honoured/g' \
        -e 's/\bHonored\b/Honoured/g' \
        -e 's/\blabored\b/laboured/g' \
        -e 's/\bLabored\b/Laboured/g' \
        \
        -e 's/ gray / grey /g' \
        -e 's/ Gray / Grey /g' \
        -e 's/^gray /grey /g' \
        -e 's/^Gray /Grey /g' \
        -e 's/ gray$/ grey/g' \
        -e 's/ Gray$/ Grey/g' \
        "$file"
    
    # Check if any changes were made
    if ! diff -q "$file" "$file.backup" > /dev/null 2>&1; then
        echo "  ✓ Changes made to $file"
        rm "$file.backup"
        return 0
    else
        echo "  - No changes needed for $file"
        rm "$file.backup"
        return 1
    fi
}

# Function to process directory recursively
process_directory() {
    local dir="$1"
    local files_changed=0
    
    find "$dir" -name "*.md" -type f | while read -r file; do
        if convert_file "$file"; then
            files_changed=$((files_changed + 1))
        fi
    done
    
    return $files_changed
}

# Main script
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <file_or_directory>"
    echo "Converts US English to UK English in markdown files"
    exit 1
fi

target="$1"

if [[ ! -e "$target" ]]; then
    echo "Error: $target does not exist"
    exit 1
fi

echo "Converting US English to UK English..."
echo "Target: $target"
echo

if [[ -f "$target" ]]; then
    if [[ "$target" == *.md ]]; then
        convert_file "$target"
    else
        echo "Error: File must be a markdown (.md) file"
        exit 1
    fi
elif [[ -d "$target" ]]; then
    process_directory "$target"
else
    echo "Error: Target must be a file or directory"
    exit 1
fi

echo
echo "Conversion complete!"