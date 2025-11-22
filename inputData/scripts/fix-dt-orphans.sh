#!/bin/bash

# Script to fix DT domain orphan concepts by adding semantic parents
# Based on the task requirements for 288 DT orphan concepts

PAGES_DIR="/home/devuser/workspace/logseq/mainKnowledgeGraph/pages"
LOG_FILE="/home/devuser/workspace/logseq/scripts/fix-dt-orphans.log"
FIXED_COUNT=0
FAILED_FILES=""

# Initialize log
echo "DT Orphan Fix Log - $(date)" > "$LOG_FILE"
echo "================================" >> "$LOG_FILE"

# Function to determine parent class based on filename and content
get_parent_class() {
    local filename="$1"
    local filepath="$2"
    local content=""

    if [ -f "$filepath" ]; then
        content=$(cat "$filepath" 2>/dev/null)
    fi

    # Named concept patterns - check filename patterns
    case "$filename" in
        *"Layer"*|*"layer"*)
            echo "[[SystemArchitectureLayer]]"
            return
            ;;
        *"Metric"*|*"metric"*|*"AUC"*|*"AUROC"*|*"F1"*|*"Precision"*|*"Recall"*)
            echo "[[PerformanceMetric]]"
            return
            ;;
        *"Telepresence"*|*"Telecollaboration"*|*"telepresence"*|*"telecollaboration"*)
            echo "[[TelecollaborationTechnology]]"
            return
            ;;
        *"Compliance"*|*"compliance"*|*"AML"*|*"KYC"*|*"ESG"*|*"Regulatory"*)
            echo "[[ComplianceFramework]]"
            return
            ;;
        *"Blockchain"*|*"blockchain"*|*"Cryptocurrency"*|*"Token"*|*"NFT"*|*"DAO"*)
            echo "[[BlockchainTechnology]]"
            return
            ;;
        *"Robot"*|*"robot"*|*"Haptic"*|*"haptic"*)
            echo "[[RoboticsTechnology]]"
            return
            ;;
        *"Neural"*|*"neural"*|*"AI"*|*"Ai"*|*"ML"*|*"Model"*|*"Learning"*|*"Multimodal"*)
            echo "[[ArtificialIntelligenceTechnology]]"
            return
            ;;
        *"VR"*|*"AR"*|*"XR"*|*"Virtual"*|*"Augmented"*|*"Immersive"*|*"Metaverse"*)
            echo "[[ImmersiveTechnology]]"
            return
            ;;
        *"Dataset"*|*"dataset"*|*"Data"*|*"data"*)
            echo "[[DataResource]]"
            return
            ;;
        *"Content"*|*"content"*|*"Moderation"*|*"Generation"*)
            echo "[[ContentManagementSystem]]"
            return
            ;;
        *"Fraud"*|*"fraud"*|*"Detection"*|*"detection"*|*"Defect"*|*"defect"*)
            echo "[[DetectionSystem]]"
            return
            ;;
        *"Decoder"*|*"Encoder"*|*"Transformer"*|*"Attention"*)
            echo "[[NeuralNetworkArchitecture]]"
            return
            ;;
        *"DALL"*|*"dall"*|*"Text"*|*"Image"*|*"Generative"*)
            echo "[[GenerativeAISystem]]"
            return
            ;;
    esac

    # Check source-domain in content
    if echo "$content" | grep -q "source-domain:: blockchain"; then
        echo "[[BlockchainTechnology]]"
        return
    elif echo "$content" | grep -q "source-domain:: ai"; then
        echo "[[ArtificialIntelligenceTechnology]]"
        return
    elif echo "$content" | grep -q "source-domain:: robotics"; then
        echo "[[RoboticsTechnology]]"
        return
    elif echo "$content" | grep -q "source-domain:: metaverse"; then
        echo "[[MetaverseTechnology]]"
        return
    fi

    # Default parent
    echo "[[DisruptiveTechnology]]"
}

# Function to add is-subclass-of property to a file
add_parent_property() {
    local filepath="$1"
    local parent="$2"
    local filename=$(basename "$filepath")

    # Check if file already has is-subclass-of
    if grep -q "is-subclass-of::" "$filepath" 2>/dev/null; then
        echo "SKIP (already has parent): $filename" >> "$LOG_FILE"
        return 1
    fi

    # Find the line after "ontology:: true" to insert is-subclass-of
    if grep -q "ontology:: true" "$filepath"; then
        # Insert is-subclass-of after ontology:: true
        sed -i "/ontology:: true/a\\    - is-subclass-of:: $parent" "$filepath"
        echo "FIXED: $filename -> $parent" >> "$LOG_FILE"
        return 0
    else
        echo "WARN (no ontology block): $filename" >> "$LOG_FILE"
        return 1
    fi
}

# Main processing loop
echo ""
echo "Processing orphan files..."

# Find all files with ontology:: true that lack is-subclass-of
for file in "$PAGES_DIR"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")

        # Check if file has ontology:: true but no is-subclass-of
        if grep -q "ontology:: true" "$file" && ! grep -q "is-subclass-of::" "$file"; then
            parent=$(get_parent_class "$filename" "$file")

            if add_parent_property "$file" "$parent"; then
                ((FIXED_COUNT++))
            fi
        fi
    fi
done

echo "" >> "$LOG_FILE"
echo "================================" >> "$LOG_FILE"
echo "Total files fixed: $FIXED_COUNT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "Processing complete!"
echo "Files fixed: $FIXED_COUNT"
echo "Log saved to: $LOG_FILE"

exit 0
