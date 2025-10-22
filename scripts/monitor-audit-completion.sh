#!/bin/bash
# Audit Completion Monitor
# Checks for Vircadia and Agent Container audit files every 60 seconds

DOCS_DIR="/home/devuser/workspace/project/docs"
VIRCADIA_AUDIT="${DOCS_DIR}/audit-vircadia-settings.md"
CONTAINER_AUDIT="${DOCS_DIR}/audit-agent-container-settings.md"
LOG_FILE="${DOCS_DIR}/audit-monitor.log"

echo "=== Audit Completion Monitor ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

check_count=0
max_checks=60  # Maximum 1 hour (60 checks * 60 seconds)

while [ $check_count -lt $max_checks ]; do
    check_count=$((check_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    vircadia_exists=false
    container_exists=false

    # Check for Vircadia audit
    if [ -f "$VIRCADIA_AUDIT" ]; then
        vircadia_exists=true
        vircadia_size=$(stat -f%z "$VIRCADIA_AUDIT" 2>/dev/null || stat -c%s "$VIRCADIA_AUDIT" 2>/dev/null)
        echo "[$timestamp] ✅ Vircadia audit found (${vircadia_size} bytes)" | tee -a "$LOG_FILE"
    else
        echo "[$timestamp] ⏳ Vircadia audit not found" | tee -a "$LOG_FILE"
    fi

    # Check for Container audit
    if [ -f "$CONTAINER_AUDIT" ]; then
        container_exists=true
        container_size=$(stat -f%z "$CONTAINER_AUDIT" 2>/dev/null || stat -c%s "$CONTAINER_AUDIT" 2>/dev/null)
        echo "[$timestamp] ✅ Container audit found (${container_size} bytes)" | tee -a "$LOG_FILE"
    else
        echo "[$timestamp] ⏳ Container audit not found" | tee -a "$LOG_FILE"
    fi

    # If both exist, trigger integration
    if [ "$vircadia_exists" = true ] && [ "$container_exists" = true ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "[$timestamp] 🎉 BOTH AUDITS COMPLETE!" | tee -a "$LOG_FILE"
        echo "[$timestamp] Ready for integration analysis" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        # Display audit statistics
        echo "=== Audit Statistics ===" | tee -a "$LOG_FILE"
        echo "Vircadia Audit: ${vircadia_size} bytes" | tee -a "$LOG_FILE"
        echo "Container Audit: ${container_size} bytes" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        # Signal integration ready
        touch "${DOCS_DIR}/.audits-complete"
        echo "[$timestamp] Created integration trigger: ${DOCS_DIR}/.audits-complete" | tee -a "$LOG_FILE"

        exit 0
    fi

    # If at least one exists, show partial progress
    if [ "$vircadia_exists" = true ] || [ "$container_exists" = true ]; then
        echo "[$timestamp] 📊 Partial progress - waiting for remaining audit(s)" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"

    # Wait 60 seconds before next check
    sleep 60
done

# Timeout reached
echo "[$timestamp] ⏱️  Monitoring timeout reached after ${max_checks} checks (60 minutes)" | tee -a "$LOG_FILE"
echo "[$timestamp] Audits still pending - manual integration required" | tee -a "$LOG_FILE"
exit 1
