#!/bin/bash

# Trigger physics parameter update for VisionFlow
echo "Sending physics update to trigger ForceComputeActor..."

# Send update to trigger UpdateSimulationParams message
curl -X POST http://172.18.0.10:4000/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "visualisation": {
      "graphs": {
        "visionflow": {
          "physics": {
            "springStrength": 5.0,
            "repulsionStrength": 50.0,
            "velocityDecay": 0.2
          }
        }
      }
    }
  }' 2>/dev/null | jq -r '.visualisation.graphs.visionflow.physics' 2>/dev/null

echo "Physics update sent. Monitoring for UpdateSimulationParams in logs..."
