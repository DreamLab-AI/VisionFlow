#!/bin/bash

echo "=== Testing Physics Stability with GPU Parameters ==="
echo ""

# Test the physics update endpoint
echo "1. Testing physics parameter update:"
curl -X POST http://localhost:4000/api/physics/update \
  -H "Content-Type: application/json" \
  -d '{
    "physics": {
      "spring_k": 0.005,
      "repel_k": 50.0,
      "damping": 0.95,
      "dt": 0.016,
      "max_velocity": 1.0,
      "max_force": 1.0,
      "separation_radius": 2.0,
      "temperature": 0.01,
      "gravity_k": 0.0001,
      "min_distance": 0.15,
      "max_repulsion_dist": 50.0,
      "boundary_margin": 0.85,
      "boundary_force_strength": 2.0,
      "warmup_iterations": 200,
      "cooling_rate": 0.0001
    }
  }' | jq '.'

echo ""
echo "2. Monitoring physics logs for stability:"
echo "Watching for explosion/bouncing patterns..."
echo ""

# Monitor logs for physics issues
timeout 10s tail -f /workspace/ext/logs/rust-error.log | \
  grep -E "PHYSICS|GPU|repel_k|damping|velocity|explosion|unstable|NaN|Infinity" || true

echo ""
echo "3. Checking current simulation status:"
curl -s http://localhost:4000/api/health | jq '.physics'

echo ""
echo "=== Test Complete ==="
echo "If no explosion/NaN errors appeared, physics is stable!"