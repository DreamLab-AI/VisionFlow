#!/bin/bash
# Debug script to check why backend isn't starting

echo "🔍 Debugging Backend Issues"
echo "=========================="
echo ""

# Check if container is running
echo "1. Checking container status..."
docker ps | grep visionflow_container > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Container is running"
else
    echo "❌ Container is not running"
    echo "   Run: ./scripts/launch.sh"
    exit 1
fi

echo ""
echo "2. Checking supervisord status..."
docker exec visionflow_container supervisorctl status

echo ""
echo "3. Checking Rust backend logs..."
echo "--- Last 20 lines of rust.log ---"
docker exec visionflow_container tail -20 /app/logs/rust.log 2>/dev/null || echo "No rust.log found"

echo ""
echo "--- Last 20 lines of rust-error.log ---"
docker exec visionflow_container tail -20 /app/logs/rust-error.log 2>/dev/null || echo "No rust-error.log found"

echo ""
echo "4. Checking if cargo-watch is installed..."
docker exec visionflow_container which cargo-watch > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ cargo-watch is installed"
else
    echo "❌ cargo-watch is NOT installed"
    echo "   Installing cargo-watch..."
    docker exec visionflow_container cargo install cargo-watch
fi

echo ""
echo "5. Testing Rust compilation..."
docker exec visionflow_container bash -c "cd /app && cargo check --features gpu 2>&1 | head -50"

echo ""
echo "6. Checking port 4000..."
docker exec visionflow_container lsof -i :4000 2>/dev/null || echo "Port 4000 is not in use"

echo ""
echo "7. Manually starting Rust backend (test)..."
echo "   Running: cargo run --features gpu"
docker exec visionflow_container bash -c "cd /app && timeout 5 cargo run --features gpu 2>&1" || true

echo ""
echo "8. Checking nginx upstream connectivity..."
docker exec visionflow_container curl -v http://127.0.0.1:4000/health 2>&1 | head -20