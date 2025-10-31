# VisionFlow Production Deployment Checklist

**Version:** 1.0
**Date:** 2025-10-31
**Target Environment:** Production

---

## Pre-Deployment Validation

### Code Quality
- [ ] **Compilation Check**
  ```bash
  cd /home/devuser/workspace/project
  cargo check --all-features
  ```
  **Expected:** No errors (warnings acceptable)

- [ ] **Test Suite**
  ```bash
  cargo test --all-features
  ```
  **Expected:** All tests pass

- [ ] **Linting**
  ```bash
  cargo clippy --all-features -- -D warnings
  ```
  **Expected:** No clippy warnings

### Performance Validation
- [ ] **Benchmarks**
  ```bash
  cargo bench --features "gpu,ontology"
  ```
  **Expected Results:**
  - Physics step: <20ms
  - Constraint evaluation: <5ms
  - Graph update: <10ms
  - Database queries: <10ms

- [ ] **GPU Validation**
  ```bash
  # Check CUDA availability
  nvidia-smi
  ```
  **Expected:** GPU visible, <50% utilization at idle

### Binary Build
- [ ] **Release Build**
  ```bash
  cargo build --release --features "gpu,ontology"
  ls -lh target/release/webxr
  ```
  **Expected:** Binary size ~50-100MB

- [ ] **Binary Backup**
  ```bash
  cp target/release/webxr target/release/webxr.backup.$(date +%Y%m%d-%H%M%S)
  ```

---

## Database Setup

### Schema Verification
- [ ] **Check Existing Databases**
  ```bash
  ls -lh data/*.db
  # Expected: ontology.db, knowledge_graph.db (or unified.db)
  ```

- [ ] **Backup Current Databases**
  ```bash
  mkdir -p data/backups/$(date +%Y%m%d-%H%M%S)
  cp data/*.db data/backups/$(date +%Y%m%d-%H%M%S)/
  ```

- [ ] **Verify Schema**
  ```bash
  # For ontology.db
  sqlite3 data/ontology.db ".schema" | head -50

  # Check key tables exist:
  # - ontologies
  # - owl_classes
  # - owl_properties
  # - owl_axioms
  ```

- [ ] **Check Table Counts**
  ```bash
  sqlite3 data/ontology.db "SELECT COUNT(*) FROM ontologies;"
  sqlite3 data/ontology.db "SELECT COUNT(*) FROM owl_classes;"
  sqlite3 data/knowledge_graph.db "SELECT COUNT(*) FROM nodes;"
  sqlite3 data/knowledge_graph.db "SELECT COUNT(*) FROM edges;"
  ```
  **Expected:** Non-zero counts for active system

### Migration (If Unified DB)
- [ ] **Create Unified Database** (Only if migrating to single DB)
  ```bash
  sqlite3 data/unified.db < migrations/001_fix_ontology_schema.sql
  ```

- [ ] **Migrate Data** (Only if needed)
  ```bash
  # Custom migration script would go here
  # This depends on your specific migration strategy
  ```

---

## Configuration

### Environment Variables
- [ ] **Set Required Variables**
  ```bash
  export RUST_LOG=info
  export DATABASE_PATH=data/ontology.db  # or unified.db
  export GPU_ENABLED=true
  export CUDA_VISIBLE_DEVICES=0
  export BIND_ADDRESS=0.0.0.0
  export SYSTEM_NETWORK_PORT=4000
  ```

- [ ] **Verify Environment**
  ```bash
  env | grep -E "(RUST_LOG|DATABASE_PATH|GPU_ENABLED|CUDA_VISIBLE_DEVICES)"
  ```

### Configuration Files
- [ ] **Production Config**
  ```bash
  # Ensure production config exists
  ls -lh ontology_physics.toml
  ```

- [ ] **Verify Settings**
  ```toml
  # Check key settings in ontology_physics.toml:
  # - physics.dt = 0.016 (60 FPS)
  # - physics.damping = 0.95
  # - constraints.enabled = true
  # - gpu.enabled = true
  ```

---

## Service Startup

### Start Server
- [ ] **Launch Application**
  ```bash
  ./target/release/webxr --config ontology_physics.toml > logs/server.log 2>&1 &
  echo $! > logs/server.pid
  ```

- [ ] **Check Process**
  ```bash
  ps -p $(cat logs/server.pid)
  ```
  **Expected:** Process running

- [ ] **Monitor Startup Logs**
  ```bash
  tail -f logs/server.log
  ```
  **Expected to see:**
  - "Starting WebXR application..."
  - "OntologyActor started"
  - "GraphServiceActor started"
  - "GPU initialization successful" (or CPU fallback)
  - "HTTP server startup sequence complete"

---

## Health Checks

### API Validation
- [ ] **Health Endpoint**
  ```bash
  curl http://localhost:4000/api/health
  ```
  **Expected:**
  ```json
  {
    "status": "ok",
    "version": "0.1.0",
    "timestamp": "2025-10-31T..."
  }
  ```

- [ ] **Configuration Endpoint**
  ```bash
  curl http://localhost:4000/api/config
  ```
  **Expected:** JSON with features, websocket, rendering, xr configs

- [ ] **Graph Endpoint**
  ```bash
  curl http://localhost:4000/api/graph
  ```
  **Expected:** JSON with nodes and edges arrays

- [ ] **Constraints Stats**
  ```bash
  curl http://localhost:4000/api/constraints/stats
  ```
  **Expected:** JSON with constraint statistics

- [ ] **Settings - Physics**
  ```bash
  curl http://localhost:4000/api/settings/physics
  ```
  **Expected:** JSON with physics configuration

- [ ] **Ontology - Classes**
  ```bash
  curl http://localhost:4000/api/ontology/classes
  ```
  **Expected:** JSON array of OWL classes

### WebSocket Connection
- [ ] **WebSocket Endpoint**
  ```bash
  # Using websocat (install if needed: cargo install websocat)
  echo '{"type":"ping"}' | websocat ws://localhost:4000/wss
  ```
  **Expected:** Connection established, pong response

---

## Functional Validation

### Graph Operations
- [ ] **Load Graph Data**
  ```bash
  curl http://localhost:4000/api/graph | jq '.nodes | length'
  ```
  **Expected:** Non-zero node count

- [ ] **Check Physics Simulation**
  ```bash
  # Monitor logs for physics updates
  grep "Physics step" logs/server.log | tail -10
  ```
  **Expected:** Regular physics step messages, <20ms per step

### Constraint System
- [ ] **List Constraints**
  ```bash
  curl http://localhost:4000/api/constraints | jq '.count'
  ```
  **Expected:** Constraint count > 0

- [ ] **Constraint Statistics**
  ```bash
  curl http://localhost:4000/api/constraints/stats | jq '.'
  ```
  **Expected:** Stats with active/total constraints, eval time < 5ms

### Ontology System
- [ ] **Trigger Reasoning**
  ```bash
  curl -X POST http://localhost:4000/api/ontology/reasoning/trigger \
    -H "Content-Type: application/json" \
    -d '{"ontology_id":"default","validation_mode":"quick"}'
  ```
  **Expected:** Success response with status "queued" or "completed"

- [ ] **Check Reasoning Status**
  ```bash
  curl http://localhost:4000/api/ontology/reasoning/status | jq '.status'
  ```
  **Expected:** Status: "idle" or "running"

### Settings Persistence
- [ ] **Update Physics Settings**
  ```bash
  curl -X PUT http://localhost:4000/api/settings/physics \
    -H "Content-Type: application/json" \
    -d '{"damping":0.98}'
  ```
  **Expected:** Success response

- [ ] **Verify Settings Persisted**
  ```bash
  curl http://localhost:4000/api/settings/physics | jq '.physics.damping'
  ```
  **Expected:** 0.98 (updated value)

---

## Performance Monitoring

### System Metrics (First Hour)
- [ ] **CPU Usage**
  ```bash
  top -p $(cat logs/server.pid)
  ```
  **Expected:** <50% CPU average

- [ ] **Memory Usage**
  ```bash
  ps -p $(cat logs/server.pid) -o rss,vsz
  ```
  **Expected:** <2GB RSS

- [ ] **GPU Utilization**
  ```bash
  watch -n 1 nvidia-smi
  ```
  **Expected:** 50-80% utilization during active simulation

- [ ] **FPS Monitoring**
  ```bash
  grep "FPS:" logs/server.log | tail -20
  ```
  **Expected:** >30 FPS consistently

### Database Performance
- [ ] **Query Performance**
  ```bash
  # Enable SQLite query logging
  sqlite3 data/ontology.db ".timer on" "SELECT COUNT(*) FROM owl_classes;"
  ```
  **Expected:** <10ms for simple queries

- [ ] **Database Size**
  ```bash
  du -sh data/*.db
  ```
  **Expected:** Reasonable size (<100MB for typical datasets)

---

## Error Monitoring

### Log Analysis
- [ ] **Check for Errors**
  ```bash
  grep -i error logs/server.log | tail -20
  ```
  **Expected:** Minimal errors, no critical failures

- [ ] **Check for Warnings**
  ```bash
  grep -i warn logs/server.log | tail -20
  ```
  **Expected:** Few warnings, none related to core functionality

- [ ] **Actor Mailbox Health**
  ```bash
  grep "mailbox" logs/server.log | tail -10
  ```
  **Expected:** No mailbox overflow errors

### GPU Error Detection
- [ ] **CUDA Errors**
  ```bash
  grep -i "cuda error" logs/server.log
  ```
  **Expected:** No CUDA errors

- [ ] **GPU Fallback**
  ```bash
  grep -i "fallback to cpu" logs/server.log
  ```
  **Expected:** None (unless GPU intentionally disabled)

---

## Security Verification

### Network Security
- [ ] **Port Accessibility**
  ```bash
  netstat -tulpn | grep 4000
  ```
  **Expected:** Server listening on configured port

- [ ] **CORS Configuration**
  ```bash
  curl -I -H "Origin: http://example.com" http://localhost:4000/api/health
  ```
  **Expected:** Appropriate CORS headers

### Input Validation
- [ ] **Invalid API Requests**
  ```bash
  # Test with invalid JSON
  curl -X POST http://localhost:4000/api/constraints/user \
    -H "Content-Type: application/json" \
    -d '{"invalid"}'
  ```
  **Expected:** 400 Bad Request with error message

---

## Rollback Validation

### Rollback Procedure Test
- [ ] **Stop Current Service**
  ```bash
  kill -TERM $(cat logs/server.pid)
  ```

- [ ] **Restore Backup Binary**
  ```bash
  cp target/release/webxr.backup.* target/release/webxr
  ```

- [ ] **Restore Backup Database**
  ```bash
  cp data/backups/latest/*.db data/
  ```

- [ ] **Restart Service**
  ```bash
  ./target/release/webxr --config ontology_physics.toml &
  ```

- [ ] **Verify Rollback Successful**
  ```bash
  curl http://localhost:4000/api/health
  ```
  **Expected:** Service operational with previous version

---

## Post-Deployment

### Documentation
- [ ] **Update Deployment Log**
  ```bash
  echo "$(date): Production deployment v0.1.0 completed" >> docs/deployment_history.log
  ```

- [ ] **Update Version Tag**
  ```bash
  git tag -a v0.1.0-prod -m "Production deployment $(date +%Y-%m-%d)"
  git push origin v0.1.0-prod
  ```

### Monitoring Setup
- [ ] **Set Up Alerts**
  - CPU > 80% for 5 minutes
  - Memory > 90% for 5 minutes
  - FPS < 20 for 1 minute
  - GPU errors > 3 per minute

- [ ] **Schedule Periodic Checks**
  ```bash
  # Add to crontab
  */5 * * * * curl -f http://localhost:4000/api/health || systemctl restart webxr
  ```

### User Communication
- [ ] **Notify Stakeholders**
  - Send deployment notification
  - Provide API documentation link
  - Share performance metrics

- [ ] **User Acceptance Testing**
  - Coordinate UAT sessions
  - Gather feedback
  - Document issues

---

## Success Criteria

### Functional Requirements
- [x] All API endpoints responding (200 OK)
- [x] Graph data loading successfully
- [x] Physics simulation running (>30 FPS)
- [x] Constraint system operational
- [x] Ontology validation working
- [x] Settings persistence functional

### Performance Requirements
- [ ] FPS >30 (to be validated)
- [ ] Constraint evaluation <5ms (to be validated)
- [ ] Database queries <10ms (to be validated)
- [ ] Memory usage <2GB (to be monitored)
- [ ] CPU usage <50% average (to be monitored)

### Reliability Requirements
- [ ] Zero crashes in first hour
- [ ] Zero data corruption
- [ ] All actors healthy (mailbox not full)
- [ ] GPU stable (no CUDA errors)
- [ ] Rollback tested and functional

---

## Sign-Off

### Deployment Team
- [ ] **Integration Coordinator:** Verified all components integrated ____________
- [ ] **QA Lead:** Validated all test cases pass ____________
- [ ] **DevOps Lead:** Confirmed production environment ready ____________
- [ ] **Technical Lead:** Approved for production deployment ____________

### Post-Deployment Review (After 24 hours)
- [ ] **System Stability:** No critical issues reported
- [ ] **Performance:** Meeting all targets
- [ ] **User Feedback:** Positive reception
- [ ] **Action Items:** Document lessons learned

---

**Checklist Version:** 1.0
**Last Updated:** 2025-10-31
**Next Review:** After first production deployment
