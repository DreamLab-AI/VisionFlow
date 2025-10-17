# Settings System Migration Checklist

## Overview

This checklist guides you through migrating from the legacy YAML-based settings system to the new SQLite-backed settings system. Follow these steps in order to ensure a smooth transition.

## Pre-Migration Phase

### 1. Backup Existing Data

- [ ] Backup current `settings.yaml` file
  ```bash
  cp /app/settings.yaml /backup/settings-$(date +%Y%m%d).yaml
  ```

- [ ] Backup all user settings files
  ```bash
  tar -czf /backup/user_settings-$(date +%Y%m%d).tar.gz /app/user_settings/
  ```

- [ ] Document any custom settings modifications
  ```bash
  diff /app/settings.yaml data/settings.yaml > /backup/custom-changes.diff
  ```

### 2. Environment Preparation

- [ ] Set `ADMIN_PUBKEY` environment variable for bootstrap admin
  ```bash
  export ADMIN_PUBKEY="your_nostr_pubkey_hex"
  ```

- [ ] Verify database directory is writable
  ```bash
  mkdir -p /app/data
  chmod 755 /app/data
  ```

- [ ] Ensure SQLite3 is installed
  ```bash
  sqlite3 --version
  ```

### 3. Review Current Settings

- [ ] Document all custom physics parameters
- [ ] List all power users who should have elevated permissions
- [ ] Identify any non-standard configuration values
- [ ] Review integration settings (RagFlow, Perplexity, OpenAI, etc.)

## Migration Phase

### 4. Database Initialization

- [ ] Stop the application server
  ```bash
  systemctl stop visionflow
  # or
  docker-compose down
  ```

- [ ] Initialize SQLite database schema
  ```bash
  sqlite3 /app/data/settings.db < schema/settings_schema.sql
  ```

- [ ] Verify schema creation
  ```bash
  sqlite3 /app/data/settings.db ".tables"
  # Expected: settings, user_settings, users, physics_settings, settings_audit_log, etc.
  ```

### 5. Data Import

- [ ] Run automatic migration script (if available)
  ```bash
  cargo run --bin migrate_settings -- \
    --yaml /backup/settings-20251017.yaml \
    --database /app/data/settings.db
  ```

- [ ] Or manually import settings using SQL
  ```bash
  # Parse YAML to JSON first
  python3 scripts/yaml_to_json.py /backup/settings.yaml > /tmp/settings.json

  # Import to SQLite
  sqlite3 /app/data/settings.db < scripts/import_settings.sql
  ```

- [ ] Migrate user-specific settings
  ```bash
  for yaml_file in /app/user_settings/*.yaml; do
    pubkey=$(basename "$yaml_file" .yaml)
    cargo run --bin migrate_user_settings -- \
      --yaml "$yaml_file" \
      --pubkey "$pubkey" \
      --database /app/data/settings.db
  done
  ```

### 6. Verification

- [ ] Verify global settings imported correctly
  ```sql
  sqlite3 /app/data/settings.db "SELECT COUNT(*) FROM settings;"
  # Expected: ~200+ rows
  ```

- [ ] Check physics settings
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT graph_type, spring_k, repel_k FROM physics_settings;"
  ```

- [ ] Verify user settings
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT user_id, COUNT(*) FROM user_settings GROUP BY user_id;"
  ```

- [ ] Check users table
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT pubkey, is_power_user FROM users;"
  ```

### 7. Bootstrap Administrator

- [ ] Create initial power user
  ```sql
  sqlite3 /app/data/settings.db \
    "INSERT INTO users (pubkey, display_name, is_power_user, created_at)
     VALUES ('$ADMIN_PUBKEY', 'Admin', TRUE, CURRENT_TIMESTAMP)
     ON CONFLICT(pubkey) DO UPDATE SET is_power_user = TRUE;"
  ```

- [ ] Verify admin user
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT * FROM users WHERE is_power_user = TRUE;"
  ```

## Post-Migration Phase

### 8. Application Startup

- [ ] Update application configuration to use SQLite
  - Remove YAML file path references
  - Set database path: `/app/data/settings.db`
  - Configure connection pool size

- [ ] Start application server
  ```bash
  systemctl start visionflow
  # or
  docker-compose up -d
  ```

- [ ] Monitor startup logs for errors
  ```bash
  tail -f /app/logs/visionflow.log | grep -i "settings\|database\|migration"
  ```

### 9. Smoke Testing

- [ ] Test GET `/api/settings`
  ```bash
  curl http://localhost:4000/api/settings | jq .
  ```

- [ ] Test GET `/api/settings/user/:pubkey` (as authenticated user)
  ```bash
  curl http://localhost:4000/api/settings/user/abc123... \
    -H "Authorization: Nostr abc123..."
  ```

- [ ] Test PUT `/api/settings` (as power user)
  ```bash
  curl -X PUT http://localhost:4000/api/settings \
    -H "Authorization: Nostr $ADMIN_PUBKEY" \
    -H "Content-Type: application/json" \
    -d '{
      "visualisation": {
        "rendering": {
          "ambientLightIntensity": 0.6
        }
      }
    }'
  ```

- [ ] Verify settings update reflected in database
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT value FROM settings
     WHERE key = 'visualisation.rendering.ambient_light_intensity';"
  ```

### 10. WebSocket Testing

- [ ] Connect WebSocket client
  ```javascript
  const ws = new WebSocket('ws://localhost:4000/ws');
  ws.onmessage = (event) => console.log('Received:', event.data);
  ```

- [ ] Subscribe to settings updates
  ```javascript
  ws.send(JSON.stringify({ type: 'settings:subscribe' }));
  ```

- [ ] Update a setting and verify WebSocket message received
- [ ] Unsubscribe
  ```javascript
  ws.send(JSON.stringify({ type: 'settings:unsubscribe' }));
  ```

### 11. Validation Testing

- [ ] Test validation endpoint with valid data
  ```bash
  curl -X POST http://localhost:4000/api/settings/validate \
    -H "Content-Type: application/json" \
    -d '{ "visualisation": { "rendering": { "ambientLightIntensity": 0.7 } } }'
  ```

- [ ] Test validation with invalid data (should fail)
  ```bash
  curl -X POST http://localhost:4000/api/settings/validate \
    -H "Content-Type: application/json" \
    -d '{ "visualisation": { "rendering": { "ambientLightIntensity": 999 } } }'
  # Expected: 400 Bad Request with validation error
  ```

### 12. Performance Testing

- [ ] Run load test on settings endpoint
  ```bash
  ab -n 1000 -c 10 http://localhost:4000/api/settings
  ```

- [ ] Measure database query performance
  ```sql
  sqlite3 /app/data/settings.db \
    "EXPLAIN QUERY PLAN SELECT * FROM user_settings WHERE user_id = 'abc123...';"
  # Verify indexes are being used
  ```

- [ ] Check connection pool metrics
  ```bash
  curl http://localhost:4000/api/metrics | grep "db_pool"
  ```

### 13. User Migration Verification

- [ ] For each existing user, verify their settings loaded correctly
- [ ] Test user-specific overrides work as expected
- [ ] Verify settings inheritance (user override → global default)

### 14. Audit Log Verification

- [ ] Check audit log entries created during migration
  ```sql
  sqlite3 /app/data/settings.db \
    "SELECT * FROM settings_audit_log ORDER BY timestamp DESC LIMIT 10;"
  ```

- [ ] Verify permission grants logged
- [ ] Test audit log API endpoint
  ```bash
  curl http://localhost:4000/api/admin/audit/permissions \
    -H "Authorization: Nostr $ADMIN_PUBKEY"
  ```

## Cleanup Phase

### 15. Remove Legacy Files

⚠️ **Only proceed after confirming everything works correctly!**

- [ ] Archive legacy YAML files (don't delete yet)
  ```bash
  mkdir -p /archive/legacy-settings
  mv /app/settings.yaml /archive/legacy-settings/
  mv /app/user_settings/ /archive/legacy-settings/
  ```

- [ ] Update Docker volumes to exclude YAML paths
  ```yaml
  # docker-compose.yml - Remove:
  # - ./data/settings.yaml:/app/settings.yaml
  # - ./data/user_settings:/app/user_settings
  ```

- [ ] Remove YAML parsing code (after 30 days of stable operation)
  - `src/models/user_settings.rs` (legacy YAML-based implementation)
  - YAML-specific deserialization functions in `src/config/mod.rs`

### 16. Documentation Updates

- [ ] Update README.md with new settings approach
- [ ] Update deployment documentation
- [ ] Update API documentation
- [ ] Add migration notes to CHANGELOG

### 17. Monitoring Setup

- [ ] Set up database size monitoring
  ```bash
  watch -n 300 'du -h /app/data/settings.db'
  ```

- [ ] Configure backup cron job
  ```bash
  crontab -e
  # Add:
  # 0 2 * * * sqlite3 /app/data/settings.db ".backup /backup/settings-$(date +\%Y\%m\%d).db"
  ```

- [ ] Set up audit log alerts for suspicious activity
  ```sql
  -- Monitor for excessive permission grants
  SELECT COUNT(*) FROM settings_audit_log
  WHERE action = 'permission_granted'
  AND timestamp > datetime('now', '-1 hour');
  ```

## Rollback Procedure

If issues occur, follow these steps to rollback:

### Emergency Rollback

- [ ] Stop application
  ```bash
  systemctl stop visionflow
  ```

- [ ] Restore YAML files from backup
  ```bash
  cp /backup/settings-20251017.yaml /app/settings.yaml
  tar -xzf /backup/user_settings-20251017.tar.gz -C /app/
  ```

- [ ] Revert code to use YAML settings
  ```bash
  git checkout legacy-yaml-settings-branch
  ```

- [ ] Restart application
  ```bash
  systemctl start visionflow
  ```

- [ ] Document rollback reason for post-mortem

## Post-Migration Monitoring (First 7 Days)

### Daily Checks

- [ ] **Day 1:** Verify all functionality working, no errors in logs
- [ ] **Day 2:** Check database size growth, verify backups running
- [ ] **Day 3:** Test all user scenarios, verify permissions working
- [ ] **Day 4:** Review audit logs, check for anomalies
- [ ] **Day 5:** Performance testing, optimize slow queries if needed
- [ ] **Day 6:** User feedback collection, address any issues
- [ ] **Day 7:** Final verification, plan legacy file removal

### Weekly Checks (Weeks 2-4)

- [ ] **Week 2:** Monitor database performance, vacuum if needed
- [ ] **Week 3:** Review security audit logs, check for suspicious activity
- [ ] **Week 4:** Final sign-off, schedule legacy file deletion

## Success Criteria

Migration is considered successful when:

- [ ] All global settings loaded correctly from SQLite
- [ ] User-specific overrides work as expected
- [ ] WebSocket updates propagate in real-time
- [ ] Validation prevents invalid settings
- [ ] Audit log tracks all changes
- [ ] Performance meets or exceeds YAML-based system
- [ ] No data loss or corruption
- [ ] All users can authenticate and modify their settings
- [ ] Power users can manage global settings
- [ ] No critical errors in logs for 7+ days

## Known Issues and Workarounds

### Issue 1: Database Locked

**Symptom:** `database is locked` errors

**Workaround:**
```bash
sqlite3 /app/data/settings.db "PRAGMA busy_timeout = 5000;"
```

### Issue 2: Large User Settings Import Slow

**Symptom:** User settings import takes >1 hour for 1000+ users

**Workaround:** Use batch inserts
```sql
BEGIN TRANSACTION;
-- Insert all user settings
INSERT INTO user_settings ...
INSERT INTO user_settings ...
COMMIT;
```

### Issue 3: WebSocket Connections Not Receiving Updates

**Symptom:** Settings change not propagated to clients

**Workaround:**
1. Check WebSocket server running: `netstat -an | grep 4000`
2. Verify settings actor receiving updates
3. Restart application if needed

## Support and Resources

- **Documentation:** `/docs/settings-*.md`
- **Migration Scripts:** `/scripts/migrate_*.sh`
- **Schema Files:** `/schema/settings_schema.sql`
- **Test Suite:** `cargo test settings_migration`
- **Rollback Guide:** See "Rollback Procedure" above

## Migration Timeline Recommendation

| Phase | Duration | Notes |
|-------|----------|-------|
| Pre-Migration | 1 day | Backup, review, prepare |
| Migration | 2-4 hours | Import data, verify |
| Post-Migration Testing | 1 day | Smoke tests, validation |
| Monitoring | 7 days | Daily checks |
| Cleanup | After 30 days | Remove legacy files |

**Total estimated time:** 30-40 days for complete migration with stable operation

## Contact

For migration support:
- GitHub Issues: https://github.com/visionflow/visionflow/issues
- Documentation: `/docs/`
- Emergency rollback: See "Rollback Procedure" section

---

**Last Updated:** 2025-10-17
**Schema Version:** 2.0
