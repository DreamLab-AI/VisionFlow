# Backup & Restore

## Database Backup

### Manual Backup

```bash
# Backup database
pg_dump -U visionflow -h localhost visionflow > backup.sql

# Compressed backup
pg_dump -U visionflow -h localhost visionflow | gzip > backup.sql.gz
```

### Automated Backup

**backup.sh**:

```bash
#!/bin/bash

BACKUP_DIR="/backup/visionflow"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="visionflow_${TIMESTAMP}.sql.gz"

# Create backup
pg_dump -U visionflow visionflow | gzip > "$BACKUP_DIR/$FILENAME"

# Keep only last 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: $FILENAME"
```

**Cron job**:

```cron
# Daily backup at 2 AM
0 2 * * * /opt/visionflow/scripts/backup.sh
```

### Docker Backup

```bash
docker-compose exec db pg_dump -U visionflow visionflow > backup.sql
```

## Storage Backup

### S3 Sync

```bash
# Sync to backup bucket
aws s3 sync s3://visionflow-data s3://visionflow-backup
```

### Local Storage

```bash
# Backup storage directory
tar -czf storage_backup.tar.gz /var/lib/visionflow/storage
```

## Restore Database

```bash
# Restore from backup
psql -U visionflow -h localhost visionflow < backup.sql

# Restore compressed backup
gunzip -c backup.sql.gz | psql -U visionflow visionflow
```

## Disaster Recovery

### Recovery Point Objective (RPO)

- Database: 1 hour (hourly backups)
- Storage: 24 hours (daily backups)

### Recovery Time Objective (RTO)

- Database: 15 minutes
- Full system: 1 hour

### Recovery Procedure

1. **Stop services**:
   ```bash
   docker-compose down
   ```

2. **Restore database**:
   ```bash
   psql -U visionflow visionflow < latest_backup.sql
   ```

3. **Restore storage**:
   ```bash
   aws s3 sync s3://visionflow-backup s3://visionflow-data
   ```

4. **Start services**:
   ```bash
   docker-compose up -d
   ```

5. **Verify**:
   ```bash
   curl http://localhost:9090/health
   ```

## Backup Testing

Regularly test backups:

```bash
# Monthly restore test
0 3 1 * * /opt/visionflow/scripts/test-restore.sh
```
