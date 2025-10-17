use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use log::{debug, info};
use rusqlite::{Connection, params, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::ontology_downloader::{OntologyBlock, OntologyRelationship, RelationshipType};

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

pub struct OntologyStorage {
    conn: Arc<Mutex<Connection>>,
}

impl OntologyStorage {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        storage.initialize_schema()?;
        Ok(storage)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
        };

        storage.initialize_schema()?;
        Ok(storage)
    }

    fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS ontology_blocks (
                id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                downloaded_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_blocks_source_file
                ON ontology_blocks(source_file);
            CREATE INDEX IF NOT EXISTS idx_ontology_blocks_content_hash
                ON ontology_blocks(content_hash);

            CREATE TABLE IF NOT EXISTS ontology_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                property_key TEXT NOT NULL,
                property_value TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES ontology_blocks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_properties_block_id
                ON ontology_properties(block_id);
            CREATE INDEX IF NOT EXISTS idx_ontology_properties_key
                ON ontology_properties(property_key);

            CREATE TABLE IF NOT EXISTS ontology_owl_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_order INTEGER NOT NULL,
                FOREIGN KEY (block_id) REFERENCES ontology_blocks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_owl_content_block_id
                ON ontology_owl_content(block_id);

            CREATE TABLE IF NOT EXISTS ontology_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                class_name TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES ontology_blocks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_classes_block_id
                ON ontology_classes(block_id);
            CREATE INDEX IF NOT EXISTS idx_ontology_classes_name
                ON ontology_classes(class_name);

            CREATE TABLE IF NOT EXISTS ontology_owl_properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                property_name TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES ontology_blocks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_owl_properties_block_id
                ON ontology_owl_properties(block_id);
            CREATE INDEX IF NOT EXISTS idx_ontology_owl_properties_name
                ON ontology_owl_properties(property_name);

            CREATE TABLE IF NOT EXISTS ontology_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                FOREIGN KEY (block_id) REFERENCES ontology_blocks(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_ontology_relationships_block_id
                ON ontology_relationships(block_id);
            CREATE INDEX IF NOT EXISTS idx_ontology_relationships_subject
                ON ontology_relationships(subject);
            CREATE INDEX IF NOT EXISTS idx_ontology_relationships_object
                ON ontology_relationships(object);
            CREATE INDEX IF NOT EXISTS idx_ontology_relationships_type
                ON ontology_relationships(relationship_type);

            CREATE TABLE IF NOT EXISTS sync_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )?;

        info!("Database schema initialized successfully");
        Ok(())
    }

    pub fn save_block(&self, block: &OntologyBlock) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let tx = conn.unchecked_transaction()?;

        tx.execute(
            r#"
            INSERT OR REPLACE INTO ontology_blocks
                (id, source_file, title, content_hash, downloaded_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                &block.id,
                &block.source_file,
                &block.title,
                &block.content_hash,
                &block.downloaded_at.to_rfc3339(),
                Utc::now().to_rfc3339(),
            ],
        )?;

        tx.execute(
            "DELETE FROM ontology_properties WHERE block_id = ?1",
            params![&block.id],
        )?;

        for (key, values) in &block.properties {
            for value in values {
                tx.execute(
                    r#"
                    INSERT INTO ontology_properties (block_id, property_key, property_value)
                    VALUES (?1, ?2, ?3)
                    "#,
                    params![&block.id, key, value],
                )?;
            }
        }

        tx.execute(
            "DELETE FROM ontology_owl_content WHERE block_id = ?1",
            params![&block.id],
        )?;

        for (idx, content) in block.owl_content.iter().enumerate() {
            tx.execute(
                r#"
                INSERT INTO ontology_owl_content (block_id, content, content_order)
                VALUES (?1, ?2, ?3)
                "#,
                params![&block.id, content, idx as i32],
            )?;
        }

        tx.execute(
            "DELETE FROM ontology_classes WHERE block_id = ?1",
            params![&block.id],
        )?;

        for class in &block.classes {
            tx.execute(
                r#"
                INSERT INTO ontology_classes (block_id, class_name)
                VALUES (?1, ?2)
                "#,
                params![&block.id, class],
            )?;
        }

        tx.execute(
            "DELETE FROM ontology_owl_properties WHERE block_id = ?1",
            params![&block.id],
        )?;

        for property in &block.properties_list {
            tx.execute(
                r#"
                INSERT INTO ontology_owl_properties (block_id, property_name)
                VALUES (?1, ?2)
                "#,
                params![&block.id, property],
            )?;
        }

        tx.execute(
            "DELETE FROM ontology_relationships WHERE block_id = ?1",
            params![&block.id],
        )?;

        for rel in &block.relationships {
            let rel_type = relationship_type_to_string(&rel.relationship_type);
            tx.execute(
                r#"
                INSERT INTO ontology_relationships
                    (block_id, subject, predicate, object, relationship_type)
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
                params![&block.id, &rel.subject, &rel.predicate, &rel.object, rel_type],
            )?;
        }

        tx.commit()?;

        debug!("Saved ontology block: {}", block.id);
        Ok(())
    }

    pub fn save_blocks(&self, blocks: &[OntologyBlock]) -> Result<usize> {
        let mut count = 0;
        for block in blocks {
            self.save_block(block)?;
            count += 1;
        }

        info!("Saved {} ontology blocks to database", count);
        Ok(count)
    }

    pub fn get_block(&self, id: &str) -> Result<Option<OntologyBlock>> {
        let conn = self.conn.lock().unwrap();

        let block_data: Option<(String, String, String, String, String)> = conn
            .query_row(
                r#"
                SELECT id, source_file, title, content_hash, downloaded_at
                FROM ontology_blocks
                WHERE id = ?1
                "#,
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .optional()?;

        if let Some((id, source_file, title, content_hash, downloaded_at)) = block_data {
            let downloaded_at = DateTime::parse_from_rfc3339(&downloaded_at)?
                .with_timezone(&Utc);

            let properties = self.get_properties(&conn, &id)?;
            let owl_content = self.get_owl_content(&conn, &id)?;
            let classes = self.get_classes(&conn, &id)?;
            let properties_list = self.get_owl_properties(&conn, &id)?;
            let relationships = self.get_relationships(&conn, &id)?;

            Ok(Some(OntologyBlock {
                id,
                source_file,
                title,
                properties,
                owl_content,
                classes,
                properties_list,
                relationships,
                downloaded_at,
                content_hash,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_all_blocks(&self) -> Result<Vec<OntologyBlock>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"
            SELECT id, source_file, title, content_hash, downloaded_at
            FROM ontology_blocks
            ORDER BY downloaded_at DESC
            "#,
        )?;

        let block_ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(conn);

        let mut blocks = Vec::new();
        for id in block_ids {
            if let Some(block) = self.get_block(&id)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    pub fn search_by_class(&self, class_name: &str) -> Result<Vec<OntologyBlock>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"
            SELECT DISTINCT block_id
            FROM ontology_classes
            WHERE class_name LIKE ?1
            "#,
        )?;

        let pattern = format!("%{}%", class_name);
        let block_ids: Vec<String> = stmt
            .query_map(params![pattern], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(conn);

        let mut blocks = Vec::new();
        for id in block_ids {
            if let Some(block) = self.get_block(&id)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    pub fn search_by_property(&self, property_key: &str) -> Result<Vec<OntologyBlock>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            r#"
            SELECT DISTINCT block_id
            FROM ontology_properties
            WHERE property_key = ?1
            "#,
        )?;

        let block_ids: Vec<String> = stmt
            .query_map(params![property_key], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(conn);

        let mut blocks = Vec::new();
        for id in block_ids {
            if let Some(block) = self.get_block(&id)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    pub fn delete_block(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();

        let rows_affected = conn.execute(
            "DELETE FROM ontology_blocks WHERE id = ?1",
            params![id],
        )?;

        Ok(rows_affected > 0)
    }

    pub fn clear_all(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute_batch(
            r#"
            DELETE FROM ontology_relationships;
            DELETE FROM ontology_owl_properties;
            DELETE FROM ontology_classes;
            DELETE FROM ontology_owl_content;
            DELETE FROM ontology_properties;
            DELETE FROM ontology_blocks;
            DELETE FROM sync_metadata;
            "#,
        )?;

        info!("Cleared all ontology data from database");
        Ok(())
    }

    pub fn set_sync_metadata(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            r#"
            INSERT OR REPLACE INTO sync_metadata (key, value, updated_at)
            VALUES (?1, ?2, ?3)
            "#,
            params![key, value, Utc::now().to_rfc3339()],
        )?;

        Ok(())
    }

    pub fn get_sync_metadata(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();

        let value: Option<String> = conn
            .query_row(
                "SELECT value FROM sync_metadata WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;

        Ok(value)
    }

    pub fn get_statistics(&self) -> Result<DatabaseStatistics> {
        let conn = self.conn.lock().unwrap();

        let total_blocks: i64 = conn.query_row(
            "SELECT COUNT(*) FROM ontology_blocks",
            [],
            |row| row.get(0),
        )?;

        let total_classes: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT class_name) FROM ontology_classes",
            [],
            |row| row.get(0),
        )?;

        let total_properties: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT property_name) FROM ontology_owl_properties",
            [],
            |row| row.get(0),
        )?;

        let total_relationships: i64 = conn.query_row(
            "SELECT COUNT(*) FROM ontology_relationships",
            [],
            |row| row.get(0),
        )?;

        let last_sync: Option<String> = self.get_sync_metadata("last_sync_time")?;
        let last_sync_time = last_sync
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        Ok(DatabaseStatistics {
            total_blocks: total_blocks as usize,
            total_classes: total_classes as usize,
            total_properties: total_properties as usize,
            total_relationships: total_relationships as usize,
            last_sync_time,
        })
    }

    fn get_properties(
        &self,
        conn: &Connection,
        block_id: &str,
    ) -> Result<std::collections::HashMap<String, Vec<String>>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT property_key, property_value
            FROM ontology_properties
            WHERE block_id = ?1
            "#,
        )?;

        let mut properties = std::collections::HashMap::new();
        let rows = stmt.query_map(params![block_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        for row in rows {
            let (key, value) = row?;
            properties
                .entry(key)
                .or_insert_with(Vec::new)
                .push(value);
        }

        Ok(properties)
    }

    fn get_owl_content(&self, conn: &Connection, block_id: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT content
            FROM ontology_owl_content
            WHERE block_id = ?1
            ORDER BY content_order
            "#,
        )?;

        let contents: Vec<String> = stmt
            .query_map(params![block_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(contents)
    }

    fn get_classes(&self, conn: &Connection, block_id: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT class_name
            FROM ontology_classes
            WHERE block_id = ?1
            "#,
        )?;

        let classes: Vec<String> = stmt
            .query_map(params![block_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(classes)
    }

    fn get_owl_properties(&self, conn: &Connection, block_id: &str) -> Result<Vec<String>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT property_name
            FROM ontology_owl_properties
            WHERE block_id = ?1
            "#,
        )?;

        let properties: Vec<String> = stmt
            .query_map(params![block_id], |row| row.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(properties)
    }

    fn get_relationships(
        &self,
        conn: &Connection,
        block_id: &str,
    ) -> Result<Vec<OntologyRelationship>> {
        let mut stmt = conn.prepare(
            r#"
            SELECT subject, predicate, object, relationship_type
            FROM ontology_relationships
            WHERE block_id = ?1
            "#,
        )?;

        let relationships: Vec<OntologyRelationship> = stmt
            .query_map(params![block_id], |row| {
                Ok(OntologyRelationship {
                    subject: row.get(0)?,
                    predicate: row.get(1)?,
                    object: row.get(2)?,
                    relationship_type: string_to_relationship_type(&row.get::<_, String>(3)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(relationships)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DatabaseStatistics {
    pub total_blocks: usize,
    pub total_classes: usize,
    pub total_properties: usize,
    pub total_relationships: usize,
    pub last_sync_time: Option<DateTime<Utc>>,
}

fn relationship_type_to_string(rel_type: &RelationshipType) -> String {
    match rel_type {
        RelationshipType::SubClassOf => "SubClassOf".to_string(),
        RelationshipType::ObjectProperty => "ObjectProperty".to_string(),
        RelationshipType::DataProperty => "DataProperty".to_string(),
        RelationshipType::DisjointWith => "DisjointWith".to_string(),
        RelationshipType::EquivalentTo => "EquivalentTo".to_string(),
        RelationshipType::InverseOf => "InverseOf".to_string(),
        RelationshipType::Domain => "Domain".to_string(),
        RelationshipType::Range => "Range".to_string(),
        RelationshipType::Other(s) => s.clone(),
    }
}

fn string_to_relationship_type(s: &str) -> RelationshipType {
    match s {
        "SubClassOf" => RelationshipType::SubClassOf,
        "ObjectProperty" => RelationshipType::ObjectProperty,
        "DataProperty" => RelationshipType::DataProperty,
        "DisjointWith" => RelationshipType::DisjointWith,
        "EquivalentTo" => RelationshipType::EquivalentTo,
        "InverseOf" => RelationshipType::InverseOf,
        "Domain" => RelationshipType::Domain,
        "Range" => RelationshipType::Range,
        _ => RelationshipType::Other(s.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_create_storage() {
        let storage = OntologyStorage::in_memory().unwrap();
        let stats = storage.get_statistics().unwrap();
        assert_eq!(stats.total_blocks, 0);
    }

    #[test]
    fn test_save_and_retrieve_block() {
        let storage = OntologyStorage::in_memory().unwrap();

        let mut properties = HashMap::new();
        properties.insert("term-id".to_string(), vec!["123".to_string()]);

        let block = OntologyBlock {
            id: "test:block:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test Block".to_string(),
            properties,
            owl_content: vec!["Declaration(Class(test:Class))".to_string()],
            classes: vec!["test:Class".to_string()],
            properties_list: vec!["test:property".to_string()],
            relationships: vec![OntologyRelationship {
                subject: "test:A".to_string(),
                predicate: "rdfs:subClassOf".to_string(),
                object: "test:B".to_string(),
                relationship_type: RelationshipType::SubClassOf,
            }],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();

        let retrieved = storage.get_block("test:block:1").unwrap();
        assert!(retrieved.is_some());

        let retrieved_block = retrieved.unwrap();
        assert_eq!(retrieved_block.id, "test:block:1");
        assert_eq!(retrieved_block.title, "Test Block");
        assert_eq!(retrieved_block.classes.len(), 1);
        assert_eq!(retrieved_block.relationships.len(), 1);
    }

    #[test]
    fn test_search_by_class() {
        let storage = OntologyStorage::in_memory().unwrap();

        let block = OntologyBlock {
            id: "test:block:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test Block".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec!["mv:Avatar".to_string()],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();

        let results = storage.search_by_class("Avatar").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test:block:1");
    }

    #[test]
    fn test_statistics() {
        let storage = OntologyStorage::in_memory().unwrap();

        let block = OntologyBlock {
            id: "test:block:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test Block".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec!["test:Class1".to_string(), "test:Class2".to_string()],
            properties_list: vec!["test:prop1".to_string()],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();

        let stats = storage.get_statistics().unwrap();
        assert_eq!(stats.total_blocks, 1);
        assert_eq!(stats.total_classes, 2);
        assert_eq!(stats.total_properties, 1);
    }

    #[test]
    fn test_clear_all() {
        let storage = OntologyStorage::in_memory().unwrap();

        let block = OntologyBlock {
            id: "test:block:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test Block".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec![],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();
        assert_eq!(storage.get_statistics().unwrap().total_blocks, 1);

        storage.clear_all().unwrap();
        assert_eq!(storage.get_statistics().unwrap().total_blocks, 0);
    }
}
