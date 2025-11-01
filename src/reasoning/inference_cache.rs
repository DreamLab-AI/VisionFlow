/
/
/
/
/
/

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sha1::{Sha1, Digest};
use std::path::Path;
use crate::reasoning::{
    custom_reasoner::{InferredAxiom, OntologyReasoner},
    ReasoningError, ReasoningResult,
};

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedInference {
    pub ontology_id: i64,
    pub ontology_checksum: String,
    pub inferred_axioms: Vec<InferredAxiom>,
    pub cached_at: i64, 
}

/
pub struct InferenceCache {
    db_path: String,
}

impl InferenceCache {
    
    pub fn new(db_path: impl AsRef<Path>) -> ReasoningResult<Self> {
        let db_path = db_path.as_ref().to_string_lossy().to_string();

        
        let conn = Connection::open(&db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS inference_cache (
                ontology_id INTEGER PRIMARY KEY,
                ontology_checksum TEXT NOT NULL,
                inferred_axioms TEXT NOT NULL,
                cached_at INTEGER NOT NULL
            )",
            [],
        )?;

        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_checksum
             ON inference_cache(ontology_checksum)",
            [],
        )?;

        Ok(Self { db_path })
    }

    
    pub fn get_or_compute(
        &self,
        ontology_id: i64,
        reasoner: &dyn OntologyReasoner,
        ontology: &crate::reasoning::custom_reasoner::Ontology,
    ) -> ReasoningResult<Vec<InferredAxiom>> {
        let checksum = self.compute_ontology_checksum(ontology);

        
        if let Some(cached) = self.load_from_cache(ontology_id)? {
            if self.is_valid(&cached, &checksum) {
                
                return Ok(cached.inferred_axioms);
            }
        }

        
        let start = std::time::Instant::now();
        let inferred = reasoner.infer_axioms(ontology)?;
        let duration = start.elapsed();

        log::info!(
            "Inference computed for ontology {} in {:?} (cache miss)",
            ontology_id,
            duration
        );

        
        self.store_to_cache(ontology_id, &checksum, &inferred)?;

        Ok(inferred)
    }

    
    fn compute_ontology_checksum(
        &self,
        ontology: &crate::reasoning::custom_reasoner::Ontology,
    ) -> String {
        let mut hasher = Sha1::new();

        
        let mut class_iris: Vec<_> = ontology.classes.keys().collect();
        class_iris.sort();
        for iri in class_iris {
            hasher.update(iri.as_bytes());
        }

        
        let mut subclass_pairs: Vec<_> = ontology.subclass_of
            .iter()
            .flat_map(|(child, parents)| {
                parents.iter().map(move |parent| (child, parent))
            })
            .collect();
        subclass_pairs.sort();
        for (child, parent) in subclass_pairs {
            hasher.update(child.as_bytes());
            hasher.update(b"->");
            hasher.update(parent.as_bytes());
        }

        
        for disjoint_set in &ontology.disjoint_classes {
            let mut classes: Vec<_> = disjoint_set.iter().collect();
            classes.sort();
            for class in classes {
                hasher.update(class.as_bytes());
                hasher.update(b",");
            }
        }

        
        let mut equiv_pairs: Vec<_> = ontology.equivalent_classes
            .iter()
            .flat_map(|(a, equivalents)| {
                equivalents.iter().map(move |b| (a, b))
            })
            .collect();
        equiv_pairs.sort();
        for (a, b) in equiv_pairs {
            hasher.update(a.as_bytes());
            hasher.update(b"==");
            hasher.update(b.as_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    
    pub(crate) fn load_from_cache(&self, ontology_id: i64) -> ReasoningResult<Option<CachedInference>> {
        let conn = Connection::open(&self.db_path)?;

        let mut stmt = conn.prepare(
            "SELECT ontology_checksum, inferred_axioms, cached_at
             FROM inference_cache
             WHERE ontology_id = ?"
        )?;

        let mut rows = stmt.query([ontology_id])?;

        if let Some(row) = rows.next()? {
            let checksum: String = row.get(0)?;
            let axioms_json: String = row.get(1)?;
            let cached_at: i64 = row.get(2)?;

            let inferred_axioms: Vec<InferredAxiom> = serde_json::from_str(&axioms_json)
                .map_err(|e| ReasoningError::Cache(format!("Failed to deserialize axioms: {}", e)))?;

            Ok(Some(CachedInference {
                ontology_id,
                ontology_checksum: checksum,
                inferred_axioms,
                cached_at,
            }))
        } else {
            Ok(None)
        }
    }

    
    fn store_to_cache(
        &self,
        ontology_id: i64,
        checksum: &str,
        inferred: &[InferredAxiom],
    ) -> ReasoningResult<()> {
        let conn = Connection::open(&self.db_path)?;

        let axioms_json = serde_json::to_string(inferred)
            .map_err(|e| ReasoningError::Cache(format!("Failed to serialize axioms: {}", e)))?;

        let cached_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO inference_cache
             (ontology_id, ontology_checksum, inferred_axioms, cached_at)
             VALUES (?, ?, ?, ?)",
            params![ontology_id, checksum, axioms_json, cached_at],
        )?;

        Ok(())
    }

    
    fn is_valid(&self, cached: &CachedInference, current_checksum: &str) -> bool {
        
        if cached.ontology_checksum != current_checksum {
            log::info!(
                "Cache invalid for ontology {} (checksum mismatch)",
                cached.ontology_id
            );
            return false;
        }

        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let ttl = 3600; 
        if now - cached.cached_at > ttl {
            log::info!(
                "Cache expired for ontology {} (TTL exceeded)",
                cached.ontology_id
            );
            return false;
        }

        true
    }

    
    pub fn invalidate(&self, ontology_id: i64) -> ReasoningResult<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "DELETE FROM inference_cache WHERE ontology_id = ?",
            [ontology_id],
        )?;
        Ok(())
    }

    
    pub fn clear_all(&self) -> ReasoningResult<()> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute("DELETE FROM inference_cache", [])?;
        Ok(())
    }

    
    pub fn get_stats(&self) -> ReasoningResult<CacheStats> {
        let conn = Connection::open(&self.db_path)?;

        let total_entries: i64 = conn.query_row(
            "SELECT COUNT(*) FROM inference_cache",
            [],
            |row| row.get(0),
        )?;

        let total_size: i64 = conn.query_row(
            "SELECT SUM(LENGTH(inferred_axioms)) FROM inference_cache",
            [],
            |row| row.get(0),
        ).unwrap_or(0);

        Ok(CacheStats {
            total_entries: total_entries as usize,
            total_size_bytes: total_size as usize,
        })
    }
}

/
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning::custom_reasoner::{CustomReasoner, Ontology, OWLClass};
    use tempfile::TempDir;
    use std::collections::{HashMap, HashSet};

    fn create_test_ontology() -> Ontology {
        let mut ontology = Ontology::default();

        ontology.classes.insert("A".to_string(), OWLClass {
            iri: "A".to_string(),
            label: Some("Class A".to_string()),
            parent_class_iri: None,
        });

        ontology.classes.insert("B".to_string(), OWLClass {
            iri: "B".to_string(),
            label: Some("Class B".to_string()),
            parent_class_iri: Some("A".to_string()),
        });

        ontology.subclass_of.insert("B".to_string(),
            vec!["A".to_string()].into_iter().collect());

        ontology
    }

    #[test]
    fn test_cache_hit() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_test_ontology();

        
        let start = std::time::Instant::now();
        let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let duration1 = start.elapsed();

        
        let start = std::time::Instant::now();
        let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let duration2 = start.elapsed();

        assert_eq!(result1, result2);
        assert!(duration2 < duration1); 

        println!("Cache miss: {:?}, Cache hit: {:?}", duration1, duration2);
    }

    #[test]
    fn test_checksum_invalidation() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let mut ontology = create_test_ontology();

        
        let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        
        ontology.classes.insert("C".to_string(), OWLClass {
            iri: "C".to_string(),
            label: Some("Class C".to_string()),
            parent_class_iri: Some("B".to_string()),
        });

        
        let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        
        assert_ne!(result1.len(), result2.len());
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_test_ontology();

        
        cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        cache.get_or_compute(2, &reasoner, &ontology).unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_size_bytes > 0);
    }

    #[test]
    fn test_cache_invalidate() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_test_ontology();

        
        cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        
        cache.invalidate(1).unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 0);
    }
}
