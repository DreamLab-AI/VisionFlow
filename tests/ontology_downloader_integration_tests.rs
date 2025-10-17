#[cfg(feature = "ontology")]
mod tests {
    use webxr::services::ontology_downloader::{OntologyDownloader, OntologyDownloaderConfig};
    use webxr::services::ontology_storage::OntologyStorage;
    use webxr::services::ontology_sync::OntologySync;

    #[test]
    fn test_config_validation() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        assert_eq!(config.github_token, "test_token");
        assert_eq!(config.repo_owner, "jjohare");
        assert_eq!(config.repo_name, "logseq");
        assert_eq!(config.base_path, "mainKnowledgeGraph/pages");
    }

    #[test]
    fn test_storage_initialization() {
        let storage = OntologyStorage::in_memory().unwrap();
        let stats = storage.get_statistics().unwrap();

        assert_eq!(stats.total_blocks, 0);
        assert_eq!(stats.total_classes, 0);
        assert_eq!(stats.total_properties, 0);
        assert_eq!(stats.total_relationships, 0);
    }

    #[test]
    fn test_downloader_creation() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        let downloader = OntologyDownloader::new(config);

        assert!(downloader.is_ok());
    }

    #[test]
    fn test_empty_token_fails() {
        let config = OntologyDownloaderConfig::with_token(String::new());
        let downloader = OntologyDownloader::new(config);

        assert!(downloader.is_err());
    }

    #[tokio::test]
    async fn test_progress_initialization() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        let downloader = OntologyDownloader::new(config).unwrap();

        let progress = downloader.get_progress().await;
        assert_eq!(progress.total_files, 0);
        assert_eq!(progress.processed_files, 0);
        assert_eq!(progress.ontology_blocks_found, 0);
        assert!(progress.errors.is_empty());
    }

    #[test]
    fn test_storage_metadata() {
        let storage = OntologyStorage::in_memory().unwrap();

        storage.set_sync_metadata("test_key", "test_value").unwrap();
        let value = storage.get_sync_metadata("test_key").unwrap();

        assert_eq!(value, Some("test_value".to_string()));
    }

    #[test]
    fn test_storage_clear() {
        let storage = OntologyStorage::in_memory().unwrap();

        use std::collections::HashMap;
        use webxr::services::ontology_downloader::OntologyBlock;
        use chrono::Utc;

        let block = OntologyBlock {
            id: "test:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec!["test:Class".to_string()],
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

    #[test]
    fn test_relationship_types() {
        use webxr::services::ontology_downloader::RelationshipType;

        let types = vec![
            RelationshipType::SubClassOf,
            RelationshipType::ObjectProperty,
            RelationshipType::DataProperty,
            RelationshipType::DisjointWith,
            RelationshipType::EquivalentTo,
            RelationshipType::InverseOf,
            RelationshipType::Domain,
            RelationshipType::Range,
            RelationshipType::Other("Custom".to_string()),
        ];

        assert_eq!(types.len(), 9);
    }

    #[tokio::test]
    async fn test_sync_creation() {
        let config = OntologyDownloaderConfig::with_token("test_token".to_string());
        let storage = OntologyStorage::in_memory().unwrap();
        let sync = OntologySync::new(config, storage);

        assert!(sync.is_ok());
    }

    #[test]
    fn test_block_search_by_class() {
        let storage = OntologyStorage::in_memory().unwrap();

        use std::collections::HashMap;
        use webxr::services::ontology_downloader::OntologyBlock;
        use chrono::Utc;

        let block1 = OntologyBlock {
            id: "test:1".to_string(),
            source_file: "test1.md".to_string(),
            title: "Test 1".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec!["mv:Avatar".to_string()],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        let block2 = OntologyBlock {
            id: "test:2".to_string(),
            source_file: "test2.md".to_string(),
            title: "Test 2".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec!["mv:Entity".to_string()],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "def456".to_string(),
        };

        storage.save_block(&block1).unwrap();
        storage.save_block(&block2).unwrap();

        let results = storage.search_by_class("Avatar").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test:1");

        let results = storage.search_by_class("Entity").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test:2");
    }

    #[test]
    fn test_block_search_by_property() {
        let storage = OntologyStorage::in_memory().unwrap();

        use std::collections::HashMap;
        use webxr::services::ontology_downloader::OntologyBlock;
        use chrono::Utc;

        let mut properties = HashMap::new();
        properties.insert("maturity".to_string(), vec!["mature".to_string()]);

        let block = OntologyBlock {
            id: "test:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test".to_string(),
            properties,
            owl_content: vec![],
            classes: vec![],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();

        let results = storage.search_by_property("maturity").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "test:1");
    }

    #[test]
    fn test_block_deletion() {
        let storage = OntologyStorage::in_memory().unwrap();

        use std::collections::HashMap;
        use webxr::services::ontology_downloader::OntologyBlock;
        use chrono::Utc;

        let block = OntologyBlock {
            id: "test:1".to_string(),
            source_file: "test.md".to_string(),
            title: "Test".to_string(),
            properties: HashMap::new(),
            owl_content: vec![],
            classes: vec![],
            properties_list: vec![],
            relationships: vec![],
            downloaded_at: Utc::now(),
            content_hash: "abc123".to_string(),
        };

        storage.save_block(&block).unwrap();
        assert!(storage.get_block("test:1").unwrap().is_some());

        let deleted = storage.delete_block("test:1").unwrap();
        assert!(deleted);

        assert!(storage.get_block("test:1").unwrap().is_none());
    }

    #[test]
    fn test_list_all_blocks() {
        let storage = OntologyStorage::in_memory().unwrap();

        use std::collections::HashMap;
        use webxr::services::ontology_downloader::OntologyBlock;
        use chrono::Utc;

        for i in 1..=5 {
            let block = OntologyBlock {
                id: format!("test:{}", i),
                source_file: format!("test{}.md", i),
                title: format!("Test {}", i),
                properties: HashMap::new(),
                owl_content: vec![],
                classes: vec![],
                properties_list: vec![],
                relationships: vec![],
                downloaded_at: Utc::now(),
                content_hash: format!("hash{}", i),
            };
            storage.save_block(&block).unwrap();
        }

        let all_blocks = storage.list_all_blocks().unwrap();
        assert_eq!(all_blocks.len(), 5);
    }
}
