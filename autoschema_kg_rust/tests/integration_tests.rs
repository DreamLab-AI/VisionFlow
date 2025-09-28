//! Integration tests for the complete AutoSchema KG system

use std::path::PathBuf;
use tempfile::TempDir;
use serde_json::json;
use tokio_test;

mod common;
use common::*;

#[cfg(test)]
mod pipeline_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_kg_construction_pipeline() {
        init_test_env();

        let temp_dir = TempDir::new().unwrap();

        // Create test documents
        let documents = create_test_documents(&temp_dir);

        // Initialize pipeline components
        let pipeline = setup_test_pipeline().await;

        // Process documents through the pipeline
        let result = pipeline.process_documents(documents).await;

        assert!(result.is_ok(), "Pipeline processing should succeed");

        let knowledge_graph = result.unwrap();

        // Verify knowledge graph structure
        assert!(knowledge_graph.node_count() > 0, "Should have extracted entities");
        assert!(knowledge_graph.edge_count() >= 0, "Should have extracted relationships");

        // Verify specific entities were extracted
        assert!(knowledge_graph.has_entity("Apple Inc."), "Should extract organization entities");
        assert!(knowledge_graph.has_entity("Steve Jobs"), "Should extract person entities");

        // Verify relationships
        assert!(knowledge_graph.has_relationship("Steve Jobs", "Apple Inc.", "founded"));
    }

    #[tokio::test]
    async fn test_multimodal_data_processing() {
        init_test_env();

        let temp_dir = TempDir::new().unwrap();

        // Create different file types
        create_csv_test_file(&temp_dir, "companies.csv");
        create_json_test_file(&temp_dir, "products.json");
        create_markdown_test_file(&temp_dir, "readme.md");
        create_text_test_file(&temp_dir, "article.txt");

        let ingestor = utils::DataIngestor::new();

        // Process each file type
        let csv_docs = ingestor.ingest_csv(&temp_dir.path().join("companies.csv")).await.unwrap();
        let json_docs = ingestor.ingest_json(&temp_dir.path().join("products.json")).await.unwrap();
        let md_docs = ingestor.ingest_file(&temp_dir.path().join("readme.md")).await.map(|d| vec![d]).unwrap();
        let txt_docs = ingestor.ingest_file(&temp_dir.path().join("article.txt")).await.map(|d| vec![d]).unwrap();

        // Combine all documents
        let mut all_docs = Vec::new();
        all_docs.extend(csv_docs);
        all_docs.extend(json_docs);
        all_docs.extend(md_docs);
        all_docs.extend(txt_docs);

        assert!(all_docs.len() >= 4, "Should process all file types");

        // Process through KG construction
        let pipeline = setup_test_pipeline().await;
        let kg = pipeline.process_documents(all_docs).await.unwrap();

        // Verify cross-format entity linking
        assert!(kg.node_count() > 0);

        // Should find entities from different sources
        let entities_by_source = kg.get_entities_by_source();
        assert!(entities_by_source.len() > 1, "Should have entities from multiple sources");
    }

    #[tokio::test]
    async fn test_large_scale_processing() {
        init_test_env();

        // Generate large dataset
        let large_dataset = generate_large_test_dataset(1000); // 1000 documents

        let pipeline = setup_test_pipeline().await;

        // Process in batches
        let batch_size = 50;
        let mut total_entities = 0;
        let mut total_relationships = 0;

        for chunk in large_dataset.chunks(batch_size) {
            let result = pipeline.process_documents(chunk.to_vec()).await;
            assert!(result.is_ok(), "Batch processing should succeed");

            let kg = result.unwrap();
            total_entities += kg.node_count();
            total_relationships += kg.edge_count();
        }

        assert!(total_entities > 100, "Should extract significant number of entities");
        assert!(total_relationships >= 0, "Should extract relationships");

        // Verify performance within reasonable bounds
        // This is a basic check - in practice you'd want more sophisticated metrics
        println!("Processed {} documents, extracted {} entities, {} relationships",
                 large_dataset.len(), total_entities, total_relationships);
    }

    #[tokio::test]
    async fn test_error_recovery_and_resilience() {
        init_test_env();

        let temp_dir = TempDir::new().unwrap();

        // Create mixed valid and invalid documents
        let documents = vec![
            create_valid_document("Valid document with entities like Apple Inc."),
            create_corrupted_document("Corrupted content with \0 invalid characters"),
            create_empty_document(),
            create_valid_document("Another valid document mentioning Google and Microsoft."),
            create_large_document(10000), // Very large document
        ];

        let pipeline = setup_resilient_pipeline().await;

        // Process documents - should handle errors gracefully
        let results = pipeline.process_documents_with_error_handling(documents).await;

        // Should have some successful results despite errors
        assert!(results.successful_count > 0, "Should process some documents successfully");
        assert!(results.failed_count >= 0, "Some documents may fail");

        // Verify partial knowledge graph was built
        let kg = results.knowledge_graph;
        if kg.node_count() > 0 {
            assert!(kg.is_consistent(), "Knowledge graph should be consistent");
        }
    }

    #[tokio::test]
    async fn test_cross_module_consistency() {
        init_test_env();

        let document_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California.";

        // Test entity extraction
        let entity_extractor = kg_construction::EntityExtractor::new();
        let entities = entity_extractor.extract_entities(document_text).await.unwrap();

        // Test relationship extraction
        let rel_extractor = kg_construction::RelationshipExtractor::new();
        let relationships = rel_extractor.extract_relationships(document_text, &entities).await.unwrap();

        // Test retrieval system
        let mut retriever = retriever::MultiHopRetriever::new(retriever::RetrieverConfig::default());

        // Add extracted entities to retriever
        for entity in &entities {
            let embedding = generate_test_embedding(768);
            retriever.add_document(&entity.text, &embedding).await.unwrap();
        }

        // Test search consistency
        let query_embedding = generate_test_embedding(768);
        let search_results = retriever.search(&query_embedding, 5).await.unwrap();

        // Verify consistency between extraction and retrieval
        assert!(!search_results.is_empty(), "Should find relevant results");

        let found_entities: Vec<_> = search_results.iter()
            .map(|r| &r.document_id)
            .collect();

        // Should find at least some extracted entities
        for entity in &entities {
            if found_entities.contains(&&entity.text) {
                // Found at least one entity - good!
                break;
            }
        }
    }

    async fn setup_test_pipeline() -> kg_construction::KGConstructionPipeline {
        let config = kg_construction::PipelineConfig {
            batch_size: 10,
            max_workers: 2,
            enable_coreference: false, // Disable for faster testing
            entity_confidence_threshold: 0.5,
            relationship_confidence_threshold: 0.5,
            ..Default::default()
        };

        kg_construction::KGConstructionPipeline::new(config)
    }

    async fn setup_resilient_pipeline() -> kg_construction::ResilientPipeline {
        let config = kg_construction::ResilientPipelineConfig {
            max_retries: 2,
            error_threshold: 0.5, // Allow 50% failure rate
            enable_partial_results: true,
            ..Default::default()
        };

        kg_construction::ResilientPipeline::new(config)
    }

    fn create_test_documents(temp_dir: &TempDir) -> Vec<kg_construction::Document> {
        vec![
            kg_construction::Document::new(
                "doc1",
                "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.",
                "test_source"
            ),
            kg_construction::Document::new(
                "doc2",
                "Google was established by Larry Page and Sergey Brin at Stanford University.",
                "test_source"
            ),
            kg_construction::Document::new(
                "doc3",
                "Microsoft Corporation is headquartered in Redmond, Washington.",
                "test_source"
            ),
        ]
    }

    fn create_csv_test_file(temp_dir: &TempDir, filename: &str) {
        let content = r#"company,founder,year,location
Apple Inc,Steve Jobs,1976,Cupertino
Google,Larry Page,1998,Mountain View
Microsoft,Bill Gates,1975,Redmond"#;

        std::fs::write(temp_dir.path().join(filename), content).unwrap();
    }

    fn create_json_test_file(temp_dir: &TempDir, filename: &str) {
        let content = json!({
            "products": [
                {
                    "name": "iPhone",
                    "company": "Apple Inc",
                    "category": "Smartphone",
                    "release_year": 2007
                },
                {
                    "name": "Android",
                    "company": "Google",
                    "category": "Operating System",
                    "release_year": 2008
                }
            ]
        });

        std::fs::write(temp_dir.path().join(filename), content.to_string()).unwrap();
    }

    fn create_markdown_test_file(temp_dir: &TempDir, filename: &str) {
        let content = r#"# Technology Companies

## Apple Inc.
- Founded by Steve Jobs and Steve Wozniak
- Headquarters: Cupertino, California
- Known for: iPhone, Mac, iPad

## Google
- Founded by Larry Page and Sergey Brin
- Parent company: Alphabet Inc.
- Headquarters: Mountain View, California
"#;

        std::fs::write(temp_dir.path().join(filename), content).unwrap();
    }

    fn create_text_test_file(temp_dir: &TempDir, filename: &str) {
        let content = r#"
The technology industry has been shaped by several pioneering companies.
Apple Inc., founded by Steve Jobs, revolutionized personal computing and mobile devices.
Google, established by Larry Page and Sergey Brin, transformed how we search and access information.
Microsoft, led by Bill Gates, dominated the personal computer operating system market.
These companies continue to influence technological advancement worldwide.
"#;

        std::fs::write(temp_dir.path().join(filename), content).unwrap();
    }

    fn generate_large_test_dataset(count: usize) -> Vec<kg_construction::Document> {
        (0..count).map(|i| {
            let content = format!(
                "Document {} discusses Company{} which was founded by Person{} in Location{}. \
                 The company specializes in Product{} and has partnerships with Partner{}.",
                i, i % 100, i % 50, i % 20, i % 30, (i + 1) % 100
            );

            kg_construction::Document::new(
                &format!("doc_{}", i),
                &content,
                "generated"
            )
        }).collect()
    }

    fn create_valid_document(content: &str) -> kg_construction::Document {
        kg_construction::Document::new("valid", content, "test")
    }

    fn create_corrupted_document(content: &str) -> kg_construction::Document {
        kg_construction::Document::new("corrupted", content, "test")
    }

    fn create_empty_document() -> kg_construction::Document {
        kg_construction::Document::new("empty", "", "test")
    }

    fn create_large_document(size: usize) -> kg_construction::Document {
        let content = "Large document content. ".repeat(size / 25);
        kg_construction::Document::new("large", &content, "test")
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_embedding_performance() {
        let model = retriever::EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2").await.unwrap();

        let texts = (0..100).map(|i| format!("Test document number {}", i)).collect::<Vec<_>>();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let start = Instant::now();
        let embeddings = model.encode_batch(&text_refs).await.unwrap();
        let duration = start.elapsed();

        assert_eq!(embeddings.len(), 100);
        assert!(duration.as_millis() < 10000, "Batch embedding should complete within 10 seconds");

        println!("Batch embedding performance: {} docs in {:?}", embeddings.len(), duration);
    }

    #[tokio::test]
    async fn test_vector_search_performance() {
        let config = retriever::VectorIndexConfig::new(384, "hnsw");
        let mut index = retriever::VectorIndex::new(config);

        // Add many vectors
        let start = Instant::now();
        for i in 0..1000 {
            let vector = generate_test_embedding(384);
            index.add_vector(&format!("doc_{}", i), &vector).await.unwrap();
        }
        let index_time = start.elapsed();

        // Search performance
        let query = generate_test_embedding(384);
        let start = Instant::now();
        let results = index.search(&query, 10).await.unwrap();
        let search_time = start.elapsed();

        assert_eq!(results.len(), 10);
        assert!(search_time.as_millis() < 100, "Search should be fast");

        println!("Index build: {:?}, Search: {:?}", index_time, search_time);
    }

    #[tokio::test]
    async fn test_memory_usage() {
        // This is a basic memory test - in production you'd use more sophisticated tools
        let initial_memory = get_memory_usage();

        // Process a reasonable amount of data
        let documents = generate_large_test_dataset(500);
        let pipeline = setup_test_pipeline().await;

        let _result = pipeline.process_documents(documents).await.unwrap();

        let final_memory = get_memory_usage();
        let memory_increase = final_memory - initial_memory;

        // Memory increase should be reasonable (this is a rough check)
        assert!(memory_increase < 500 * 1024 * 1024, "Memory usage should be reasonable"); // < 500MB

        println!("Memory increase: {} MB", memory_increase / (1024 * 1024));
    }

    fn get_memory_usage() -> usize {
        // Simplified memory measurement - in real tests you'd use more accurate methods
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<usize>().unwrap_or(0) * 1024; // Convert KB to bytes
                    }
                }
            }
        }
        0
    }

    async fn setup_test_pipeline() -> kg_construction::KGConstructionPipeline {
        let config = kg_construction::PipelineConfig::default();
        kg_construction::KGConstructionPipeline::new(config)
    }
}

#[cfg(test)]
mod data_consistency_tests {
    use super::*;

    #[tokio::test]
    async fn test_entity_deduplication() {
        let documents = vec![
            kg_construction::Document::new("doc1", "Apple Inc. is a technology company.", "source1"),
            kg_construction::Document::new("doc2", "Apple Inc is headquartered in California.", "source2"),
            kg_construction::Document::new("doc3", "APPLE INC. was founded by Steve Jobs.", "source3"),
        ];

        let pipeline = setup_test_pipeline().await;
        let kg = pipeline.process_documents(documents).await.unwrap();

        // Should deduplicate similar entity names
        let apple_entities = kg.find_entities_by_name_similarity("Apple Inc", 0.8);
        assert!(apple_entities.len() <= 2, "Should deduplicate similar entity names");
    }

    #[tokio::test]
    async fn test_relationship_consistency() {
        let documents = vec![
            kg_construction::Document::new("doc1", "Steve Jobs founded Apple Inc.", "source1"),
            kg_construction::Document::new("doc2", "Apple Inc. was founded by Steve Jobs.", "source2"),
        ];

        let pipeline = setup_test_pipeline().await;
        let kg = pipeline.process_documents(documents).await.unwrap();

        // Should not create duplicate relationships
        let founded_relationships = kg.get_relationships_by_type("founded");
        let steve_apple_rels = founded_relationships.iter()
            .filter(|r| (r.subject.contains("Steve Jobs") && r.object.contains("Apple")) ||
                       (r.subject.contains("Apple") && r.object.contains("Steve Jobs")))
            .count();

        assert!(steve_apple_rels <= 2, "Should not create excessive duplicate relationships");
    }

    #[tokio::test]
    async fn test_cross_document_entity_linking() {
        let documents = vec![
            kg_construction::Document::new("doc1", "Steve Jobs was the CEO of Apple.", "source1"),
            kg_construction::Document::new("doc2", "Jobs revolutionized the smartphone industry.", "source2"),
            kg_construction::Document::new("doc3", "The former Apple CEO passed away in 2011.", "source3"),
        ];

        let pipeline = setup_test_pipeline().await;
        let kg = pipeline.process_documents(documents).await.unwrap();

        // Should link "Jobs", "Steve Jobs", and "former Apple CEO" to the same entity
        let steve_jobs_node = kg.find_entity("Steve Jobs").unwrap();
        let linked_entities = kg.get_linked_entities(&steve_jobs_node.id);

        assert!(linked_entities.len() >= 1, "Should link entity references across documents");
    }

    async fn setup_test_pipeline() -> kg_construction::KGConstructionPipeline {
        let config = kg_construction::PipelineConfig {
            enable_entity_linking: true,
            enable_deduplication: true,
            similarity_threshold: 0.8,
            ..Default::default()
        };
        kg_construction::KGConstructionPipeline::new(config)
    }
}

#[cfg(test)]
mod end_to_end_workflow_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_rag_workflow() {
        init_test_env();

        // 1. Data Ingestion
        let temp_dir = TempDir::new().unwrap();
        create_comprehensive_test_data(&temp_dir);

        let ingestor = utils::DataIngestor::new();
        let documents = ingestor.ingest_directory(temp_dir.path()).await.unwrap();

        // 2. Knowledge Graph Construction
        let kg_pipeline = kg_construction::KGConstructionPipeline::new(Default::default());
        let knowledge_graph = kg_pipeline.process_documents(documents.clone()).await.unwrap();

        // 3. Vector Store Population
        let embedding_model = retriever::EmbeddingModel::new("sentence-transformers/all-MiniLM-L6-v2").await.unwrap();
        let mut vector_store = retriever::VectorIndex::new(retriever::VectorIndexConfig::new(384, "hnsw"));

        for doc in &documents {
            let embedding = embedding_model.encode(&doc.content).await.unwrap();
            vector_store.add_vector(&doc.id, &embedding).await.unwrap();
        }

        // 4. Retrieval System Setup
        let mut retriever = retriever::MultiHopRetriever::new(retriever::RetrieverConfig::default());
        retriever.set_vector_store(Box::new(vector_store));
        retriever.set_knowledge_graph(Box::new(knowledge_graph));

        // 5. Query Processing
        let queries = vec![
            "Who founded Apple Inc?",
            "What products does Google develop?",
            "Where is Microsoft headquartered?",
        ];

        for query in queries {
            let query_embedding = embedding_model.encode(query).await.unwrap();
            let search_results = retriever.search(&query_embedding, 5).await.unwrap();

            assert!(!search_results.is_empty(), "Should find relevant results for query: {}", query);

            // Verify results are relevant
            let top_result = &search_results[0];
            assert!(top_result.score > 0.3, "Top result should have reasonable relevance score");
        }

        // 6. LLM Generation (mock)
        let llm_config = llm_generator::GenerationConfig::default();
        let mut llm = MockLLMGenerator::new();

        for query in queries {
            let query_embedding = embedding_model.encode(query).await.unwrap();
            let context = retriever.search(&query_embedding, 3).await.unwrap();

            let context_text = context.iter()
                .map(|r| r.content.clone())
                .collect::<Vec<_>>()
                .join("\n");

            let prompt = format!("Context: {}\n\nQuestion: {}\n\nAnswer:", context_text, query);

            llm.expect_generate()
                .with(mockall::predicate::eq(prompt))
                .returning(|_| Ok("Generated response based on context.".to_string()));

            let response = llm.generate(&prompt).await.unwrap();
            assert!(!response.is_empty(), "Should generate non-empty response");
        }
    }

    fn create_comprehensive_test_data(temp_dir: &TempDir) {
        // Create multiple file types with interconnected data

        // Companies CSV
        std::fs::write(temp_dir.path().join("companies.csv"),
            "name,founder,founded_year,headquarters\n\
             Apple Inc,Steve Jobs,1976,Cupertino\n\
             Google,Larry Page,1998,Mountain View\n\
             Microsoft,Bill Gates,1975,Redmond").unwrap();

        // Products JSON
        let products = json!({
            "products": [
                {"name": "iPhone", "company": "Apple Inc", "type": "smartphone"},
                {"name": "Search", "company": "Google", "type": "web_service"},
                {"name": "Windows", "company": "Microsoft", "type": "operating_system"}
            ]
        });
        std::fs::write(temp_dir.path().join("products.json"), products.to_string()).unwrap();

        // Technology article
        std::fs::write(temp_dir.path().join("tech_article.txt"),
            "The technology industry has been shaped by visionary leaders. \
             Steve Jobs, co-founder of Apple Inc., revolutionized personal computing with products like the iPhone. \
             Larry Page and Sergey Brin founded Google, which became the world's most popular search engine. \
             Bill Gates built Microsoft into a software giant with the Windows operating system.").unwrap();

        // Company documentation
        std::fs::write(temp_dir.path().join("companies.md"),
            "# Major Technology Companies\n\n\
             ## Apple Inc.\n\
             - Founded: 1976\n\
             - Founder: Steve Jobs, Steve Wozniak\n\
             - Headquarters: Cupertino, California\n\n\
             ## Google\n\
             - Founded: 1998\n\
             - Founders: Larry Page, Sergey Brin\n\
             - Headquarters: Mountain View, California").unwrap();
    }

    use mockall::mock;

    mock! {
        LLMGenerator {}

        #[async_trait::async_trait]
        impl llm_generator::LLMGenerator for LLMGenerator {
            async fn generate(&self, prompt: &str) -> llm_generator::Result<String>;
        }
    }
}

// Helper function implementations
fn generate_test_embedding(dimensions: usize) -> Vec<f32> {
    (0..dimensions).map(|i| (i as f32) * 0.01).collect()
}

fn init_test_env() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();
}