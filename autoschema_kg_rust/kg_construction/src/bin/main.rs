//! Main binary for running the knowledge graph extraction pipeline

use clap::{Arg, Command};
use env_logger;
use kg_construction::{
    config::{Device, InferenceBackend, ProcessingConfig},
    extractor::KnowledgeGraphExtractor,
    Result,
};
use log::info;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let matches = Command::new("kg-extraction")
        .version("1.0.0")
        .about("Knowledge Graph Extraction Pipeline")
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL_PATH")
                .help("Model path for knowledge extraction")
                .required(true),
        )
        .arg(
            Arg::new("data_dir")
                .long("data-dir")
                .value_name("DIR")
                .help("Directory containing input data")
                .default_value("./data"),
        )
        .arg(
            Arg::new("file_name")
                .long("file-name")
                .value_name("PATTERN")
                .help("Filename pattern to match")
                .default_value("en_simple_wiki_v0"),
        )
        .arg(
            Arg::new("batch_size")
                .short('b')
                .long("batch-size")
                .value_name("SIZE")
                .help("Batch size for processing")
                .default_value("16"),
        )
        .arg(
            Arg::new("output_dir")
                .long("output-dir")
                .value_name("DIR")
                .help("Output directory for results")
                .default_value("./generation_result"),
        )
        .arg(
            Arg::new("total_shards")
                .long("total-shards")
                .value_name("N")
                .help("Total number of data shards")
                .default_value("1"),
        )
        .arg(
            Arg::new("shard")
                .long("shard")
                .value_name("N")
                .help("Current shard index (0-based)")
                .default_value("0"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .value_name("BACKEND")
                .help("ML inference backend (candle, ort, remote)")
                .default_value("candle"),
        )
        .arg(
            Arg::new("device")
                .long("device")
                .value_name("DEVICE")
                .help("Device to use (cpu, cuda, metal, auto)")
                .default_value("auto"),
        )
        .arg(
            Arg::new("quantize")
                .long("quantize")
                .help("Use 8-bit quantization")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("debug")
                .long("debug")
                .help("Enable debug mode")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("resume")
                .long("resume")
                .value_name("N")
                .help("Resume from specific batch")
                .default_value("0"),
        )
        .arg(
            Arg::new("max_tokens")
                .long("max-tokens")
                .value_name("N")
                .help("Maximum tokens to generate")
                .default_value("8192"),
        )
        .arg(
            Arg::new("workers")
                .long("workers")
                .value_name("N")
                .help("Maximum number of worker threads")
                .default_value("8"),
        )
        .arg(
            Arg::new("record")
                .long("record")
                .help("Record usage statistics")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("api_endpoint")
                .long("api-endpoint")
                .value_name("URL")
                .help("Remote API endpoint (for remote backend)")
                .requires("api_key"),
        )
        .arg(
            Arg::new("api_key")
                .long("api-key")
                .value_name("KEY")
                .help("API key for remote backend"),
        )
        .get_matches();

    // Parse inference backend
    let inference_backend = match matches.get_one::<String>("backend").unwrap().as_str() {
        "candle" => InferenceBackend::Candle,
        "ort" => InferenceBackend::Ort,
        "remote" => {
            let endpoint = matches
                .get_one::<String>("api_endpoint")
                .ok_or_else(|| kg_construction::error::KgConstructionError::ConfigError(
                    "Remote backend requires --api-endpoint".to_string()
                ))?
                .clone();
            let api_key = matches.get_one::<String>("api_key").cloned();
            InferenceBackend::RemoteApi { endpoint, api_key }
        }
        backend => {
            return Err(kg_construction::error::KgConstructionError::ConfigError(
                format!("Unknown backend: {}", backend)
            ));
        }
    };

    // Parse device
    let device = match matches.get_one::<String>("device").unwrap().as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        "metal" => Device::Metal,
        "auto" => Device::Auto,
        device => {
            if device.starts_with("cuda:") {
                if let Ok(gpu_id) = device[5..].parse::<usize>() {
                    Device::Cuda(gpu_id)
                } else {
                    return Err(kg_construction::error::KgConstructionError::ConfigError(
                        format!("Invalid CUDA device: {}", device)
                    ));
                }
            } else {
                return Err(kg_construction::error::KgConstructionError::ConfigError(
                    format!("Unknown device: {}", device)
                ));
            }
        }
    };

    // Create configuration
    let config = ProcessingConfig {
        model_path: matches.get_one::<String>("model").unwrap().clone(),
        data_directory: PathBuf::from(matches.get_one::<String>("data_dir").unwrap()),
        filename_pattern: matches.get_one::<String>("file_name").unwrap().clone(),
        batch_size_triple: matches
            .get_one::<String>("batch_size")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid batch size: {}", e)
            ))?,
        output_directory: PathBuf::from(matches.get_one::<String>("output_dir").unwrap()),
        total_shards_triple: matches
            .get_one::<String>("total_shards")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid total shards: {}", e)
            ))?,
        current_shard_triple: matches
            .get_one::<String>("shard")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid shard index: {}", e)
            ))?,
        use_8bit: matches.get_flag("quantize"),
        debug_mode: matches.get_flag("debug"),
        resume_from: matches
            .get_one::<String>("resume")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid resume batch: {}", e)
            ))?,
        record: matches.get_flag("record"),
        max_new_tokens: matches
            .get_one::<String>("max_tokens")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid max tokens: {}", e)
            ))?,
        max_workers: matches
            .get_one::<String>("workers")
            .unwrap()
            .parse()
            .map_err(|e| kg_construction::error::KgConstructionError::ConfigError(
                format!("Invalid worker count: {}", e)
            ))?,
        inference_backend,
        device,
        ..Default::default()
    };

    // Validate configuration
    config.validate()?;

    info!("Starting knowledge graph extraction with configuration:");
    info!("  Model: {}", config.model_path);
    info!("  Data directory: {:?}", config.data_directory);
    info!("  Output directory: {:?}", config.output_directory);
    info!("  Batch size: {}", config.batch_size_triple);
    info!("  Shard: {}/{}", config.current_shard_triple + 1, config.total_shards_triple);
    info!("  Backend: {:?}", config.inference_backend);
    info!("  Device: {:?}", config.device);
    info!("  Debug mode: {}", config.debug_mode);

    // Create and run extractor
    let mut extractor = KnowledgeGraphExtractor::new(config).await?;

    info!("Backend info: {:?}", extractor.get_backend_info());

    let output_file = extractor.run_extraction().await?;

    // Print final statistics
    let stats = extractor.get_stats();
    let parser_stats = extractor.get_parser_stats();

    info!("=== EXTRACTION COMPLETED ===");
    info!("Output file: {}", output_file);
    info!("Total chunks processed: {}", stats.total_chunks_processed);
    info!("Total batches processed: {}", stats.total_batches_processed);
    info!("Total entities extracted: {}", stats.total_entities_extracted);
    info!("Total events extracted: {}", stats.total_events_extracted);
    info!("Total relations extracted: {}", stats.total_relations_extracted);
    info!("Processing time: {:.2} seconds", stats.processing_time_seconds);
    info!("Average time per chunk: {:.2} ms", stats.average_chunk_time_ms);
    info!("Parser success rate: {:.1}%", parser_stats.successful_parses as f64 / parser_stats.total_parsed as f64 * 100.0);
    info!("JSON repairs performed: {}", parser_stats.json_repairs);

    Ok(())
}