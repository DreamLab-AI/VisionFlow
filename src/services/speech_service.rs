use crate::config::AppFullSettings;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task;
use tokio_tungstenite::{connect_async, tungstenite, MaybeTlsStream, WebSocketStream};
use tungstenite::http::Request;
// use crate::config::Settings; // AppFullSettings is used from self.settings
use crate::actors::voice_commands::VoiceCommand;
use crate::errors::{SpeechError as VisionSpeechError, VisionFlowError, VisionFlowResult};
use crate::types::speech::{
    STTProvider, SpeechCommand, SpeechOptions, TTSProvider, TranscriptionOptions,
};
use crate::utils::mcp_connection::{
    call_agent_list, call_agent_spawn, call_swarm_init, call_task_orchestrate,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use log::{debug, error, info};
use tokio::net::TcpStream;
use url::Url;
// DEPRECATED: call_task_orchestrate_docker removed - use TaskOrchestratorActor
use crate::services::voice_context_manager::VoiceContextManager;
use crate::services::voice_tag_manager::{TaggedVoiceResponse, VoiceTagManager};
use chrono;
use reqwest::Client;
use uuid::Uuid;

/// Centralized speech service managing both Text-to-Speech (TTS) and Speech-to-Text (STT) operations
///
/// This service orchestrates real-time voice interactions by:
/// - Managing TTS via Kokoro API for generating speech from text
/// - Managing STT via Whisper API for transcribing audio to text
/// - Broadcasting audio and transcription data to multiple WebSocket clients
/// - Handling provider switching and configuration management
///
/// The service uses async channels for command processing and broadcast channels
/// for distributing results to multiple subscribers simultaneously.
pub struct SpeechService {
    /// Command sender for internal message passing to the service task
    sender: Arc<Mutex<mpsc::Sender<SpeechCommand>>>,
    /// Shared application settings containing API configurations
    settings: Arc<RwLock<AppFullSettings>>,
    /// Current Text-to-Speech provider (Kokoro, OpenAI, etc.)
    tts_provider: Arc<RwLock<TTSProvider>>,
    /// Current Speech-to-Text provider (Whisper, OpenAI, etc.)
    stt_provider: Arc<RwLock<STTProvider>>,
    /// Broadcast channel for distributing TTS audio data to all connected WebSocket clients
    /// Buffer size of 100 allows multiple clients without blocking
    audio_tx: broadcast::Sender<Vec<u8>>,
    /// Broadcast channel for distributing STT transcription results to all connected clients
    /// Each transcription result is sent as a String to all subscribers
    transcription_tx: broadcast::Sender<String>,
    /// Shared HTTP client for making API requests to external services (Kokoro, Whisper)
    /// Reused across all requests for connection pooling and efficiency
    http_client: Arc<Client>,
    /// Voice conversation context manager for multi-turn interactions
    context_manager: Arc<VoiceContextManager>,
    /// Voice tag manager for tracking commands through hive mind
    tag_manager: Arc<VoiceTagManager>,
    /// TTS response receiver for tagged responses
    tts_response_rx: Option<Arc<Mutex<mpsc::Receiver<TaggedVoiceResponse>>>>,
}

impl SpeechService {
    /// Creates a new SpeechService instance with default configurations
    ///
    /// # Arguments
    /// * `settings` - Shared application settings containing API configurations for TTS/STT providers
    ///
    /// # Returns
    /// * `SpeechService` - A new service instance ready for speech operations
    ///
    /// # Behavior
    /// - Initializes internal command channel with buffer size of 100 commands
    /// - Creates broadcast channels for audio (TTS output) and transcriptions (STT output)
    /// - Sets up shared HTTP client for efficient API communication
    /// - Defaults to Kokoro TTS and Whisper STT providers
    /// - Automatically starts the internal service task for command processing
    ///
    /// # Channel Buffers
    /// - Command channel: 100 commands (prevents blocking on rapid command submission)
    /// - Audio broadcast: 100 audio chunks (handles multiple clients with buffering)
    /// - Transcription broadcast: 100 transcriptions (handles multiple clients with buffering)
    pub fn new(settings: Arc<RwLock<AppFullSettings>>) -> Self {
        // Create internal command channel for async command processing
        let (tx, rx) = mpsc::channel(100);
        let sender = Arc::new(Mutex::new(tx));

        // Create broadcast channel for TTS audio data with buffer size of 100
        // This allows multiple WebSocket clients to receive the same audio simultaneously
        let (audio_tx, _) = broadcast::channel(100);

        // Create shared HTTP client for API requests to external services
        // Reuses connections for better performance across multiple requests
        let http_client = Arc::new(Client::new());

        // Create broadcast channel for STT transcription results
        // Multiple clients can subscribe to receive transcription text
        let (transcription_tx, _) = broadcast::channel(100);

        // Create TTS response channel for tagged responses
        let (tts_response_tx, tts_response_rx) = mpsc::channel(100);

        // Create voice tag manager and configure TTS sender
        let mut tag_manager = VoiceTagManager::new();
        tag_manager.set_tts_sender(tts_response_tx);
        let tag_manager = Arc::new(tag_manager);

        let service = SpeechService {
            sender,
            settings,
            tts_provider: Arc::new(RwLock::new(TTSProvider::Kokoro)), // Default to Kokoro for TTS
            stt_provider: Arc::new(RwLock::new(STTProvider::Whisper)), // Default to Whisper for STT
            audio_tx,
            transcription_tx,
            http_client,
            context_manager: Arc::new(VoiceContextManager::new()),
            tag_manager,
            tts_response_rx: Some(Arc::new(Mutex::new(tts_response_rx))),
        };

        // Start the internal service task for async command processing
        service.start(rx);

        // Start the tagged TTS response handler
        service.start_tagged_tts_handler();

        service
    }

    fn start(&self, mut receiver: mpsc::Receiver<SpeechCommand>) {
        let settings: Arc<RwLock<AppFullSettings>> = Arc::clone(&self.settings);
        let http_client = Arc::clone(&self.http_client);
        let tts_provider = Arc::clone(&self.tts_provider);
        let stt_provider = Arc::clone(&self.stt_provider);
        let audio_tx = self.audio_tx.clone();
        let transcription_tx = self.transcription_tx.clone();

        task::spawn(async move {
            let mut ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>> = None;

            while let Some(command) = receiver.recv().await {
                match command {
                    SpeechCommand::Initialize => {
                        let settings_read = settings.read().await;

                        // Safely get OpenAI API key
                        let openai_api_key = match settings_read
                            .openai
                            .as_ref()
                            .and_then(|o| o.api_key.as_ref())
                        {
                            Some(key) if !key.is_empty() => key.clone(),
                            _ => {
                                error!("OpenAI API key not configured or empty. Cannot initialize OpenAI Realtime API.");
                                continue; // Skip initialization if key is missing
                            }
                        };

                        let url_str = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01";
                        let url = match Url::parse(url_str) {
                            Ok(url) => url,
                            Err(e) => {
                                error!("Failed to parse OpenAI URL '{}': {}", url_str, e);
                                continue;
                            }
                        };

                        let request = match Request::builder()
                            .uri(url.as_str())
                            .header("Authorization", format!("Bearer {}", openai_api_key))
                            .header("OpenAI-Beta", "realtime=v1")
                            .header("Content-Type", "application/json")
                            .header("User-Agent", "WebXR Graph")
                            .header("Sec-WebSocket-Version", "13")
                            .header(
                                "Sec-WebSocket-Key",
                                tungstenite::handshake::client::generate_key(),
                            )
                            .header("Connection", "Upgrade")
                            .header("Upgrade", "websocket")
                            .body(())
                        {
                            Ok(req) => req,
                            Err(e) => {
                                error!("Failed to build request: {}", e);
                                continue;
                            }
                        };

                        match connect_async(request).await {
                            Ok((mut stream, _)) => {
                                info!("Connected to OpenAI Realtime API");

                                let init_event = json!({
                                    "type": "response.create",
                                    "response": {
                                        "modalities": ["text", "audio"],
                                        "instructions": "You are a helpful AI assistant. Respond naturally and conversationally."
                                    }
                                });

                                if let Err(e) = stream
                                    .send(tungstenite::Message::Text(init_event.to_string()))
                                    .await
                                {
                                    error!("Failed to send initial response.create event: {}", e);
                                    continue;
                                }

                                ws_stream = Some(stream);
                            }
                            Err(e) => error!("Failed to connect to OpenAI Realtime API: {}", e),
                        }
                    }
                    SpeechCommand::SendMessage(msg) => {
                        if let Some(stream) = &mut ws_stream {
                            let msg_event = json!({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{
                                        "type": "input_text",
                                        "text": msg
                                    }]
                                }
                            });

                            if let Err(e) = stream
                                .send(tungstenite::Message::Text(msg_event.to_string()))
                                .await
                            {
                                error!("Failed to send message to OpenAI: {}", e);
                                continue;
                            }

                            let response_event = json!({
                                "type": "response.create"
                            });

                            if let Err(e) = stream
                                .send(tungstenite::Message::Text(response_event.to_string()))
                                .await
                            {
                                error!("Failed to request response from OpenAI: {}", e);
                                continue;
                            }

                            while let Some(message) = stream.next().await {
                                match message {
                                    Ok(tungstenite::Message::Text(text)) => {
                                        let event = match serde_json::from_str::<serde_json::Value>(
                                            &text,
                                        ) {
                                            Ok(event) => event,
                                            Err(e) => {
                                                error!("Failed to parse server event: {}", e);
                                                continue;
                                            }
                                        };

                                        match event["type"].as_str() {
                                            Some("conversation.item.created") => {
                                                if let Some(content) =
                                                    event["item"]["content"].as_array()
                                                {
                                                    for item in content {
                                                        if item["type"] == "audio" {
                                                            if let Some(audio_data) =
                                                                item["audio"].as_str()
                                                            {
                                                                match BASE64.decode(audio_data) {
                                                                    Ok(audio_bytes) => {
                                                                        debug!("Received audio data of size: {}", audio_bytes.len());
                                                                    },
                                                                    Err(e) => error!("Failed to decode audio data: {}", e),
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            Some("error") => {
                                                error!("OpenAI Realtime API error: {:?}", event);
                                                break;
                                            }
                                            Some("response.completed") => break,
                                            _ => {}
                                        }
                                    }
                                    Ok(tungstenite::Message::Close(_)) => break,
                                    Err(e) => {
                                        error!("Error receiving from OpenAI: {}", e);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        } else {
                            error!("OpenAI WebSocket not initialized");
                        }
                    }
                    SpeechCommand::Close => {
                        if let Some(mut stream) = ws_stream.take() {
                            if let Err(e) = stream.send(tungstenite::Message::Close(None)).await {
                                error!("Failed to send close frame: {}", e);
                            }
                        }
                        break;
                    }
                    SpeechCommand::SetTTSProvider(provider) => {
                        let mut current_provider = tts_provider.write().await;
                        *current_provider = provider.clone();
                        info!("TTS provider updated to: {:?}", provider);
                    }
                    SpeechCommand::TextToSpeech(text, options) => {
                        let provider = tts_provider.read().await.clone();

                        match provider {
                            TTSProvider::OpenAI => {
                                info!("Processing TextToSpeech command with OpenAI provider");
                                let openai_config = {
                                    let s = settings.read().await;
                                    s.openai.clone()
                                };

                                if let Some(config) = openai_config {
                                    if let Some(api_key) = config.api_key.as_ref() {
                                        let api_url = "https://api.openai.com/v1/audio/speech";
                                        info!("Sending TTS request to OpenAI API: {}", api_url);

                                        let request_body = json!({
                                            "model": "tts-1",
                                            "input": text,
                                            "voice": options.voice.clone(),
                                            "response_format": "mp3",
                                            "speed": options.speed
                                        });

                                        let response = match http_client
                                            .post(api_url)
                                            .header("Authorization", format!("Bearer {}", api_key))
                                            .header("Content-Type", "application/json")
                                            .body(request_body.to_string())
                                            .send()
                                            .await
                                        {
                                            Ok(response) => {
                                                if !response.status().is_success() {
                                                    let status = response.status();
                                                    let error_text =
                                                        response.text().await.unwrap_or_default();
                                                    error!(
                                                        "OpenAI TTS API error {}: {}",
                                                        status, error_text
                                                    );
                                                    continue;
                                                }
                                                response
                                            }
                                            Err(e) => {
                                                error!(
                                                    "Failed to connect to OpenAI TTS API: {}",
                                                    e
                                                );
                                                continue;
                                            }
                                        };

                                        match response.bytes().await {
                                            Ok(bytes) => {
                                                if let Err(e) = audio_tx.send(bytes.to_vec()) {
                                                    error!(
                                                        "Failed to send OpenAI audio data: {}",
                                                        e
                                                    );
                                                } else {
                                                    debug!(
                                                        "Sent {} bytes of OpenAI audio data",
                                                        bytes.len()
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to get OpenAI audio bytes: {}", e);
                                            }
                                        }
                                    } else {
                                        error!("OpenAI API key not configured");
                                    }
                                } else {
                                    error!("OpenAI configuration not found");
                                }
                            }
                            TTSProvider::Kokoro => {
                                info!("Processing TextToSpeech command with Kokoro provider");
                                let kokoro_config = {
                                    let s = settings.read().await;
                                    s.kokoro.clone()
                                };

                                if let Some(config) = kokoro_config {
                                    let api_url_base = match config.api_url.as_deref() {
                                        Some(url) if !url.is_empty() => url,
                                        _ => {
                                            // Use default Kokoro URL on Docker network
                                            info!("Using default Kokoro API URL on Docker network");
                                            "http://kokoro-tts-container:8880"
                                        }
                                    };
                                    let api_url = format!(
                                        "{}/v1/audio/speech",
                                        api_url_base.trim_end_matches('/')
                                    );
                                    info!("Sending TTS request to Kokoro API: {}", api_url);

                                    let response_format =
                                        config.default_format.as_deref().unwrap_or("mp3");

                                    let request_body = json!({
                                        "model": "kokoro",
                                        "input": text,
                                        "voice": options.voice.clone(),
                                        "response_format": response_format,
                                        "speed": options.speed,
                                        "stream": options.stream
                                    });

                                    let response = match http_client
                                        .post(&api_url)
                                        .header("Content-Type", "application/json")
                                        .body(request_body.to_string())
                                        .send()
                                        .await
                                    {
                                        Ok(response) => {
                                            if !response.status().is_success() {
                                                let status = response.status();
                                                let error_text =
                                                    response.text().await.unwrap_or_default();
                                                error!(
                                                    "Kokoro API error {}: {}",
                                                    status, error_text
                                                );
                                                continue;
                                            }
                                            response
                                        }
                                        Err(e) => {
                                            error!("Failed to connect to Kokoro API: {}", e);
                                            continue;
                                        }
                                    };

                                    if options.stream {
                                        let stream = response.bytes_stream();
                                        let audio_broadcaster = audio_tx.clone();

                                        tokio::spawn(async move {
                                            let mut stream = Box::pin(stream);

                                            while let Some(item) = stream.next().await {
                                                match item {
                                                    Ok(bytes) => {
                                                        if let Err(e) =
                                                            audio_broadcaster.send(bytes.to_vec())
                                                        {
                                                            error!("Failed to broadcast audio chunk: {}", e);
                                                        }
                                                    }
                                                    Err(e) => {
                                                        error!(
                                                            "Error receiving audio stream: {}",
                                                            e
                                                        );
                                                        break;
                                                    }
                                                }
                                            }
                                            debug!("Finished streaming audio from Kokoro");
                                        });
                                    } else {
                                        match response.bytes().await {
                                            Ok(bytes) => {
                                                if let Err(e) = audio_tx.send(bytes.to_vec()) {
                                                    error!("Failed to send audio data: {}", e);
                                                } else {
                                                    debug!(
                                                        "Sent {} bytes of audio data",
                                                        bytes.len()
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to get audio bytes: {}", e);
                                            }
                                        }
                                    }
                                } else {
                                    error!("Kokoro configuration not found");
                                }
                            }
                        }
                    }
                    SpeechCommand::SetSTTProvider(provider) => {
                        let mut current_provider = stt_provider.write().await;
                        *current_provider = provider.clone();
                        info!("STT provider updated to: {:?}", provider);
                    }
                    SpeechCommand::StartTranscription(options) => {
                        let provider = stt_provider.read().await.clone();

                        match provider {
                            STTProvider::Whisper => {
                                info!("Starting Whisper transcription with options: {:?}", options);

                                let whisper_config = {
                                    let s = settings.read().await;
                                    s.whisper.clone()
                                };

                                if let Some(config) = whisper_config {
                                    let api_url = config
                                        .api_url
                                        .as_deref()
                                        .unwrap_or("http://whisper-webui-backend:8000");
                                    info!("Whisper STT initialized with API URL: {}", api_url);

                                    let _ = transcription_tx.send("Whisper STT ready".to_string());
                                } else {
                                    error!("Whisper configuration not found");
                                    let _ = transcription_tx
                                        .send("Whisper STT configuration missing".to_string());
                                }
                            }
                            STTProvider::OpenAI => {
                                info!("Starting OpenAI transcription with options: {:?}", options);
                                let openai_config = {
                                    let s = settings.read().await;
                                    s.openai.clone()
                                };

                                if let Some(config) = openai_config {
                                    if config.api_key.is_some() {
                                        info!("OpenAI STT initialized with API key configured");
                                        let _ =
                                            transcription_tx.send("OpenAI STT ready".to_string());
                                    } else {
                                        error!("OpenAI API key not configured for STT");
                                        let _ = transcription_tx
                                            .send("OpenAI STT API key missing".to_string());
                                    }
                                } else {
                                    error!("OpenAI configuration not found for STT");
                                    let _ = transcription_tx
                                        .send("OpenAI STT configuration missing".to_string());
                                }
                            }
                        }
                    }
                    SpeechCommand::StopTranscription => {
                        info!("Stopping transcription");

                        // Notify subscribers that transcription has stopped
                        let _ = transcription_tx.send("Transcription stopped".to_string());

                        // Additional cleanup could be added here for specific providers
                        match stt_provider.read().await.clone() {
                            STTProvider::Whisper => {
                                debug!("Whisper transcription stopped");
                            }
                            STTProvider::OpenAI => {
                                debug!("OpenAI transcription stopped");
                            }
                        }
                    }
                    SpeechCommand::ProcessAudioChunk(audio_data) => {
                        debug!("Processing audio chunk of size: {} bytes", audio_data.len());

                        let provider = stt_provider.read().await.clone();

                        match provider {
                            STTProvider::Whisper => {
                                let whisper_config = {
                                    let s = settings.read().await;
                                    s.whisper.clone()
                                };

                                if let Some(config) = whisper_config {
                                    let api_url_base = config
                                        .api_url
                                        .as_deref()
                                        .unwrap_or("http://whisper-webui-backend:8000");
                                    let api_url = format!(
                                        "{}/transcription/",
                                        api_url_base.trim_end_matches('/')
                                    );

                                    // Detect audio format from the data
                                    // WebM files start with 0x1A45DFA3 (EBML header)
                                    // WAV files start with "RIFF" (0x52494646)
                                    let (mime_type, file_ext) = if audio_data.len() >= 4 {
                                        let header = &audio_data[0..4];
                                        if header == [0x1A, 0x45, 0xDF, 0xA3] {
                                            // WebM/Opus format from browser
                                            ("audio/webm", "audio.webm")
                                        } else if header == [0x52, 0x49, 0x46, 0x46] {
                                            // WAV format
                                            ("audio/wav", "audio.wav")
                                        } else {
                                            // Default to webm as that's what browser sends
                                            info!("Unknown audio format, header: {:?}, defaulting to webm", header);
                                            ("audio/webm", "audio.webm")
                                        }
                                    } else {
                                        ("audio/webm", "audio.webm")
                                    };

                                    info!(
                                        "Detected audio format: {} for upload to Whisper",
                                        mime_type
                                    );

                                    // Build the form with all required fields
                                    let mut form = reqwest::multipart::Form::new().part(
                                        "file",
                                        reqwest::multipart::Part::bytes(audio_data)
                                            .file_name(file_ext)
                                            .mime_str(mime_type)
                                            .unwrap_or_else(|_| {
                                                reqwest::multipart::Part::bytes(vec![])
                                                    .mime_str("audio/webm")
                                                    .unwrap()
                                            }),
                                    );

                                    // Add optional parameters
                                    if let Some(model) = config.default_model.clone() {
                                        form = form.text("model_size", model);
                                    }
                                    if let Some(language) = config.default_language.clone() {
                                        form = form.text("lang", language);
                                    }
                                    if let Some(temperature) = config.temperature {
                                        form = form.text("temperature", temperature.to_string());
                                    }
                                    if let Some(vad_filter) = config.vad_filter {
                                        form = form.text("vad_filter", vad_filter.to_string());
                                    }
                                    if let Some(word_timestamps) = config.word_timestamps {
                                        form = form
                                            .text("word_timestamps", word_timestamps.to_string());
                                    }
                                    if let Some(initial_prompt) = config.initial_prompt.clone() {
                                        form = form.text("initial_prompt", initial_prompt);
                                    }

                                    // Process Whisper transcription directly without spawning
                                    let http_client_clone = Arc::clone(&http_client);
                                    let transcription_broadcaster = transcription_tx.clone();
                                    let api_url_clone = api_url.clone();

                                    // Process inline to avoid Send issues
                                    {
                                        // Submit audio to Whisper API
                                        match http_client_clone
                                            .post(&api_url_clone)
                                            .multipart(form)
                                            .send()
                                            .await
                                        {
                                            Ok(response) => {
                                                if response.status().is_success() {
                                                    match response.json::<serde_json::Value>().await
                                                    {
                                                        Ok(json) => {
                                                            // Whisper returns a task ID, not the transcription directly
                                                            if let Some(identifier) = json
                                                                .get("identifier")
                                                                .and_then(|t| t.as_str())
                                                            {
                                                                info!("Whisper task queued with ID: {}", identifier);

                                                                // Poll for task completion
                                                                let task_url = format!(
                                                                    "{}/task/{}",
                                                                    api_url_clone.trim_end_matches(
                                                                        "/transcription/"
                                                                    ),
                                                                    identifier
                                                                );
                                                                let mut attempts = 0;
                                                                const MAX_ATTEMPTS: u32 = 30;
                                                                const POLL_DELAY_MS: u64 = 200;

                                                                loop {
                                                                    attempts += 1;
                                                                    if attempts > MAX_ATTEMPTS {
                                                                        error!("Timeout waiting for Whisper task {}", identifier);
                                                                        break;
                                                                    }

                                                                    tokio::time::sleep(tokio::time::Duration::from_millis(POLL_DELAY_MS)).await;

                                                                    match http_client_clone
                                                                        .get(&task_url)
                                                                        .send()
                                                                        .await
                                                                    {
                                                                        Ok(task_response) => {
                                                                            if task_response
                                                                                .status()
                                                                                .is_success()
                                                                            {
                                                                                if let Ok(task_json) = task_response.json::<serde_json::Value>().await {
                                                                                    if let Some(status) = task_json.get("status").and_then(|s| s.as_str()) {
                                                                                        match status {
                                                                                            "completed" => {
                                                                                                // Extract transcription from result array
                                                                                                if let Some(result) = task_json.get("result").and_then(|r| r.as_array()) {
                                                                                                    let mut full_text = String::new();
                                                                                                    for segment in result {
                                                                                                        if let Some(text) = segment.get("text").and_then(|t| t.as_str()) {
                                                                                                            full_text.push_str(text);
                                                                                                            full_text.push(' ');
                                                                                                        }
                                                                                                    }

                                                                                                    let transcription_text = full_text.trim().to_string();
                                                                                                    if !transcription_text.is_empty() {
                                                                                                        info!("Whisper transcription: {}", transcription_text);
                                                                                                        let _ = transcription_broadcaster.send(transcription_text.clone());

                                                                                                        // Check if this is a voice command and process it
                                                                                                        if Self::is_voice_command(&transcription_text) {
                                                                                                            let session_id = Uuid::new_v4().to_string();
                                                                                                            debug!("Processing as voice command: {}", transcription_text);

                                                                                                            // Parse and execute voice command
                                                                                                            if let Ok(voice_cmd) = VoiceCommand::parse(&transcription_text, session_id) {
                                                                                                                debug!("Executing voice command: {:?}", voice_cmd.parsed_intent);

                                                                                                                // Execute actual agent commands via MCP with context
                                                                                                                let context_manager = Arc::new(VoiceContextManager::new());
                                                                                                                let response_text = Self::execute_voice_command_with_context(voice_cmd, context_manager).await;

                                                                                                                // Broadcast the response text
                                                                                                                let _ = transcription_broadcaster.send(format!("Response: {}", response_text));
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                                break;
                                                                                            },
                                                                                            "failed" => {
                                                                                                error!("Whisper task {} failed: {:?}", identifier, task_json.get("error"));
                                                                                                break;
                                                                                            },
                                                                                            "queued" | "in_progress" => {
                                                                                                // Continue polling
                                                                                                debug!("Whisper task {} status: {}", identifier, status);
                                                                                            },
                                                                                            _ => {
                                                                                                debug!("Unknown Whisper task status: {}", status);
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                        Err(e) => {
                                                                            error!("Failed to poll Whisper task {}: {}", identifier, e);
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            } else {
                                                                error!("No identifier field in Whisper response: {:?}", json);
                                                            }
                                                        }
                                                        Err(e) => {
                                                            error!("Failed to parse Whisper response JSON: {}", e);
                                                        }
                                                    }
                                                } else {
                                                    let status = response.status();
                                                    let error_text =
                                                        response.text().await.unwrap_or_default();
                                                    error!(
                                                        "Whisper API error {}: {}",
                                                        status, error_text
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                error!("Failed to connect to Whisper API: {}", e);
                                            }
                                        }
                                    }
                                } else {
                                    error!("Whisper configuration not found for audio processing");
                                }
                            }
                            STTProvider::OpenAI => {
                                debug!("Processing audio chunk with OpenAI STT");
                                let openai_config = {
                                    let s = settings.read().await;
                                    s.openai.clone()
                                };

                                if let Some(config) = openai_config {
                                    if let Some(api_key) = config.api_key.as_ref() {
                                        let api_url =
                                            "https://api.openai.com/v1/audio/transcriptions";

                                        let form = reqwest::multipart::Form::new()
                                            .part(
                                                "file",
                                                reqwest::multipart::Part::bytes(audio_data)
                                                    .file_name("audio.wav")
                                                    .mime_str("audio/wav")
                                                    .unwrap_or_else(|_| {
                                                        reqwest::multipart::Part::bytes(vec![])
                                                            .mime_str("audio/wav")
                                                            .unwrap()
                                                    }),
                                            )
                                            .text("model", "whisper-1")
                                            .text("response_format", "json");

                                        let http_client_clone = Arc::clone(&http_client);
                                        let transcription_broadcaster = transcription_tx.clone();
                                        let api_key_clone = api_key.clone();

                                        tokio::spawn(async move {
                                            match http_client_clone
                                                .post(api_url)
                                                .header(
                                                    "Authorization",
                                                    format!("Bearer {}", api_key_clone),
                                                )
                                                .multipart(form)
                                                .send()
                                                .await
                                            {
                                                Ok(response) => {
                                                    if response.status().is_success() {
                                                        match response
                                                            .json::<serde_json::Value>()
                                                            .await
                                                        {
                                                            Ok(json) => {
                                                                if let Some(text) = json
                                                                    .get("text")
                                                                    .and_then(|t| t.as_str())
                                                                {
                                                                    if !text.trim().is_empty() {
                                                                        debug!("OpenAI transcription: {}", text);
                                                                        let _ = transcription_broadcaster.send(text.to_string());
                                                                    }
                                                                } else {
                                                                    error!("No text field in OpenAI response: {:?}", json);
                                                                }
                                                            }
                                                            Err(e) => {
                                                                error!("Failed to parse OpenAI response JSON: {}", e);
                                                            }
                                                        }
                                                    } else {
                                                        let status = response.status();
                                                        let error_text = response
                                                            .text()
                                                            .await
                                                            .unwrap_or_default();
                                                        error!(
                                                            "OpenAI STT API error {}: {}",
                                                            status, error_text
                                                        );
                                                    }
                                                }
                                                Err(e) => {
                                                    error!(
                                                        "Failed to connect to OpenAI STT API: {}",
                                                        e
                                                    );
                                                }
                                            }
                                        });
                                    } else {
                                        error!(
                                            "OpenAI API key not configured for audio processing"
                                        );
                                    }
                                } else {
                                    error!("OpenAI configuration not found for audio processing");
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Start background task to handle tagged TTS responses
    fn start_tagged_tts_handler(&self) {
        if let Some(rx) = &self.tts_response_rx {
            let rx = Arc::clone(rx);
            let sender = Arc::clone(&self.sender);
            let tag_manager = Arc::clone(&self.tag_manager);

            task::spawn(async move {
                let mut receiver = rx.lock().await;

                while let Some(tagged_response) = receiver.recv().await {
                    info!(
                        "Processing tagged TTS response: {} (tag: {})",
                        tagged_response.response.text,
                        tagged_response.tag.short_id()
                    );

                    // Convert tagged response to regular TTS command
                    let tts_command = SpeechCommand::TextToSpeech(
                        tagged_response.response.text.clone(),
                        SpeechOptions::default(),
                    );

                    // Send to TTS processing
                    if let Err(e) = sender.lock().await.send(tts_command).await {
                        error!("Failed to send tagged response to TTS: {}", e);
                    } else {
                        debug!(
                            "Successfully routed tagged response {} to TTS",
                            tagged_response.tag.short_id()
                        );
                    }

                    // Clean up expired tags periodically
                    tag_manager.cleanup_expired_commands().await;
                }

                info!("Tagged TTS response handler terminated");
            });
        }
    }

    pub async fn initialize(&self) -> VisionFlowResult<()> {
        let command = SpeechCommand::Initialize;
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::InitializationFailed(e.to_string()))
        })?;
        Ok(())
    }

    pub async fn send_message(&self, message: String) -> VisionFlowResult<()> {
        let command = SpeechCommand::SendMessage(message);
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::TTSFailed {
                text: "message".to_string(),
                reason: e.to_string(),
            })
        })?;
        Ok(())
    }

    /// Converts text to speech using the configured TTS provider
    ///
    /// # Arguments
    /// * `text` - The text to be converted to speech
    /// * `options` - Speech generation options including voice, speed, and streaming preferences
    ///
    /// # Returns
    /// * `Ok(())` if the command was successfully queued for processing
    /// * `Err` if the command channel is closed or other error occurs
    ///
    /// # Behavior
    /// - Queues the TTS request for async processing by the service task
    /// - Audio output is broadcast to all subscribers via the audio channel
    /// - Supports both streaming and non-streaming audio generation
    /// - Uses Kokoro API by default with fallback error handling
    pub async fn text_to_speech(
        &self,
        text: String,
        options: SpeechOptions,
    ) -> VisionFlowResult<()> {
        let command = SpeechCommand::TextToSpeech(text.clone(), options);
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::TTSFailed {
                text,
                reason: e.to_string(),
            })
        })?;
        Ok(())
    }

    pub async fn close(&self) -> VisionFlowResult<()> {
        let command = SpeechCommand::Close;
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::InitializationFailed(format!(
                "Failed to close speech service: {}",
                e
            )))
        })?;
        Ok(())
    }

    pub async fn set_tts_provider(&self, provider: TTSProvider) -> VisionFlowResult<()> {
        let command = SpeechCommand::SetTTSProvider(provider.clone());
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::ProviderConfigError {
                provider: format!("{:?}", provider),
                reason: e.to_string(),
            })
        })?;
        Ok(())
    }

    /// Creates a new subscriber to the audio broadcast channel for receiving TTS audio data
    ///
    /// # Returns
    /// * `broadcast::Receiver<Vec<u8>>` - A receiver that will get all audio chunks from TTS operations
    ///
    /// # Usage
    /// Multiple WebSocket connections can subscribe to receive the same audio data simultaneously.
    /// Each subscriber gets its own independent receiver with a buffer to handle temporary disconnections.
    /// Audio data is broadcast as raw bytes (typically MP3 or WAV format from Kokoro TTS).
    pub fn subscribe_to_audio(&self) -> broadcast::Receiver<Vec<u8>> {
        self.audio_tx.subscribe()
    }

    // Current provider
    pub async fn get_tts_provider(&self) -> TTSProvider {
        self.tts_provider.read().await.clone()
    }

    pub async fn set_stt_provider(&self, provider: STTProvider) -> VisionFlowResult<()> {
        let command = SpeechCommand::SetSTTProvider(provider.clone());
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::ProviderConfigError {
                provider: format!("{:?}", provider),
                reason: e.to_string(),
            })
        })?;
        Ok(())
    }

    pub async fn start_transcription(&self, options: TranscriptionOptions) -> VisionFlowResult<()> {
        let command = SpeechCommand::StartTranscription(options);
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::STTFailed {
                reason: format!("Failed to start transcription: {}", e),
            })
        })?;
        Ok(())
    }

    pub async fn stop_transcription(&self) -> VisionFlowResult<()> {
        let command = SpeechCommand::StopTranscription;
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::STTFailed {
                reason: format!("Failed to stop transcription: {}", e),
            })
        })?;
        Ok(())
    }

    /// Processes audio data for speech-to-text transcription using the configured STT provider
    ///
    /// # Arguments
    /// * `audio_data` - Raw audio bytes in WAV format from client microphone input
    ///
    /// # Returns
    /// * `Ok(())` if the audio chunk was successfully queued for processing
    /// * `Err` if the command channel is closed or other error occurs
    ///
    /// # Behavior
    /// - Queues audio data for async STT processing by the service task
    /// - Sends audio to Whisper API at configured endpoint (default: http://whisper-webui-backend:8000)
    /// - Transcription results are broadcast to all subscribers via transcription channel
    /// - Supports configurable Whisper parameters (model, language, temperature, etc.)
    /// - Handles multipart form upload format required by Whisper-WebUI-Backend
    pub async fn process_audio_chunk(&self, audio_data: Vec<u8>) -> VisionFlowResult<()> {
        let command = SpeechCommand::ProcessAudioChunk(audio_data);
        self.sender.lock().await.send(command).await.map_err(|e| {
            VisionFlowError::Speech(VisionSpeechError::AudioProcessingFailed {
                reason: format!("Failed to process audio chunk: {}", e),
            })
        })?;
        Ok(())
    }

    /// Creates a new subscriber to the transcription broadcast channel for receiving STT results
    ///
    /// # Returns
    /// * `broadcast::Receiver<String>` - A receiver that will get all transcription text from STT operations
    ///
    /// # Usage
    /// Multiple WebSocket connections can subscribe to receive the same transcription results simultaneously.
    /// Each subscriber gets its own independent receiver with a buffer to handle temporary disconnections.
    /// Transcription results are broadcast as plain text strings from Whisper STT processing.
    pub fn subscribe_to_transcriptions(&self) -> broadcast::Receiver<String> {
        self.transcription_tx.subscribe()
    }

    /// Execute voice command and return response for TTS
    pub async fn process_voice_command(&self, text: String) -> VisionFlowResult<String> {
        let session_id = Uuid::new_v4().to_string();

        if Self::is_voice_command(&text) {
            if let Ok(voice_cmd) = VoiceCommand::parse(&text, session_id) {
                let response = Self::execute_voice_command_with_context(
                    voice_cmd.clone(),
                    Arc::clone(&self.context_manager),
                )
                .await;

                // Add conversation turn to context
                let _ = self
                    .context_manager
                    .add_conversation_turn(
                        &voice_cmd.session_id,
                        text,
                        response.clone(),
                        Some(voice_cmd.parsed_intent),
                    )
                    .await;

                // Generate contextual response
                let contextual_response = self
                    .context_manager
                    .generate_contextual_response(&voice_cmd.session_id, &response)
                    .await;

                Ok(contextual_response)
            } else {
                Ok("Sorry, I couldn't understand that command.".to_string())
            }
        } else {
            Ok("That doesn't appear to be a voice command.".to_string())
        }
    }

    /// Get conversation context for a session
    pub async fn get_conversation_context(
        &self,
        session_id: &str,
    ) -> Option<crate::actors::voice_commands::ConversationContext> {
        self.context_manager.get_context(session_id).await
    }

    /// Check if a session needs follow-up
    pub async fn session_needs_follow_up(&self, session_id: &str) -> bool {
        self.context_manager.needs_follow_up(session_id).await
    }

    /// Get access to the voice tag manager
    pub fn get_tag_manager(&self) -> Arc<VoiceTagManager> {
        Arc::clone(&self.tag_manager)
    }

    /// Get access to the transcription sender for broadcasting text
    pub fn get_transcription_sender(&self) -> broadcast::Sender<String> {
        self.transcription_tx.clone()
    }

    /// Process voice command with tag tracking through hive mind
    pub async fn process_voice_command_with_tags(
        &self,
        text: String,
        session_id: String,
    ) -> VisionFlowResult<String> {
        use crate::services::speech_voice_integration::VoiceSwarmIntegration;

        match VoiceSwarmIntegration::process_voice_command_with_tags(
            self,
            text,
            session_id,
            Arc::clone(&self.tag_manager),
        )
        .await
        {
            Ok(tag) => Ok(format!(
                "Voice command processed with tag: {}",
                tag.short_id()
            )),
            Err(e) => {
                error!("Failed to process tagged voice command: {}", e);
                Err(VisionFlowError::Speech(
                    VisionSpeechError::AudioProcessingFailed {
                        reason: format!("Tagged voice command failed: {}", e),
                    },
                ))
            }
        }
    }

    /// Check if text looks like a voice command for the swarm
    fn is_voice_command(text: &str) -> bool {
        let command_keywords = [
            "spawn",
            "agent",
            "status",
            "list",
            "stop",
            "add",
            "remove",
            "help",
            "show",
            "create",
            "delete",
            "query",
            "execute",
            "run",
            "node",
            "graph",
            "connect",
            "researcher",
            "coder",
            "analyst",
        ];

        let lower = text.to_lowercase();
        command_keywords
            .iter()
            .any(|keyword| lower.contains(keyword))
    }

    /// Execute voice command via MCP task orchestration with context management
    async fn execute_voice_command_with_context(
        voice_cmd: VoiceCommand,
        context_manager: Arc<VoiceContextManager>,
    ) -> String {
        let mcp_host =
            std::env::var("MCP_HOST").unwrap_or_else(|_| "multi-agent-container".to_string());
        let mcp_port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

        // Get or create session for context
        let session_id = context_manager
            .get_or_create_session(Some(voice_cmd.session_id.clone()), None)
            .await;

        match voice_cmd.parsed_intent {
            crate::actors::voice_commands::SwarmIntent::SpawnAgent { agent_type, .. } => {
                info!("Executing spawn agent command for type: {}", agent_type);

                // First ensure we have a swarm initialized
                match call_swarm_init(&mcp_host, &mcp_port, "mesh", 10, "balanced").await {
                    Ok(swarm_result) => {
                        let swarm_id = swarm_result.get("swarmId")
                            .and_then(|s| s.as_str())
                            .unwrap_or("default-swarm");

                        // Spawn the agent
                        match call_agent_spawn(&mcp_host, &mcp_port, &agent_type, swarm_id).await {
                            Ok(_) => {
                                // Track the operation in context
                                let mut params = std::collections::HashMap::new();
                                params.insert("agent_type".to_string(), agent_type.clone());
                                params.insert("swarm_id".to_string(), swarm_id.to_string());

                                let _ = context_manager.add_pending_operation(
                                    &session_id,
                                    "spawn_agent".to_string(),
                                    params,
                                    None,
                                ).await;

                                format!("Successfully spawned {} agent in swarm {}.", agent_type, swarm_id)
                            }
                            Err(e) => {
                                error!("Failed to spawn agent: {}", e);
                                format!("Failed to spawn {} agent. Error: {}", agent_type, e)
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to initialize swarm: {}", e);
                        format!("Failed to initialize swarm for agent spawning. Error: {}", e)
                    }
                }
            },

            crate::actors::voice_commands::SwarmIntent::QueryStatus { target } => {
                info!("Executing status query for target: {:?}", target);

                match call_agent_list(&mcp_host, &mcp_port, "all").await {
                    Ok(agent_result) => {
                        // Parse agent count from result
                        let agent_count = agent_result.get("content")
                            .and_then(|c| c.as_array())
                            .map(|arr| arr.len())
                            .unwrap_or(0);

                        if agent_count > 0 {
                            format!("System status: {} agents active and operational.", agent_count)
                        } else {
                            "System status: No active agents found.".to_string()
                        }
                    }
                    Err(e) => {
                        error!("Failed to query agent status: {}", e);
                        format!("Failed to query system status. Error: {}", e)
                    }
                }
            },

            crate::actors::voice_commands::SwarmIntent::ListAgents => {
                info!("Executing list agents command");

                match call_agent_list(&mcp_host, &mcp_port, "all").await {
                    Ok(agent_result) => {
                        // Parse agents from result
                        if let Some(content) = agent_result.get("content").and_then(|c| c.as_array()) {
                            let mut agent_names: Vec<String> = Vec::new();
                            for agent in content.iter() {
                                if let Some(text) = agent.get("text").and_then(|t| t.as_str()) {
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                                        if let Some(agents) = parsed.get("agents").and_then(|a| a.as_array()) {
                                            for a in agents {
                                                if let Some(name) = a.get("name").and_then(|n| n.as_str()) {
                                                    agent_names.push(name.to_string());
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if agent_names.is_empty() {
                                "No agents are currently active.".to_string()
                            } else {
                                format!("Active agents: {}.", agent_names.join(", "))
                            }
                        } else {
                            "No agents found in the system.".to_string()
                        }
                    }
                    Err(e) => {
                        error!("Failed to list agents: {}", e);
                        format!("Failed to list agents. Error: {}", e)
                    }
                }
            },

            crate::actors::voice_commands::SwarmIntent::ExecuteTask { description, priority } => {
                info!("Executing task: {} with priority: {:?}", description, priority);

                let priority_str = match priority {
                    crate::actors::voice_commands::TaskPriority::Critical => "critical",
                    crate::actors::voice_commands::TaskPriority::High => "high",
                    crate::actors::voice_commands::TaskPriority::Medium => "medium",
                    crate::actors::voice_commands::TaskPriority::Low => "low",
                };

                match call_task_orchestrate(&mcp_host, &mcp_port, &description, Some(priority_str), Some("balanced")).await {
                    Ok(task_result) => {
                        let task_id = task_result.get("taskId")
                            .and_then(|id| id.as_str())
                            .unwrap_or("unknown");

                        // Track the task in context
                        let mut params = std::collections::HashMap::new();
                        params.insert("task_id".to_string(), task_id.to_string());
                        params.insert("description".to_string(), description.clone());
                        params.insert("priority".to_string(), priority_str.to_string());

                        let _ = context_manager.add_pending_operation(
                            &session_id,
                            "execute_task".to_string(),
                            params,
                            Some(chrono::Utc::now() + chrono::Duration::minutes(30)), // Expected completion
                        ).await;

                        format!("Task '{}' has been assigned to the swarm with ID: {}.", description, task_id)
                    }
                    Err(e) => {
                        error!("Failed to orchestrate task: {}", e);
                        format!("Failed to execute task '{}'. Error: {}", description, e)
                    }
                }
            },

            crate::actors::voice_commands::SwarmIntent::Help => {
                "You can ask me to spawn agents, check status, list agents, or execute tasks. Just speak naturally!".to_string()
            },

            _ => {
                "Command received but not yet implemented.".to_string()
            }
        }
    }
}
