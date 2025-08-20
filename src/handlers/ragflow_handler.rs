use actix_web::{web, HttpResponse, ResponseError, Responder, Result};
use crate::AppState;
use crate::handlers::validation_handler::ValidationService;
use crate::utils::validation::rate_limit::{RateLimiter, EndpointRateLimits, extract_client_id};
use crate::utils::validation::sanitization::Sanitizer;
use crate::utils::validation::errors::DetailedValidationError;
use crate::utils::validation::MAX_REQUEST_SIZE;
use serde::{Serialize, Deserialize};
use log::{error, info, warn, debug};
use serde_json::{json, Value};
use futures::StreamExt;
use actix_web::web::Bytes;
use crate::services::ragflow_service::RAGFlowError;
use actix_web::web::ServiceConfig;
use crate::types::speech::SpeechOptions;
use crate::models::ragflow_chat::{RagflowChatRequest, RagflowChatResponse};
use actix_web::HttpRequest;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    pub user_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionResponse {
    pub success: bool,
    pub session_id: String,
    pub message: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SendMessageRequest {
    pub question: String,
    pub stream: Option<bool>,
    pub session_id: Option<String>,
    pub enable_tts: Option<bool>,
}

// Implement ResponseError for RAGFlowError
impl ResponseError for RAGFlowError {
    fn error_response(&self) -> HttpResponse {
        HttpResponse::InternalServerError()
            .json(json!({"error": self.to_string()}))
    }
}

/// Handler for sending a message to the RAGFlow service.
pub async fn send_message(
    state: web::Data<AppState>,
    request: web::Json<SendMessageRequest>,
) -> impl Responder {
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    // Get session ID from request or use the default one from app state if not provided
    let session_id = match &request.session_id {
        Some(id) => id.clone(),
        None => state.ragflow_session_id.clone(),
    };

    let enable_tts = request.enable_tts.unwrap_or(false);
    // The quote and doc_ids parameters are not used in the new API
    match ragflow_service.send_message(
        session_id,
        request.question.clone(),
        false, // quote parameter (unused)
        None,  // doc_ids parameter (unused)
        request.stream.unwrap_or(true),
    ).await {
        Ok(response_stream) => {
            // Check if TTS is enabled and speech service exists
            if enable_tts {
                if let Some(speech_service) = &state.speech_service {
                    let speech_service = speech_service.clone();
                    // Clone the question to pass to TTS
                    let question = request.question.clone();
                    // Spawn a task to process TTS in the background
                    actix_web::rt::spawn(async move {
                        let speech_options = SpeechOptions::default();
                        // The exact question will be sent to TTS
                        if let Err(e) = speech_service.text_to_speech(question, speech_options).await {
                            error!("Error processing TTS: {:?}", e);
                        }
                    });
                }
            }
            
            // Continue with normal text response handling
            let enable_tts = enable_tts; // Clone for capture in closure
            let mapped_stream = response_stream.map(move |result| {
                result.map(|answer| {
                    // Skip empty messages (like the end marker)
                    if answer.is_empty() {
                        return Bytes::new();
                    }
                    
                    // If TTS is enabled, send answer to speech service
                    if enable_tts {
                        if let Some(speech_service) = &state.speech_service {
                            let speech_service = speech_service.clone();
                            let speech_options = SpeechOptions::default();
                            let answer_clone = answer.clone();
                            actix_web::rt::spawn(async move {
                                if let Err(e) = speech_service.text_to_speech(answer_clone, speech_options).await {
                                    error!("Error processing TTS for answer: {:?}", e);
                                }
                            });
                        }
                    }
                    
                    let json_response = json!({
                        "answer": answer,
                        "success": true
                    });
                    Bytes::from(json_response.to_string())
                })
                .map_err(|e| actix_web::error::ErrorInternalServerError(e))
            });
            HttpResponse::Ok().streaming(mapped_stream)
        },
        Err(e) => {
            error!("Error sending message: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to send message: {}", e)
            }))
        }
    }
}

/// Handler for initiating a new session with RAGFlow agent.
pub async fn create_session(
    state: web::Data<AppState>,
    request: web::Json<CreateSessionRequest>,
) -> impl Responder {
    let user_id = request.user_id.clone();
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    match ragflow_service.create_session(user_id.clone()).await {
        Ok(session_id) => {
            // Store the session ID in the AppState for future use
            // We can't directly modify AppState through an Arc, but we can clone it and create a new state
            // For now, we'll log this situation but not update the shared state
            // In a production environment, you'd want a better solution like using RwLock for the session_id
            info!(
                "Created new RAGFlow session: {}. Note: session ID cannot be stored in shared AppState.",
                session_id
            );
            // Use the session_id directly from the request in subsequent calls
            
            HttpResponse::Ok().json(CreateSessionResponse {
                success: true,
                session_id,
                message: None,
            })
        },
        Err(e) => {
            error!("Failed to initialize chat: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to initialize chat: {}", e)
            }))
        }
    }
}

/// Handler for retrieving session history.
pub async fn get_session_history(
    state: web::Data<AppState>,
    session_id: web::Path<String>,
) -> impl Responder {
    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => return HttpResponse::ServiceUnavailable().json(json!({
            "error": "RAGFlow service is not available"
        }))
    };

    match ragflow_service.get_session_history(session_id.to_string()).await {
        Ok(history) => HttpResponse::Ok().json(history),
        Err(e) => {
            error!("Failed to get session history: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to get chat history: {}", e)
            }))
        }
    }
}

/// Configure RAGFlow API routes
async fn handle_ragflow_chat(
    state: web::Data<AppState>,
    req: HttpRequest, // To get headers for auth
    payload: web::Json<RagflowChatRequest>,
) -> impl Responder {
    // Authentication: Check for power user
    let pubkey = match req.headers().get("X-Nostr-Pubkey").and_then(|v| v.to_str().ok()) {
        Some(pk) => pk.to_string(),
        None => return HttpResponse::Unauthorized().json(json!({"error": "Missing X-Nostr-Pubkey header"})),
    };
    let token = match req.headers().get("Authorization").and_then(|v| v.to_str().ok().map(|s| s.trim_start_matches("Bearer "))) {
        Some(t) => t.to_string(),
        None => return HttpResponse::Unauthorized().json(json!({"error": "Missing Authorization token"})),
    };

    if let Some(nostr_service) = &state.nostr_service {
        if !nostr_service.validate_session(&pubkey, &token).await {
            return HttpResponse::Unauthorized().json(json!({"error": "Invalid session token"}));
        }
        // Accessing feature checks through AppState methods
        let has_ragflow_specific_access = state.has_feature_access(&pubkey, "ragflow");
        let is_power_user = state.is_power_user(&pubkey);

        if !is_power_user && !has_ragflow_specific_access {
            return HttpResponse::Forbidden().json(json!({"error": "This feature requires power user access or specific RAGFlow permission"}));
        }
    } else {
        // This case should ideally not be reached if nostr_service is integral
        // and initialized properly. Consider logging a warning or error.
        error!("Nostr service not available during chat handling for pubkey: {}", pubkey);
        return HttpResponse::InternalServerError().json(json!({"error": "Nostr service not available"}));
    }

    info!("[handle_ragflow_chat] Checking RAGFlow service availability. Is Some: {}", state.ragflow_service.is_some()); // ADDED LOG

    let ragflow_service = match &state.ragflow_service {
        Some(service) => service,
        None => {
            error!("[handle_ragflow_chat] RAGFlow service is None, returning 503."); // ADDED LOG
            return HttpResponse::ServiceUnavailable().json(json!({"error": "RAGFlow service not available"}));
        }
    };

    info!("[handle_ragflow_chat] RAGFlow service is Some. Proceeding."); // ADDED LOG

    let mut session_id = payload.session_id.clone();
    if session_id.is_none() {
        // Create a new session if none provided. Using pubkey as user_id for RAGFlow session.
        match ragflow_service.create_session(pubkey.clone()).await {
            Ok(new_sid) => {
                info!("Created new RAGFlow session {} for pubkey {}", new_sid, pubkey);
                session_id = Some(new_sid);
            }
            Err(e) => {
                error!("Failed to create RAGFlow session for pubkey {}: {}", pubkey, e);
                return HttpResponse::InternalServerError().json(json!({"error": format!("Failed to create RAGFlow session: {}", e)}));
            }
        }
    }

    // We've ensured it's Some by now, or returned an error.
    let current_session_id = session_id.expect("Session ID should be Some at this point");

    let stream_preference = payload.stream.unwrap_or(false); // Default to false if not provided
    match ragflow_service.send_chat_message(current_session_id.clone(), payload.question.clone(), stream_preference).await {
        Ok((answer, final_session_id)) => {
            HttpResponse::Ok().json(RagflowChatResponse {
                answer,
                session_id: final_session_id, // RAGFlow service send_chat_message returns the session_id it used
            })
        }
        Err(e) => {
            error!("Error communicating with RAGFlow for session {}: {}", current_session_id, e);
            HttpResponse::InternalServerError().json(json!({"error": format!("RAGFlow communication error: {}", e)}))
        }
    }
}
/// Enhanced RAGFlow handler with comprehensive validation and security
pub struct EnhancedRagFlowHandler {
    validation_service: ValidationService,
    rate_limiter: Arc<RateLimiter>,
}

impl EnhancedRagFlowHandler {
    pub fn new() -> Self {
        let config = EndpointRateLimits::ragflow_chat();
        let rate_limiter = Arc::new(RateLimiter::new(config));

        Self {
            validation_service: ValidationService::new(),
            rate_limiter,
        }
    }

    /// Enhanced chat endpoint with full validation
    pub async fn chat_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);
        
        // Rate limiting check
        if !self.rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for RAGFlow chat from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many chat requests. Please wait before sending another message.",
                "retry_after": self.rate_limiter.reset_time(&client_id).as_secs()
            })));
        }

        // Request size validation
        let payload_size = serde_json::to_vec(&*payload).unwrap_or_default().len();
        if payload_size > MAX_REQUEST_SIZE {
            error!("RAGFlow chat payload too large: {} bytes", payload_size);
            return Ok(HttpResponse::PayloadTooLarge().json(json!({
                "error": "payload_too_large",
                "message": "Chat message too long",
                "max_size": MAX_REQUEST_SIZE
            })));
        }

        info!("Processing enhanced RAGFlow chat from client: {} (size: {} bytes)", client_id, payload_size);

        // Authentication validation
        let pubkey = match req.headers().get("X-Nostr-Pubkey").and_then(|v| v.to_str().ok()) {
            Some(pk) => pk.to_string(),
            None => {
                warn!("Missing authentication header from client: {}", client_id);
                return Ok(HttpResponse::Unauthorized().json(json!({
                    "error": "authentication_required",
                    "message": "X-Nostr-Pubkey header is required"
                })));
            }
        };

        let token = match req.headers().get("Authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_start_matches("Bearer ")) {
            Some(t) => t.to_string(),
            None => {
                warn!("Missing authorization token from client: {}", client_id);
                return Ok(HttpResponse::Unauthorized().json(json!({
                    "error": "authorization_required",
                    "message": "Authorization token is required"
                })));
            }
        };

        // Validate session and permissions
        if let Some(nostr_service) = &state.nostr_service {
            if !nostr_service.validate_session(&pubkey, &token).await {
                warn!("Invalid session for pubkey: {} from client: {}", pubkey, client_id);
                return Ok(HttpResponse::Unauthorized().json(json!({
                    "error": "invalid_session",
                    "message": "Invalid session token"
                })));
            }

            let has_ragflow_access = state.has_feature_access(&pubkey, "ragflow");
            let is_power_user = state.is_power_user(&pubkey);

            if !is_power_user && !has_ragflow_access {
                warn!("Insufficient permissions for pubkey: {} from client: {}", pubkey, client_id);
                return Ok(HttpResponse::Forbidden().json(json!({
                    "error": "insufficient_permissions",
                    "message": "RAGFlow access requires power user privileges or specific permission"
                })));
            }
        } else {
            error!("Nostr service not available for authentication");
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "authentication_service_unavailable",
                "message": "Authentication service is not available"
            })));
        }

        // Comprehensive validation and sanitization
        let validated_payload = match self.validation_service.validate_ragflow_chat(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("RAGFlow chat validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("RAGFlow chat validation passed for client: {}", client_id);

        // Extract validated parameters
        let question = validated_payload.get("question")
            .and_then(|q| q.as_str())
            .ok_or_else(|| {
                error!("Question field missing from validated payload");
                DetailedValidationError::missing_required_field("question")
            })?;

        let session_id = validated_payload.get("session_id")
            .and_then(|s| s.as_str())
            .map(String::from);

        let stream = validated_payload.get("stream")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);

        let enable_tts = validated_payload.get("enable_tts")
            .and_then(|t| t.as_bool())
            .unwrap_or(false);

        // Additional question content validation
        self.validate_question_content(question)?;

        // Get or create session
        let ragflow_service = match &state.ragflow_service {
            Some(service) => service,
            None => {
                error!("RAGFlow service not available");
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "ragflow_service_unavailable",
                    "message": "RAGFlow service is currently not available"
                })));
            }
        };

        let current_session_id = match session_id {
            Some(id) => id,
            None => {
                debug!("Creating new RAGFlow session for pubkey: {}", pubkey);
                match ragflow_service.create_session(pubkey.clone()).await {
                    Ok(new_session_id) => {
                        info!("Created new RAGFlow session {} for pubkey {}", new_session_id, pubkey);
                        new_session_id
                    }
                    Err(e) => {
                        error!("Failed to create RAGFlow session: {}", e);
                        return Ok(HttpResponse::InternalServerError().json(json!({
                            "error": "session_creation_failed",
                            "message": "Failed to create new chat session"
                        })));
                    }
                }
            }
        };

        // Process TTS if requested
        if enable_tts {
            self.process_tts_request(&state, question).await;
        }

        // Send message to RAGFlow
        match ragflow_service.send_chat_message(current_session_id.clone(), question.to_string(), stream).await {
            Ok((answer, final_session_id)) => {
                info!("RAGFlow response received for client: {} (session: {})", client_id, final_session_id);

                // Process TTS for response if requested
                if enable_tts {
                    self.process_tts_request(&state, &answer).await;
                }

                Ok(HttpResponse::Ok().json(RagflowChatResponse {
                    answer,
                    session_id: final_session_id,
                }))
            }
            Err(e) => {
                error!("RAGFlow communication error for session {}: {}", current_session_id, e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "ragflow_communication_failed",
                    "message": "Failed to communicate with RAGFlow service",
                    "session_id": current_session_id
                })))
            }
        }
    }

    /// Enhanced session creation with validation
    pub async fn create_session_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // Rate limiting
        if !self.rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many session creation requests"
            })));
        }

        info!("Processing enhanced session creation from client: {}", client_id);

        // Validate user_id field
        let user_id = payload.get("user_id")
            .and_then(|u| u.as_str())
            .ok_or_else(|| DetailedValidationError::missing_required_field("user_id"))?;

        // Sanitize user_id
        let sanitized_user_id = Sanitizer::sanitize_string(user_id)
            .map_err(|e| {
                warn!("User ID sanitization failed: {}", e);
                e
            })?;

        let ragflow_service = match &state.ragflow_service {
            Some(service) => service,
            None => {
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "ragflow_service_unavailable",
                    "message": "RAGFlow service is not available"
                })));
            }
        };

        match ragflow_service.create_session(sanitized_user_id.clone()).await {
            Ok(session_id) => {
                info!("RAGFlow session created: {} for user: {} (client: {})", session_id, sanitized_user_id, client_id);
                Ok(HttpResponse::Ok().json(json!({
                    "success": true,
                    "session_id": session_id,
                    "user_id": sanitized_user_id,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })))
            }
            Err(e) => {
                error!("Failed to create RAGFlow session for user {}: {}", sanitized_user_id, e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "session_creation_failed",
                    "message": "Failed to create new session"
                })))
            }
        }
    }

    /// Get session history with validation
    pub async fn get_session_history_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        session_id: web::Path<String>,
    ) -> Result<HttpResponse> {
        let client_id = extract_client_id(&req);

        // More permissive rate limiting for history requests
        let history_rate_limiter = Arc::new(RateLimiter::new(
            crate::utils::validation::rate_limit::RateLimitConfig {
                requests_per_minute: 30,
                burst_size: 10,
                ..Default::default()
            }
        ));

        if !history_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many history requests"
            })));
        }

        // Validate and sanitize session ID
        let sanitized_session_id = Sanitizer::sanitize_string(&session_id)
            .map_err(|e| {
                warn!("Session ID sanitization failed: {}", e);
                e
            })?;

        debug!("Getting session history for session: {} (client: {})", sanitized_session_id, client_id);

        let ragflow_service = match &state.ragflow_service {
            Some(service) => service,
            None => {
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "ragflow_service_unavailable",
                    "message": "RAGFlow service is not available"
                })));
            }
        };

        match ragflow_service.get_session_history(sanitized_session_id.clone()).await {
            Ok(history) => {
                debug!("Session history retrieved for session: {}", sanitized_session_id);
                Ok(HttpResponse::Ok().json(json!({
                    "session_id": sanitized_session_id,
                    "history": history,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })))
            }
            Err(e) => {
                error!("Failed to get session history for {}: {}", sanitized_session_id, e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "history_retrieval_failed",
                    "message": "Failed to retrieve session history"
                })))
            }
        }
    }

    /// Validate question content for potentially harmful input
    fn validate_question_content(&self, question: &str) -> Result<(), DetailedValidationError> {
        // Check for prompt injection attempts
        let injection_patterns = [
            "ignore previous instructions",
            "forget everything above",
            "new instructions:",
            "system:",
            "\\n\\nUser:",
            "\\n\\nAssistant:",
            "<|im_start|>",
            "<|im_end|>",
        ];

        let question_lower = question.to_lowercase();
        for pattern in &injection_patterns {
            if question_lower.contains(pattern) {
                warn!("Potential prompt injection detected: {}", pattern);
                return Err(DetailedValidationError::malicious_content(
                    "question", 
                    "prompt_injection"
                ));
            }
        }

        // Check for excessive repetition (spam detection)
        if self.has_excessive_repetition(question) {
            return Err(DetailedValidationError::new(
                "question",
                "Question contains excessive repetition", 
                "EXCESSIVE_REPETITION"
            ));
        }

        // Check for extremely long questions
        if question.len() > 8000 {
            return Err(DetailedValidationError::new(
                "question",
                "Question is too long",
                "QUESTION_TOO_LONG"
            ));
        }

        Ok(())
    }

    /// Check for excessive repetition in question
    fn has_excessive_repetition(&self, text: &str) -> bool {
        if text.len() < 50 {
            return false;
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_counts = std::collections::HashMap::new();

        for word in &words {
            *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }

        // Check if any word appears more than 30% of total words
        let total_words = words.len();
        word_counts.values().any(|&count| count as f64 / total_words as f64 > 0.3)
    }

    /// Process TTS request asynchronously
    async fn process_tts_request(&self, state: &web::Data<AppState>, text: &str) {
        if let Some(speech_service) = &state.speech_service {
            let speech_service = speech_service.clone();
            let text = text.to_string();
            
            tokio::spawn(async move {
                if let Err(e) = speech_service.text_to_speech(text, Default::default()).await {
                    error!("TTS processing failed: {}", e);
                }
            });
        }
    }
}

impl Default for EnhancedRagFlowHandler {
    fn default() -> Self {
        Self::new()
    }
}

pub fn config(cfg: &mut ServiceConfig) {
    let handler = web::Data::new(EnhancedRagFlowHandler::new());
    
    cfg.app_data(handler.clone())
        .service(
            web::scope("/ragflow")
                .route("/session", web::post().to(create_session)) // Existing
                .route("/message", web::post().to(send_message))   // Existing (streaming)
                .route("/chat", web::post().to(|req, state, payload, handler: web::Data<EnhancedRagFlowHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    match handler.chat_enhanced(req, state, payload).await {
                        Ok(response) => response,
                        Err(_) => handle_ragflow_chat(state, HttpRequest::from_parts(
                            actix_web::dev::RequestHead::default(),
                            actix_web::dev::Payload::None
                        ).unwrap(), web::Json(RagflowChatRequest {
                            question: "fallback".to_string(),
                            session_id: None,
                            stream: Some(false)
                        })).await
                    }
                })) // Enhanced chat endpoint with fallback
                .route("/session/enhanced", web::post().to(|req, state, payload, handler: web::Data<EnhancedRagFlowHandler>| {
                    handler.create_session_enhanced(req, state, payload)
                })) // Enhanced session creation
                .route("/history/{session_id}", web::get().to(get_session_history)) // Existing
                .route("/history/enhanced/{session_id}", web::get().to(|req, state, session_id, handler: web::Data<EnhancedRagFlowHandler>| {
                    handler.get_session_history_enhanced(req, state, session_id)
                })) // Enhanced history
        );
}
