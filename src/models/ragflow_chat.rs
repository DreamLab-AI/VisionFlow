// src/models/ragflow_chat.rs
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RagflowChatRequest {
    pub question: String,
    pub session_id: Option<String>, 
    pub stream: Option<bool>,       
                                    
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RagflowChatResponse {
    pub answer: String,
    pub session_id: String, 
                            
}
