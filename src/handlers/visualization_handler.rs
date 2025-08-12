use crate::config::AppFullSettings;
use crate::AppState;
use crate::actors::messages::{GetSettings, UpdateSettings};
use actix_web::{web, HttpResponse};
use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// Internal helper function to convert camelCase or kebab-case to snake_case
fn to_snake_case(s: &str) -> String {
    // First handle kebab-case by replacing hyphens with underscores
    let s = s.replace('-', "_");
    
    // Then handle camelCase by adding underscores before uppercase letters
    let mut result = String::with_capacity(s.len() + 4);
    let mut chars = s.chars().peekable();
    
    while let Some(c) = chars.next() {
        if c.is_ascii_uppercase() {
            // If this is an uppercase letter, add an underscore before it
            // unless it's at the beginning of the string
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingResponse {
    pub category: String,
    pub setting: String,
    pub value: Value,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategorySettingsResponse {
    pub category: String,
    pub settings: Value,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn get_category_settings_value(settings: &AppFullSettings, category: &str) -> Result<Value, String> {
    debug!("Getting settings for category: {}", category);
    let value = match category {
        "visualisation.nodes" => serde_json::to_value(&settings.visualisation.nodes)
            .map_err(|e| format!("Failed to serialize node settings: {}", e))?,
        "visualisation.edges" => serde_json::to_value(&settings.visualisation.edges)
            .map_err(|e| format!("Failed to serialize edge settings: {}", e))?,
        "visualisation.rendering" => serde_json::to_value(&settings.visualisation.rendering)
            .map_err(|e| format!("Failed to serialize rendering settings: {}", e))?,
        "visualisation.labels" => serde_json::to_value(&settings.visualisation.labels)
            .map_err(|e| format!("Failed to serialize labels settings: {}", e))?,
        "visualisation.bloom" => serde_json::to_value(&settings.visualisation.bloom)
            .map_err(|e| format!("Failed to serialize bloom settings: {}", e))?,
        "visualisation.animations" => serde_json::to_value(&settings.visualisation.animations)
            .map_err(|e| format!("Failed to serialize animations settings: {}", e))?,
        "visualisation.physics" => serde_json::to_value(&settings.visualisation.graphs.logseq.physics)
            .map_err(|e| format!("Failed to serialize physics settings: {}", e))?,
        "visualisation.hologram" => serde_json::to_value(&settings.visualisation.hologram)
            .map_err(|e| format!("Failed to serialize hologram settings: {}", e))?,
        "system.network" => serde_json::to_value(&settings.system.network)
            .map_err(|e| format!("Failed to serialize network settings: {}", e))?,
        "system.websocket" => serde_json::to_value(&settings.system.websocket)
            .map_err(|e| format!("Failed to serialize websocket settings: {}", e))?,
        "system.security" => serde_json::to_value(&settings.system.security)
            .map_err(|e| format!("Failed to serialize security settings: {}", e))?,
        "system.debug" => {
            // Return empty object for debug settings (controlled by env vars)
            serde_json::to_value(&serde_json::json!({}))
                .map_err(|e| format!("Failed to serialize debug settings: {}", e))?
        },
        "xr" => serde_json::to_value(&settings.xr)
            .map_err(|e| format!("Failed to serialize xr settings: {}", e))?,
        "github" => serde_json::to_value(&settings.github)
            .map_err(|e| format!("Failed to serialize github settings: {}", e))?,
        "ragflow" => serde_json::to_value(&settings.ragflow)
            .map_err(|e| format!("Failed to serialize ragflow settings: {}", e))?,
        "perplexity" => serde_json::to_value(&settings.perplexity)
            .map_err(|e| format!("Failed to serialize perplexity settings: {}", e))?,
        "openai" => serde_json::to_value(&settings.openai)
            .map_err(|e| format!("Failed to serialize openai settings: {}", e))?,
        _ => return Err(format!("Invalid category: {}", category)),
    };
    debug!("Successfully retrieved settings for category: {}", category);
    Ok(value)
}

pub async fn get_setting(
    app_state: web::Data<AppState>,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!(
        "Getting setting for category: {}, setting: {}",
        category, setting
    );

    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Use AppFullSettings directly - no conversion needed
    let debug_mode = crate::utils::logging::is_debug_enabled();

    // Get the category data
    let category_value = match get_category_settings_value(&settings, &category) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to get category settings: {}", e);
            return HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some(e),
            });
        }
    };

    // Extract specific setting from the category
    let setting_snake = to_snake_case(&setting);
    let setting_value = category_value.get(&setting_snake).cloned().unwrap_or(Value::Null);

    if debug_mode {
        debug!(
            "Retrieved setting '{}' from category '{}': {:?}",
            setting_snake, category, setting_value
        );
    }

    HttpResponse::Ok().json(SettingResponse {
        category,
        setting,
        value: setting_value,
        success: true,
        error: None,
    })
}

pub async fn update_setting(
    app_state: web::Data<AppState>,
    path: web::Path<(String, String)>,
    value: web::Json<Value>,
) -> HttpResponse {
    let (category, setting) = path.into_inner();
    info!(
        "Updating setting for category: {}, setting: {} with value: {:?}",
        category, setting, value
    );

    // Get current settings
    let mut settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error: {}", e);
            return HttpResponse::InternalServerError().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Update the specific setting
    let category_snake = to_snake_case(&category);
    let setting_snake = to_snake_case(&setting);
    
    // Navigate to the correct nested structure and update
    let settings_json = serde_json::to_value(&settings).unwrap();
    let mut settings_map = settings_json.as_object().unwrap().clone();
    
    // Special handling for debug settings - ignore them
    if category == "system" && setting == "debug" {
        return HttpResponse::Ok().json(SettingResponse {
            category,
            setting,
            value: value.into_inner(),
            success: true,
            error: Some("Debug settings are controlled via environment variables".to_string()),
        });
    }
    
    // Navigate through the path and update the value
    if let Some(category_obj) = settings_map.get_mut(&category_snake) {
        if let Some(category_map) = category_obj.as_object_mut() {
            category_map.insert(setting_snake.clone(), value.clone().into_inner());
        }
    }

    // Convert back to AppFullSettings
    match serde_json::from_value::<AppFullSettings>(Value::Object(settings_map)) {
        Ok(new_settings) => {
            // Update settings via actor
            match app_state.settings_addr.send(UpdateSettings { settings: new_settings }).await {
                Ok(Ok(())) => {
                    HttpResponse::Ok().json(SettingResponse {
                        category,
                        setting,
                        value: value.into_inner(),
                        success: true,
                        error: None,
                    })
                }
                Ok(Err(e)) => {
                    error!("Failed to update settings: {}", e);
                    HttpResponse::InternalServerError().json(SettingResponse {
                        category,
                        setting,
                        value: Value::Null,
                        success: false,
                        error: Some("Failed to update settings".to_string()),
                    })
                }
                Err(e) => {
                    error!("Settings actor mailbox error: {}", e);
                    HttpResponse::InternalServerError().json(SettingResponse {
                        category,
                        setting,
                        value: Value::Null,
                        success: false,
                        error: Some("Settings service unavailable".to_string()),
                    })
                }
            }
        }
        Err(e) => {
            error!("Failed to deserialize updated settings: {}", e);
            HttpResponse::BadRequest().json(SettingResponse {
                category,
                setting,
                value: Value::Null,
                success: false,
                error: Some(format!("Invalid settings format: {}", e)),
            })
        }
    }
}

pub async fn get_category_settings(
    app_state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let category = path.into_inner();
    info!("Getting all settings for category: {}", category);

    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return HttpResponse::InternalServerError().json(CategorySettingsResponse {
                category,
                settings: Value::Null,
                success: false,
                error: Some("Failed to get settings".to_string()),
            });
        }
        Err(e) => {
            error!("Settings actor mailbox error: {}", e);
            return HttpResponse::InternalServerError().json(CategorySettingsResponse {
                category,
                settings: Value::Null,
                success: false,
                error: Some("Settings service unavailable".to_string()),
            });
        }
    };

    // Get category settings
    match get_category_settings_value(&settings, &category) {
        Ok(settings_value) => {
            let debug_enabled = crate::utils::logging::is_debug_enabled();
            let log_json = debug_enabled;

            if log_json {
                debug!(
                    "Retrieved settings for category '{}': {}",
                    category,
                    serde_json::to_string(&settings_value).unwrap_or_else(|_| "Unable to serialize".to_string())
                );
            }

            HttpResponse::Ok().json(CategorySettingsResponse {
                category,
                settings: settings_value,
                success: true,
                error: None,
            })
        }
        Err(e) => {
            error!("Failed to get category settings: {}", e);
            HttpResponse::BadRequest().json(CategorySettingsResponse {
                category,
                settings: Value::Null,
                success: false,
                error: Some(e),
            })
        }
    }
}