use actix::prelude::*;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use uuid::Uuid;

use crate::services::settings_broadcast::{SettingsBroadcastManager, SettingsWebSocket};

///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
///
pub async fn settings_websocket(
    req: HttpRequest,
    stream: web::Payload,
) -> Result<HttpResponse, actix_web::Error> {
    
    let client_id = Uuid::new_v4().to_string();

    
    let broadcast_manager = SettingsBroadcastManager::from_registry();

    
    let ws_session = SettingsWebSocket::new(client_id.clone(), broadcast_manager);

    log::info!("New settings WebSocket connection: {}", client_id);

    
    ws::start(ws_session, &req, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn test_websocket_endpoint() {
        let app = test::init_service(
            App::new().route("/api/settings/ws", web::get().to(settings_websocket)),
        )
        .await;

        
        let req = test::TestRequest::get()
            .uri("/api/settings/ws")
            .to_request();

        
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success() || resp.status().is_client_error());
    }
}
