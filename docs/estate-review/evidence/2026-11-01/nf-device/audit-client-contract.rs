
#[cfg(test)]
mod audit_client_contract {
 use super::*;
 #[test]
 fn client_registration_0_omits_required_proof() {
  let device_pubkey = "11".repeat(32); let label = "audit";
  let body = serde_json::json!({ "device_pubkey": device_pubkey, "label": label }).to_string();
  let parsed: RegisterDeviceBody = serde_json::from_str(&body).unwrap();
  assert!(normalize_register(&parsed).is_ok());
  assert!(parsed.device_proof.is_none(), "current producer does not send the proof required by handle_register");
 }
 #[test]
 fn client_registration_1_omits_required_proof() {
  let device_pubkey = "11".repeat(32); let label = "audit";
  let body = serde_json::json!({ "device_pubkey": device_pubkey, "label": label }).to_string();
  let parsed: RegisterDeviceBody = serde_json::from_str(&body).unwrap();
  assert!(normalize_register(&parsed).is_ok());
  assert!(parsed.device_proof.is_none(), "current producer does not send the proof required by handle_register");
 }

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct DeviceKey {
    /// 64-hex device/child pubkey.
    pub device_pubkey: String,
    /// Human label ("My phone"). May be absent on legacy rows.
    #[serde(default)]
    pub label: Option<String>,
    /// Unix seconds the device was registered.
    #[serde(default)]
    pub created_at: u64,
    /// 0 = active, 1 = revoked.
    #[serde(default)]
    pub revoked: u8,
}
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("no signer available")]
    NoSigner,
    #[error("master key unavailable")]
    NoMasterKey,
    #[error("subkey derivation failed: {0}")]
    Derive(String),
    #[error("nsec encode failed: {0}")]
    Encode(String),
    #[error("request failed: {0}")]
    Request(String),
    #[error("invalid response: {0}")]
    Parse(String),
}
fn parse_device_list(raw: &str) -> Result<Vec<DeviceKey>, DeviceError> {
    if let Ok(list) = serde_json::from_str::<Vec<DeviceKey>>(raw) {
        return Ok(list);
    }
    #[derive(Deserialize)]
    struct Envelope {
        #[serde(default)]
        devices: Vec<DeviceKey>,
    }
    serde_json::from_str::<Envelope>(raw)
        .map(|e| e.devices)
        .map_err(|e| DeviceError::Parse(e.to_string()))
}

 #[test] fn nonempty_server_list_is_rejected_by_client() {
  let row = DeviceRow {device_pubkey: "11".repeat(32),owner_pubkey: "22".repeat(32),label: None,created_at: 100.0,revoked: 0.0};
  let response = serde_json::json!({"devices": vec![row]}).to_string();
  let err = parse_device_list(&response).unwrap_err().to_string();
  assert!(err.contains("floating point"), "{err}");
 }
 #[test] fn empty_server_list_is_accepted_control() {
  let rows: Vec<DeviceRow> = vec![];
  let response = serde_json::json!({"devices": rows}).to_string();
  assert!(parse_device_list(&response).unwrap().is_empty());
 }
 #[test] fn integer_created_at_still_rejects_float_revoked() {
  let row = DeviceRow {device_pubkey: "11".repeat(32),owner_pubkey: "22".repeat(32),label: None,created_at: 100.0,revoked: 0.0};
  let mut response = serde_json::json!({"devices": vec![row]});
  response["devices"][0]["created_at"] = serde_json::json!(100);
  assert!(parse_device_list(&response.to_string()).is_err());
  response["devices"][0]["revoked"] = serde_json::json!(0);
  assert_eq!(parse_device_list(&response.to_string()).unwrap().len(),1);
 }
}
