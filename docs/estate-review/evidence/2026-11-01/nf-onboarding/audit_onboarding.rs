#![allow(dead_code,unused_variables)]
use std::cell::RefCell;
#[derive(Clone, PartialEq, Debug)]
enum ConnectState {Working,Success,Error(String)}
#[derive(Clone)]
struct World { window:bool, history:bool, replace_ok:bool, print_ok:bool, hash:String, trace:Vec<&'static str>, result:ConnectState, printed:bool, confirmed:bool, ready:bool, timeout:Option<i32>, imported:Option<String> }
impl Default for World {fn default()->Self{Self{window:true,history:true,replace_ok:true,print_ok:true,hash:String::new(),trace:vec![],result:ConnectState::Working,printed:false,confirmed:false,ready:false,timeout:None,imported:None}}}
thread_local!{static WORLD:RefCell<World>=RefCell::new(World::default());}
fn reset(hash:&str){WORLD.with(|w|*w.borrow_mut()=World{hash:hash.into(),..World::default()});}
fn snapshot()->World{WORLD.with(|w|w.borrow().clone())}
fn base_href(path:&str)->String{format!("/community{path}")}
fn current_app_path(path:&str)->String{base_href(path)}
mod wasm_bindgen {pub struct JsValue;impl JsValue{pub const NULL:Self=Self;}}
mod web_sys {
 use super::*;
 pub struct Window;pub struct Location;pub struct History;
 pub fn window()->Option<Window>{WORLD.with(|w|if w.borrow().window{Some(Window)}else{None})}
 impl Window{pub fn location(&self)->Location{Location} pub fn history(&self)->Result<History,()>{WORLD.with(|w|if w.borrow().history{Ok(History)}else{Err(())})} pub fn print(&self)->Result<(),()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("print");if w.print_ok{Ok(())}else{Err(())}})}}
 impl Location{pub fn hash(&self)->Result<String,()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("read");Ok(w.hash.clone())})}}
 impl History{pub fn replace_state_with_url(&self,_:&wasm_bindgen::JsValue,_:&str,_:Option<&str>)->Result<(),()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("strip");if w.replace_ok{w.hash.clear();Ok(())}else{Err(())}})}}
}
#[derive(Clone,Copy)]struct Auth;impl Auth{fn clear_error(&self){} fn login_with_local_key(&self,k:&str)->Result<(),String>{WORLD.with(|w|w.borrow_mut().trace.push("import"));let p=validate_key(k)?;WORLD.with(|w|w.borrow_mut().imported=Some(p));Ok(())}}
#[derive(Clone,Copy)]struct State;impl State{fn set(&self,s:ConnectState){WORLD.with(|w|w.borrow_mut().result=s)}}
#[derive(Clone,Copy)]struct Navigate;impl Navigate{fn with_value<F:FnOnce(&fn(&str,NavigateOptions))>(&self,f:F){let n:fn(&str,NavigateOptions)=|_,_|{};f(&n)}}
#[derive(Default)]struct NavigateOptions;
fn set_timeout_once<F:FnOnce()+'static>(_:F,ms:i32){WORLD.with(|w|w.borrow_mut().timeout=Some(ms));}
struct Printed;impl Printed{fn get(&self)->bool{snapshot().printed}fn set(&self,v:bool){WORLD.with(|w|w.borrow_mut().printed=v)}}
struct Confirmed;impl Confirmed{fn get(&self)->bool{snapshot().confirmed}}
struct Ready;impl Ready{fn run(&self,_:()){WORLD.with(|w|w.borrow_mut().ready=true)}}
struct RecoveryDownload<'a> {
    display_name: &'a str,
    created: &'a str,
    nip05: Option<&'a str>,
    connect_url: Option<&'a str>,
    connect_qr: &'a str,
    nsec: &'a str,
    nsec_qr: &'a str,
    relay_url: &'a str,
    relay_qr: &'a str,
    npub: &'a str,
    npub_qr: &'a str,
}

fn download_recovery_html(sheet: RecoveryDownload<'_>) -> String {
    let RecoveryDownload {
        display_name,
        created,
        nip05,
        connect_url,
        connect_qr,
        nsec,
        nsec_qr,
        relay_url,
        relay_qr,
        npub,
        npub_qr,
    } = sheet;
    let nip05_line = nip05
        .filter(|h| !h.is_empty())
        .map(|h| format!(" &middot; {h}"))
        .unwrap_or_default();

    let connect_section = connect_url
        .map(|url| {
            format!(
                r#"<div style="border:2px solid #dc2626;border-radius:12px;padding:16px;background:#fef2f2">
  <p style="font-size:14px;font-weight:bold;color:#b91c1c;text-transform:uppercase;letter-spacing:0.05em">📱 Sign in on your phone</p>
  <p style="font-size:12px;color:#1f2937;margin:6px 0">Scan this code with your phone's camera — the forum opens and signs you in.</p>
  <p style="font-size:11px;color:#b91c1c;font-weight:bold;margin-bottom:10px">⚠ Treat this code like a key to your account.</p>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    <div style="flex-shrink:0">{connect_qr}</div>
    <div style="min-width:0;flex:1"><p style="font-size:10px;color:#dc2626;text-transform:uppercase;margin-bottom:4px">your sign-in link</p><code style="font-size:10px;word-break:break-all">{url}</code></div>
  </div>
</div>"#
            )
        })
        .unwrap_or_default();

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recovery Sheet – {display_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; color: #111; background: #fff; }}
  code {{ font-family: "SF Mono", "Cascadia Code", "Fira Code", monospace; word-break: break-all; }}
  svg {{ width: 180px; height: 180px; }}
  .section {{ border: 1px solid #d1d5db; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
  .secret {{ border: 2px solid #dc2626; background: #fef2f2; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  @media print {{ @page {{ margin: 12mm; }} }}
</style>
</head>
<body>
<div style="border-bottom:1px solid #d1d5db;padding-bottom:12px;margin-bottom:16px">
  <h1>Your account &amp; sign-in sheet</h1>
  <p style="font-size:12px;color:#4b5563">Save this file and keep it somewhere safe — it's how you get back into your account. Everything here is private to you; don't share it.</p>
  <p style="font-size:11px;color:#6b7280;margin-top:4px">Account: {display_name}{nip05_line} &middot; Created {created}</p>
</div>

{connect_section}

<div class="section secret">
  <p style="font-size:14px;font-weight:bold;color:#b91c1c;text-transform:uppercase;letter-spacing:0.05em">🔑 Your recovery key</p>
  <p style="font-size:12px;color:#1f2937;margin:6px 0">This is the master key to your account. Anyone who reads it can sign in as you.</p>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    <div style="flex-shrink:0">{nsec_qr}</div>
    <div style="min-width:0;flex:1"><p style="font-size:10px;color:#dc2626;text-transform:uppercase;margin-bottom:4px">recovery key (nsec)</p><code style="font-size:12px">{nsec}</code></div>
  </div>
</div>

<div class="section">
  <p style="font-size:14px;font-weight:bold;text-transform:uppercase;letter-spacing:0.05em">📡 Server address</p>
  <p style="font-size:12px;color:#4b5563;margin:6px 0">Only needed if you connect a separate messaging app.</p>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    <div style="flex-shrink:0">{relay_qr}</div>
    <div style="min-width:0;flex:1"><p style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-bottom:4px">address (relay)</p><code style="font-size:12px">{relay_url}</code></div>
  </div>
</div>

<div class="section">
  <p style="font-size:14px;font-weight:bold;text-transform:uppercase;letter-spacing:0.05em">🪪 Your public profile</p>
  <p style="font-size:12px;color:#4b5563;margin:6px 0">Safe to share — it's how people find you.</p>
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    <div style="flex-shrink:0">{npub_qr}</div>
    <div style="min-width:0;flex:1">
      <p style="font-size:12px"><strong>Name:</strong> {display_name}</p>
      <p style="font-size:10px;color:#6b7280;text-transform:uppercase;margin-top:6px">public ID (npub)</p>
      <code style="font-size:12px">{npub}</code>
    </div>
  </div>
</div>

<div class="section" style="background:#f9fafb">
  <p style="font-size:14px;font-weight:bold;text-transform:uppercase;letter-spacing:0.05em">📖 How to restore</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px">
    <div>
      <p style="font-size:13px;font-weight:600;margin-bottom:4px">On your phone (easiest)</p>
      <ol style="font-size:11px;color:#374151;padding-left:16px;margin:0">
        <li>Point your phone's camera at the 📱 code.</li>
        <li>The forum opens and signs you in automatically.</li>
      </ol>
    </div>
    <div>
      <p style="font-size:13px;font-weight:600;margin-bottom:4px">On a computer</p>
      <ol style="font-size:11px;color:#374151;padding-left:16px;margin:0">
        <li>Open the forum's sign-in page.</li>
        <li>Paste your recovery key (the 🔑 key above).</li>
        <li>You're back in — same account.</li>
      </ol>
    </div>
  </div>
</div>
</body>
</html>"##
    );

    html
}
fn read_key_from_hash() -> Option<String> {
    let window = web_sys::window()?;
    let hash = window.location().hash().ok()?;
    // `hash()` includes the leading '#'. Strip it.
    let frag = hash.strip_prefix('#').unwrap_or(&hash).trim();
    if frag.is_empty() {
        return None;
    }
    // Canonical form is `k=<key>`; tolerate a bare `<key>` too.
    let key = frag.strip_prefix("k=").unwrap_or(frag).trim();
    if key.is_empty() {
        None
    } else {
        Some(key.to_string())
    }
}
fn strip_fragment_from_history() {
    if let Some(window) = web_sys::window() {
        if let Ok(history) = window.history() {
            let clean = base_href("/connect");
            // `replace_state` with the same data and an explicit URL rewrites the
            // visible URL (dropping the fragment) without adding a history entry.
            let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&clean));
        }
    }
}
fn decode_nsec(nsec: &str) -> Result<Vec<u8>, String> {
    let (hrp, data) = bech32::decode(nsec).map_err(|e| format!("Invalid bech32 encoding: {e}"))?;
    if hrp.as_str() != "nsec" {
        return Err(format!("Expected nsec prefix, got {}", hrp.as_str()));
    }
    if data.len() != 32 {
        return Err(format!("nsec data must be 32 bytes, got {}", data.len()));
    }
    Ok(data)
}

fn validate_key(key_input: &str) -> Result<String,String> {
        let bytes = if key_input.starts_with("nsec1") {
            decode_nsec(key_input)?
        } else {
            hex::decode(key_input).map_err(|_| {
                "Invalid key. Paste a 64-char hex key or nsec1... bech32 key.".to_string()
            })?
        };
        if bytes.len() != 32 {
            return Err("Key must be 32 bytes (64 hex characters or nsec1 bech32)".to_string());
        }
        let privkey_hex = &hex::encode(&bytes);
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&bytes);

        let sk = nostr_bbs_core::SecretKey::from_bytes(key_bytes)
            .map_err(|e| format!("Invalid secp256k1 key: {e}"))?;
        let pubkey = sk.public_key().to_hex();
Ok(pubkey)
}
fn run_connect() {let auth=Auth;let navigate=Navigate;let state=State;

        // (1) read first — we need the value before we wipe the URL.
        let key = read_key_from_hash();

        // (2) strip immediately, unconditionally, before any import.
        strip_fragment_from_history();

        // (3) validate + import.
        match key {
            Some(k) => {
                auth.clear_error();
                match auth.login_with_local_key(&k) {
                    Ok(()) => {
                        state.set(ConnectState::Success);
                        // (4) brief pause so the user reads the bearer warning,
                        // then SPA-navigate into the forum.
                        let nav = navigate;
                        set_timeout_once(
                            move || {
                                let dest = current_app_path("/forums");
                                nav.with_value(|n| n(&dest, NavigateOptions::default()));
                            },
                            1500,
                        );
                    }
                    Err(e) => state.set(ConnectState::Error(e)),
                }
            }
            None => state.set(ConnectState::Error(
                "No sign-in key found in this link. Open it from the QR on your recovery sheet, \
                 or sign in manually."
                    .to_string(),
            )),
        }
}
fn run_print(){let printed=Printed;

        if let Some(window) = web_sys::window() {
            window.print().ok();
        }
        printed.set(true);
}
fn run_ready(){let printed=Printed;let confirmed=Confirmed;let on_ready=Ready;

        if printed.get() && confirmed.get() {
            on_ready.run(());
        }
}
    #[test]
    fn decode_nsec_valid() {
        // Generate a valid nsec from known bytes
        let secret = [0x42u8; 32];
        let nsec =
            bech32::encode::<bech32::Bech32>(bech32::Hrp::parse("nsec").unwrap(), &secret).unwrap();
        let decoded = decode_nsec(&nsec).unwrap();
        assert_eq!(decoded.len(), 32);
        assert_eq!(decoded, secret.to_vec());
    }

    #[test]
    fn decode_nsec_wrong_prefix() {
        let data = [0x01u8; 32];
        let npub =
            bech32::encode::<bech32::Bech32>(bech32::Hrp::parse("npub").unwrap(), &data).unwrap();
        let result = decode_nsec(&npub);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected nsec prefix"));
    }

    #[test]
    fn decode_nsec_invalid_bech32() {
        let result = decode_nsec("nsec1notvalidbech32");
        assert!(result.is_err());
    }

    #[test]
    fn decode_nsec_completely_invalid() {
        let result = decode_nsec("not a bech32 string at all");
        assert!(result.is_err());
    }

    #[test]
    fn decode_nsec_empty_string() {
        let result = decode_nsec("");
        assert!(result.is_err());
    }

fn key()->String{"01".repeat(32)}
#[test]fn connect_orders_read_strip_import(){reset(&format!("#k={}",key()));run_connect();let s=snapshot();assert_eq!(s.trace,vec!["read","strip","import"]);assert!(s.hash.is_empty());assert_eq!(s.result,ConnectState::Success);assert_eq!(s.timeout,Some(1500));}
#[test]fn failed_history_rewrite_still_imports(){reset(&format!("#k={}",key()));WORLD.with(|w|w.borrow_mut().replace_ok=false);run_connect();let s=snapshot();assert_eq!(s.result,ConnectState::Success);assert!(s.hash.contains(&key()));assert!(s.imported.is_some());}
#[test]fn unavailable_history_still_imports(){reset(&key());WORLD.with(|w|w.borrow_mut().history=false);run_connect();assert_eq!(snapshot().result,ConnectState::Success);assert_eq!(snapshot().trace,vec!["read","import"]);}
#[test]fn invalid_fragment_cleared_before_error(){reset("#k=not-a-key");run_connect();let s=snapshot();assert!(s.hash.is_empty());assert!(matches!(s.result,ConnectState::Error(_)));assert!(s.imported.is_none());}
#[test]fn empty_fragment_errors_without_import(){reset("#k=");run_connect();assert_eq!(snapshot().trace,vec!["read","strip"]);assert!(matches!(snapshot().result,ConnectState::Error(_)));}
#[test]fn hash_parser_accepts_bare_and_prefix_but_not_url_decoding(){for h in [key(),format!("#{}",key()),format!("k={}",key()),format!("#k= {} ",key())]{reset(&h);assert_eq!(read_key_from_hash(),Some(key()));}reset("#k=%30%31");assert_eq!(read_key_from_hash(),Some("%30%31".into()));assert!(validate_key(&read_key_from_hash().unwrap()).is_err());}
#[test]fn key_validation_accepts_hex_and_nsec_and_rejects_invalid_scalars(){let nsec=nostr_bbs_core::encode_nsec(&key()).unwrap();assert_eq!(validate_key(&key()).unwrap(),validate_key(&nsec).unwrap());for bad in ["00".repeat(32),"ff".repeat(32),"01".repeat(31),"invalid".into()]{assert!(validate_key(&bad).is_err());}}
#[test]fn uppercase_nsec_is_rejected_by_both_layers(){let nsec=nostr_bbs_core::encode_nsec(&key()).unwrap();assert!(decode_nsec(&nsec.to_uppercase()).is_err());assert!(validate_key(&nsec.to_uppercase()).is_err());}
#[test]fn failed_print_still_marks_printed(){reset("");WORLD.with(|w|w.borrow_mut().print_ok=false);run_print();assert!(snapshot().printed);}
#[test]fn absent_window_still_marks_printed(){reset("");WORLD.with(|w|w.borrow_mut().window=false);run_print();assert!(snapshot().printed);}
#[test]fn ready_callback_latches_after_checkbox_cleared(){reset("");WORLD.with(|w|{let mut w=w.borrow_mut();w.printed=true;w.confirmed=true;});run_ready();assert!(snapshot().ready);WORLD.with(|w|w.borrow_mut().confirmed=false);run_ready();assert!(snapshot().ready);assert!(!snapshot().confirmed);}
fn sheet(name:&str,connect:Option<&str>)->String{download_recovery_html(RecoveryDownload{display_name:name,created:"2026-11-01",nip05:None,connect_url:connect,connect_qr:"<svg></svg>",nsec:"SYNTHETIC-RECOVERY-KEY",nsec_qr:"<svg></svg>",relay_url:"wss://example.invalid",relay_qr:"<svg></svg>",npub:"SYNTHETIC-PUBLIC-KEY",npub_qr:"<svg></svg>"})}
#[test]fn export_contains_unescaped_signup_name(){let name="<svg onload=window.auditXss=1>";assert!((2..=50).contains(&name.trim().len()));let h=sheet(name,Some("https://example.invalid/community/connect#k=SYNTHETIC"));assert!(h.contains(name));assert!(h.contains("SYNTHETIC-RECOVERY-KEY"));assert!(!h.contains("&lt;svg"));}
#[test]fn export_has_master_link_and_no_device_or_sweep_inputs(){let h=sheet("Test",Some("https://example.invalid/community/connect#k=SYNTHETIC-MASTER"));assert!(h.contains("SYNTHETIC-MASTER"));assert!(!h.contains("rs-tearoff"));assert!(!h.contains("Remove every address"));}
#[test]fn emit_browser_artifacts(){let out=std::path::PathBuf::from(std::env::var("AUDIT_ONBOARDING_OUTPUT").expect("explicit synthetic output directory"));std::fs::create_dir_all(&out).unwrap();std::fs::write(out.join("plain.html"),sheet("Synthetic account",None)).unwrap();std::fs::write(out.join("marker.html"),sheet("<svg onload=window.auditXss=1>",None)).unwrap();}
