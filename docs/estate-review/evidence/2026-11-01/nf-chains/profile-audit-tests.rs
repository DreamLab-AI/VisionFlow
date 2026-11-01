
#[cfg(test)] mod estate_audit {
use super::*;
fn r(s: &str)->(Vec<ChainProfile>,Vec<String>){resolve(Some(s),Legacy{mirror:None,relays:None})}
#[test] fn dream_asset_can_be_replaced_without_warning(){let a="ab".repeat(32);let(p,w)=r(&format!(r#"[{{"id":"sidestr:dreamlab","asset_id":"{a}"}}]"#));assert_eq!(p[0].asset_id.as_deref(),Some(a.as_str()));assert_eq!(p[0].ticker,"DREAM");assert!(w.is_empty());}
#[test] fn wrong_typed_mirror_drops_txbt4_and_falls_back_to_dream(){let(p,w)=r(r#"[{"id":"sidestr:dreamlab-txbt4","mirror":17}]"#);assert_eq!(p.len(),1);assert_eq!(p[0].id,chain::CHAIN_ID);assert!(!w.is_empty());}
#[test] fn slash_only_https_is_accepted_then_becomes_non_url(){let(p,w)=r(r#"[{"id":"sidestr:dreamlab-txbt4","mirror":"https:///"}]"#);assert_eq!(p[0].mirror,"https:");assert!(w.is_empty());}
#[test] fn bare_wss_is_accepted(){let(p,w)=r(r#"[{"id":"sidestr:dreamlab-txbt4","relays":["wss://"]}]"#);assert_eq!(p[0].relays,vec!["wss://"]);assert!(w.is_empty());}
#[test] fn unpinned_chain_still_cannot_be_added(){let(p,w)=r(r#"[{"id":"sidestr:untrusted"}]"#);assert_eq!(p[0].id,chain::CHAIN_ID);assert!(!w.is_empty());}
}
