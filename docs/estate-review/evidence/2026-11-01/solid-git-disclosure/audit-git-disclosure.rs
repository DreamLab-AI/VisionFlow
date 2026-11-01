//! Disposable observation probe: synthetic data, local Git, no remote endpoints.
#![cfg(feature = "git")]
use actix_web::{test, http::Method};
use bytes::Bytes;
use solid_pod_rs::{auth::nip98, storage::{fs::FsBackend, Storage}};
use solid_pod_rs_server::{build_app, AppState};
use std::{path::Path, process::{Command, Stdio}, io::Write, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

fn git(root: &Path, args: &[&str]) -> String {
    let o=Command::new("git").arg("-C").arg(root).args(args).output().unwrap();
    assert!(o.status.success(), "git {args:?}: {}",String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().trim().to_owned()
}
fn decode_pack(body: &[u8], destination: &Path, commit: &str, file: &str) -> String {
    let start=body.windows(4).position(|s|s==b"PACK").expect("real pack response");
    git(destination,&["init","-q"]);
    let mut p=Command::new("git").arg("-C").arg(destination).args(["index-pack","--stdin"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    p.stdin.take().unwrap().write_all(&body[start..]).unwrap();
    let out=p.wait_with_output().unwrap();
    assert!(out.status.success(),"index pack: {}",String::from_utf8_lossy(&out.stderr));
    git(destination,&["show",&format!("{commit}:{file}")])
}
fn policy(owner: &str,target: &str,public: bool)->String {
    let mut graph=vec![serde_json::json!({"@id":"#owner","acl:agent":{"@id":owner},"acl:accessTo":{"@id":target},"acl:default":{"@id":target},"acl:mode":[{"@id":"acl:Read"},{"@id":"acl:Write"},{"@id":"acl:Control"}]})];
    if public {graph.push(serde_json::json!({"@id":"#public","acl:agentClass":{"@id":"foaf:Agent"},"acl:accessTo":{"@id":target},"acl:default":{"@id":target},"acl:mode":{"@id":"acl:Read"}}));}
    serde_json::json!({"@graph":graph}).to_string()
}
#[actix_web::test]
async fn observe_normal_and_nested_git_roots() {
    let tmp=tempfile::tempdir().unwrap();
    let root=tmp.path();let pod=root.join("alice");std::fs::create_dir(&pod).unwrap();
    git(&pod,&["init","-q","-b","main"]);
    let key=k256::schnorr::SigningKey::from_slice(&[0x53;32]).unwrap();
    let secret=hex::encode(key.to_bytes());let owner=format!("did:nostr:{}",hex::encode(key.verifying_key().to_bytes()));
    let storage=Arc::new(FsBackend::new(root).await.unwrap());
    storage.put("/alice.acl",Bytes::from(policy(&owner,"/alice/",true)),"application/ld+json").await.unwrap();
    storage.put("/alice/secret.txt.acl",Bytes::from(policy(&owner,"/alice/secret.txt",false)),"application/ld+json").await.unwrap();
    let mut state=AppState::new(storage.clone());state.data_root=Some(root.to_path_buf());
    let app=test::init_service(build_app(state)).await;
    let now=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    for (i,body) in ["synthetic-history-before","synthetic-current-private"].iter().enumerate() {
        let token=nip98::mint_with_payload("http://pod.example/alice/secret.txt","PUT",Some(body.as_bytes()),&secret,now+i as u64).unwrap();
        let req=test::TestRequest::default().method(Method::PUT).uri("/alice/secret.txt").insert_header(("host","pod.example")).insert_header(("authorization",nip98::authorization_header(&token))).insert_header(("content-type","text/plain")).set_payload(*body).to_request();
        let res=test::call_service(&app,req).await;
        assert!(res.status().is_success(),"owner PUT {}",res.status());
    }
    let head=git(&pod,&["rev-parse","HEAD"]);let prior=git(&pod,&["rev-parse","HEAD^"]);
    assert_eq!(git(&pod,&["show",&format!("{prior}:secret.txt")]),"synthetic-history-before");
    let denied=test::call_service(&app,test::TestRequest::get().uri("/alice/secret.txt").to_request()).await.status().as_u16();assert_eq!(denied,401);
    let res=test::call_service(&app,test::TestRequest::get().uri("/alice/info/refs?service=git-upload-pack").to_request()).await;
    let status=res.status().as_u16();let text=test::read_body(res).await;println!("normal pod: private LDP={denied}; Git discovery={status}; response={}",String::from_utf8_lossy(&text));assert_eq!(status,404);
    // Positive control: the same Git service with the correct enclosing root.
    let want=format!("want {head}\n");let request=format!("{:04x}{want}00000009done\n",want.len()+4);
    let svc=solid_pod_rs_git::GitHttpService::new(root.to_path_buf());
    let res=svc.handle(solid_pod_rs_git::GitRequest{method:"POST".into(),path:"/alice/git-upload-pack".into(),query:String::new(),headers:vec![("content-type".into(),"application/x-git-upload-pack-request".into())],body:Bytes::from(request),host_url:None}).await.unwrap();assert_eq!(res.status,200);
    let receiver=tempfile::tempdir().unwrap();assert_eq!(decode_pack(&res.body,receiver.path(),&prior,"secret.txt"),"synthetic-history-before");assert_eq!(git(receiver.path(),&["show",&format!("{head}:secret.txt")]),"synthetic-current-private");
    println!("correct-root transport control: decoded old and current synthetic bytes from returned pack");
    // Existing nested repositories are the layout the server currently reaches.
    let nested=pod.join("alice");std::fs::create_dir(&nested).unwrap();git(&nested,&["init","-q","-b","main"]);
    storage.put("/alice/alice/secret.txt",Bytes::from_static(b"synthetic-nested-private"),"text/plain").await.unwrap();
    storage.put("/alice/alice/secret.txt.acl",Bytes::from(policy(&owner,"/alice/alice/secret.txt",false)),"application/ld+json").await.unwrap();
    git(&nested,&["add","secret.txt"]);git(&nested,&["-c","user.name=Audit Fixture","-c","user.email=audit@example.invalid","commit","-qm","synthetic"]);let nested_head=git(&nested,&["rev-parse","HEAD"]);
    let denied=test::call_service(&app,test::TestRequest::get().uri("/alice/alice/secret.txt").to_request()).await.status().as_u16();assert_eq!(denied,401);
    let res=test::call_service(&app,test::TestRequest::get().uri("/alice/info/refs?service=git-upload-pack").to_request()).await;assert_eq!(res.status().as_u16(),200);
    let want=format!("want {nested_head}\n");let request=format!("{:04x}{want}00000009done\n",want.len()+4);
    let res=test::call_service(&app,test::TestRequest::post().uri("/alice/git-upload-pack").insert_header(("content-type","application/x-git-upload-pack-request")).set_payload(request).to_request()).await;let status=res.status().as_u16();assert_eq!(status,200);let body=test::read_body(res).await;
    let receiver=tempfile::tempdir().unwrap();assert_eq!(decode_pack(&body,receiver.path(),&nested_head,"secret.txt"),"synthetic-nested-private");
    println!("nested pod: private LDP={denied}; anonymous Git={status}; decoded protected synthetic bytes from returned pack");
}
