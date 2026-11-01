use crate::presence_actor::{
    BroadcastFrame, JoinRejection, JoinRoom, LeaveRoom, ListMembers, PresenceActor,
    RoomEventEnvelope,
};
use actix::{Actor, Context, Handler};
use sha2::{Digest, Sha256};
use visionclaw_xr_presence::{AvatarMetadata, Did, RoomId};

struct Sink;
impl Actor for Sink {
    type Context = Context<Self>;
}
impl Handler<BroadcastFrame> for Sink {
    type Result = ();
    fn handle(&mut self, _: BroadcastFrame, _: &mut Context<Self>) {}
}
impl Handler<RoomEventEnvelope> for Sink {
    type Result = ();
    fn handle(&mut self, _: RoomEventEnvelope, _: &mut Context<Self>) {}
}
fn identity(i: u8) -> Did {
    Did::parse(format!("did:nostr:{i:064x}")).unwrap()
}
fn metadata(did: &Did) -> AvatarMetadata {
    AvatarMetadata {
        did: did.clone(),
        display_name: "audit".into(),
        model_uri: None,
    }
}

#[actix::test]
async fn actor_accepts_33_members_and_preserves_duplicate_and_rejoin_controls() {
    let room =
        PresenceActor::new(RoomId::parse("urn:visionclaw:room:sha256-12-123456abcdef").unwrap())
            .start();
    let sink = Sink.start();
    let mut ids = vec![];
    for i in 1..=33 {
        let did = identity(i);
        let ack = room
            .send(JoinRoom {
                did: did.clone(),
                metadata: metadata(&did),
                frame_recipient: sink.clone().recipient(),
                event_recipient: sink.clone().recipient(),
            })
            .await
            .unwrap()
            .unwrap();
        ids.push(ack.avatar_id);
    }
    assert_eq!(room.send(ListMembers).await.unwrap().len(), 33);
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"actor_accepts_33_members","members":33,"classification":"missing_proposed_cap_not_auth_bypass"})
    );
    let did = identity(1);
    let duplicate = room
        .send(JoinRoom {
            did: did.clone(),
            metadata: metadata(&did),
            frame_recipient: sink.clone().recipient(),
            event_recipient: sink.clone().recipient(),
        })
        .await
        .unwrap();
    assert!(matches!(duplicate, Err(JoinRejection::DuplicateMember)));
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"duplicate_did_refused","passed":true})
    );
    room.send(LeaveRoom {
        avatar_id: ids.remove(0),
    })
    .await
    .unwrap();
    assert_eq!(room.send(ListMembers).await.unwrap().len(), 32);
    let rejoined = room
        .send(JoinRoom {
            did: did.clone(),
            metadata: metadata(&did),
            frame_recipient: sink.clone().recipient(),
            event_recipient: sink.clone().recipient(),
        })
        .await
        .unwrap()
        .unwrap();
    ids.push(rejoined.avatar_id);
    assert_eq!(room.send(ListMembers).await.unwrap().len(), 33);
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"leave_rejoin_control","members":33})
    );
    for avatar_id in ids {
        room.send(LeaveRoom { avatar_id }).await.unwrap();
    }
}

#[test]
fn proposed_room_vector_and_current_parser_boundary() {
    let domain = b"visionclaw/room/nip29/v1";
    let key: Vec<u8> = (0..32).collect();
    let group = b"general";
    let mut input = domain.to_vec();
    input.extend(&key);
    input.extend((group.len() as u16).to_be_bytes());
    input.extend(group);
    let digest = Sha256::digest(&input);
    let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    let urn = format!("urn:visionclaw:room:nip29-{hex}");
    assert_eq!(
        urn,
        "urn:visionclaw:room:nip29-e5488c85b92647026b5d54e422122a6f"
    );
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"proposal_golden_vector","domain_bytes":domain.len(),"urn":urn,"matches_specified_hex_input":true})
    );
    assert!(RoomId::parse(&urn).is_err());
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"proposed_nip29_room_rejected_by_current_parser","rejected":true})
    );
    let mixed = RoomId::parse("urn:visionclaw:room:sha256-12-ABCDEF123456").unwrap();
    let lower = RoomId::parse("urn:visionclaw:room:sha256-12-abcdef123456").unwrap();
    assert_ne!(mixed, lower);
    assert_ne!(mixed.wire_hash(), lower.wire_hash());
    println!(
        "AUDIT_CASE {}",
        serde_json::json!({"case":"legacy_room_case_is_not_canonicalised","distinct_rooms_and_wire_hashes":true,"classification":"legacy_parser_boundary"})
    );
}
