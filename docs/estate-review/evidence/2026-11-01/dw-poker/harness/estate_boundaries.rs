
use nostr_bbs_poker_citizen::{table::{Citizen,Config,House,Balances,Effect},ledger::{Ledger,Owed}};
use nostr_bbs_poker::{protocol::{ToCitizen,ToHero,VERSION,Asset},engine::Action};
fn make(ledger:Ledger)->Citizen { let mut n=0u8; Citizen::new(Config{stakes_bb:vec![2],buyin_bb:100,daily_cap:200,claim_grace_secs:600,..Config::default()},House{pubkey:"c".repeat(64),script_hex:format!("5120{}","c".repeat(64))},ledger,Box::new(move||{n=n.wrapping_add(1);[n;32]})) }
fn bal()->Balances{Balances{hero_asset:200,house_asset:200}}
fn hello(c:&mut Citizen, who:&str, now:u64)->String{c.handle(who,ToCitizen::Hello{v:VERSION},now,bal()).into_iter().find_map(|e|match e{Effect::Send(_,ToHero::Offer{commit,..})=>Some(commit),_=>None}).unwrap()}
fn sit(c:&mut Citizen,who:&str,commit:&str,now:u64)->Vec<Effect>{c.handle(who,ToCitizen::Sit{v:VERSION,commit:commit.into(),asset:Asset::Dream,bb:2,nonce:"ab".repeat(32)},now,bal())}
fn debt(who:&str)->Owed{Owed{hero:who.into(),to:"c".repeat(64),root:"1".repeat(64),amount:20,since:1,claimed:None}}
#[test] fn arbitrary_payment_claim_can_be_renewed_after_grace_expires(){
 let who="a".repeat(64);let mut l=Ledger::default();l.owed.push(debt(&who));let mut c=make(l);
 for now in [1000,5000,50000]{
  let out=c.handle(&who,ToCitizen::Paid{root:"1".repeat(64),txid:"z".repeat(64)},now,bal());
  assert_eq!(out,vec![Effect::Persist]);assert_eq!(c.ledger.owed[0].claimed.as_ref().unwrap().at,now);
 }
 let commit=hello(&mut c,&who,50001);let out=sit(&mut c,&who,&commit,50001);
 assert!(out.iter().any(|e|matches!(e,Effect::Send(_,ToHero::State{..}))));assert_eq!(c.ledger.owed.len(),1);
}
#[test] fn parallel_hands_do_not_reserve_house_balance_or_daily_cap(){
 let mut c=make(Ledger::default());for who in ["a".repeat(64),"b".repeat(64)]{
  let commit=hello(&mut c,&who,1000);let out=sit(&mut c,&who,&commit,1000);
  assert!(out.iter().any(|e|matches!(e,Effect::Send(_,ToHero::State{..}))));assert!(c.in_hand(&who));
 }assert_eq!(c.ledger.owing_total(),0);assert_eq!(c.ledger.paid_today(1000),0);
}
#[test] fn live_hand_has_no_timeout_and_is_lost_by_ledger_restart(){
 let who="a".repeat(64);let mut c=make(Ledger::default());let commit=hello(&mut c,&who,1000);sit(&mut c,&who,&commit,1000);
 c.tick(1000+86400*30);assert!(c.in_hand(&who));let restored=make(Ledger::parse(&c.ledger.to_json()).unwrap());assert!(!restored.in_hand(&who));assert_eq!(restored.ledger.hands,0);
}
#[test] fn acceptance_does_not_recheck_challenger_debt(){
 let a="a".repeat(64);let b="b".repeat(64);let mut c=make(Ledger::default());hello(&mut c,&a,1000);hello(&mut c,&b,1000);
 let out=c.handle(&a,ToCitizen::Challenge{v:VERSION,opponent:b.clone(),bb:2,nonce:"aa".repeat(32)},1000,bal());
 let commit=out.iter().find_map(|e|match e{Effect::Send(_,ToHero::Waiting{commit,..})=>Some(commit.clone()),_=>None}).unwrap();
 c.ledger.owed.push(debt(&a));
 let out=c.handle(&b,ToCitizen::Accept{commit,nonce:"bb".repeat(32)},1001,bal());
 assert!(out.iter().any(|e|matches!(e,Effect::Send(_,ToHero::State{..}))));assert!(c.in_hand(&a));
}
#[test] fn finished_hand_is_sent_before_first_persist_effect(){
 let who="a".repeat(64);let mut c=make(Ledger::default());let commit=hello(&mut c,&who,1000);sit(&mut c,&who,&commit,1000);
 let out=c.handle(&who,ToCitizen::Act{commit,action:Action{seat:0,action:"fold".into(),amount:None}},1001,bal());
 let done=out.iter().position(|e|matches!(e,Effect::Send(_,ToHero::Done{..}))).unwrap();let save=out.iter().position(|e|matches!(e,Effect::Persist)).unwrap();assert!(done<save);assert_eq!(c.ledger.owed.len(),1);
}
#[test] fn version_only_ledger_is_accepted_as_empty_book(){
 let l=Ledger::parse(r#"{"version":1}"#).unwrap();assert_eq!(l,Ledger::default());
}
