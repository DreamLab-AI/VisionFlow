
#[cfg(test)] mod estate_audit {
 use super::*;
 use bitcoin::hashes::Hash;
 #[test] fn fragmented_inputs_exceed_displayed_fee_hint(){
  let snap=replay(&DREAMLAB,DREAM_ASSET_ID.parse().ok(),include_bytes!("testdata/blocks.dat"),None).unwrap();
  let pk=sidestr_agent::parse_pubkey("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798").unwrap();
  let signer=sidestr_wallet::external::ExternalSigner::new(pk);
  let coins:Vec<Coin>=(1..=30u8).map(|i|Coin{outpoint:OutPoint{txid:Txid::from_byte_array([i;32]),vout:0},value:1500,height:1,coinbase:false}).collect();
  let to=script_of("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9").unwrap();
  let spend=build_provision(&snap,&coins,&signer,&to,0,30000,"DREAM").unwrap();
  println!("audit fee: {} sats, {} inputs, compared with UI hint 250",spend.fee,spend.tx.input.len());
  assert!(spend.fee>250);
 }
}
