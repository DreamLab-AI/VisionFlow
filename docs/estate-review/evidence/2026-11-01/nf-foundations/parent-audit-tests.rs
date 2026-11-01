
#[cfg(test)]
mod estate_audit {
 use super::*;
 #[test] fn empty_utxo_marks_nonzero_balance_spendable(){assert_eq!(split_maturity(100,Some("[]"),Some(160000)),Maturity::Split{spendable:100,immature:0});}
 #[test] fn inconsistent_ordinary_utxo_total_is_not_detected(){let c=r#"[{"value":1,"coinbase":false,"status":{"confirmed":true,"block_height":159999}}]"#;assert_eq!(split_maturity(100,Some(c),Some(160000)),Maturity::Split{spendable:100,immature:0});}
 #[test] fn future_height_ordinary_coin_is_still_spendable(){let c=r#"[{"value":100,"coinbase":false,"status":{"confirmed":true,"block_height":170000}}]"#;assert_eq!(split_maturity(100,Some(c),Some(160000)),Maturity::Split{spendable:100,immature:0});}
 fn body(f:u64,s:u64)->String{serde_json::json!({"address":"a","chain_stats":{"funded_txo_sum":100,"spent_txo_sum":0},"mempool_stats":{"funded_txo_sum":f,"spent_txo_sum":s}}).to_string()}
 #[test] fn unsigned_max_inflow_becomes_negative_one(){assert_eq!(parse_balance("a",&body(u64::MAX,0),None,None).unwrap().unconfirmed,-1);}
 #[test] #[cfg(debug_assertions)] fn mempool_subtraction_panics_with_out_of_range_backend_values(){assert!(std::panic::catch_unwind(||parse_balance("a",&body(i64::MAX as u64,u64::MAX),None,None)).is_err());}
 #[test] fn duplicate_pair_outpoints_are_counted_twice(){let b=r#"{"address":"a","txbt4":{"only":{"value":0}},"tbtc4":{"only":{"value":0}},"both":[{"txid":"same","vout":0,"value":500},{"txid":"same","vout":0,"value":500}]}"#;assert_eq!(parse_pair("a",b).unwrap().both,1000);}
 #[test] #[cfg(debug_assertions)] fn pair_sum_panics_with_out_of_range_backend_values(){let b=format!(r#"{{"address":"a","txbt4":{{"only":{{"value":0}}}},"tbtc4":{{"only":{{"value":0}}}},"both":[{{"value":{}}},{{"value":1}}]}}"#,u64::MAX);assert!(std::panic::catch_unwind(||parse_pair("a",&b)).is_err());}
 #[test] fn wrong_address_is_rejected(){assert!(parse_balance("other",&body(0,0),None,None).is_err());}
}
