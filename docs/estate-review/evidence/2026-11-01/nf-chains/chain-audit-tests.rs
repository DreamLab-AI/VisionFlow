
#[cfg(test)] mod estate_audit {
use super::*;
#[test] fn valid_old_prefix_replays_after_the_full_snapshot(){let dat=include_bytes!("testdata/txbt4-blocks.dat");let full=replay(&DREAMLAB_TXBT4,None,dat,None).unwrap();let first=&sidestr_core::mirror::records(dat).unwrap()[0];let short=sidestr_core::mirror::encode_record(first.height,&first.bytes);let old=replay(&DREAMLAB_TXBT4,None,&short,None).unwrap();assert!(full.height()>=26);assert_eq!(old.height(),0);}
}
