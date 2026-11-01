from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--kit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert not (a.kit/'.git').exists();a.out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent;rel='crates/nostr-bbs-auth-worker/src/webauthn.rs';src=(here/'source'/rel).read_text();target=a.kit/rel;assert target.read_text()==src
start=src.index('        let mut h = Sha256::new();',src.index('    let prf_salt = {',src.index('pub async fn register_verify')));end=src.index('\n    };',start);block=src[start:end]
tests=(here/'audit-tests.rs').read_text().replace('__STORED_SALT_BLOCK__',block)
target.write_text(src+'\n'+tests);(a.out/'appended-tests.rs').write_text(tests)
(a.out/'extraction.json').write_text(json.dumps({'path':rel,'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'slice_sha256':hashlib.sha256(block.encode()).hexdigest(),'slice':block,'adaptation':'Server-secret lookup replaced with explicit synthetic parameter. Hash statements unchanged. Other probes call actual worker functions in native unit-test build.'},indent=2)+'\n')
