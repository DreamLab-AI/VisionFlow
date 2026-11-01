from pathlib import Path
import json,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--kit',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();assert not (a.kit/'.git').exists();a.out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent;s=here/'source';recovery=(s/'crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs').read_text();connect=(s/'crates/nostr-bbs-forum-client/src/pages/connect.rs').read_text();auth=(s/'crates/nostr-bbs-forum-client/src/auth/mod.rs').read_text();signup=(s/'crates/nostr-bbs-forum-client/src/pages/signup.rs').read_text()
parts={}
def take(name,t,start,end):
 i=t.index(start);j=t.index(end,i+len(start));v=t[i:j];parts[name]=dict(sha256=hashlib.sha256(v.encode()).hexdigest(),source=v);return v
render=take('download_format_body',recovery,'struct RecoveryDownload', '    let Some(window) = web_sys::window() else')
render=render.replace("fn download_recovery_html(sheet: RecoveryDownload<'_>) {", "fn download_recovery_html(sheet: RecoveryDownload<'_>) -> String {")+ '    html\n}\n'
read=take('read_hash_function',connect,'fn read_key_from_hash()', '\n/// Replace the current history')
strip=take('strip_history_function',connect,'fn strip_fragment_from_history()', '\n#[component]')
effect=take('connect_effect_body',connect,'    Effect::new(move |_| {','\n\n    let continue_now')
effect=effect.removeprefix('    Effect::new(move |_| {').rsplit('    });',1)[0]
validation=take('local_key_validation_prefix',auth,'        let bytes = if key_input.starts_with("nsec1")','        self.privkey.set_value')
decode=take('client_decode_nsec',auth,'fn decode_nsec(', '\n// -- Context providers')
oldtests=take('existing_decode_tests',auth,'    #[test]\n    fn decode_nsec_valid()', '    // ── StoredSession serialization')
printclosure=take('print_callback',recovery,'    let on_print = move |_| {','\n\n    let on_toggle_confirm')
printbody=printclosure.removeprefix('    let on_print = move |_| {').rsplit('    };',1)[0]
gate=take('ready_effect',recovery,'    Effect::new(move |_| {','\n\n    // Capture values')
gatebody=gate.removeprefix('    Effect::new(move |_| {').rsplit('    });',1)[0]
code=(here/'fixture-prefix.rs').read_text()+render+read+strip+decode+'\nfn validate_key(key_input: &str) -> Result<String,String> {\n'+validation+'Ok(pubkey)\n}\nfn run_connect() {let auth=Auth;let navigate=Navigate;let state=State;\n'+effect+'}\nfn run_print(){let printed=Printed;\n'+printbody+'}\nfn run_ready(){let printed=Printed;let confirmed=Confirmed;let on_ready=Ready;\n'+gatebody+'}\n'+oldtests+(here/'audit-cases.rs').read_text()
dest=a.kit/'crates/nostr-bbs-core/tests/audit_onboarding.rs';assert not dest.exists();dest.write_text(code);(a.out/'extracted-source.json').write_text(json.dumps(parts,indent=2)+'\n');(a.out/'audit_onboarding.rs').write_text(code)
print('Prepared exact extracted bodies with documented mocks',len(parts))
