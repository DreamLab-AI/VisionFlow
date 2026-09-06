#!/usr/bin/env python3
"""Compile the actual sensing-server parser with a firmware-layout fixture."""
from pathlib import Path
import subprocess,tempfile,json,hashlib
w=Path(__file__).resolve().parents[4]
p=w/'RuView/rust-port/wifi-densepose-rs/crates/wifi-densepose-sensing-server/src/main.rs'
s=p.read_text()
def block(marker):
 start=s.index(marker);begin=s.index('{',start);depth=1;end=begin+1
 while depth:
  if s[end]=='{':depth+=1
  elif s[end]=='}':depth-=1
  end+=1
 return s[start:end]
code=block('struct Esp32Frame {')+'\n'+block('fn parse_esp32_frame(')+r'''
fn main() {
 let mut b=vec![0u8;148];
 b[0..4].copy_from_slice(&0xC5110001u32.to_le_bytes());
 b[4]=1;b[5]=1;b[6..8].copy_from_slice(&64u16.to_le_bytes());
 b[8..12].copy_from_slice(&2412u32.to_le_bytes());
 b[12..16].copy_from_slice(&42u32.to_le_bytes());
 b[16]=(-45i8) as u8;b[17]=(-95i8) as u8;
 for iq in b[20..].chunks_exact_mut(2){iq[0]=3;iq[1]=4;}
 let f=parse_esp32_frame(&b).unwrap();
 println!("{{\"sequence\":{},\"rssi\":{},\"noise_floor\":{},\"subcarriers\":{},\"first_amplitude\":{}}}",f.sequence,f.rssi,f.noise_floor,f.n_subcarriers,f.amplitudes[0]);
}
'''
with tempfile.TemporaryDirectory(prefix='estate-ruview-parser-') as td:
 t=Path(td);(t/'probe.rs').write_text(code)
 c=subprocess.run(['rustc','--edition=2021','-Awarnings',str(t/'probe.rs'),'-o',str(t/'probe')],capture_output=True,text=True)
 if c.returncode:raise RuntimeError(c.stderr)
 run=subprocess.run([str(t/'probe')],capture_output=True,text=True,check=True)
 result={'source':str(p.relative_to(w)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'method':'actual extracted Rust struct/parser; synthetic bytes matching firmware/csi_collector.c layout; no device','expected':{'sequence':42,'rssi':-45,'noise_floor':-95,'subcarriers':64,'first_amplitude':5},'observed':json.loads(run.stdout)}
 out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
