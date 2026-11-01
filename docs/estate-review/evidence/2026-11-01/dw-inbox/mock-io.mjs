import fs from 'node:fs';
import * as nip17 from 'nostr-tools/nip17';
const dir=process.env.INBOX_WATCH_DIR;
const f=JSON.parse(fs.readFileSync(dir+'/fixture.json','utf8'));
const events=[...f.events];
const record=(data)=>fs.appendFileSync(dir+'/io.jsonl',JSON.stringify(data)+'\n');
globalThis.fetch=async(url,opts={})=>{
 record({transport:'fetch',url,method:opts.method??'GET'});
 if(String(url).includes('/api/check-whitelist'))return new Response(JSON.stringify({isWhitelisted:f.whitelisted??false,ingressRecipient:true,isAdmin:f.privileged,cohorts:f.privileged?['agent']:[]}));
 if(String(url).endsWith('/api/agents/disclosure'))return new Response(JSON.stringify({agents:[{pubkey:f.monitorPk,name:'website-info',registered_by:'b'.repeat(64)}]}));
 if(url==='https://fixture.invalid/')return new Response('<script src="/assets/index-fixture.js"></script>');
 if(url==='https://fixture.invalid/assets/index-fixture.js')return new Response([f.monitorPk,f.adminPk,f.jarvisPk,'dreamlab-nostr-relay','DreamLab website signup'].join(' '));
 throw Error('unexpected fixture fetch '+url);
};
globalThis.WebSocket=class {
 constructor(url){record({transport:'ws-open',url});setImmediate(()=>this.emit(['AUTH','fixture-challenge']));}
 emit(frame){this.onmessage?.({data:JSON.stringify(frame)});}
 send(data){const frame=JSON.parse(data);record({transport:'ws-send',type:frame[0],eventKind:frame[1]?.kind});
  if(frame[0]==='AUTH'){setImmediate(()=>this.emit(['OK',frame[1].id,true,'']));return;}
  if(frame[0]==='EVENT'){
   const e=frame[1];
   if(e.tags.some(t=>t[0]==='p'&&t[1]===f.monitorPk))events.push(e);
   if(e.tags.some(t=>t[0]==='p'&&t[1]===f.operatorPk)){
    const rumor=nip17.unwrapEvent(e,Uint8Array.from(Buffer.from(f.operatorSk,'hex')));
    record({transport:'digest',content:rumor.content});
   }
   setImmediate(()=>this.emit(['OK',e.id,!f.rejectDigest||!e.tags.some(t=>t[0]==='p'&&t[1]===f.operatorPk),'fixture-result']));return;
  }
  if(frame[0]==='REQ'){
   const filter=frame[2];const matches=events.filter(e=>e.created_at>=filter.since&&(filter.until===undefined||e.created_at<=filter.until)).sort((a,b)=>b.created_at-a.created_at).slice(0,filter.limit);
   setImmediate(()=>{for(const e of matches)this.emit(['EVENT',frame[1],e]);this.emit(['EOSE',frame[1]]);});
  }
 }
 close(){}
};
