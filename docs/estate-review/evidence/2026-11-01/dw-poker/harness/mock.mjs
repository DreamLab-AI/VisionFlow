import fs from 'node:fs';
globalThis.fetch=()=>{throw Error('network forbidden')};
const frames=[];const record=()=>fs.writeFileSync(process.env.IO_FILE,JSON.stringify(frames));
globalThis.WebSocket=class{constructor(){setImmediate(()=>this.onmessage({data:JSON.stringify(['AUTH','fixture'])}));}send(raw){const f=JSON.parse(raw);frames.push({type:f[0],kind:f[1]?.kind,closed:this.closed??false});record();if(f[0]==='AUTH'&&process.env.CASE==='auth-ack')setImmediate(()=>this.onmessage({data:JSON.stringify(['OK',f[1].id,true,'auth accepted'])}));if(f[0]==='EVENT')setImmediate(()=>this.onmessage({data:JSON.stringify(['OK',f[1].id,process.env.CASE!=='rejected','fixture rejection'])}));}close(){this.closed=true;}}
