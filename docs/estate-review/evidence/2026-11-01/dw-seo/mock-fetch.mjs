import fs from 'node:fs';
let modelCalls=0;
globalThis.fetch=async(url,options={})=>{
 fs.appendFileSync(process.env.FIXTURE_CALLS,JSON.stringify({url,method:options.method??'GET'})+'\n');
 if(url==='https://fixture.invalid/v1/messages'){
  modelCalls++;
  if(process.env.FIXTURE_MODE==='error')return new Response(JSON.stringify({error:process.env.ZAI_API_KEY}),{status:400});
  if(process.env.FIXTURE_MODE==='retry'&&modelCalls===1)return new Response(JSON.stringify({content:[],stop_reason:'max_tokens',usage:{input_tokens:111,output_tokens:222}}));
  return new Response(JSON.stringify({content:[{type:'text',text:'## Summary\nFixture only.\nPROPOSE: NO'}],usage:{input_tokens:7,output_tokens:9}}));
 }
 if(!String(url).startsWith('https://dreamlab-ai.com/'))throw Error('unexpected fixture URL');
 return new Response('',{status:503});
};
