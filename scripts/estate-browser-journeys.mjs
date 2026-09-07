#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const out = process.argv[2] || 'docs/estate-review/evidence/execution-2026-09-07/browser-journeys';
mkdirSync(out, { recursive: true });
const ip = execFileSync('python3', ['-c', 'import socket;print(socket.gethostbyname("browsercontainer"))'], {encoding:'utf8'}).trim();
const host = `${ip}:9223`;
const pause = ms => new Promise(r => setTimeout(r, ms));
const http = async (path, method='GET') => (await fetch(`http://${host}${path}`, {method})).json();
class Session {
  constructor(ws) { this.ws=ws; this.n=0; this.pending=new Map(); this.events=[]; ws.addEventListener('message', e => { const m=JSON.parse(e.data); const p=this.pending.get(m.id); if(p){this.pending.delete(m.id);clearTimeout(p.timer);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}else this.events.push(m); }); }
  static async open(url) { const ws=new WebSocket(url.replace(/^ws:\/\/[^/]+/,`ws://${host}`)); await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});return new Session(ws); }
  send(method,params={}) { const id=++this.n; return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.pending.delete(id);reject(Error(`timeout ${method}`));},20000);this.pending.set(id,{resolve,reject,timer});this.ws.send(JSON.stringify({id,method,params}));}); }
  async evaluate(expression) {const r=await this.send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value;}
}
const surfaces=[
 ['candidate','http://agentbox:8108/',true],
 ['canon','https://www.visionflow.info/',true],
 ['explorer',process.env.ESTATE_EXPLORER_URL||'https://narrativegoldmine.com/',false],
 ['notes','https://narrativegoldmine.com/notes/',false],
 ['commercial','https://www.dreamlab-ai.com/',false],
 ['forum','https://www.dreamlab-ai.com/community/',false],
 ['dream',process.env.ESTATE_DREAM_URL||'https://dreamlab-ai.github.io/dream-engine/',false],
];
const records=[];
for(const [name,url,canon] of surfaces.filter(x=>!process.argv[3]||process.argv[3].split(',').includes(x[0]))){
 const t=await http('/json/new?about:blank','PUT');const s=await Session.open(t.webSocketDebuggerUrl);const record={name,url,checks:[]};
 const check=(name,ok,detail)=>record.checks.push({name,ok,detail});
 try{
  for(const method of ['Page.enable','Runtime.enable','Network.enable'])await s.send(method);
  await s.send('Network.setCacheDisabled',{cacheDisabled:true});
  record.cacheDisabled=true;
  await s.send('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
  await s.send('Page.navigate',{url});await pause(1500);
  for(let attempt=0;attempt<20;attempt++){try{if(await s.evaluate('!!document.body && document.body.innerText.length>100'))break;}catch{}await pause(500);}
  record.page=await s.evaluate(`({url:location.href,title:document.title,text:document.body.innerText.length,headings:[...document.querySelectorAll('h1,h2')].map(e=>e.textContent.trim()).slice(0,8),brokenImages:[...document.images].filter(i=>i.complete&&!i.naturalWidth).map(i=>i.src)})`);
  check('content',record.page.text>150,record.page.text);check('images',!record.page.brokenImages.length,record.page.brokenImages);
  if(canon){
   record.reveals=await s.evaluate(`(async()=>{const result=[];for(const el of document.querySelectorAll('.reveal')){el.scrollIntoView({block:'start',behavior:'instant'});for(let attempt=0;attempt<20&&!el.classList.contains('visible');attempt++)await new Promise(r=>setTimeout(r,100));result.push({id:el.id||el.tagName,visible:el.classList.contains('visible'),height:el.offsetHeight});}return result;})()`);
   check('all-sections-reveal',record.reveals.every(x=>x.visible),record.reveals.filter(x=>!x.visible));
   record.toggle=await s.evaluate(`(()=>{const b=document.querySelector('#reading-switch [data-mode="plain"]')||[...document.querySelectorAll('#reading-switch button')].find(b=>/Plain/.test(b.textContent));if(!b)return null;b.click();return {plain:document.body.classList.contains('reading-plain'),pressed:b.getAttribute('aria-pressed')};})()`);
   check('plain-reading-toggle',record.toggle?.plain===true,record.toggle);
   // A regression fixture exceeding ten viewports proves the threshold boundary.
   if(name==='candidate'){
    await s.send('Page.addScriptToEvaluateOnNewDocument',{source:`document.addEventListener('DOMContentLoaded',()=>{const el=document.querySelector('.reveal');if(el){el.style.minHeight='15000px';el.dataset.tallFixture='true';}}, {once:true});`});
    await s.send('Page.reload',{ignoreCache:true});await pause(1500);
    record.tall=await s.evaluate(`(async()=>{const e=document.querySelector('[data-tall-fixture]');e.scrollIntoView({block:'start',behavior:'instant'});await new Promise(r=>setTimeout(r,250));return {height:e.offsetHeight,visible:e.classList.contains('visible')};})()`);
    check('tall-section-regression',record.tall.height>=15000&&record.tall.visible,record.tall);
    await s.send('Page.navigate',{url});await pause(1200);
   }
  }
  if(name==='explorer'){
   await s.send('Emulation.setDeviceMetricsOverride',{width:375,height:812,deviceScaleFactor:1,mobile:true});await pause(500);
   record.homeMobile=await s.evaluate(`({requestedWidth:375,viewport:innerWidth,clientWidth:document.documentElement.clientWidth,width:document.documentElement.scrollWidth})`);
   check('home-mobile-no-overflow',record.homeMobile.clientWidth===375&&record.homeMobile.width<=376,record.homeMobile);
   await s.send('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});await pause(250);

   record.searchStarted=await s.evaluate(`(()=>{const e=document.querySelector('input[type=search]');if(!e)return false;e.focus();return true;})()`);
   if(record.searchStarted){
    // Native input updates React's controlled state; assigning .value does not.
    await s.send('Input.insertText',{text:'Agent'});
    record.searchSubmit=await s.evaluate(`(()=>{const b=document.querySelector('input[type=search]')?.closest('form')?.querySelector('button[type=submit]');if(!b)return null;b.scrollIntoView({block:'center',behavior:'instant'});const r=b.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
    if(record.searchSubmit){
     await s.send('Input.dispatchMouseEvent',{type:'mousePressed',...record.searchSubmit,button:'left',clickCount:1});
     await s.send('Input.dispatchMouseEvent',{type:'mouseReleased',...record.searchSubmit,button:'left',clickCount:1});
    }
   }
   for(let attempt=0;attempt<20;attempt++){
    record.search=await s.evaluate(`({url:location.href,text:document.body.innerText.slice(0,5000)})`);
    if(new URL(record.search.url).pathname==='/search'&&/\b[1-9]\d* results for/.test(record.search.text))break;
    await pause(500);
   }
   const searchUrl=new URL(record.search.url);
   check('search-journey',record.searchStarted&&searchUrl.pathname==='/search'&&searchUrl.searchParams.get('q')==='Agent'&&/\b[1-9]\d* results for/.test(record.search.text),record.search);
  }

  if(name==='notes'&&process.argv[4]){
   const css=readFileSync(process.argv[4],'utf8');record.candidateStyle={path:process.argv[4],sha256:createHash('sha256').update(css).digest('hex')};
   await s.evaluate(`(()=>{const style=document.createElement('style');style.textContent=${JSON.stringify(css)};document.head.append(style);})()`);
  }
  await s.send('Emulation.setDeviceMetricsOverride',{width:375,height:812,deviceScaleFactor:1,mobile:true});await pause(500);
  record.mobile=await s.evaluate(`({requestedWidth:375,viewport:innerWidth,clientWidth:document.documentElement.clientWidth,width:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth})`);
  check('mobile-no-overflow',record.mobile.clientWidth===375&&record.mobile.width<=376,record.mobile);
  if(record.mobile.width>376)record.overflow=await s.evaluate(`([...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>376&&getComputedStyle(e).opacity!=='0').slice(0,12).map(e=>({tag:e.tagName,class:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width})))`);
  await s.evaluate('scrollTo(0,0)');await pause(200);
  const shot=await s.send('Page.captureScreenshot',{format:'png'});const bytes=Buffer.from(shot.data,'base64');writeFileSync(`${out}/${name}-mobile.png`,bytes);record.screenshot={path:`${name}-mobile.png`,sha256:createHash('sha256').update(bytes).digest('hex')};
  record.exceptions=s.events.filter(e=>e.method==='Runtime.exceptionThrown').map(e=>e.params.exceptionDetails.exception?.description||e.params.exceptionDetails.text);
  check('no-uncaught-exceptions',!record.exceptions.length,record.exceptions);
  record.httpStatuses=s.events.filter(e=>e.method==='Network.responseReceived'&&e.params.type==='Document').map(e=>({url:e.params.response.url,status:e.params.response.status}));
 }catch(error){record.error=String(error);check('journey-completed',false,String(error));}
 finally{s.ws.close();await fetch(`http://${host}/json/close/${t.id}`);}
 records.push(record);console.log(name,record.checks.filter(x=>!x.ok));writeFileSync(`${out}/receipt.json`,JSON.stringify({at:new Date().toISOString(),browser:'sidecar Chrome via CDP 9223',scope:'Read-only public/candidate journeys; no identity credentials, submissions or device acceptance',records},null,2)+'\n');
}
process.exitCode=records.some(r=>r.checks.some(c=>!c.ok))?1:0;
