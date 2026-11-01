import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('./js/push.js',import.meta.url),'utf8');
test('actual pushStoreCfg writes IndexedDB but not logout localStorage copy', async () => {
 const records=new Map(), localWrites=[];
 const indexedDB={open(){const req={};queueMicrotask(()=>{
  req.result={close(){},transaction(){const tx={};tx.objectStore=()=>({put(v,k){records.set(k,v);queueMicrotask(()=>tx.oncomplete?.());return {result:k};}});return tx;}};
  req.onsuccess();
 });return req;}};
 const context=vm.createContext({indexedDB,Promise,Uint8Array,atob,localStorage:{setItem:(...x)=>localWrites.push(x)}});
 vm.runInContext(source.replace(/^export /gm,''),context);
 await context.pushStoreCfg('{"id":"device","token":"fake"}');
 assert.equal(records.get('cfg'),'{"id":"device","token":"fake"}');
 assert.deepEqual(localWrites,[]);
});
