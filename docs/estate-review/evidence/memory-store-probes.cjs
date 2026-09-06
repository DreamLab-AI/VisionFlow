'use strict';
// No database calls: execute the production factory against injected mocks.
const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'../../../../project/agentbox');
process.env.RUVECTOR_TYPED_METADATA='true';
process.env.RUVECTOR_EPISODIC_TTL_SWEEP='true';
const {createMemoryTools}=require(path.join(root,'mcp/servers/lib/memory-tools.js'));
(async()=>{
 const queries=[];
 const t=createMemoryTools({backend:'external-pg',deps:{
 pool:{query:async(sql,params)=>{queries.push({sql,params});return {rows:[],rowCount:0};}},getPgOk:()=>true,
 getEmbedding:async()=>{throw new Error('synthetic embedding outage');},xinfEnsure:async()=>true,
 vecToSql:a=>'['+a.join(',')+']',entryId:(ns,key)=>`agentbox:${ns}:${key}`,parseVal:v=>v,
 notifyMemoryFlash:()=>{},notifyMemoryFlashBatch:()=>{},log:()=>{},writeSourceType:'agentbox'}});
 const stored=await t.memStore('synthetic','new value','probe',{ttl_seconds:1});
 assert.equal(stored.success,true);assert.equal(stored.embedded,false);
 assert(queries[0].sql.includes('COALESCE(EXCLUDED.embedding, memory_entries.embedding)'));
 const metadata=JSON.parse(queries[0].params.at(-1));assert.equal(metadata.memory_type,'semantic');assert(metadata.expires_at);
 await t.memRetrieve('synthetic','probe');await t.memList('probe',2);await t.memSweepEpisodic('probe');
 const out={method:'production factory, injected mock database and embedding failure; no external mutation',stored,metadata,queries,observations:{nullEmbeddingInsert:queries[0].sql.includes('NULL'),retainOldEmbeddingOnReplacement:true,retrieveFiltersExpiry:queries[1].sql.includes('expires_at'),listFiltersExpiry:queries[2].sql.includes('expires_at'),sweepRequiresEpisodic:queries[3].sql.includes("= 'episodic'")}};
 fs.writeFileSync(path.join(__dirname,'memory-store-probes.json'),JSON.stringify(out,null,2)+'\n');
 console.log(JSON.stringify(out.observations));
})().catch(e=>{console.error(e);process.exitCode=1;});
