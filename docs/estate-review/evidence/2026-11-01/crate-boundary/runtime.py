import pathlib,subprocess,json,datetime,hashlib,urllib.request,re,tomllib
ab=pathlib.Path('/home/devuser/workspace/project/agentbox')
result=subprocess.run(['supervisorctl','status'],text=True,capture_output=True)
status=result.stdout
assert status, 'supervisor returned no status output'
rows=[]
for line in status.splitlines():
 if not line.startswith('sidestr-'):continue
 name,state,*_=line.split();m=re.search(r'pid (\d+)',line);r={'name':name,'state':state}
 if m:
  pid=int(m.group(1));p=pathlib.Path('/proc')/str(pid);args=p.joinpath('cmdline').read_bytes().decode().split('\0');r.update(pid=pid,executable=str(p.joinpath('exe').resolve()),entrypoint=args[:3] if 'producer' in name else args[:2])
  # Select public routing/state flags only; never capture key/cookie contents or environment.
  r['public_flags']={a:args[i+1] for i,a in enumerate(args[:-1]) if a in ['--chain','--dir','--port','--checkpoint-every']}
 rows.append(r)
observed={'supervisor_status_exit':result.returncode,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'programs':rows,'upstream_pins':{},'local_endpoints':[],'configuration':{}}
for name in ['spec','schema','blaketestnode']:
 p=pathlib.Path('/opt/agentbox/sidestr/upstream')/name/'.pin-commit';observed['upstream_pins'][name]=p.read_text().strip() if p.exists() else None
cfg=tomllib.load(open(ab/'agentbox.toml','rb'))['sidechain']
for name in ['dreamlab','dreamlab-txbt4']:
 c=cfg if name=='dreamlab' else cfg[name]
 observed['configuration'][name]={k:c.get(k) for k in ['enabled','mirror','faucet','checkpoint_every']}
 port=3450 if name=='dreamlab' else 3451;item={'chain':name,'port':port}
 source=json.load(open(ab/'config/sidechain'/name/'chain.json'))
 item['sealed_document']={k:source.get(k) for k in ['id','parent','level','rules','genesisHash']}
 for route in ['tip','chain.json']:
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}',timeout=4) as response:
    data=response.read(65537);assert len(data)<=65536;v=json.loads(data)
   if route=='tip':item[route]={k:v.get(k) for k in ['height','hash']}
   else:item[route]={'matches_sealed_document':v==source,'fields':{k:v.get(k) for k in ['id','parent','level','rules','genesisHash']}}
  except Exception as e:item[route]={'error':type(e).__name__+': '+str(e)}
 observed['local_endpoints'].append(item)
for f in ['config/sidechain/run-producer.sh','config/sidechain/run-faucet.sh']:
 live=pathlib.Path('/opt/agentbox')/f;observed.setdefault('baked_runner_matches_checkout',{})[f]=live.read_bytes()==(ab/f).read_bytes()
observed['limits']='Point-in-time process and public loopback endpoint observation. No relay, mirror freshness, grants, payment/channel lifecycle or parent-node acceptance. No secrets or environment captured; no restart or mutation.'
out=pathlib.Path('/tmp/vf-boundary-audit/runtime.json');out.write_text(json.dumps(observed,indent=2)+'\n');print(out.read_text())
