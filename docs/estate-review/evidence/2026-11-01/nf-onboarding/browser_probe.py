import json,re,base64,sys
from pathlib import Path
from mcp_client import Client
out=Path('/tmp/vf-nf-onboarding-artifacts');c=Client();results=[];page=None

def call(name,args):
 r=c.tool(name,args)
 if r.get('isError'):raise RuntimeError(r)
 return r

def evaluate(name,code):
 r=call('evaluate_script',{'function':code});text='\n'.join(x.get('text','') for x in r.get('content',[]) if x.get('type')=='text');blocks=re.findall(r'```(?:json)?\s*([\s\S]*?)```',text)
 value=json.loads(blocks[-1]) if blocks else text
 results.append({'case':name,'result':value});return value
try:
 r=call('new_page',{'url':'http://172.20.0.10:18763/plain.html#k=SYNTHETIC-FRAGMENT','isolatedContext':'vf-onboarding-audit-20261101'});text='\n'.join(x.get('text','') for x in r['content']);m=re.search(r'(?m)^(\d+):.*\[selected\]',text)
 if not m:raise RuntimeError('No selected-page id in new_page response: '+text)
 page=int(m[1]);(out/'owned-browser-page.json').write_text(json.dumps({'page_id':page,'isolated_context':'vf-onboarding-audit-20261101'}))
 v=evaluate('plain_html',"() => ({marker: window.auditXss ?? null, syntheticSecretVisible:document.body.innerText.includes('SYNTHETIC-RECOVERY-KEY'), fragment:location.hash})")
 assert v['marker'] is None and v['syntheticSecretVisible'] and v['fragment']=='#k=SYNTHETIC-FRAGMENT'
 v=evaluate('history_success',"() => { history.replaceState(null,'','/plain.html');return {fragment:location.hash}; }");assert v['fragment']==''
 v=evaluate('history_cross_origin_failure',"() => { history.replaceState(null,'','/plain.html#k=SYNTHETIC-FRAGMENT');let error='';try{history.replaceState(null,'','https://example.invalid/connect');}catch(e){error=e.name;}return {error,fragment:location.hash};}");assert v['error']=='SecurityError' and v['fragment']=='#k=SYNTHETIC-FRAGMENT'
 call('navigate_page',{'type':'url','url':'http://172.20.0.10:18763/marker.html'})
 v=evaluate('display_name_html_executes',"() => ({marker:window.auditXss ?? null, syntheticSecretVisible:document.body.innerText.includes('SYNTHETIC-RECOVERY-KEY'), nameLength:'<svg onload=window.auditXss=1>'.length, scripts:document.scripts.length})");assert v['marker']==1 and v['syntheticSecretVisible'] and v['nameLength']<=50
 screenshot=call('take_screenshot',{'format':'png','fullPage':False})
 for item in screenshot.get('content',[]):
  if item.get('type')=='image':(out/'recovery-marker-browser.png').write_bytes(base64.b64decode(item['data']))
 call('navigate_page',{'type':'url','url':'http://172.20.0.10:18763/escaped-control.html'})
 v=evaluate('escaped_name_control',"() => ({marker:window.auditXss ?? null, escapedNameVisible:document.body.innerText.includes('<svg onload=window.auditXss=1>'), syntheticSecretVisible:document.body.innerText.includes('SYNTHETIC-RECOVERY-KEY')})");assert v['marker'] is None and v['escapedNameVisible'] and v['syntheticSecretVisible']
 print(json.dumps({'browser_cases':len(results),'all_expected':True,'owned_page':page}))
finally:
 if page is not None:
  call('close_page',{'pageId':page});(out/'owned-browser-page.json').write_text(json.dumps({'page_id':page,'closed':True,'isolated_context':'vf-onboarding-audit-20261101'}))
 (out/'browser-results.json').write_text(json.dumps({'transport':'browsercontainer chrome-devtools MCP SSE','cases':results,'limits':['Synthetic HTML from extracted production formatter; not the full Leptos signup or connect app.','Inert marker executes only assignment; no exfiltration or external requests.','History error is deliberately cross-origin, showing browser failure semantics; production route failure controlled separately by native mock.','Escaped control is a test-only derivative, not a production fix.']},indent=2)+'\n')
