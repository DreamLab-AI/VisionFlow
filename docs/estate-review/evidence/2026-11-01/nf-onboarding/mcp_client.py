import json,urllib.request,urllib.parse,threading,queue
class Client:
 def __init__(self):
  self.base='http://browsercontainer:8931';self.messages=queue.Queue();self.endpoint=queue.Queue();self.seq=0
  self.stream=urllib.request.urlopen(self.base+'/sse',timeout=45)
  def read():
   event='';data=[]
   for raw in self.stream:
    line=raw.decode().rstrip('\r\n')
    if not line:
     if event=='endpoint':self.endpoint.put(''.join(data))
     elif event=='message':self.messages.put(json.loads('\n'.join(data)))
     event='';data=[]
    elif line.startswith('event:'):event=line[6:].strip()
    elif line.startswith('data:'):data.append(line[5:].strip())
  threading.Thread(target=read,daemon=True).start();self.url=urllib.parse.urljoin(self.base,self.endpoint.get(timeout=10));self.request('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'visionflow-onboarding-audit','version':'1'}});self.notify('notifications/initialized',{})
 def send(self,obj):
  req=urllib.request.Request(self.url,data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,timeout=35) as r:r.read()
 def notify(self,method,params):self.send({'jsonrpc':'2.0','method':method,'params':params})
 def request(self,method,params):
  self.seq+=1;i=self.seq;self.send({'jsonrpc':'2.0','id':i,'method':method,'params':params})
  while True:
   msg=self.messages.get(timeout=45)
   if msg.get('id')==i:
    if 'error' in msg:raise RuntimeError(msg['error'])
    return msg['result']
 def tool(self,name,args):return self.request('tools/call',{'name':name,'arguments':args})
if __name__=='__main__':
 c=Client();r=c.request('tools/list',{});open('/tmp/vf-nf-onboarding-artifacts/browser-tools.json','w').write(json.dumps(r,indent=2))
 for t in r['tools']:
  if t['name'] in ['new_page','close_page','evaluate_script','navigate_page','take_screenshot','list_pages']:print(json.dumps(t))
