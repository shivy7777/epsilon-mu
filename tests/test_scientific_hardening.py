import json
from urllib.error import HTTPError

import pytest

from epsilonmu.ai.client import Client
from epsilonmu.ai.configuration import AIError, Config
from epsilonmu.analysis import inspect_source
from epsilonmu.investigation import investigate
from epsilonmu.physics_rules import verify_relationships, relationship_coverage


def config(**options):
    return Config('ollama','synthetic','http://127.0.0.1','none',
                  share=('measurements','hypotheses','conclusions'),**options)


@pytest.mark.parametrize('allow_retry',[False,True])
def test_each_dispatch_has_fresh_approval(allow_retry):
    events=[]
    def approve(preview):
        events.append(('ask',json.loads(preview)))
        return len(events)==1 or allow_retry
    def transport(url,headers,body,timeout):
        assert events[-1]==('ask',{'destination':url,'payload':body})
        events.append(('send',None))
        if len(events)==2:
            raise HTTPError(url,429,'',{'Retry-After':'0'},None)
        return {'message':{'content':'{"commentary":"ok"}'}}
    client=Client(config(retries=1),transport)
    if allow_retry:
        client.test_connection(approve)
    else:
        with pytest.raises(AIError,match='declined'):
            client.test_connection(approve)
    assert [e for e,_ in events]==['ask','send','ask']+(['send'] if allow_retry else [])
    assert client.budget.attempts==1+allow_retry
    assert client.authorizations==[{'retry':0,'approved':True},{'retry':1,'approved':allow_retry}]


def oscillator(force,context='harmonic oscillator',gain='7'):
    return f'''"""{context}"""
lever={gain}
inertia=3
delta=.02
offset=2
rate=0
for index in range(4):
    push={force}
    change=push/inertia
    rate=rate+delta*change
    offset+=rate*delta
'''


@pytest.mark.parametrize(('expression','expected'),[
    ('lever*offset','FAIL'),('-(offset*lever)','PASS'),('-(offset+offset)*lever/2','PASS')])
def test_linear_direction(expression,expected):
    analysis=inspect_source(oscillator(expression))
    checks=verify_relationships(analysis)
    assert any(c.name=='restoring acceleration sign' and c.status.value==expected for c in checks)
    assert 'restoring acceleration sign' in relationship_coverage(analysis,checks)['checked']


def test_unestablished_sign_and_alternative_intent():
    analysis=inspect_source(oscillator('lever*offset',gain='input_gain()'))
    checks=verify_relationships(analysis)
    assert 'restoring acceleration sign' in relationship_coverage(analysis,checks)['unverified']
    toy=inspect_source(oscillator('lever*offset',context='repulsive field'))
    assert not any(c.status.value=='FAIL' for c in verify_relationships(toy))


def test_retry_over_loopback_http():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from dataclasses import replace
    approvals=[]
    received=[]
    class Fixture(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def do_POST(self):
            received.append((self.path,json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            if len(received)==1:
                self.send_response(429)
                self.send_header('Retry-After','0')
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"message":{"content":"{\\"commentary\\":\\"ok\\"}"}}')

    with HTTPServer(('127.0.0.1',0),Fixture) as server:
        endpoint=f'http://127.0.0.1:{server.server_port}'
        thread=Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            client=Client(replace(config(retries=1),endpoint=endpoint))
            def approve(preview):
                assert len(received)==len(approvals)
                approvals.append(json.loads(preview))
                return True
            assert client.test_connection(approve).commentary=='ok'
        finally:
            server.shutdown()
            thread.join()
    assert len(approvals)==len(received)==2
    assert approvals==[{'destination':endpoint+path,'payload':body} for path,body in received]


@pytest.mark.parametrize(('check','prediction','expected'),[
    ('Earth surface gravity','WARNING','WARNING'),('parameter plausibility','INCONCLUSIVE','UNKNOWN')])
def test_reference_provenance(tmp_path,check,prediction,expected):
    p=tmp_path/'motion.py'
    p.write_text('# projectile near earth surface in SI units\ng=29.0\nh=.01\nz=w=0.\n'
                 'for i in range(12):\n    w-=g*h\n    z+=w*h\n')
    action={'tool':'parameter','arguments':{'check':check},'prediction':prediction}
    client=Client(config(),lambda *a:{'message':{'content':json.dumps({'commentary':json.dumps(action)})}})
    state=investigate(p,client,lambda p:True,lambda p:True,max_steps=1)
    h=state['hypotheses'][0]
    assert h['assessment']=='supported' and h['category']=='MODEL_HYPOTHESIS'
    call=state['calls'][0]
    assert call['category']=='CONTEXTUAL_REFERENCE' and not call['new_evidence']
    assert call['output'][0]['status']==expected
    assert state['overall']=='INCONCLUSIVE'
