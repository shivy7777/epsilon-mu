import json
from dataclasses import replace
from urllib.error import HTTPError

import pytest

from epsilonmu.ai.client import Client, Evidence
from epsilonmu.ai.configuration import Config, AIError, DEFAULTS, load, path
from epsilonmu.cli import main


def config(provider='ollama',**changes):
    url,_=DEFAULTS[provider]
    c=Config(provider,'model',url or 'https://example.org/v1',
             'none' if provider=='ollama' else 'env',provider=='ollama')
    return replace(c,**changes)


def response(provider):
    text='{"commentary":"uncertain"}'
    if provider=='ollama':
        return {'message':{'content':text},'prompt_eval_count':4,'eval_count':2}
    if provider=='anthropic':
        return {'content':[{'type':'text','text':text}],'usage':{'input_tokens':4,'output_tokens':2}}
    if provider=='gemini':
        return {'candidates':[{'content':{'parts':[{'text':text}]}}],
                'usageMetadata':{'promptTokenCount':4,'candidatesTokenCount':2}}
    return {'choices':[{'message':{'content':text}}],'usage':{'prompt_tokens':4,'completion_tokens':2}}


@pytest.mark.parametrize('provider',DEFAULTS)
def test_providers_preview_and_usage(monkeypatch,provider):
    c=config(provider,share=('conclusions',))
    if c.auth=='env':
        monkeypatch.setenv(DEFAULTS[provider][1],'private-credential')
    sent=[]
    def transport(url,headers,body,timeout):
        sent.append(body)
        return response(provider)
    previews=[]
    client=Client(c,transport)
    result=client.interpret(Evidence(conclusions=('INCONCLUSIVE token=hidden',)),
                            lambda p: previews.append(p) is None)
    assert (client.reported_input_tokens,client.reported_output_tokens)==(4,2)
    assert json.loads(previews[0])['payload']==sent[0]
    assert all(x not in previews[0] for x in ('hidden','private-credential'))
    assert result.commentary=='uncertain' and (result.input_tokens,result.output_tokens)==(4,2)
    assert float(result.estimated_cost(1,3))==0.000010


def test_consent_and_policy_fail_closed():
    client=Client(config(),lambda *a: pytest.fail('unexpected transmission'))
    with pytest.raises(AIError,match='prohibits'):
        client.interpret(Evidence(measurements=('sensitive',)),lambda p:True)
    with pytest.raises(AIError,match='declined'):
        client.test_connection(lambda p:False)
    assert client.attempts==0


@pytest.mark.parametrize('changes',[{'endpoint':'https://example.org'},{'model':''},
    {'endpoint':'http://name:password@127.0.0.1'},{'requests':0},{'context':8},
    {'model':'example:cloud'},{'timeout':0}])
def test_invalid_settings(changes):
    with pytest.raises(AIError):
        config(**changes).validate()


def test_no_credential_leak_on_failure():
    def fail(*args):
        raise HTTPError('secret',401,'private detail',{},None)
    with pytest.raises(AIError,match='^Provider authentication failed.$'):
        Client(config(),fail).test_connection(lambda p:True)


def test_timeout_is_sanitized():
    def fail(*args):
        raise TimeoutError('secret')
    with pytest.raises(AIError,match='^Provider connection failed or timed out.$'):
        Client(config(),fail).test_connection(lambda p:True)


def test_rate_limit_retries_count_against_budget():
    def fail(*args):
        raise HTTPError('url',429,'',{'Retry-After':'0'},None)
    client=Client(config(retries=2,requests=1),fail)
    with pytest.raises(AIError,match='budget exhausted'):
        client.test_connection(lambda p:True)
    assert client.attempts==1


def test_reject_provider_verdict():
    def transport(*args):
        return {'message':{'content':'{"commentary":"ok","status":"VERIFIED"}'}}
    with pytest.raises(AIError,match='Invalid structured'):
        Client(config(),transport).test_connection(lambda p:True)


def test_input_budget_precedes_consent():
    with pytest.raises(AIError,match='input exceeds'):
        Client(config(max_input=1)).test_connection(lambda p:pytest.fail('oversize'))


def test_configuration_cli(monkeypatch,tmp_path,capsys):
    monkeypatch.setenv('XDG_CONFIG_HOME',str(tmp_path))
    answers=iter(['openai','test-model','','env','no',''])
    monkeypatch.setattr('builtins.input',lambda *a:next(answers))
    monkeypatch.setenv('OPENAI_API_KEY','never-persist-this')
    assert main(['configure'])==0
    assert load().provider=='openai'
    assert 'never-persist-this' not in path().read_text()
    assert main(['config','show'])==0
    assert 'never-persist-this' not in capsys.readouterr().out
    assert main(['config','reset'])==0 and not path().exists()
    assert main(['providers','list'])==0


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(AIError,match='credential unavailable'):
        Client(config('openai')).test_connection(lambda p:True)


@pytest.mark.parametrize('think',[False,True,None])
def test_ollama_thinking_request(think):
    from epsilonmu.ai.providers import prepare
    c=config() if think is False else config(ollama_think=think)
    body=prepare(c.validate(),'synthetic','')[2]
    if think is None:
        assert 'think' not in body
    else:
        assert body['think'] is think
    assert 'think' not in body['options']


@pytest.mark.parametrize(('content','thinking','error'),[
    ('{"commentary":"ready"}',None,None),
    ('{"commentary":"ready"}','synthetic trace',None),
    ('','{"commentary":"ready"}','output limit'),
    ('not json','{"commentary":"ready"}','Invalid structured'),
    ('{"answer":"ready"}',None,'Invalid structured'),
])
def test_ollama_final_answer(content,thinking,error):
    from epsilonmu.ai.providers import normalize
    message={'content':content}
    if thinking is not None:
        message['thinking']=thinking
    envelope={'message':message,'done_reason':'length' if not content else 'stop'}
    if error:
        with pytest.raises(AIError,match=error):
            normalize('ollama',envelope)
    else:
        assert normalize('ollama',envelope).commentary=='ready'


def test_native_storage_and_plaintext_backend_refusal(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    from epsilonmu.ai.credentials import credential
    from epsilonmu.ai.configuration import save
    monkeypatch.setenv('XDG_CONFIG_HOME',str(tmp_path))
    values={}
    class Native:
        __module__='keyring.backends.Windows'
        def set_password(self,service,account,value):
            values[service,account]=value
        def get_password(self,service,account):
            return values.get((service,account))
        def delete_password(self,service,account):
            del values[service,account]
    backend=Native()
    monkeypatch.setitem(sys.modules,'keyring',SimpleNamespace(get_keyring=lambda:backend))
    c=config('openai',auth='keyring')
    credential(c,'write','private-test-key')
    save(c)
    assert credential(load())=='private-test-key' and 'private-test-key' not in path().read_text()
    assert main(['config','reset'])==0 and not values
    backend=object()
    with pytest.raises(AIError,match='native OS'):
        credential(c,'write','private-test-key')
