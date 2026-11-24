from pathlib import Path

import pytest

from epsilonmu.ai.client import Client
from epsilonmu.ai.configuration import AIError, Config
from epsilonmu.agent import investigate_timestep
from epsilonmu.execution import run_script


def test_runner_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','synthetic')
    monkeypatch.setenv('LAB_OPTION','enabled')
    script=tmp_path/'envcheck.py'
    script.write_text("import os, tempfile, math\nassert 'OPENAI_API_KEY' not in os.environ\nprint(os.getenv('LAB_OPTION'),math.sqrt(4),bool(tempfile.gettempdir()))")
    assert run_script(script).stdout.strip()=='None 2.0 True'
    assert run_script(script,forward_env=('LAB_OPTION',)).stdout.strip()=='enabled 2.0 True'


def test_refinement_children_isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','synthetic')
    p=tmp_path/'simulation.py'
    p.write_text("import os\nassert 'GEMINI_API_KEY' not in os.environ\n"+
                 Path('examples/orbit/bad_timestep.py').read_text())
    assert len(investigate_timestep(p).experiments)==2


def test_rejected_usage_and_unknown():
    c=Config('ollama','synthetic','http://127.0.0.1','none')
    replies=iter([{'prompt_eval_count':8,'eval_count':3,'message':{'content':'bad'}},
                  {'message':{'content':'{"commentary":"ok"}'}}])
    client=Client(c,lambda *args:next(replies))
    with pytest.raises(AIError):
        client.test_connection(lambda p:True)
    assert (client.reported_input_tokens,client.reported_output_tokens)==(8,3)
    assert client.test_connection(lambda p:True).input_tokens is None
    b=client.budget
    assert (b.attempts,b.successful,b.rejected,b.unknown_usage)==(2,1,1,1)


@pytest.mark.parametrize('limits',[{'max_attempts':1},{'max_tokens':380},
                                    {'rates':(1,1),'max_cost':'0.000380'}])
def test_shared_budget_retains_unknown_reservation(limits):
    from epsilonmu.ai.budget import Budget
    b=Budget(**limits)
    c=Config('ollama','synthetic','http://127.0.0.1','none')
    def reply(*args):
        return {'message':{'content':'{"commentary":"ok"}'}}
    Client(c,reply,budget=b).test_connection(lambda _:True)
    with pytest.raises(AIError,match='budget exhausted'):
        Client(c,reply,budget=b).test_connection(lambda _:True)
    assert b.attempts==1 and b.unknown_usage==1


def test_shared_deadline_rejects_late_results():
    from epsilonmu.ai.budget import Budget
    tick=[0.]
    b=Budget(deadline_seconds=2,clock=lambda:tick[0])
    def late(url,headers,body,timeout):
        assert timeout==2
        tick[0]=3
        return {'prompt_eval_count':7,'message':{'content':'{"commentary":"ok"}'}}
    client=Client(Config('ollama','m','http://127.0.0.1','none'),late,budget=b)
    for _ in range(2):
        with pytest.raises(AIError,match='deadline'):
            client.test_connection(lambda _:True)
    assert (b.attempts,b.rejected,b.unknown_usage,b.reported_input_tokens)==(1,1,1,7)


def test_retry_records_each_attempt_once():
    from io import BytesIO
    from urllib.error import HTTPError
    from epsilonmu.ai.budget import Budget
    b=Budget(max_attempts=2)
    def reply(*args):
        if b.attempts==1:
            raise HTTPError('url',429,'',{'Retry-After':'0'},BytesIO(b'{"eval_count":2}'))
        return {'prompt_eval_count':5,'eval_count':3,'message':{'content':'{"commentary":"ok"}'}}
    Client(Config('ollama','m','http://127.0.0.1','none',retries=1),reply,budget=b).test_connection(lambda _:True)
    assert (b.attempts,b.failed,b.successful,b.unknown_usage)==(2,1,1,1)
    assert (b.reported_input_tokens,b.reported_output_tokens)==(5,5)


def test_provider_specific_usage_categories():
    from epsilonmu.ai.providers import reported_usage
    assert reported_usage('anthropic',{'usage':{'input_tokens':2,'cache_creation_input_tokens':4,
        'cache_read_input_tokens':7,'output_tokens':1}})==(13,1)
    assert reported_usage('gemini',{'usageMetadata':{'promptTokenCount':2,'thoughtsTokenCount':4,
        'candidatesTokenCount':1,'totalTokenCount':7}})==(2,5)


def test_cli_offline_without_provider(monkeypatch,tmp_path):
    import socket
    from epsilonmu.cli import main
    monkeypatch.setenv('XDG_CONFIG_HOME',str(tmp_path))
    monkeypatch.setattr(socket.socket,'connect',lambda *args:pytest.fail('network access'))
    for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','EPSILONMU_API_KEY','GEMINI_API_KEY'):
        monkeypatch.delenv(key,raising=False)
    for command in ('inspect','verify','debug'):
        assert main([command,'examples/orbit/correct.py'])==0
