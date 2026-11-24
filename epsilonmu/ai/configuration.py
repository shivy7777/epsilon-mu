import json
import os
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urlsplit

DEFAULTS = {
    'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'anthropic': ('https://api.anthropic.com/v1', 'ANTHROPIC_API_KEY'),
    'gemini': ('https://generativelanguage.googleapis.com/v1beta', 'GEMINI_API_KEY'),
    'ollama': ('http://127.0.0.1:11434', ''),
    'compatible': ('', 'EPSILONMU_API_KEY'),
}

class AIError(ValueError):
    """User-facing error without sensitive request or response details."""


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    endpoint: str
    auth: str = 'env'
    local_only: bool = True
    share: tuple = ()
    max_input: int = 4096
    max_output: int = 256
    context: int = 8192
    requests: int = 4
    timeout: int = 20
    retries: int = 0

    @property
    def local(self):
        return urlsplit(self.endpoint).hostname in ('127.0.0.1','::1')

    def validate(self):
        if self.provider not in DEFAULTS or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,180}', self.model) or '..' in self.model:
            raise AIError('Invalid provider or model.')
        if not re.fullmatch(r'https?://[A-Za-z0-9.\[\]:-]+(?:/[A-Za-z0-9._/-]*)?', self.endpoint):
            raise AIError('Use an endpoint without credentials, queries or fragments.')
        u=urlsplit(self.endpoint)
        if not u.hostname or (not self.local and (u.scheme!='https' or self.local_only)):
            raise AIError('Remote endpoints require HTTPS and explicit remote configuration.')
        if self.provider in ('openai','anthropic','gemini') and self.endpoint.rstrip('/')!=DEFAULTS[self.provider][0]:
            raise AIError('Use the official endpoint for this provider.')
        if self.provider=='ollama' and (not self.local or self.model.endswith(('-cloud',':cloud')) or self.auth!='none'):
            raise AIError('Ollama requires a local model and a loopback endpoint.')
        if self.auth not in ('env','keyring','none') or (self.auth=='none' and self.provider not in ('ollama','compatible')):
            raise AIError('Invalid authentication method.')
        if type(self.local_only)!=bool or not isinstance(self.share,(tuple,list)) or set(self.share)-{'measurements','hypotheses','conclusions'}:
            raise AIError('Invalid sharing policy.')
        for value,limit in ((self.max_input,1000000),(self.max_output,100000),(self.context,2000000),(self.requests,1000),(self.timeout,120)):
            if type(value)!=int or not 1<=value<=limit:
                raise AIError('Invalid resource limit.')
        if self.max_input+self.max_output>self.context or type(self.retries)!=int or not 0<=self.retries<=3:
            raise AIError('Invalid context or retry limit.')
        return self


def path():
    return Path(os.getenv('XDG_CONFIG_HOME') or os.getenv('APPDATA') or Path.home()/'.config')/'epsilonmu'/'ai.json'


def load():
    try:
        return Config(**json.loads(path().read_text(encoding='utf8'))).validate()
    except (OSError,TypeError,ValueError):
        raise AIError('Missing or invalid AI configuration. Run epsilonmu configure.') from None


def save(config):
    config.validate()
    dest=path()
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent,mode='w',encoding='utf8',delete=False) as f:
        json.dump(asdict(config),f,indent=2)
    try:
        os.replace(f.name,dest)
    finally:
        Path(f.name).unlink(missing_ok=True)
