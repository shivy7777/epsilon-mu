"""Explicitly consented, bounded requests. All provider text is advisory."""
import json
import re
import time
from dataclasses import dataclass
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .configuration import AIError, DEFAULTS
from .credentials import credential
from .providers import normalize, prepare, reported_usage
from .budget import Budget


@dataclass(frozen=True)
class Evidence:
    measurements: tuple[str,...] = ()
    hypotheses: tuple[str,...] = ()
    conclusions: tuple[str,...] = ()


class RejectedResponse(AIError):
    """An HTTP response arrived but its body could not be accepted."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args):
        return None


def send(url,headers,body,timeout):
    request=Request(url,data=json.dumps(body).encode(),headers=headers,method='POST')
    with build_opener(ProxyHandler({}),NoRedirect()).open(request,timeout=timeout) as response:
        raw=response.read(1048577)
    if len(raw)>1048576:
        raise RejectedResponse('Provider response exceeded the size limit.')
    try:
        return json.loads(raw)
    except (ValueError,RecursionError):
        raise RejectedResponse('Invalid provider response.') from None


class Client:
    def __init__(self,config,transport=send,*,budget=None):
        self.config=config.validate()
        self.transport=transport
        self.attempts=0
        self.budget=budget if budget is not None else Budget(max_attempts=config.requests)
        self.lock=Lock()

    @property
    def reported_input_tokens(self):
        return self.budget.reported_input_tokens

    @property
    def reported_output_tokens(self):
        return self.budget.reported_output_tokens

    def interpret(self,evidence,approve):
        return self._request(evidence,approve,False)

    def test_connection(self,approve):
        return self._request(Evidence(),approve,True)

    def _request(self,evidence,approve,probe):
        import os
        c=self.config
        key=credential(c)
        content={}
        for kind in ('measurements','hypotheses','conclusions'):
            items=getattr(evidence,kind)
            if items and kind not in c.share:
                raise AIError('Configuration prohibits sharing this category.')
            if not isinstance(items,tuple) or any(not isinstance(v,str) for v in items):
                raise AIError('Expected tuples of evidence strings.')
            if items:
                content[kind]=list(items)
        prompt='Return only a JSON object with one string field named commentary. '
        prompt+=('Respond with ready.' if probe else 'Provide advisory interpretation. Preserve uncertainty. '+json.dumps(content))
        for secret in [key,*[os.getenv(name) for _,name in DEFAULTS.values() if name]]:
            if secret:
                prompt=prompt.replace(secret,'[REDACTED]')
        prompt=re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|password|token)\s*[:=]\s*)[^\s",}]+',r'\1[REDACTED]',prompt)
        if len(prompt.encode())+16>c.max_input:
            raise AIError('Estimated input exceeds the configured token limit.')
        url,headers,body=prepare(c,prompt,key)
        preview=json.dumps({'destination':url,'payload':body},ensure_ascii=True,indent=2)
        if approve(preview) is not True:
            raise AIError('Request declined.')
        for attempt in range(c.retries+1):
            with self.lock:
                if self.attempts>=c.requests:
                    raise AIError('Client request budget exhausted.')
                ticket=self.budget.reserve(len(prompt.encode())+16,c.max_output)
                self.attempts+=1
            usage=(None,None)
            outcome='failed'
            try:
                data=self.transport(url,headers,body,min(c.timeout,self.budget.remaining()))
                usage=reported_usage(c.provider,data)
                outcome='rejected'
                result=normalize(c.provider,data)
                self.budget.remaining()
                outcome='successful'
                return result
            except HTTPError as exc:
                code=exc.code
                try:
                    if exc.fp is not None:
                        raw=exc.read(1048577)
                        if raw and len(raw)<=1048576:
                            usage=reported_usage(c.provider,json.loads(raw))
                except (ValueError,OSError):
                    pass
                finally:
                    exc.close()
                if code in (401,403):
                    raise AIError('Provider authentication failed.') from None
                if code!=429 or attempt==c.retries:
                    raise AIError(f'Provider returned HTTP {code}.') from None
                delay=exc.headers.get('Retry-After','1') if exc.headers else '1'
                if not delay.isdigit() or int(delay)>5:
                    raise AIError('Rate limited. Retry later.') from None
                if int(delay)>=self.budget.remaining():
                    raise AIError('Retry would exceed investigation deadline.') from None
                time.sleep(int(delay))
            except (URLError,OSError):
                raise AIError('Provider connection failed or timed out.') from None
            except RejectedResponse:
                outcome='rejected'
                raise
            except AIError:
                raise
            except (ValueError,TypeError):
                raise AIError('Invalid provider response.') from None
            finally:
                self.budget.settle(ticket,usage,outcome)
