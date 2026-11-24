"""Wire adapters for documented JSON APIs; no provider SDK required."""
from dataclasses import dataclass
from urllib.parse import quote
from .configuration import AIError


@dataclass(frozen=True)
class Interpretation:
    commentary: str
    input_tokens: int | None
    output_tokens: int | None

    def estimated_cost(self, input_per_million, output_per_million):
        """Use caller-verified model rates in the same currency, excluding discounts."""
        from decimal import Decimal
        if None in (self.input_tokens,self.output_tokens,input_per_million,output_per_million):
            return None
        rates=[Decimal(str(v)) for v in (input_per_million,output_per_million)]
        if any(not r.is_finite() or r<0 for r in rates):
            raise AIError('Invalid token price.')
        return (self.input_tokens*rates[0]+self.output_tokens*rates[1])/1_000_000


def prepare(c, prompt, key):
    headers={'Content-Type':'application/json'}
    messages=[{'role':'user','content':prompt}]
    if c.provider=='anthropic':
        headers.update({'x-api-key':key,'anthropic-version':'2023-06-01'})
        route='/messages'
        body=dict(model=c.model,messages=messages,max_tokens=c.max_output)
    elif c.provider=='gemini':
        headers['x-goog-api-key']=key
        route='/models/'+quote(c.model,safe='')+':generateContent'
        body={'contents':[{'parts':[{'text':prompt}]}],
              'generationConfig':{'maxOutputTokens':c.max_output,'responseMimeType':'application/json'}}
    elif c.provider=='ollama':
        route='/api/chat'
        body=dict(model=c.model,messages=messages,stream=False,format='json',
                  options=dict(num_ctx=c.context,num_predict=c.max_output))
    else:
        route='/chat/completions'
        if key:
            headers['Authorization']='Bearer '+key
        body=dict(model=c.model,messages=messages,response_format={'type':'json_object'})
        body['max_completion_tokens' if c.provider=='openai' else 'max_tokens']=c.max_output
    return c.endpoint.rstrip('/')+route,headers,body


def reported_usage(provider,data):
    """Recover each valid directional count even from an otherwise invalid envelope."""
    if not isinstance(data,dict):
        return None,None
    field,keys={
        'anthropic':('usage',('input_tokens','output_tokens')),
        'gemini':('usageMetadata',('promptTokenCount','candidatesTokenCount')),
        'ollama':(None,('prompt_eval_count','eval_count')),
    }.get(provider,('usage',('prompt_tokens','completion_tokens')))
    counts=data if field is None else data.get(field)
    if not isinstance(counts,dict):
        return None,None
    inputs,outputs=(n if type(n) is int and n>=0 else None for n in map(counts.get,keys))
    if provider=='anthropic' and inputs is not None:
        for name in ('cache_creation_input_tokens','cache_read_input_tokens'):
            n=counts.get(name,0)
            if type(n)!=int or n<0:
                inputs=None
                break
            inputs+=n
    if provider=='gemini':
        for name,direction in [('toolUsePromptTokenCount','input'),('thoughtsTokenCount','output')]:
            n=counts.get(name,0)
            if direction=='input':
                inputs=inputs+n if inputs is not None and type(n)==int and n>=0 else None
            else:
                outputs=outputs+n if outputs is not None and type(n)==int and n>=0 else None
    return inputs,outputs


def normalize(provider,data):
    import json
    try:
        if provider=='anthropic':
            text=''.join(x['text'] for x in data['content'] if x['type']=='text')
            counts=data.get('usage',{})
            keys=('input_tokens','output_tokens')
        elif provider=='gemini':
            text=''.join(x['text'] for x in data['candidates'][0]['content']['parts'] if 'text' in x)
            counts=data.get('usageMetadata',{})
            keys=('promptTokenCount','candidatesTokenCount')
        elif provider=='ollama':
            text=data['message']['content']
            counts=data
            keys=('prompt_eval_count','eval_count')
        else:
            text=data['choices'][0]['message']['content']
            counts=data.get('usage') or {}
            keys=('prompt_tokens','completion_tokens')
        value=json.loads(text)
        if not isinstance(value,dict) or set(value)!={'commentary'} or not isinstance(value['commentary'],str):
            raise ValueError
        usage=[counts.get(k) for k in keys]
        if any(n is not None and (type(n)!=int or n<0) for n in usage):
            raise ValueError
        return Interpretation(value['commentary'],*reported_usage(provider,data))
    except (TypeError,ValueError,KeyError,IndexError,AttributeError):
        raise AIError('Invalid structured provider response.') from None
