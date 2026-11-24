"""Interactive AI commands. Verification commands never construct a client."""
import json
from dataclasses import asdict
from getpass import getpass

from .client import Client
from .configuration import AIError, Config, DEFAULTS, load, path, save
from .credentials import credential


def register(sub):
    sub.add_parser('configure')
    for command,choices in [('config',('show','reset')),('providers',('list','test'))]:
        parser=sub.add_parser(command)
        parser.add_argument('action',choices=choices)


def ask(label,default):
    return input(f'{label} [{default}]: ').strip() or default


def configure():
    provider=ask('Provider ('+', '.join(DEFAULTS)+')','ollama')
    if provider not in DEFAULTS:
        raise AIError('Unknown provider.')
    endpoint,env=DEFAULTS[provider]
    model=input('Model name: ').strip()
    endpoint=ask('API base URL',endpoint)
    auth=ask('Authentication (env, keyring, none)','none' if provider=='ollama' else 'env')
    local_only=ask('Restrict all requests to loopback? (yes/no)','yes')
    if local_only not in ('yes','no'):
        raise AIError('Choose yes or no.')
    categories=input('Allowed evidence categories (measurements,hypotheses,conclusions; blank for none): ')
    config=Config(provider,model,endpoint,auth,local_only=='yes',
                  tuple(k.strip() for k in categories.split(',') if k.strip())).validate()
    if auth=='keyring':
        secret=getpass('API key for native OS credential storage: ')
        if not secret:
            raise AIError('Empty credential.')
        credential(config,'write',secret)
    save(config)
    print(f'Saved configuration to {path()}')
    if auth=='env':
        print(f'Set {env} in your environment.')


def dispatch(args):
    try:
        if args.command=='configure':
            configure()
        elif args.command=='providers' and args.action=='list':
            print('\n'.join(DEFAULTS))
        elif args.command=='config' and args.action=='show':
            config=load()
            print(json.dumps(asdict(config),indent=2))
        elif args.command=='config':
            if path().exists():
                config=load()
                if config.auth=='keyring':
                    credential(config,'delete')
                path().unlink()
            print('Configuration reset.')
        else:
            def approve(preview):
                print(preview)
                return input('Send this network request? Cloud usage may incur charges [y/N]: ').lower()=='y'
            result=Client(load()).test_connection(approve)
            print(f'Connection successful. Reported tokens: input={result.input_tokens}, output={result.output_tokens}')
        return 0
    except (AIError,OSError,EOFError,ValueError):
        # Only controlled AIError messages may surface provider-related details.
        import sys
        exc=sys.exc_info()[1]
        print(str(exc) if isinstance(exc,AIError) else 'Configuration or input error.',file=sys.stderr)
        return 2
