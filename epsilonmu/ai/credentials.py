import os
from .configuration import AIError, DEFAULTS


def credential(c, action='read', value=None):
    if c.auth=='none':
        return ''
    if c.auth=='env':
        key=os.getenv(DEFAULTS[c.provider][1])
    else:
        account=c.provider+'|'+c.endpoint
        try:
            import keyring
            backend=keyring.get_keyring()
            if type(backend).__module__ not in {'keyring.backends.Windows','keyring.backends.macOS','keyring.backends.SecretService'}:
                raise AIError('Select a native OS keyring or use environment authentication.')
            if action=='write':
                backend.set_password('epsilonmu',account,value)
                return
            key=backend.get_password('epsilonmu',account)
            if action=='delete':
                if key:
                    backend.delete_password('epsilonmu',account)
                return
        except AIError:
            raise
        except Exception:
            raise AIError('Native credential store unavailable. Install epsilonmu[credentials] or use environment authentication.') from None
    if not key:
        raise AIError('Provider credential unavailable.')
    return key
