"""Child-process tracer for a copied standalone script. Internal entry point."""

import json
import runpy
import sys
from pathlib import Path

PREFIX = 'EPSILONMU_INSTRUMENTED:'
target = str(Path(sys.argv[1]).resolve())
records: dict[tuple[str, str], dict] = {}
events = 0


def numbers(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if type(value).__module__.startswith('numpy') and hasattr(value, 'tolist'):
        try:
            value = value.tolist()
        except (ValueError, TypeError):
            return None
    if isinstance(value, (list, tuple)) and len(value) <= 4096:
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
            return [float(x) for x in value]
    return None


def capture(mapping, scope, line, final=False):
    if len(records) >= 128:
        return
    for name, value in list(mapping.items()):
        if name.startswith('__') or len(records) >= 128:
            continue
        if not final and not isinstance(value, (int, float)):
            continue
        sequence = numbers(value)
        if sequence is None or not sequence:
            continue
        key = (scope, name)
        record = records.setdefault(key, {'name': name, 'scope': scope, 'line': line, 'values': []})
        current = record['values']
        if len(sequence) == 1:
            if (not current or current[-1] != sequence[0]) and len(current) < 512:
                current.append(sequence[0])
        elif final:
            record['values'] = sequence


def trace(frame, event, arg):
    global events
    if frame.f_code.co_filename != target:
        return None
    events += 1
    if events > 100000:
        return None
    if event == 'line':
        capture(frame.f_locals, frame.f_code.co_name, frame.f_lineno)
    elif event == 'return':
        capture(frame.f_locals, frame.f_code.co_name, frame.f_lineno, final=True)
    return trace


sys.settrace(trace)
try:
    namespace = runpy.run_path(target, run_name='__main__')
    capture(namespace, '<module>', 0, final=True)
finally:
    sys.settrace(None)
    print(PREFIX + json.dumps(list(records.values())))
