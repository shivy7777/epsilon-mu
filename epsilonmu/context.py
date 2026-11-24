"""Textual source hints are fallible evidence, never physical assertions."""

import ast
import io
import tokenize
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceContext:
    labels: tuple[str, ...]
    evidence: tuple[str, ...]


def extract_context(source: str) -> SourceContext:
    tree = ast.parse(source)
    fragments = []
    docstring = ast.get_docstring(tree)
    if docstring:
        fragments.append(('module docstring', docstring.lower()))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fragments.append(('function name', node.name.lower().replace('_', ' ')))
            value = ast.get_docstring(node)
            if value:
                fragments.append(('function docstring', value.lower()))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 500:
            fragments.append(('string literal', node.value.lower()))
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                fragments.append(('comment', token.string.lower()))
    except tokenize.TokenError:
        pass
    labels, evidence = set(), []
    for origin, text in fragments:
        if 'newtonian' in text and ('gravit' in text or 'mechanic' in text or 'two-body' in text):
            label = 'newtonian_gravity' if ('gravit' in text or 'two-body' in text) else 'newtonian_mechanics'
            labels.add(label)
            evidence.append(f'{origin}: {label}')
        if 'two-body orbit' in text:
            labels.add('orbital_context')
            evidence.append(f'{origin}: orbital_context')
        if 'earth' in text and ('surface' in text or 'near earth' in text):
            labels.add('earth_surface')
            evidence.append(f'{origin}: earth_surface')
        if 'si units' in text or 'international system of units' in text:
            labels.add('si_units')
            evidence.append(f'{origin}: si_units')
        if 'harmonic oscillator' in text:
            labels.add('harmonic_oscillator')
            evidence.append(f'{origin}: harmonic_oscillator')
        if 'projectile' in text:
            labels.add('projectile')
            evidence.append(f'{origin}: projectile')
    return SourceContext(tuple(sorted(labels)), tuple(dict.fromkeys(evidence)))
