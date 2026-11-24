"""Candidate physical roles from paired discrete derivative updates in one scope."""
import ast
from collections import defaultdict
from dataclasses import dataclass

from .relationships.symbolic import dependencies, equivalent, multiply, power, symbol, symbol_power


@dataclass(frozen=True)
class Binding:
    variable: str
    role: str
    evidence: str


def infer_roles(tree, relationships, labels):
    roles = set()
    models = set()
    radial_checks = []
    def bind(name, role, reason):
        roles.add(Binding(name, role, reason))

    unit_roles = {'kg': ('mass',), 's': ('time','timestep'), 'm': ('position','radius'),
                  'm/s': ('velocity',), 'm/s^2': ('acceleration',), 'N': ('force',),
                  'J': ('energy',), 'kg*m/s': ('momentum',)}
    for node in ast.walk(tree):
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and isinstance(node.annotation,ast.Constant):
            for role in unit_roles.get(node.annotation.value, ()):
                bind(node.target.id, role, 'declared SI unit annotation')

    # Isolate straight-line integration bodies. Never connect two functions' locals.
    bodies = [node.body for node in ast.walk(tree)
              if isinstance(node, (ast.For, ast.While, ast.FunctionDef))]
    for body in bodies:
        lines = {stmt.lineno for stmt in body}
        rows = [r for r in relationships if r.line in lines]
        increments = {}
        definitions = {}
        central_components = defaultdict(list)
        for row in rows:
            expr = row.expression
            terms = list(expr.args) if expr.op == 'add' else []
            if symbol(row.target) in terms:
                terms.remove(symbol(row.target))
                if len(terms) == 1:
                    increments[row.target] = terms[0]
            else:
                definitions[row.target] = row
        for pos, delta in increments.items():
            if delta.op != 'mul' or len(delta.args) != 2 or any(a.op != 'symbol' for a in delta.args):
                continue
            for rate, step in (delta.args, delta.args[::-1]):
                vel, h = rate.value, step.value
                if vel not in increments or h in increments or h in definitions:
                    continue
                if symbol_power(increments[vel], h) != 1:
                    continue
                bind(pos,'position','integral of a changing rate')
                bind(vel,'velocity','shared step in two successive integrals')
                bind(h,'timestep','common integration multiplier')
                for clock, tick in increments.items():
                    if tick == step:
                        bind(clock,'time','increments by one timestep')
                acc = multiply(increments[vel], power(step,-1))
                if acc.op == 'symbol':
                    bind(acc.value,'acceleration','velocity increment divided by step')
                def expand(expr, seen=frozenset()):
                    if expr.op == 'symbol' and expr.value in definitions and expr.value not in seen:
                        row = definitions[expr.value]
                        if row.target in radii:
                            return expr
                        return expand(row.expression, seen | {row.target})
                    if expr.op == 'mul':
                        return multiply(*(expand(a,seen) for a in expr.args))
                    if expr.op == 'pow':
                        return power(expand(expr.args[0],seen),expr.value)
                    return expr
                radii = {name for name,r in definitions.items() if r.expression.op == 'pow'
                         and r.expression.value == 0.5 and r.expression.args[0].op == 'add'
                         and all(a.op == 'pow' and a.value == 2 for a in r.expression.args[0].args)
                         and pos in r.dependencies}
                observed = expand(acc)
                deps = set(dependencies(observed))
                model = 'mechanics'
                for radius in radii & deps:
                    if symbol_power(observed,pos) != 1:
                        continue
                    bind(radius,'radius','Euclidean norm of position components')
                    exponent = symbol_power(observed,radius)
                    model = 'central-force system'
                    if 'newtonian_gravity' in labels or exponent == -3 and str(observed).startswith('-'):
                        model = 'orbital mechanics'
                    line = (definitions[acc.value].line if acc.op=='symbol' and acc.value in definitions
                            else next(r.line for r in rows if r.target==vel))
                    central_components[radius].append((pos, multiply(observed,power(symbol(pos),-1)),
                                                       model, (vel,line,exponent)))
                    break
                else:
                    if pos in deps and (('harmonic_oscillator' in labels) or
                            (symbol_power(observed,pos)==1 and str(observed).startswith('-') and
                             not ((deps & increments.keys())-{pos}))):
                        model = 'harmonic oscillator'
                    elif not deps & (increments.keys() | definitions.keys()):
                        axes = [other for other, change in increments.items() if other != pos and
                                change.op == 'mul' and step in change.args and
                                any(a.op=='symbol' and a != step for a in change.args)]
                        model = 'projectile motion' if any(a!=vel for a in axes) or 'projectile' in labels else 'constant-acceleration motion'
                if model not in {'central-force system','orbital mechanics'}:
                    models.add(model)
                if acc.op=='symbol' and acc.value in definitions:
                    expr = definitions[acc.value].expression
                    for name in dependencies(expr):
                        if symbol_power(expr,name)==-1:
                            bind(name,'mass','candidate divisor converting force to acceleration')
                        elif symbol_power(expr,name)==1 and name in definitions:
                            bind(name,'force','candidate numerator of acceleration')
                for name in deps - increments.keys() - radii:
                    bind(name,'physical coefficient','factor in the acceleration law')
                for row in rows:
                    if vel in row.dependencies and row.target not in increments:
                        if symbol_power(row.expression,vel)==2:
                            bind(row.target,'energy','quadratic velocity diagnostic')
                        elif symbol_power(row.expression,vel)==1:
                            bind(row.target,'momentum','linear velocity diagnostic')
        for radius, components in central_components.items():
            positions = {entry[0] for entry in components}
            common_factor = components[0][1]
            if positions != set(definitions[radius].dependencies) or not all(
                    equivalent(entry[1], common_factor) for entry in components):
                models.add('mechanics')
                continue
            models.add(components[0][2])
            if 'newtonian_gravity' in labels:
                radial_checks.extend(entry[3] for entry in components if entry[3][2] is not None)
    model = next(iter(models)) if len(models)==1 else 'mechanics' if models else None
    return tuple(sorted(roles,key=lambda b:(b.variable,b.role))), model, tuple(radial_checks)
