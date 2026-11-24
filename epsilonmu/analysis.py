"""Static, deliberately conservative Python source inspection."""

import ast
from dataclasses import dataclass
from pathlib import Path

from .context import SourceContext, extract_context
from .roles import Binding, infer_roles
from .relationships import Relationship, DependencyGraph, build_graph, extract_relationships


@dataclass(frozen=True)
class Analysis:
    imports: tuple[str, ...]
    solver: str
    timestep: float | None
    functions: tuple[str, ...]
    domain: str
    classification: 'Classification'
    relationships: tuple[Relationship, ...] = ()
    graph: DependencyGraph = DependencyGraph(())
    context: SourceContext = SourceContext((), ())
    roles: tuple[Binding, ...] = ()
    radial_checks: tuple = ()


@dataclass(frozen=True)
class Classification:
    path: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...] = ()


def _names(node: ast.AST) -> set[str]:
    return {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}


def _negative_product(node: ast.AST) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and
        (_negative_product(node.left) or _negative_product(node.right)))


def classify(tree: ast.AST, names: set[str], solver: str) -> Classification:
    updates = [node for node in ast.walk(tree) if isinstance(node, ast.AugAssign)
               and isinstance(node.target, ast.Name)]
    update_map = {node.target.id: _names(node.value) for node in updates}
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    negative_aliases = {target.id: _names(node.value) for node in assignments
                        if _negative_product(node.value)
                        for target in node.targets if isinstance(target, ast.Name)}
    restoring = []
    for node in [*assignments, *[n for n in ast.walk(tree) if isinstance(n, ast.Return)]]:
        value = node.value
        if value is not None and _negative_product(value) and _names(value):
            restoring.append(node)
    radial = any(isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
                 and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) in {'hypot', 'sqrt'}
                 for node in ast.walk(tree))
    central = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
               and any(isinstance(part, ast.BinOp) and isinstance(part.op, ast.Pow)
                       and isinstance(part.right, ast.Constant) and part.right.value in (2, 3, 4)
                       for part in ast.walk(node.right))]
    inverse_square_vectors = [node for node in central
                              if any(isinstance(part, ast.BinOp) and isinstance(part.op, ast.Pow)
                                     and isinstance(part.right, ast.Constant) and part.right.value == 3
                                     for part in ast.walk(node.right))]
    if radial and len(inverse_square_vectors) >= 2 and len(update_map) >= 2:
        return Classification(('physics', 'mechanics', 'orbital mechanics'), 0.85,
                              ('radial distance expression', 'central inverse-power terms', 'state updates'))
    neg_constant_updates = [node for node in updates if _negative_product(node.value)
                            and any(isinstance(part, ast.Constant) and isinstance(part.value, (int, float))
                                    for part in ast.walk(node.value))]
    linked = [target for target, sources in update_map.items() if sources & (set(update_map) - {target})]
    if neg_constant_updates and len(linked) >= 2 and len(update_map) >= 4:
        return Classification(('physics', 'mechanics', 'projectile motion'), 0.76,
                              ('constant negative velocity update', 'two-axis coupled position updates'))
    coupled_restoring = False
    for update in updates:
        velocity = update.target.id
        origins = _names(update.value) if isinstance(update.op, ast.Sub) or _negative_product(update.value) else set()
        for alias in _names(update.value):
            origins |= negative_aliases.get(alias, set())
        if any(position != velocity and position in update_map and velocity in update_map[position]
               for position in origins):
            coupled_restoring = True
    if coupled_restoring:
        return Classification(('physics', 'mechanics', 'harmonic oscillator'), 0.78,
                              ('negative restoring expression', 'coupled state updates'))
    mechanics_terms = names & {'force', 'velocity', 'acceleration', 'momentum'}
    if restoring or central or (len(mechanics_terms) >= 2 and update_map) or (solver == 'solve_ivp' and mechanics_terms):
        reasons = ('mechanical update or derivative evidence',) if restoring or central else ('solver and mechanics terms',)
        return Classification(('physics', 'mechanics'), 0.52, reasons)
    return Classification((), 0.0, ())


def inspect_source(source: str) -> Analysis:
    tree = ast.parse(source)
    imports, functions, names = set(), set(), set()
    timestep = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in {'dt', 'timestep'} for t in targets):
                try:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, (int, float)):
                        timestep = float(value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    solver = 'solve_ivp' if any(isinstance(c.func, (ast.Name, ast.Attribute)) and
        (c.func.id if isinstance(c.func, ast.Name) else c.func.attr) == 'solve_ivp' for c in calls) else 'custom/unknown'
    words = names | {f.lower() for f in functions}
    relationships = extract_relationships(source)
    graph = build_graph(relationships)
    context = extract_context(source)
    classification = classify(tree, words, solver)
    if 'newtonian_gravity' in context.labels:
        radial = [item.target for item in relationships if item.expression.op == 'pow'
                  and item.expression.value == 1 / 2 and item.expression.args[0].op == 'add']
        force = [item.target for item in relationships if any(radius in item.dependencies for radius in radial)
                 and set(item.dependencies) & {'G', 'gravitational_constant'}]
        acceleration = [item.target for item in relationships if any(f in item.dependencies for f in force)
                        and any(radius in item.dependencies for radius in radial)]
        if any(graph.path(radius, f) and graph.path(f, a)
               for radius in radial for f in force for a in acceleration):
            classification = Classification(('physics', 'mechanics', 'orbital mechanics'), 0.82,
                                            ('declared Newtonian gravity context', 'radius-to-force-to-acceleration dependency'),
                                            ('two-body Newtonian model intended',))
    roles, model, radial_checks = infer_roles(tree, relationships, context.labels)
    if model:
        path = ('physics', 'mechanics') if model == 'mechanics' else ('physics','mechanics',model)
        evidence = ('coupled state updates', 'common timestep in successive derivative updates')
        if model == 'harmonic oscillator':
            evidence += ('position-dependent restoring law or declared oscillator intent',)
        classification = Classification(path, 0.82, evidence)
    domain = classification.path[-1] if classification.path else 'unknown'
    return Analysis(tuple(sorted(imports)), solver, timestep, tuple(sorted(functions)), domain,
                    classification, relationships, graph, context, roles, radial_checks)


def inspect_file(path: str | Path) -> Analysis:
    return inspect_source(Path(path).read_text(encoding='utf-8'))
