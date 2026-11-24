"""Assignment and update relationships from supported Python AST arithmetic."""

import ast
from dataclasses import dataclass

from .symbolic import Expr, add, dependencies, from_ast, multiply, number, symbol


@dataclass(frozen=True)
class Relationship:
    target: str
    expression: Expr
    dependencies: tuple[str, ...]
    line: int
    kind: str
    provenance: str = 'ast'
    confidence: float = 1.0


def extract_relationships(source: str) -> tuple[Relationship, ...]:
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if (len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List))
                    and isinstance(node.value, (ast.Tuple, ast.List))
                    and len(targets[0].elts) == len(node.value.elts)):
                for target, source in zip(targets[0].elts, node.value.elts):
                    value = from_ast(source)
                    if isinstance(target, ast.Name) and value is not None:
                        found.append(Relationship(target.id, value, dependencies(value), node.lineno, 'assignment'))
                continue
            value = from_ast(node.value)
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    found.append(Relationship(target.id, value, dependencies(value), node.lineno, 'assignment'))
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            value = from_ast(node.value)
            if value is None:
                continue
            current = symbol(node.target.id)
            if isinstance(node.op, ast.Add):
                value = add(current, value)
            elif isinstance(node.op, ast.Sub):
                value = add(current, multiply(number(-1), value))
            elif isinstance(node.op, ast.Mult):
                value = multiply(current, value)
            else:
                continue
            found.append(Relationship(node.target.id, value, dependencies(value), node.lineno, 'update'))
    return tuple(sorted(found, key=lambda item: item.line))
