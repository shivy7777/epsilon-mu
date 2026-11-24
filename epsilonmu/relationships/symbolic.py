"""Inert symbolic subset of Python arithmetic; never evaluates source expressions."""

import ast
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple['Expr', ...] = ()
    value: str | Fraction | None = None

    def __str__(self) -> str:
        if self.op == 'symbol':
            return str(self.value)
        if self.op == 'number':
            return str(self.value)
        if self.op == 'add':
            return '(' + ' + '.join(map(str, self.args)) + ')'
        if self.op == 'mul':
            return '*'.join(map(str, self.args))
        if self.op == 'pow':
            return f'{self.args[0]}^{self.value}'
        return f'{self.op}(' + ', '.join(map(str, self.args)) + ')'


def symbol(name: str) -> Expr:
    return Expr('symbol', value=name)


def number(value: int | float | Fraction) -> Expr:
    return Expr('number', value=Fraction(str(value)))


def power(base: Expr, exponent: Fraction | int) -> Expr:
    exponent = Fraction(exponent)
    if exponent == 0:
        return number(1)
    if exponent == 1:
        return base
    if base.op == 'number':
        if exponent.denominator == 1:
            return number(base.value ** exponent.numerator)
    if base.op == 'pow' and exponent.denominator == 1:
        return power(base.args[0], base.value * exponent)
    return Expr('pow', (base,), exponent)


def multiply(*items: Expr) -> Expr:
    coefficient = Fraction(1)
    powers: dict[Expr, Fraction] = {}

    def add(item: Expr, exponent: Fraction = Fraction(1)):
        nonlocal coefficient
        if item.op == 'number':
            if exponent.denominator != 1:
                powers[item] = powers.get(item, Fraction(0)) + exponent
            else:
                coefficient *= item.value ** exponent.numerator
        elif item.op == 'mul':
            for part in item.args:
                add(part, exponent)
        elif item.op == 'pow':
            add(item.args[0], exponent * item.value)
        else:
            powers[item] = powers.get(item, Fraction(0)) + exponent

    for item in items:
        add(item)
    if coefficient == 0:
        return number(0)
    parts = [power(item, exponent) for item, exponent in powers.items() if exponent]
    parts.sort(key=str)
    if coefficient != 1 or not parts:
        parts.insert(0, number(coefficient))
    return parts[0] if len(parts) == 1 else Expr('mul', tuple(parts))


def add(*items: Expr) -> Expr:
    parts = []
    constant = Fraction(0)
    for item in items:
        expanded = item.args if item.op == 'add' else (item,)
        for part in expanded:
            if part.op == 'number':
                constant += part.value
            else:
                parts.append(part)
    if constant:
        parts.append(number(constant))
    parts.sort(key=str)
    return number(0) if not parts else parts[0] if len(parts) == 1 else Expr('add', tuple(parts))


def from_ast(node: ast.AST) -> Expr | None:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
        return symbol(f'{node.value.id}[{node.slice.value}]')
    if isinstance(node, ast.Name):
        return symbol(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return number(node.value)
    if isinstance(node, ast.UnaryOp):
        value = from_ast(node.operand)
        if value is None:
            return None
        return multiply(number(-1), value) if isinstance(node.op, ast.USub) else value if isinstance(node.op, ast.UAdd) else None
    if isinstance(node, ast.BinOp):
        left, right = from_ast(node.left), from_ast(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return add(left, right)
        if isinstance(node.op, ast.Sub):
            return add(left, multiply(number(-1), right))
        if isinstance(node.op, ast.Mult):
            return multiply(left, right)
        if isinstance(node.op, ast.Div):
            return multiply(left, power(right, -1))
        if isinstance(node.op, ast.Pow) and right.op == 'number':
            return power(left, right.value)
        return None
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
        args = [from_ast(arg) for arg in node.args]
        if any(arg is None for arg in args) or node.keywords:
            return None
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func.value) not in {'np', 'numpy', 'math', 'np.linalg', 'numpy.linalg'}:
            return None
        if name in {'sin', 'cos', 'exp', 'norm'} and len(args) == 1:
            return Expr(name, tuple(args))
        if name == 'square' and len(args) == 1:
            return power(args[0], 2)
        if name == 'pow' and len(args) == 2 and args[1].op == 'number':
            return power(args[0], args[1].value)
        if name == 'sqrt' and len(args) == 1:
            return power(args[0], Fraction(1, 2))
        if name == 'hypot' and len(args) == 2:
            return power(add(power(args[0], 2), power(args[1], 2)), Fraction(1, 2))
    return None


def parse_expression(source: str) -> Expr:
    result = from_ast(ast.parse(source, mode='eval').body)
    if result is None:
        raise ValueError('unsupported symbolic expression')
    return result


def dependencies(expr: Expr) -> tuple[str, ...]:
    found = {part.value for part in walk(expr) if part.op == 'symbol'}
    return tuple(sorted(found))


def equivalent(left: Expr, right: Expr) -> bool:
    import sympy as sp

    def convert(expr):
        if expr.op == 'symbol':
            return sp.Symbol(expr.value, real=True)
        if expr.op == 'number':
            return sp.Rational(expr.value)
        args = [convert(arg) for arg in expr.args]
        if expr.op == 'pow':
            return args[0] ** sp.Rational(expr.value)
        operations = {'add': sp.Add, 'mul': sp.Mul, 'sin': sp.sin,
                      'cos': sp.cos, 'exp': sp.exp, 'norm': sp.Function('norm')}
        return operations[expr.op](*args)

    return left == right or sp.expand(convert(left) - convert(right)) == 0


def walk(expr: Expr):
    yield expr
    for arg in expr.args:
        yield from walk(arg)


def symbol_power(expr: Expr, name: str) -> Fraction | None:
    if expr.op == 'symbol':
        return Fraction(1 if expr.value == name else 0)
    if expr.op == 'number':
        return Fraction(0)
    if expr.op == 'pow':
        inner = symbol_power(expr.args[0], name)
        return None if inner is None else inner * expr.value
    if expr.op == 'mul':
        values = [symbol_power(part, name) for part in expr.args]
        return None if any(value is None for value in values) else sum(values, Fraction(0))
    return None
