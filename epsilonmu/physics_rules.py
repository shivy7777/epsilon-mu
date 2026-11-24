"""Context-gated, extensible mechanics relationship specifications."""

from dataclasses import dataclass
from fractions import Fraction

from .analysis import Analysis
from .dimensions import ACCELERATION, ENERGY, FORCE, MOMENTUM, VELOCITY, Dimension
from .relationships import Relationship
from .relationships.symbolic import equivalent
from .relationships.symbolic import Expr, add, dependencies, multiply, number, parse_expression, power, symbol, symbol_power
from .verification import CheckResult, Status


@dataclass(frozen=True)
class RelationshipRule:
    key: str
    domain: str
    model: str
    symbolic_structure: str
    target_aliases: tuple[str, ...]
    required_quantities: tuple[tuple[str, tuple[str, ...]], ...]
    optional_quantities: tuple[str, ...]
    assumptions: tuple[str, ...]
    expected_dimensions: Dimension | None
    verification_method: str
    severity: str
    explanation: str


MECHANICS_RELATIONSHIPS = (
    RelationshipRule('Newton second law', 'mechanics', 'particle', 'F=m*a', ('F', 'force', 'net_force'),
                     (('m', ('m', 'mass')), ('a', ('a', 'acceleration'))), (),
                     ('F denotes net force', 'same particle and frame'), FORCE, 'symbolic equality', 'high',
                     'Net force equals mass times acceleration.'),
    RelationshipRule('linear momentum', 'mechanics', 'particle', 'p=m*v', ('p', 'momentum'),
                     (('m', ('m', 'mass')), ('v', ('v', 'velocity'))), (),
                     ('classical particle',), MOMENTUM, 'symbolic equality', 'medium',
                     'Linear momentum equals mass times velocity.'),
    RelationshipRule('kinetic energy', 'mechanics', 'particle', 'K=1/2*m*v**2', ('K', 'kinetic_energy'),
                     (('m', ('m', 'mass')), ('v', ('v', 'velocity'))), (),
                     ('nonrelativistic speed',), ENERGY, 'symbolic equality', 'medium',
                     'Classical kinetic energy equals one half mass times speed squared.'),
    RelationshipRule('uniform-field potential', 'mechanics', 'projectile', 'U=m*g*h', ('U', 'potential_energy'),
                     (('m', ('m', 'mass')), ('g', ('g', 'gravity')), ('h', ('h', 'height'))), (),
                     ('uniform field', 'reference height zero'), ENERGY, 'symbolic equality', 'medium',
                     'Near a uniform field, potential energy changes by mass times gravity times height.'),
    RelationshipRule('spring force', 'mechanics', 'oscillator', 'F=-k*x', ('F', 'spring_force'),
                     (('k', ('k', 'stiffness', 'spring_constant')), ('x', ('x', 'displacement'))), (),
                     ('linear spring', 'displacement from equilibrium'), FORCE, 'symbolic equality', 'medium',
                     'A linear spring force opposes displacement.'),
    RelationshipRule('spring potential', 'mechanics', 'oscillator', 'U=1/2*k*x**2', ('U', 'spring_potential'),
                     (('k', ('k', 'stiffness', 'spring_constant')), ('x', ('x', 'displacement'))), (),
                     ('linear spring', 'zero potential at equilibrium'), ENERGY, 'symbolic equality', 'medium',
                     'A linear spring stores one half stiffness times displacement squared.'),
    RelationshipRule('gravitational radial dependence', 'mechanics', 'orbit', 'F=G*m1*m2/r**2',
                     ('F', 'force', 'gravitational_force'), (('G', ('G', 'gravitational_constant')),),
                     ('mass names',), ('Newtonian two-body gravity intended', 'radius is Euclidean distance'),
                     FORCE, 'radial power', 'high', 'Newtonian gravitational force magnitude scales as radius to -2.'),
    RelationshipRule('gravitational vector acceleration', 'mechanics', 'orbit', 'ax=-G*M*x/r**3',
                     ('ax', 'acceleration_x'), (('G', ('G', 'gravitational_constant')),
                     ('M', ('M', 'central_mass')), ('x', ('x',)), ('r', ('r', 'radius'))),
                     ('force intermediate',), ('Newtonian central gravity intended',), ACCELERATION,
                     'symbolic equality with force substitution', 'high',
                     'Newtonian vector acceleration points inward with inverse-cube vector denominator.'),
    RelationshipRule('mechanical energy', 'mechanics', 'particle', 'E=K+U', ('E', 'total_energy'),
                     (('K', ('K', 'kinetic_energy')), ('U', ('U', 'potential_energy'))), (),
                     ('K and U share a reference and instant',), ENERGY, 'symbolic equality', 'medium',
                     'Mechanical energy is kinetic plus potential energy.'),
    RelationshipRule('constant acceleration velocity', 'mechanics', 'projectile', 'v=v0+a*t', ('v', 'velocity'),
                     (('v0', ('v0', 'initial_velocity')), ('a', ('a', 'acceleration')), ('t', ('t', 'time'))), (),
                     ('constant acceleration',), VELOCITY, 'symbolic equality', 'medium',
                     'Velocity changes linearly under constant acceleration.'),
    RelationshipRule('constant acceleration position', 'mechanics', 'projectile', 'x=x0+v0*t+1/2*a*t**2',
                     ('x', 'position'), (('x0', ('x0', 'initial_position')),
                     ('v0', ('v0', 'initial_velocity')), ('a', ('a', 'acceleration')), ('t', ('t', 'time'))), (),
                     ('constant acceleration',), None, 'symbolic equality', 'medium',
                     'Position is quadratic in time under constant acceleration.'),
)


def _active(rule: RelationshipRule, analysis: Analysis) -> bool:
    labels = set(analysis.context.labels)
    model = analysis.classification.path[-1] if analysis.classification.path else ''
    if rule.model == 'orbit':
        return 'newtonian_gravity' in labels
    if rule.model == 'oscillator':
        return 'harmonic_oscillator' in labels or model == 'harmonic oscillator'
    if rule.model == 'projectile':
        return 'projectile' in labels or model == 'projectile motion'
    return bool(labels & {'newtonian_mechanics', 'newtonian_gravity', 'harmonic_oscillator', 'projectile'}) or (
        analysis.classification.confidence >= 0.75 and model in {'harmonic oscillator', 'projectile motion', 'orbital mechanics'})


def _substitute(expr: Expr, mapping: dict[str, Expr]) -> Expr:
    if expr.op == 'symbol':
        return mapping.get(str(expr.value), expr)
    if expr.op == 'number':
        return expr
    args = tuple(_substitute(arg, mapping) for arg in expr.args)
    if expr.op == 'add':
        return add(*args)
    if expr.op == 'mul':
        return multiply(*args)
    if expr.op == 'pow':
        return power(args[0], expr.value)
    return Expr(expr.op, args, expr.value)


def _radius(analysis: Analysis) -> str | None:
    for item in analysis.relationships:
        expr = item.expression
        if (expr.op == 'pow' and expr.value == Fraction(1, 2)
                and expr.args[0].op == 'add' and len(expr.args[0].args) >= 2):
            return item.target
    return None


def _check_radial(rule: RelationshipRule, analysis: Analysis) -> CheckResult | None:
    radius = _radius(analysis)
    if radius is None:
        return None
    candidates = [item for item in analysis.relationships if radius in item.dependencies
                  and set(item.dependencies) & {'G', 'gravitational_constant'}]
    for item in candidates:
        deps = set(item.dependencies)
        if radius not in deps or not deps & {'G', 'gravitational_constant'}:
            continue
        acceleration = [other.target for other in analysis.relationships
                        if item.target in other.dependencies and radius in other.dependencies]
        if not any(analysis.graph.path(item.target, target) for target in acceleration):
            continue
        observed = symbol_power(item.expression, radius)
        if observed is None:
            continue
        status = Status.PASS if observed == -2 else Status.FAIL
        return CheckResult(rule.key, status,
                           f'Expected radius power -2; observed {observed} at line {item.line}; '
                           'Newtonian intent from source context and radius-to-acceleration dependency',
                           {'observed_radial_power': float(observed)}, 'physical relationship')
    return None


def _check_equality(rule: RelationshipRule, analysis: Analysis) -> tuple[CheckResult, ...]:
    expected = parse_expression(rule.symbolic_structure.split('=', 1)[1])
    results = []
    for item in analysis.relationships:
        if item.kind != 'assignment' or item.target not in rule.target_aliases:
            continue
        actual = item.expression
        if rule.key == 'gravitational vector acceleration':
            force = next((candidate for candidate in analysis.relationships
                          if candidate.target in {'F', 'force', 'gravitational_force'}), None)
            if force is not None:
                actual = _substitute(actual, {force.target: force.expression})
        actual_deps = set(dependencies(actual))
        mapping: dict[str, Expr] = {}
        for role, aliases in rule.required_quantities:
            match = next((alias for alias in aliases if alias in actual_deps), None)
            role_name = {'m':'mass','v':'velocity','a':'acceleration','x':'position','r':'radius',
                         't':'time','h':'position'}.get(role)
            bound = {b.variable for b in analysis.roles if b.role==role_name}
            candidates = bound & actual_deps
            if match is None and len(candidates)==1:
                match = next(iter(candidates))
            # A uniquely declared dimension can identify a missing factor.
            if match is None:
                declared = {b.variable for b in analysis.roles if b.role==role_name
                            and b.evidence=='declared SI unit annotation'}
                if len(declared)==1 and role_name in {'mass','acceleration','velocity'}:
                    match = next(iter(declared))
            if match is None:
                break
            mapping[role] = symbol(match)
        else:
            target_expected = _substitute(expected, mapping)
            status = Status.PASS if equivalent(actual, target_expected) else Status.FAIL
            results.append(CheckResult(rule.key, status,
                                       f'Expected {target_expected}; observed {actual} at line {item.line}; '
                                       f'assumptions: {", ".join(rule.assumptions)}',
                                       category='physical relationship'))
    return tuple(results)


def verify_relationships(analysis: Analysis) -> tuple[CheckResult, ...]:
    results = []
    for velocity, line, exponent in analysis.radial_checks:
        results.append(CheckResult('Newtonian vector radial dependence',
                                   Status.PASS if exponent == -3 else Status.FAIL,
                                   f'Component advancing {velocity} at line {line}: expected position times r^-3, observed r^{exponent}',
                                   category='physical relationship'))
    for rule in MECHANICS_RELATIONSHIPS:
        if not _active(rule, analysis):
            continue
        if rule.verification_method == 'radial power':
            result = _check_radial(rule, analysis)
            if result is not None:
                results.append(result)
        else:
            results.extend(_check_equality(rule, analysis))
    return tuple(results)
