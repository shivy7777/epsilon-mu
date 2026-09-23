from epsilonmu.analysis import inspect_source
from epsilonmu.knowledge import CONSTANTS, evaluate_rules
from epsilonmu.observation import Observation, Provenance
from epsilonmu.verification import Status
from epsilonmu.cli import main
from contextlib import redirect_stdout
from io import StringIO


def test_restoring_pattern_with_renamed_variables():
    source = 'def rhs(q, u, step):\n    push = -3.0*q\n    u += push*step\n    q += u*step\n    return q, u\n'
    item = inspect_source(source)
    assert item.classification.path[-1] == 'harmonic oscillator'
    assert item.classification.confidence > 0.5
    assert any('restoring' in e for e in item.classification.evidence)


def test_damping_is_not_harmonic_restoring_force():
    source = 'def step(q,u,h):\n    u += -0.4*u*h\n    q += u*h\n    return q,u\n'
    assert inspect_source(source).classification.path[-1:] != ('harmonic oscillator',)


def test_subtractive_update_is_restoring_coupling():
    source = 'def step(q,u,h):\n    u -= 2*q*h\n    q += u*h\n    return q,u\n'
    assert inspect_source(source).classification.path[-1] == 'harmonic oscillator'


def test_projectile_pattern_with_renamed_variables():
    source = 'def move(a,b,c,d,h):\n    c += 0*h\n    d += -9.8*h\n'
    source += '    a += c*h\n    b += d*h\n    return a,b,c,d\n'
    assert inspect_source(source).classification.path[-1] == 'projectile motion'


def test_orbit_pattern_and_insufficient_evidence():
    source = 'from math import hypot\ndef step(a,b,u,v,h):\n'
    source += '    r = hypot(a,b)\n    fx = -a/r**3\n    fy = -b/r**3\n'
    source += '    u += fx*h\n    v += fy*h\n    a += u*h\n    b += v*h\n'
    assert inspect_source(source).classification.path[-1] == 'orbital mechanics'
    assert inspect_source(source.replace('**3', '**4')).classification.path[-1:] != ('orbital mechanics',)
    assert inspect_source('x = 3\ny = x + 1\n').classification.path == ()


def test_ode_solver_alone_is_not_mechanics():
    source = 'from scipy.integrate import solve_ivp\n'
    source += 'def growth(t, population): return population\n'
    source += 'solve_ivp(growth, (0, 1), [1])\n'
    assert inspect_source(source).classification.path == ()
    assert inspect_source('force = 2\nvelocity = 3\n').classification.path == ()


def test_inspect_cli_exposes_classification_evidence(tmp_path):
    path = tmp_path / 'motion.py'
    path.write_text('def step(q,u,h):\n    u -= q*h\n    q += u*h\n')
    output = StringIO()
    with redirect_stdout(output):
        assert main(['inspect', str(path)]) == 0
    assert 'Confidence:' in output.getvalue()
    assert 'coupled state updates' in output.getvalue()


def test_redundant_force_relation_and_environment_gravity():
    def obs(name, value):
        return Observation(name, (value,), Provenance.EXPLICIT)
    good = evaluate_rules([obs('mass', 2), obs('acceleration', 3), obs('force', 6)])
    bad = evaluate_rules([obs('mass', 2), obs('acceleration', 3), obs('force', 9)])
    assert any(x.name == 'F = ma' and x.status == Status.PASS for x in good)
    assert any(x.name == 'F = ma' and x.status == Status.FAIL for x in bad)
    gravity = evaluate_rules([obs('g', 1.6)])
    assert not any(x.status == Status.FAIL for x in gravity)


def test_instrumented_redundant_values_warn_without_declared_context():
    values = [Observation(name, (value,), Provenance.INSTRUMENTED) for name, value in
              [('force', 9), ('mass', 2), ('acceleration', 3)]]
    assert any(x.name == 'F = ma' and x.status == Status.WARNING for x in evaluate_rules(values))


def test_declared_dimensions_and_universal_constant():
    mass = Observation('mass', (2,), Provenance.EXPLICIT, unit='kg')
    force_bad_unit = Observation('force', (6,), Provenance.EXPLICIT, unit='kg')
    results = evaluate_rules([mass, force_bad_unit])
    assert any(x.name == 'dimensions' and x.status == Status.FAIL for x in results)
    gravity_constant = Observation('G', (CONSTANTS['G'].value * 1.2,), Provenance.EXPLICIT,
                                   unit='m^3/(kg*s^2)')
    results = evaluate_rules([gravity_constant])
    assert any(x.name == 'universal constant G' and x.status == Status.FAIL for x in results)


def test_gravity_relationship_requires_redundant_evidence():
    data = [Observation(name, (value,), Provenance.EXPLICIT) for name, value in
            [('g', 10), ('central_mass', 100), ('radius', 2), ('gravitational_constant', 0.4)]]
    assert any(x.name == 'g = GM/r^2' and x.status == Status.PASS for x in evaluate_rules(data))
    conventional = [Observation('G', (0.4,), Provenance.EXPLICIT) if x.name == 'gravitational_constant' else x
                    for x in data]
    assert any(x.name == 'g = GM/r^2' and x.status == Status.PASS for x in evaluate_rules(conventional))


def test_earth_gravity_reference_needs_declared_context():
    item = Observation('g', (1.6,), Provenance.EXPLICIT, unit='m/s^2')
    assert not any(x.name == 'Earth surface gravity' for x in evaluate_rules([item]))
    contextual = evaluate_rules([item], context={'environment': 'earth_surface'})
    assert any(x.name == 'Earth surface gravity' and x.status == Status.WARNING for x in contextual)
