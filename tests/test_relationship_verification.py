from pathlib import Path

from epsilonmu.analysis import inspect_source
from epsilonmu.pipeline import OverallOutcome, verify_file
from epsilonmu.physics_rules import MECHANICS_RELATIONSHIPS, verify_relationships
from epsilonmu.verification import Status
from epsilonmu.agent import investigate_timestep
from epsilonmu.cli import main
from contextlib import redirect_stdout
from io import StringIO


def gravity_program(denominator: str, context: str = 'Newtonian two-body gravity in SI units') -> str:
    return f'''"""{context}"""
import math
G = 6.67430e-11
M = 5.0e24
m = 2.0
x, y = 7.0e6, 1.0e6
r = math.sqrt(x*x+y*y)
F = G*M*m/({denominator})
ax = -F*x/(m*r)
'''


def test_equivalent_gravity_laws_pass_and_inverse_r_fails():
    for denominator in ('r**2', 'r*r', 'pow(r,2)'):
        result = verify_relationships(inspect_source(gravity_program(denominator)))
        assert any(c.name == 'gravitational radial dependence' and c.status == Status.PASS for c in result)
        assert any(c.name == 'gravitational vector acceleration' and c.status == Status.PASS for c in result)
    result = verify_relationships(inspect_source(gravity_program('r')))
    assert any(c.name == 'gravitational radial dependence' and c.status == Status.FAIL for c in result)


def test_inverse_r_without_gravity_context_is_not_failed():
    result = verify_relationships(inspect_source(gravity_program('r', 'inverse-r toy central force')))
    assert not any(c.status == Status.FAIL for c in result)
    result = verify_relationships(inspect_source(gravity_program('r', 'two-body orbit with inverse-r toy force')))
    assert not any(c.status == Status.FAIL for c in result)


def test_gravity_radial_check_with_uninformative_variable_names():
    source = '''"""Newtonian two-body gravity in SI units"""
import math
G = 6.67430e-11
a, b = 5.0, 2.0
c, d = 7.0, 1.0
e = math.sqrt(c*c+d*d)
f = G*a*b/(e*e)
u = -f*c/(b*e)
'''
    item = inspect_source(source)
    assert item.classification.path[-1] == 'orbital mechanics'
    assert any(c.name == 'gravitational radial dependence' and c.status == Status.PASS
               for c in verify_relationships(item))


def test_direct_newton_relation_uses_symbolic_equivalence():
    source = '"""Newtonian particle mechanics"""\nm=2\na=3\nF=a*m\n'
    result = verify_relationships(inspect_source(source))
    assert any(c.name == 'Newton second law' and c.status == Status.PASS for c in result)
    source = source.replace('F=a*m', 'F=a+m')
    result = verify_relationships(inspect_source(source))
    assert any(c.name == 'Newton second law' and c.status == Status.FAIL for c in result)


def test_momentum_energy_and_spring_equivalent_forms():
    mechanics = inspect_source('"""Newtonian particle mechanics"""\n'
                               'p = velocity*mass\nK = mass*velocity*velocity/2\nE = U+K\n')
    checks = verify_relationships(mechanics)
    assert {c.name for c in checks if c.status == Status.PASS} >= {
        'linear momentum', 'kinetic energy', 'mechanical energy'}
    spring = inspect_source('"""Harmonic oscillator"""\n'
                            'F = -(displacement*stiffness)\nU = stiffness*displacement*displacement/2\n')
    checks = verify_relationships(spring)
    assert {c.name for c in checks if c.status == Status.PASS} >= {'spring force', 'spring potential'}


def test_uniform_field_and_constant_acceleration_formulas():
    source = '"""Projectile in uniform gravity"""\n'
    source += 'U = height*mass*gravity\nvelocity = initial_velocity + acceleration*time\n'
    source += 'position = initial_position + initial_velocity*time + acceleration*time*time/2\n'
    checks = verify_relationships(inspect_source(source))
    assert {c.name for c in checks if c.status == Status.PASS} >= {
        'uniform-field potential', 'constant acceleration velocity', 'constant acceleration position'}


def test_registry_has_requested_mechanics_relationships():
    assert len(MECHANICS_RELATIONSHIPS) >= 9
    assert all(rule.domain == 'mechanics' and rule.assumptions and rule.verification_method
               for rule in MECHANICS_RELATIONSHIPS)


def test_finite_only_is_inconclusive_and_explicit_failure_is_failed(tmp_path):
    plain = tmp_path / 'plain.py'
    plain.write_text('positions = [0, 1, 2]\n')
    report = verify_file(plain)
    assert report.outcome == OverallOutcome.INCONCLUSIVE
    broken = tmp_path / 'bad.py'
    broken.write_text(gravity_program('r'))
    report = verify_file(broken)
    assert report.outcome == OverallOutcome.FAILED
    assert report.status == Status.FAIL


def test_unit_match_alone_does_not_verify_physics(tmp_path):
    path = tmp_path / 'units.py'
    path.write_text('import json\nprint("EPSILONMU_JSON:" + json.dumps('
                    '{"mass": 2, "units": {"mass": "kg"}}))\n')
    report = verify_file(path)
    assert report.outcome == OverallOutcome.INCONCLUSIVE
    assert report.status == Status.UNKNOWN


def test_one_declared_check_without_identified_model_is_inconclusive(tmp_path):
    path = tmp_path / 'single.py'
    path.write_text('print("EPSILONMU_JSON:{\\"states\\":[1,2],\\"constraints\\":['
                    '{\\"name\\":\\"bound\\",\\"value\\":1,\\"minimum\\":0,\\"maximum\\":2}]}" )\n')
    report = verify_file(path)
    assert report.outcome == OverallOutcome.INCONCLUSIVE
    orbit = Path(__file__).parents[1] / 'examples/orbit/correct.py'
    assert verify_file(orbit).outcome == OverallOutcome.VERIFIED


def test_relationship_failure_becomes_agent_hypothesis(tmp_path):
    path = tmp_path / 'gravity.py'
    path.write_text(gravity_program('r'))
    baseline = verify_file(path)
    result = investigate_timestep(path, baseline=baseline)
    assert any('incorrect force law' in item.claim for item in result.state.hypotheses)
    assert 'radius power' in result.explanation
    assert not result.experiments


def test_cli_shows_relationships_graph_and_inconclusive(tmp_path):
    path = tmp_path / 'gravity.py'
    path.write_text(gravity_program('r'))
    out = StringIO()
    with redirect_stdout(out):
        assert main(['inspect', str(path)]) == 0
    content = out.getvalue()
    assert 'Relationships:' in content
    assert 'Dependencies:' in content
    assert 'gravitational radial dependence' in content
    plain = tmp_path / 'plain.py'
    plain.write_text('positions = [1, 2, 3]\n')
    out = StringIO()
    with redirect_stdout(out):
        assert main(['verify', str(plain)]) == 0
    assert 'Overall: INCONCLUSIVE' in out.getvalue()


def test_earth_reference_from_source_context_is_warning(tmp_path):
    path = tmp_path / 'earth.py'
    path.write_text('"""Near Earth\'s surface in SI units."""\ng = 1.6\n')
    report = verify_file(path)
    assert any(c.name == 'Earth surface gravity' and c.status == Status.WARNING for c in report.checks)
    other = tmp_path / 'other.py'
    other.write_text('g = 1.6\n')
    report = verify_file(other)
    assert not any(c.name == 'Earth surface gravity' for c in report.checks)
