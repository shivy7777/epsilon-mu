import pytest

from epsilonmu.agent import _modified_source, investigate_timestep
from epsilonmu.analysis import inspect_source
from epsilonmu.physics_rules import verify_relationships
from epsilonmu.verification import Status


def test_units_preserve_competing_interpretations():
    item = inspect_source('span: "m" = 2')
    assert {b.role for b in item.roles} == {'radius','position'}
    assert not item.classification.path


@pytest.mark.parametrize(('formula','expected'), [
    ('bulk*rate*rate/2', Status.PASS), ('bulk*rate**2', Status.FAIL),
    ('rate**2/2', Status.FAIL), ('-bulk*rate**2/2', Status.FAIL)])
def test_dimensional_role_inputs(formula, expected):
    source = ('"""Newtonian particle mechanics"""\nbulk: "kg" = 3\n'
              'rate: "m/s" = 4\nK = ' + formula)
    checks = verify_relationships(inspect_source(source))
    assert any(c.name=='kinetic energy' and c.status==expected for c in checks)


def test_declared_and_alternative_central_laws():
    body = """def advance(i,j,u,w,h):
    d = hypot(i,j)
    u = u - i*h/d**4
    w = w - j*h/d**4
    i = i + u*h
    j = j + w*h
"""
    toy = inspect_source(body)
    assert toy.domain=='central-force system'
    assert not any(c.status==Status.FAIL for c in verify_relationships(toy))
    anisotropic = inspect_source(body.replace('w - j*h','w - 3*j*h'))
    assert anisotropic.domain=='mechanics'
    gravity = inspect_source('"""Newtonian gravity"""\n'+body)
    assert gravity.domain=='orbital mechanics'
    assert any(c.status==Status.FAIL for c in verify_relationships(gravity))


CLOCK = """h = 0.125
tick = 0
q = 1.
u = 0.
kinetic_energy = []
potential_energy = []
while tick < 4:
    a = -q
    q = q + u*h
    u = u + a*h
    kinetic_energy.append(u*u/2)
    potential_energy.append(q*q/2)
    tick += h
"""


def test_clock_refinement_measures_drift_at_same_duration(tmp_path):
    for factor in (2,4):
        code, dt = _modified_source(CLOCK,factor)
        state = {}
        exec(code,state)
        assert state['tick']==4 and dt==0.125/factor
    script = tmp_path/'spring.py'
    script.write_text(CLOCK)
    investigation = investigate_timestep(script)
    assert len(investigation.experiments)==2
    assert investigation.experiments[-1].relative_drift < investigation.baseline_drift
    assert script.read_text()==CLOCK


@pytest.mark.parametrize('source',[CLOCK.replace('tick += h','if q: tick += h'),
                                   CLOCK.replace('tick < 4','tick < 0.3'),
                                   CLOCK.replace('a = -q','if q: break\n    a = -q')])
def test_refuses_ambiguous_duration(source):
    assert _modified_source(source,2) is None
