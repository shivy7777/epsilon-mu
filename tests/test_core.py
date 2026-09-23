from pathlib import Path

from epsilonmu.analysis import inspect_source
from epsilonmu.execution import run_script
from epsilonmu.verification import Status, check_conservation, check_stability, check_convergence
from epsilonmu.physics import MechanicsSkill
from epsilonmu.pipeline import verify_file


def test_inspection_and_skill():
    source = 'import numpy as np\nfrom scipy.integrate import solve_ivp\ndt = 0.1\ndef acceleration(x): return -x\nresult = solve_ivp(acceleration, (0, 1), [1])\n'
    analysis = inspect_source(source)
    assert analysis.solver == 'solve_ivp'
    assert analysis.timestep == 0.1
    assert 'numpy' in analysis.imports
    assert MechanicsSkill().detect(analysis)


def test_conservation_and_unknown():
    assert check_conservation([1, 1.001, 0.999], 'energy', 0.01).status == Status.PASS
    result = check_conservation([1, 1.2], 'energy', 0.01)
    assert result.status == Status.FAIL
    assert result.metrics['relative_drift'] > 0.19
    assert check_conservation([0, 0], 'energy', 0.01).status == Status.UNKNOWN


def test_stability_and_convergence():
    assert check_stability([1, float('nan')]).status == Status.FAIL
    assert check_stability([1, 2, 3]).status == Status.PASS
    assert check_convergence([0.1, 0.05, 0.025], [1.1, 1.05, 1.025]).status == Status.PASS
    assert check_convergence([0.1, 0.05, 0.025], [1.1, 1.4, 1.8]).status == Status.FAIL


def test_runner_timeout(tmp_path):
    script = tmp_path / 'slow.py'
    script.write_text('import time\ntime.sleep(5)')
    assert run_script(script, timeout=0.1).timed_out


def test_example_pairs():
    base = Path(__file__).parents[1] / 'examples'
    for domain, good, bad in [
        ('projectile', 'correct.py', 'broken_units.py'),
        ('oscillator', 'correct.py', 'unstable.py'),
        ('orbit', 'correct.py', 'bad_timestep.py'),
        ('orbit', 'correct.py', 'bad_force.py'),
        ('collision', 'correct.py', 'broken_momentum.py'),
    ]:
        assert verify_file(base / domain / good).status == Status.PASS
        assert verify_file(base / domain / bad).status == Status.FAIL


def test_no_data_is_unknown(tmp_path):
    script = tmp_path / 'plain.py'
    script.write_text('print("hello")')
    assert verify_file(script).status == Status.UNKNOWN
