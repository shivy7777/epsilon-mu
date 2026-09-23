from pathlib import Path

from epsilonmu.observation import Provenance, observe_file
from epsilonmu.pipeline import verify_file
from epsilonmu.verification import Status


def test_instrumented_standalone_series_and_original_unchanged(tmp_path):
    path = tmp_path / 'sim.py'
    path.write_text('h = 0.1\na = 1.0\nb = 0.0\ntrack = []\nfor _ in range(12):\n'
                    '    track.append(a)\n    b -= h*a\n    a += h*b\nprint("finished")\n')
    original = path.read_bytes()
    observed = observe_file(path)
    assert path.read_bytes() == original
    assert observed.run.exit_code == 0
    assert any(o.name == 'track' and o.provenance == Provenance.INSTRUMENTED and len(o.values) == 12
               for o in observed.observations)
    assert any(o.name == 'h' and o.provenance == Provenance.STATIC for o in observed.observations)


def test_function_local_numeric_array_is_observed(tmp_path):
    path = tmp_path / 'function.py'
    path.write_text('def simulate():\n    values = []\n    for i in range(5):\n'
                    '        values.append(i*i)\n    return values\nsimulate()\n')
    observed = observe_file(path)
    assert any(o.name == 'values' and o.scope == 'simulate' and o.values == (0, 1, 4, 9, 16)
               for o in observed.observations)


def test_stdout_and_explicit_provenance(tmp_path):
    path = tmp_path / 'out.py'
    path.write_text('import json\nprint("speed = 2.5")\n'
                    'print("EPSILONMU_JSON:" + json.dumps({"states": [1, 2]}))\n')
    result = observe_file(path)
    assert any(o.name == 'speed' and o.provenance == Provenance.STDOUT for o in result.observations)
    assert any(o.name == 'states' and o.provenance == Provenance.EXPLICIT for o in result.observations)


def test_ordinary_script_gets_finite_check_without_contract(tmp_path):
    path = tmp_path / 'ordinary.py'
    path.write_text('positions = [0.0, 1.0, 2.0]\nprint("done")\n')
    report = verify_file(path)
    assert report.run.exit_code == 0
    assert any(c.name == 'finite states' for c in report.checks)
    assert report.status in (Status.PASS, Status.UNKNOWN)


def test_ordinary_oscillator_energy_and_agent_experiment(tmp_path):
    from epsilonmu.agent import investigate_timestep

    path = tmp_path / 'osc.py'
    path.write_text('dt = 0.2\nx = 1.0\nv = 0.0\nkinetic = []\npotential = []\n'
                    'for i in range(100):\n    kinetic.append(0.5*v*v)\n'
                    '    potential.append(0.5*x*x)\n    v -= dt*x\n    x += dt*v\n')
    report = verify_file(path)
    assert report.status == Status.FAIL
    assert any(c.name == 'energy' and c.status == Status.FAIL for c in report.checks)
    result = investigate_timestep(path, max_experiments=2)
    assert result.state.observations
    assert len(result.experiments) == 2


def test_energy_decay_without_conservative_context_is_warning(tmp_path):
    path = tmp_path / 'damped.py'
    path.write_text('kinetic = [5, 4, 3]\npotential = [0, 0, 0]\n')
    report = verify_file(path)
    assert any(c.name == 'energy' and c.status == Status.WARNING for c in report.checks)
    assert report.status == Status.WARNING
