from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

from epsilonmu.agent import DebugState, investigate_timestep
from epsilonmu.cli import main
from epsilonmu.verification import Status


def test_state_does_not_repeat_hypothesis():
    state = DebugState(target='example.py', max_experiments=2)
    assert state.start_hypothesis('timestep too large', 'Energy drift')
    assert not state.start_hypothesis('timestep too large', 'Energy drift')


def test_timestep_experiment_uses_copies_and_evidence():
    path = Path(__file__).parents[1] / 'examples/orbit/bad_timestep.py'
    before = path.read_bytes()
    result = investigate_timestep(path, max_experiments=2)
    assert path.read_bytes() == before
    assert result.status == Status.PASS
    assert len(result.experiments) == 2
    assert result.experiments[-1].relative_drift < result.baseline_drift


def test_debug_cli_reports_experiment():
    path = Path(__file__).parents[1] / 'examples/orbit/bad_timestep.py'
    output = StringIO()
    with redirect_stdout(output):
        code = main(['debug', str(path), '--max-experiments', '2'])
    assert code == 1
    assert 'dt=' in output.getvalue()
    assert 'supports timestep-related numerical error' in output.getvalue()


def test_bad_force_does_not_support_timestep_hypothesis():
    path = Path(__file__).parents[1] / 'examples/orbit/bad_force.py'
    result = investigate_timestep(path, max_experiments=2)
    assert result.status == Status.UNKNOWN
    assert result.state.hypotheses[0].status == 'inconclusive'
