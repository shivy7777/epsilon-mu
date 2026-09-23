"""Bounded, evidence-driven timestep investigation on temporary copies."""

import ast
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .pipeline import VerificationReport, verify_file
from .verification import Status


class AgentModel(Protocol):
    """Optional explanation provider; verification never depends on it."""

    def reason(self, evidence: str) -> str: ...


class RuleBasedModel:
    """Local provider that states only what the measured evidence supports."""

    def reason(self, evidence: str) -> str:
        return evidence


@dataclass
class Hypothesis:
    claim: str
    evidence: str
    confidence: str = 'untested'
    experiment: str = ''
    result: str = ''
    status: str = 'pending'


@dataclass
class DebugState:
    target: str
    max_experiments: int = 2
    observations: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    iteration: int = 0

    def start_hypothesis(self, claim: str, evidence: str) -> bool:
        if any(item.claim == claim for item in self.hypotheses):
            return False
        self.hypotheses.append(Hypothesis(claim, evidence))
        return True


@dataclass(frozen=True)
class Experiment:
    timestep: float
    relative_drift: float
    status: Status


@dataclass(frozen=True)
class Investigation:
    status: Status
    explanation: str
    baseline_drift: float | None
    experiments: tuple[Experiment, ...]
    state: DebugState


def _energy_drift(report) -> float | None:
    return next((item.metrics['relative_drift'] for item in report.checks
                 if item.name == 'energy' and 'relative_drift' in item.metrics), None)


def _modified_source(source: str, factor: int) -> tuple[str, float] | None:
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'dt' for t in node.targets)
                   and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float))]
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == 'range' and len(node.iter.args) == 1
             and isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[0].value, int)]
    if len(assignments) != 1 or len(loops) != 1 or assignments[0].value.value <= 0:
        return None
    dt = float(assignments[0].value.value) / factor
    assignments[0].value = ast.Constant(dt)
    loops[0].iter.args[0] = ast.Constant(loops[0].iter.args[0].value * factor)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), dt


def investigate_timestep(path: str | Path, max_experiments: int = 2,
                         timeout: float = 10, model: AgentModel | None = None,
                         baseline: VerificationReport | None = None) -> Investigation:
    if not 0 <= max_experiments <= 10:
        raise ValueError('max_experiments must be between 0 and 10')
    path = Path(path).resolve()
    state = DebugState(str(path), max_experiments)
    baseline = baseline if baseline is not None else verify_file(path, timeout)
    drift = _energy_drift(baseline)
    state.checks.extend(f'{item.name}: {item.status.value}' for item in baseline.checks)
    state.observations.extend(f'{item.name} ({item.provenance.value}, {item.kind})'
                              for item in baseline.observations)
    if drift is None or drift <= 0.01:
        return Investigation(Status.UNKNOWN, 'No failed energy check with numeric drift to investigate',
                             drift, (), state)
    source = path.read_text(encoding='utf-8')
    if _modified_source(source, 2) is None:
        return Investigation(Status.UNKNOWN, 'Timestep experiment requires one literal dt and one range(N) loop',
                             drift, (), state)
    state.start_hypothesis('timestep too large', f'Baseline energy drift {drift:.3%}')
    experiments = []
    with tempfile.TemporaryDirectory(prefix='epsilonmu-experiment-') as directory:
        for index in range(max_experiments):
            modified = _modified_source(source, 2 ** (index + 1))
            assert modified is not None
            code, dt = modified
            candidate = Path(directory) / f'experiment_{index}.py'
            candidate.write_text(code, encoding='utf-8')
            report = verify_file(candidate, timeout)
            measured = _energy_drift(report)
            if measured is None:
                break
            experiments.append(Experiment(dt, measured, report.status))
            state.experiments.append(f'dt={dt:g}; drift={measured:.6g}')
            state.iteration += 1
    hypothesis = state.hypotheses[0]
    supported = bool(experiments) and all(
        item.relative_drift < prior for item, prior in zip(experiments, [drift] + [x.relative_drift for x in experiments[:-1]]))
    hypothesis.status = 'supported' if supported else 'inconclusive'
    hypothesis.confidence = 'moderate' if supported and len(experiments) >= 2 else 'low'
    hypothesis.experiment = 'Halve dt and double step count to keep the simulated duration fixed'
    hypothesis.result = '; '.join(state.experiments)
    explanation = ('Energy drift fell as dt decreased over the same simulated duration; '
                   'this supports timestep-related numerical error, but does not prove the force law is correct.'
                   if supported else 'Timestep experiments did not consistently reduce energy drift.')
    return Investigation(Status.PASS if supported else Status.UNKNOWN,
                         (model or RuleBasedModel()).reason(explanation), drift, tuple(experiments), state)
