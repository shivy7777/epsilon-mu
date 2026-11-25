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
    all_loops = [n for n in ast.walk(tree) if isinstance(n,(ast.For,ast.While,ast.AsyncFor))]
    if len(all_loops) != 1 or all_loops[0] not in tree.body:
        return None
    loop = all_loops[0]
    if any(isinstance(n,(ast.Break,ast.Continue,ast.Return,ast.Raise)) for n in ast.walk(loop)):
        return None
    stores = [n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)]
    step_name = 'dt'
    if isinstance(loop,ast.While):
        test = loop.test
        if not (isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Lt)
                and isinstance(test.left,ast.Name)):
            return None
        clock = test.left.id
        def literal(node):
            if isinstance(node, ast.Name):
                matches = [s.value for s in tree.body[:tree.body.index(loop)] if isinstance(s,ast.Assign)
                           and len(s.targets)==1 and isinstance(s.targets[0],ast.Name) and s.targets[0].id==node.id]
                if len(matches)!=1 or stores.count(node.id)!=(2 if node.id==clock else 1):
                    return None
                node = matches[0]
            return node.value if isinstance(node,ast.Constant) and type(node.value) in (int,float) else None
        start, stop = literal(test.left), literal(test.comparators[0])
        ticks = []
        for n in loop.body:
            if isinstance(n,ast.AugAssign) and isinstance(n.op,ast.Add) and isinstance(n.target,ast.Name) and n.target.id==clock and isinstance(n.value,ast.Name):
                ticks.append(n.value)
            elif (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                  and n.targets[0].id==clock and isinstance(n.value,ast.BinOp) and isinstance(n.value.op,ast.Add)
                  and isinstance(n.value.left,ast.Name) and n.value.left.id==clock and isinstance(n.value.right,ast.Name)):
                ticks.append(n.value.right)
        if len(ticks)!=1 or start is None or stop is None or start>=stop:
            return None
        step_name = ticks[0].id
        step = literal(ticks[0])
        if step is None or step<=0:
            return None
        # Replay only the clock arithmetic. Floating-point termination can add
        # an extra step even when the real-valued endpoints divide evenly.
        def steps(delta):
            time = start
            for count in range(1_000_001):
                if time >= stop:
                    return count
                time += delta
            return None
        original, refined = steps(step), steps(step/factor)
        if original is None or refined != original*factor:
            return None
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==step_name
                   and isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float))]
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == 'range' and len(node.iter.args) == 1
             and isinstance(node.iter.args[0], ast.Constant) and isinstance(node.iter.args[0].value, int)]
    if (len(assignments) != 1 or stores.count(step_name)!=1 or assignments[0] not in tree.body[:tree.body.index(loop)]
            or (isinstance(loop,ast.For) and len(loops)!=1) or assignments[0].value.value <= 0):
        return None
    if not any(isinstance(n,ast.Name) and n.id==step_name and isinstance(n.ctx,ast.Load)
               for stmt in loop.body for n in ast.walk(stmt)):
        return None
    if loops and (loop.iter.args[0].value <= 0 or not isinstance(loop.target,ast.Name)
                  or any(isinstance(n,ast.Name) and n.id==loop.target.id and isinstance(n.ctx,ast.Load)
                         for stmt in loop.body for n in ast.walk(stmt))):
        return None
    dt = float(assignments[0].value.value) / factor
    assignments[0].value = ast.Constant(dt)
    if loops:
        loops[0].iter.args[0] = ast.Constant(loops[0].iter.args[0].value * factor)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), dt


def refinement_unavailable_reason(report, path):
    """Preflight the existing one-halving experiment without executing a copy."""
    drift = _energy_drift(report)
    if drift is None or drift <= 0.01:
        return 'No baseline energy drift above the refinement threshold.'
    if any(c.category == 'physical relationship' and c.status == Status.FAIL for c in report.checks):
        return 'A failed governing-law check takes precedence in the timestep investigator.'
    if _modified_source(Path(path).read_text(encoding='utf-8'), 2) is None:
        return 'Could not establish a fixed duration for timestep refinement.'
    return None


def investigate_timestep(path: str | Path, max_experiments: int = 2,
                         timeout: float = 10, model: AgentModel | None = None,
                         baseline: VerificationReport | None = None, *, forward_env=()) -> Investigation:
    if not 0 <= max_experiments <= 10:
        raise ValueError('max_experiments must be between 0 and 10')
    path = Path(path).resolve()
    state = DebugState(str(path), max_experiments)
    baseline = baseline if baseline is not None else verify_file(path, timeout, forward_env=forward_env)
    drift = _energy_drift(baseline)
    state.checks.extend(f'{item.name}: {item.status.value}' for item in baseline.checks)
    state.observations.extend(f'{item.name} ({item.provenance.value}, {item.kind})'
                              for item in baseline.observations)
    relationship_failures = [item for item in baseline.checks
                             if item.category == 'physical relationship' and item.status == Status.FAIL]
    if relationship_failures:
        failure = relationship_failures[0]
        state.observations.append(f'{failure.name}: {failure.reason}')
        force_related = any(term in failure.name.lower() for term in ('force','gravit','radial'))
        for claim in (('incorrect force law' if force_related else f'incorrect {failure.name} relationship'), 'alternative intended physical model'):
            state.start_hypothesis(claim, failure.reason)
        state.hypotheses[0].status = 'supported by symbolic comparison'
        state.hypotheses[0].confidence = 'moderate'
        return Investigation(Status.FAIL,
                             f'{failure.name}: {failure.reason}. Confirmed discrepancy under the stated assumptions.',
                             drift, (), state)
    if drift is None or drift <= 0.01:
        for item in baseline.checks:
            if item.category == 'plausibility' and item.status in (Status.WARNING,Status.FAIL):
                state.start_hypothesis('parameter inconsistent with reference', item.reason)
                state.hypotheses[-1].status = 'suspected' if item.status==Status.WARNING else 'supported'
                return Investigation(item.status, f'{item.name}: {item.reason}', drift, (), state)
        return Investigation(Status.UNKNOWN, 'No failed energy check with numeric drift to investigate',
                             drift, (), state)
    source = path.read_text(encoding='utf-8')
    if _modified_source(source, 2) is None:
        return Investigation(Status.UNKNOWN, 'Could not establish a fixed duration for timestep refinement',
                             drift, (), state)
    state.start_hypothesis('timestep too large', f'Baseline energy drift {drift:.3%}')
    experiments = []
    with tempfile.TemporaryDirectory(prefix='epsilonmu-experiment-') as directory:
        for index in range(max_experiments):
            modified = _modified_source(source, 2 ** (index + 1))
            if modified is None:
                break
            code, dt = modified
            candidate = Path(directory) / f'experiment_{index}.py'
            candidate.write_text(code, encoding='utf-8')
            report = verify_file(candidate, timeout, forward_env=forward_env)
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
