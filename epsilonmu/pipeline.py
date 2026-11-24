"""Inspect, observe and verify a script's explicit or recovered evidence."""

import json
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from pathlib import Path

from .analysis import Analysis, inspect_file
from .execution import RunResult
from .observation import Observation, Provenance, observe_file
from .knowledge import evaluate_rules
from .physics_rules import verify_relationships
from .verification import (CheckResult, Status, check_angular_momentum,
                           check_convergence, check_energy_conservation,
                           check_linear_momentum, check_stability)

PREFIX = 'EPSILONMU_JSON:'


class OverallOutcome(str, Enum):
    VERIFIED = 'VERIFIED'
    FAILED = 'FAILED'
    INCONCLUSIVE = 'INCONCLUSIVE'


@dataclass(frozen=True)
class VerificationReport:
    analysis: Analysis
    run: RunResult
    checks: tuple[CheckResult, ...]
    status: Status
    observations: tuple[Observation, ...] = ()
    outcome: OverallOutcome = OverallOutcome.INCONCLUSIVE


def verify_file(path: str | Path, timeout: float = 10, *, forward_env=()) -> VerificationReport:
    analysis = inspect_file(path)
    observed = observe_file(path, timeout, forward_env=forward_env)
    run = observed.run
    checks: list[CheckResult] = []
    if run.timed_out:
        checks.append(CheckResult('execution', Status.FAIL, 'Execution timed out', category='runtime'))
    elif run.exit_code != 0:
        checks.append(CheckResult('execution', Status.FAIL, f'Process exited {run.exit_code}: {run.stderr[-500:]}', category='runtime'))
    else:
        lines = [line[len(PREFIX):] for line in run.stdout.splitlines() if line.startswith(PREFIX)]
        if lines:
            try:
                data = json.loads(lines[-1])
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
                if 'states' in data:
                    checks.append(check_stability(data['states']))
                if 'kinetic' in data and 'potential' in data:
                    checks.append(check_energy_conservation(data['kinetic'], data['potential']))
                if 'linear_momentum' in data:
                    checks.append(check_linear_momentum(data['linear_momentum']))
                if 'angular_momentum' in data:
                    checks.append(check_angular_momentum(data['angular_momentum']))
                if 'timesteps' in data and 'observables' in data:
                    checks.append(check_convergence(data['timesteps'], data['observables']))
                for item in data.get('constraints', []):
                    value, lower, upper = item['value'], item.get('minimum'), item.get('maximum')
                    passed = (lower is None or value >= lower) and (upper is None or value <= upper)
                    checks.append(CheckResult(item.get('name', 'constraint'), Status.PASS if passed else Status.FAIL,
                                              f'Value {value}; bounds [{lower}, {upper}]',
                                              category='physical relationship'))
                if not checks:
                    checks.append(CheckResult('evidence', Status.UNKNOWN, 'No recognized verification fields'))
            except (ValueError, TypeError, KeyError) as exc:
                checks.append(CheckResult('evidence', Status.FAIL, f'Invalid result data: {exc}'))
        else:
            series = [item for item in observed.observations
                      if item.provenance == Provenance.INSTRUMENTED and len(item.values) >= 2]
            if series:
                from math import isfinite
                finite = all(isfinite(value) for item in series for value in item.values)
                checks.append(CheckResult('finite states', Status.PASS if finite else Status.FAIL,
                                          f'{len(series)} recovered numeric series checked for finiteness'))
                by_name = {item.name.lower(): item for item in series}
                kinetic = by_name.get('kinetic') or by_name.get('kinetic_energy')
                potential = by_name.get('potential') or by_name.get('potential_energy')
                if kinetic and potential:
                    energy = check_energy_conservation(kinetic.values, potential.values)
                    if energy.status == Status.FAIL and analysis.domain not in {'harmonic oscillator', 'orbital mechanics'}:
                        energy = CheckResult(energy.name, Status.WARNING,
                                             energy.reason + '; conservation was not established for this model',
                                             energy.metrics)
                    checks.append(energy)
                momentum = by_name.get('linear_momentum')
                if momentum:
                    checks.append(check_linear_momentum(momentum.values))
            if not checks:
                checks.append(CheckResult('evidence', Status.UNKNOWN, 'No checkable numeric series recovered'))
        physical_names = {'F = ma', 'g = GM/r^2', 'p = mv', 'K = 1/2 mv^2', 'F = -kx'}
        context = dict(observed.explicit_data or {})
        if 'earth_surface' in analysis.context.labels:
            context.setdefault('environment', 'earth_surface')
        if 'si_units' in analysis.context.labels:
            context['source_si_hint'] = True
        checks.extend(replace(item, category='physical relationship' if item.name in physical_names
                              else 'plausibility') for item in
                      evaluate_rules(observed.observations, context=context))
        checks.extend(verify_relationships(analysis))
    substantive_pass = any(item.status == Status.PASS and
                           (item.category == 'physical relationship' or
                            item.name in {'energy', 'linear momentum', 'angular momentum'})
                           for item in checks)
    status = (Status.FAIL if any(c.status == Status.FAIL for c in checks) else
              Status.WARNING if any(c.status == Status.WARNING for c in checks) else
              Status.PASS if substantive_pass else Status.UNKNOWN)
    model_supported = analysis.classification.confidence >= 0.75
    outcome = (OverallOutcome.FAILED if status == Status.FAIL else
               OverallOutcome.VERIFIED if status == Status.PASS and substantive_pass and model_supported else
               OverallOutcome.INCONCLUSIVE)
    return VerificationReport(analysis, run, tuple(checks), status, observed.observations, outcome)
