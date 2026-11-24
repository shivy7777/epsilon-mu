"""Checks operate on declared numerical evidence, never inferred truth."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Sequence


class Status(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    WARNING = 'WARNING'
    UNKNOWN = 'UNKNOWN'
    NOT_APPLICABLE = 'NOT_APPLICABLE'


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    category: str = 'numerical'


def check_conservation(values: Sequence[float], name: str, tolerance: float = 0.01) -> CheckResult:
    if len(values) < 2 or not all(isfinite(v) for v in values):
        return CheckResult(name, Status.UNKNOWN, 'Need at least two finite samples')
    scale = max(abs(values[0]), 1e-12)
    if abs(values[0]) < 1e-12:
        return CheckResult(name, Status.UNKNOWN, 'Initial value is too near zero for relative drift')
    drift = max(abs(v - values[0]) for v in values) / scale
    return CheckResult(name, Status.PASS if drift <= tolerance else Status.FAIL,
                       f'Maximum relative drift {drift:.3%}; tolerance {tolerance:.3%}',
                       {'relative_drift': drift})


def check_energy_conservation(kinetic: Sequence[float], potential: Sequence[float], tolerance: float = 0.01) -> CheckResult:
    if len(kinetic) != len(potential):
        return CheckResult('energy', Status.UNKNOWN, 'Kinetic and potential series have different lengths')
    return check_conservation([k + p for k, p in zip(kinetic, potential)], 'energy', tolerance)


def check_linear_momentum(values: Sequence[float], tolerance: float = 0.01) -> CheckResult:
    return check_conservation(values, 'linear momentum', tolerance)


def check_angular_momentum(values: Sequence[float], tolerance: float = 0.01) -> CheckResult:
    return check_conservation(values, 'angular momentum', tolerance)


def check_stability(values: Sequence[float]) -> CheckResult:
    if not values:
        return CheckResult('finite states', Status.UNKNOWN, 'No state samples')
    if not all(isfinite(v) for v in values):
        return CheckResult('finite states', Status.FAIL, 'Nonfinite state detected')
    return CheckResult('finite states', Status.PASS, 'All supplied state samples are finite')


def check_convergence(steps: Sequence[float], observables: Sequence[float], tolerance: float = 0.05) -> CheckResult:
    if len(steps) < 3 or len(steps) != len(observables) or not all(isfinite(x) for x in [*steps, *observables]):
        return CheckResult('convergence', Status.UNKNOWN, 'Need three finite timestep and observable pairs')
    if any(a <= b or b <= 0 for a, b in zip(steps, steps[1:])):
        return CheckResult('convergence', Status.UNKNOWN, 'Timesteps must strictly decrease')
    coarse = abs(observables[1] - observables[0])
    fine = abs(observables[2] - observables[1])
    if coarse < 1e-12:
        return CheckResult('convergence', Status.UNKNOWN, 'Coarse difference too small to estimate convergence')
    ratio = fine / coarse
    status = Status.PASS if ratio < 0.8 else Status.FAIL
    return CheckResult('convergence', status, f'Fine/coarse difference ratio {ratio:.3g}', {'ratio': ratio})
