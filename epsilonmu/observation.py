"""Structured observations from static source, stdout and runtime tracing."""

import ast
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .execution import RunResult, run_script

EXPLICIT_PREFIX = 'EPSILONMU_JSON:'
PROBE_PREFIX = 'EPSILONMU_INSTRUMENTED:'
STDOUT_NUMBER = re.compile(r'^\s*([A-Za-z_][\w ]{0,50})\s*[:=]\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$')


class Provenance(str, Enum):
    STATIC = 'static'
    STDOUT = 'stdout'
    INSTRUMENTED = 'instrumented'
    EXPLICIT = 'explicit'


@dataclass(frozen=True)
class Observation:
    name: str
    values: tuple[float, ...]
    provenance: Provenance
    line: int | None = None
    scope: str | None = None
    unit: str | None = None
    role: str = 'unknown'

    @property
    def kind(self) -> str:
        return 'series' if len(self.values) > 1 else 'scalar'


@dataclass(frozen=True)
class ObservationRun:
    run: RunResult
    observations: tuple[Observation, ...]
    explicit_data: dict | None


def static_observations(source: str) -> tuple[Observation, ...]:
    observations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                for target in targets:
                    if isinstance(target, ast.Name):
                        observations.append(Observation(target.id, (float(value.value),), Provenance.STATIC,
                                                        node.lineno, role=infer_role(target.id)))
    return tuple(observations)


def infer_role(name: str) -> str:
    lowered = name.lower()
    if lowered in {'dt', 'timestep', 'time_step'}:
        return 'timestep'
    if lowered in {'t', 'time', 'times'}:
        return 'time'
    for role, keys in (
        ('position', ('position', 'positions', 'displacement', 'trajectory')),
        ('velocity', ('velocity', 'velocities', 'speed')),
        ('acceleration', ('acceleration', 'accelerations')),
        ('force', ('force', 'forces')),
        ('energy', ('energy', 'kinetic', 'potential')),
        ('momentum', ('momentum', 'momenta')),
        ('parameter', ('mass', 'gravity', 'radius', 'stiffness')),
    ):
        if any(key in lowered for key in keys):
            return role
    return 'unknown'


def observe_file(path: str | Path, timeout: float = 10) -> ObservationRun:
    path = Path(path)
    observations = list(static_observations(path.read_text(encoding='utf-8')))
    raw = run_script(path, timeout, instrument=True)
    visible_lines = []
    explicit = None
    for line in raw.stdout.splitlines():
        if line.startswith(PROBE_PREFIX):
            try:
                for record in json.loads(line[len(PROBE_PREFIX):]):
                    values = record['values']
                    if values and len(values) <= 4096:
                        observations.append(Observation(record['name'], tuple(values), Provenance.INSTRUMENTED,
                                                        record.get('line'), record.get('scope'),
                                                        role=infer_role(record['name'])))
            except (ValueError, TypeError, KeyError):
                pass
        else:
            visible_lines.append(line)
            if line.startswith(EXPLICIT_PREFIX):
                try:
                    data = json.loads(line[len(EXPLICIT_PREFIX):])
                    if isinstance(data, dict):
                        explicit = data
                        units = data.get('units', {})
                        if not isinstance(units, dict):
                            units = {}
                        for name, value in data.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                observations.append(Observation(name, (float(value),), Provenance.EXPLICIT,
                                                                unit=units.get(name), role=infer_role(name)))
                            elif isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
                                observations.append(Observation(name, tuple(float(x) for x in value), Provenance.EXPLICIT,
                                                                unit=units.get(name), role=infer_role(name)))
                except (ValueError, TypeError):
                    pass
            match = STDOUT_NUMBER.match(line)
            if match:
                name = match.group(1).strip()
                observations.append(Observation(name, (float(match.group(2)),), Provenance.STDOUT,
                                                role=infer_role(name)))
    run = RunResult('\n'.join(visible_lines), raw.stderr, raw.exit_code, raw.runtime, raw.timed_out)
    return ObservationRun(run, tuple(observations), explicit)
