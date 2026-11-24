"""Context-bound physics relationships with explicit assumptions and SI references."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence

from .observation import Observation, Provenance
from .verification import CheckResult, Status


@dataclass(frozen=True)
class PhysicalConstant:
    symbol: str
    value: float
    unit: str
    relative_tolerance: float
    universal: bool
    context: str


CONSTANTS = {
    'G': PhysicalConstant('G', 6.67430e-11, 'm^3/(kg*s^2)', 1e-3, True, 'SI gravitational constant'),
    'c': PhysicalConstant('c', 299792458.0, 'm/s', 0.0, True, 'SI speed of light in vacuum'),
    'g0': PhysicalConstant('g0', 9.80665, 'm/s^2', 0.05, False, 'standard Earth surface gravity reference'),
}


class Rule(Protocol):
    name: str
    def evaluate(self, data: dict[str, Observation], context: dict) -> CheckResult | None: ...


def _selected(observations: Sequence[Observation]) -> dict[str, Observation]:
    order = {Provenance.STATIC: 0, Provenance.STDOUT: 1,
             Provenance.INSTRUMENTED: 2, Provenance.EXPLICIT: 3}
    selected: dict[str, Observation] = {}
    for item in observations:
        key = item.name.replace(' ', '_')
        previous = selected.get(key)
        if previous is None or order[item.provenance] >= order[previous.provenance]:
            selected[key] = item
    return selected


def _scalar(data: dict[str, Observation], *names: str) -> float | None:
    for name in names:
        item = data.get(name)
        if item and len(item.values) == 1 and isfinite(item.values[0]):
            return item.values[0]
    return None


def _comparison(name: str, actual: float, expected: float, tolerance: float,
                evidence: str) -> CheckResult:
    error = abs(actual - expected) / max(abs(expected), 1e-12)
    return CheckResult(name, Status.PASS if error <= tolerance else Status.FAIL,
                       f'{evidence}; relative discrepancy {error:.3%}', {'relative_error': error})


def _relationship_result(data: dict[str, Observation], keys: tuple[str, ...],
                         name: str, actual: float, expected: float, tolerance: float,
                         evidence: str) -> CheckResult:
    result = _comparison(name, actual, expected, tolerance, evidence)
    if result.status == Status.FAIL and not all(data[key].provenance == Provenance.EXPLICIT for key in keys):
        return CheckResult(name, Status.WARNING,
                           result.reason + '; inferred roles or sampling times need confirmation', result.metrics)
    return result


class ForceMassAcceleration:
    name = 'F = ma'
    def evaluate(self, data, context):
        force = _scalar(data, 'force', 'f')
        mass = _scalar(data, 'mass', 'm')
        acceleration = _scalar(data, 'acceleration', 'a')
        if None in (force, mass, acceleration):
            return None
        keys = tuple(key for key in ('force', 'f', 'mass', 'm', 'acceleration', 'a') if key in data)
        return _relationship_result(data, keys, self.name, force, mass * acceleration, 0.02,
                                    'Observed force, mass and acceleration; assumes net force and common SI units')


class GravityRelation:
    name = 'g = GM/r^2'
    def evaluate(self, data, context):
        gravity = _scalar(data, 'g', 'gravity')
        mass = _scalar(data, 'central_mass', 'planet_mass')
        radius = _scalar(data, 'radius', 'r')
        gravitational = _scalar(data, 'gravitational_constant', 'big_g', 'G')
        if None in (gravity, mass, radius, gravitational) or radius <= 0:
            return None
        keys = tuple(key for key in ('g', 'gravity', 'central_mass', 'planet_mass', 'radius', 'r',
                                     'gravitational_constant', 'big_g', 'G') if key in data)
        return _relationship_result(data, keys, self.name, gravity, gravitational * mass / radius**2, 0.03,
                                    'Observed central mass, radius and gravitational constant; assumes Newtonian gravity and SI units')


class MomentumDefinition:
    name = 'p = mv'
    def evaluate(self, data, context):
        momentum = _scalar(data, 'momentum', 'p')
        mass = _scalar(data, 'mass', 'm')
        velocity = _scalar(data, 'velocity', 'v')
        if None in (momentum, mass, velocity):
            return None
        keys = tuple(key for key in ('momentum', 'p', 'mass', 'm', 'velocity', 'v') if key in data)
        return _relationship_result(data, keys, self.name, momentum, mass * velocity, 0.02,
                                    'Observed momentum, mass and velocity; assumes same object and frame')


class KineticEnergyDefinition:
    name = 'K = 1/2 mv^2'
    def evaluate(self, data, context):
        kinetic = _scalar(data, 'kinetic_energy', 'kinetic')
        mass = _scalar(data, 'mass', 'm')
        velocity = _scalar(data, 'velocity', 'v')
        if None in (kinetic, mass, velocity):
            return None
        keys = tuple(key for key in ('kinetic_energy', 'kinetic', 'mass', 'm', 'velocity', 'v') if key in data)
        return _relationship_result(data, keys, self.name, kinetic, 0.5 * mass * velocity**2, 0.02,
                                    'Observed kinetic energy, mass and speed; assumes classical mechanics')


class SpringForce:
    name = 'F = -kx'
    def evaluate(self, data, context):
        force = _scalar(data, 'spring_force')
        stiffness = _scalar(data, 'spring_constant', 'stiffness')
        displacement = _scalar(data, 'displacement')
        if None in (force, stiffness, displacement):
            return None
        keys = tuple(key for key in ('spring_force', 'spring_constant', 'stiffness', 'displacement') if key in data)
        return _relationship_result(data, keys, self.name, force, -stiffness * displacement, 0.02,
                                    'Observed spring force, stiffness and displacement; assumes Hooke law')


class ParameterPlausibility:
    name = 'parameter plausibility'
    def evaluate(self, data, context):
        mass = _scalar(data, 'mass')
        if mass is not None and mass < 0:
            return CheckResult(self.name, Status.WARNING,
                               'Negative mass is unusual; check model convention and units')
        gravity = _scalar(data, 'g', 'gravity')
        if gravity is not None and abs(gravity) > 100:
            return CheckResult(self.name, Status.WARNING,
                               'Large gravity value; environment and units were not established')
        if gravity is not None:
            return CheckResult(self.name, Status.UNKNOWN,
                               'Local gravity is environment-dependent; no Earth reference assumed')
        return None


class DimensionRule:
    name = 'dimensions'
    expected = {
        'mass': {'kg'}, 'force': {'N', 'kg*m/s^2'},
        'acceleration': {'m/s^2'}, 'velocity': {'m/s'},
        'momentum': {'kg*m/s'}, 'kinetic_energy': {'J', 'kg*m^2/s^2'},
        'potential_energy': {'J', 'kg*m^2/s^2'},
    }

    def evaluate(self, data, context):
        declared = [(name, item.unit) for name, item in data.items()
                    if name in self.expected and item.unit is not None]
        if not declared:
            return None
        mismatches = [(name, unit) for name, unit in declared if unit not in self.expected[name]]
        if mismatches:
            return CheckResult(self.name, Status.FAIL,
                               f'Declared units conflict with named SI quantities: {mismatches}')
        return CheckResult(self.name, Status.PASS, 'Declared units match supported SI dimensions')


class UniversalConstantsRule:
    name = 'universal constants'

    def evaluate(self, data, context):
        item = data.get('G') or data.get('gravitational_constant')
        if item is None or item.unit is None or len(item.values) != 1:
            return None
        reference = CONSTANTS['G']
        if item.unit != reference.unit:
            return CheckResult('universal constant G', Status.UNKNOWN,
                               'G was declared in units that are not directly comparable to the SI reference')
        return _comparison('universal constant G', item.values[0], reference.value,
                           reference.relative_tolerance, 'Declared SI gravitational constant')


class EarthGravityReference:
    name = 'Earth surface gravity'

    def evaluate(self, data, context):
        if context.get('environment') != 'earth_surface':
            return None
        item = data.get('g') or data.get('gravity')
        if item is None or len(item.values) != 1 or (item.unit != 'm/s^2' and not (item.unit is None and context.get('source_si_hint'))):
            return CheckResult(self.name, Status.UNKNOWN,
                               'Earth surface context declared, but comparable SI gravity was not observed')
        reference = CONSTANTS['g0']
        error = abs(item.values[0] - reference.value) / reference.value
        return CheckResult(self.name, Status.PASS if error <= 0.1 else Status.WARNING,
                           f'Compared with approximate Earth surface reference using declared units or SI source hint; discrepancy {error:.3%}',
                           {'relative_error': error})


RULES: tuple[Rule, ...] = (ForceMassAcceleration(), GravityRelation(), MomentumDefinition(),
                           KineticEnergyDefinition(), SpringForce(), ParameterPlausibility(),
                           DimensionRule(), UniversalConstantsRule(), EarthGravityReference())


def evaluate_rules(observations: Sequence[Observation], rules: Sequence[Rule] = RULES,
                   context: dict | None = None) -> tuple[CheckResult, ...]:
    data = _selected(observations)
    return tuple(result for rule in rules if (result := rule.evaluate(data, context or {})) is not None)
