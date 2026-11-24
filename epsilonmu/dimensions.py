"""Five-base-dimension exponent algebra, independent of a units library."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Dimension:
    mass: Fraction = Fraction(0)
    length: Fraction = Fraction(0)
    time: Fraction = Fraction(0)
    current: Fraction = Fraction(0)
    temperature: Fraction = Fraction(0)

    def __mul__(self, other: 'Dimension') -> 'Dimension':
        return Dimension(*(a + b for a, b in zip(self.components, other.components)))

    def __truediv__(self, other: 'Dimension') -> 'Dimension':
        return Dimension(*(a - b for a, b in zip(self.components, other.components)))

    def __pow__(self, exponent: int | Fraction) -> 'Dimension':
        return Dimension(*(part * Fraction(exponent) for part in self.components))

    @property
    def components(self) -> tuple[Fraction, ...]:
        return (self.mass, self.length, self.time, self.current, self.temperature)


MASS = Dimension(mass=Fraction(1))
LENGTH = Dimension(length=Fraction(1))
TIME = Dimension(time=Fraction(1))
CURRENT = Dimension(current=Fraction(1))
TEMPERATURE = Dimension(temperature=Fraction(1))
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
MOMENTUM = MASS * VELOCITY
