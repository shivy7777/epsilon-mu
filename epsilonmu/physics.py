"""Extensible physics domain detection independent of an LLM."""

from typing import Protocol
from .analysis import Analysis


class PhysicsSkill(Protocol):
    name: str

    def detect(self, analysis: Analysis) -> bool: ...
    def applicable_checks(self, analysis: Analysis) -> tuple[str, ...]: ...


class MechanicsSkill:
    name = 'mechanics'

    def detect(self, analysis: Analysis) -> bool:
        return analysis.domain != 'unknown'

    def applicable_checks(self, analysis: Analysis) -> tuple[str, ...]:
        return ('finite states', 'energy', 'linear momentum', 'angular momentum')


SKILLS: tuple[PhysicsSkill, ...] = (MechanicsSkill(),)
