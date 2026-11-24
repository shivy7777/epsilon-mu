"""Dependency edges point from input quantity to derived quantity."""

from collections import deque
from dataclasses import dataclass

from .extractor import Relationship


@dataclass(frozen=True)
class DependencyGraph:
    edges: tuple[tuple[str, str], ...]

    def path(self, source: str, target: str) -> tuple[str, ...]:
        adjacency: dict[str, set[str]] = {}
        for start, end in self.edges:
            adjacency.setdefault(start, set()).add(end)
        queue = deque([(source, (source,))])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, (*path, neighbor)))
        return ()


def build_graph(relationships: tuple[Relationship, ...]) -> DependencyGraph:
    edges = {(dependency, item.target) for item in relationships
             for dependency in item.dependencies if dependency != item.target}
    return DependencyGraph(tuple(sorted(edges)))
