# v0.3 Physics Relationship Engine Plan

1. Parse supported arithmetic AST nodes into inert symbolic trees. Normalize commutative products, divisions, powers, `pow`, and square roots without evaluating target code. Store source lines, dependencies, and update relationships.
2. Build a dependency graph from extracted relationships. Extract textual context from docstrings, comments, function names, and string metadata as fallible evidence.
3. Define mechanics relationship specifications with assumptions, dimensions, and verification methods. Apply a rule only when structural evidence and context establish its intended model. Start with radial gravity and direct algebraic mechanics relationships; return unknown for ambiguous equations.
4. Add a small five-base-dimension algebra. Improve model recognition with relationship and context evidence, retaining conservative unknown classifications.
5. Introduce `VERIFIED`, `FAILED`, and `INCONCLUSIVE` as report outcomes. Only supported physical checks can yield `VERIFIED`; numerical finiteness alone remains `INCONCLUSIVE`. Keep individual `PASS`, `FAIL`, `WARNING`, `UNKNOWN` checks and preserve existing APIs where possible.
6. Show relationships, dependencies, grouped checks, and diagnoses in the CLI. Add varied equivalence and negative-control tests. Run all tests without inspecting or editing `blind_tests`.
