# Runtime observation and physics recognition plan

1. Add `Observation` records with name, numeric value or bounded series, role, units if declared, and exact provenance (`static`, `stdout`, `instrumented`, `explicit`). Instrument a temporary copy through a tracing wrapper, keeping the original source untouched. Limit captured variables and samples. A timeout still bounds execution time; this is not a security sandbox.
2. Parse AST relationships into conservative hierarchical model evidence. Recognize restoring acceleration, two-axis constant-gravity integration, and radial central forces when multiple independent features agree. Keep `UNKNOWN` when evidence is weak. Preserve the old `Analysis.domain` interface as a derived label.
3. Add a registry of deterministic physics rules. Only compare redundant relationships when role assignments and values are clear; warn for environment-dependent or implausible parameters. Expose constants with SI units. Record each rule's evidence and assumptions.
4. Feed observed series and rule findings into `verify_file` and the timestep investigation. The experiment remains bounded and operates on temporary copies. No source patching.
5. Test varied names, organization, and equivalent formulas with valid negative controls; run the full existing suite and new tests. Do not inspect or modify `blind_tests`.
