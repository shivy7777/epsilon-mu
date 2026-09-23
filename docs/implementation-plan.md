# EpsilonMu implementation plan

1. Build a zero-dependency package with AST inspection, structured check results, a timed subprocess runner, and a small physics skill interface.
2. Require scripts to emit declared JSON evidence. Return `UNKNOWN` when no usable evidence exists; never infer correctness from successful execution.
3. Add correct and broken mechanics examples and pytest coverage for inspection, conservation, stability, convergence, timeouts, and end-to-end classification.
4. Run the full suite and every example pair. After Milestone 1 is stable, add a bounded timestep hypothesis experiment on temporary copies and a replaceable explanation-provider protocol.

This plan intentionally keeps arbitrary code patching and broad physical-model inference out of the first release. Both need stronger evidence contracts and isolation before they can be described as reliable.
