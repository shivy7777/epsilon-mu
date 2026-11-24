"""Command line entry point."""

import argparse
from pathlib import Path

from .analysis import inspect_file
from .agent import investigate_timestep
from .pipeline import verify_file
from .physics_rules import verify_relationships


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='epsilonmu')
    sub = parser.add_subparsers(dest='command', required=True)
    from .ai.cli import register, dispatch
    register(sub)
    for name in ('inspect', 'test', 'verify', 'debug'):
        command = sub.add_parser(name)
        command.add_argument('script', type=Path)
        if name != 'inspect':
            command.add_argument('--timeout', type=float, default=10)
        if name == 'debug':
            command.add_argument('--max-experiments', type=int, default=2)
    args = parser.parse_args(argv)
    if args.command in ('configure','config','providers'):
        return dispatch(args)
    if not args.script.is_file():
        parser.error(f'not a file: {args.script}')
    try:
        if args.command == 'inspect':
            item = inspect_file(args.script)
            model = item.classification.path[-1].replace(' ', '_') if item.classification.path else 'unknown'
            domain = item.classification.path[1] if len(item.classification.path) > 1 else 'unknown'
            print(f'Domain: {domain}\nModel: {model}\nSolver: {item.solver}\nTimestep: {item.timestep}')
            print(f'Confidence: {item.classification.confidence:.2f}')
            if item.classification.evidence:
                print('Evidence: ' + '; '.join(item.classification.evidence))
            if item.classification.assumptions:
                print('Assumptions: ' + '; '.join(item.classification.assumptions))
            print('Relationships:')
            for binding in item.roles:
                print(f'  {binding.variable}: candidate {binding.role} ({binding.evidence})')
            displayed = sorted(item.relationships, key=lambda relation: relation.expression.op == 'number')
            for relationship in displayed[:8]:
                print(f'  {relationship.target} = {relationship.expression}  [line {relationship.line}]')
            if len(displayed) > 8:
                print(f'  ... {len(displayed) - 8} more')
            print('Physics checks:')
            physical = verify_relationships(item)
            for check in physical[:8]:
                print(f'  {check.status.value} {check.name}: {check.reason}')
            if not physical:
                print('  UNKNOWN: no context-supported relationship comparison')
            print('Dependencies:')
            for source, target in item.graph.edges[:8]:
                print(f'  {source} -> {target}')
            if len(item.graph.edges) > 8:
                print(f'  ... {len(item.graph.edges) - 8} more')
            return 0
        report = verify_file(args.script, args.timeout)
        print(f'EpsilonMu | {args.script.name} | {report.analysis.domain}')
        for category, heading in (('runtime', 'Runtime checks'), ('physical relationship', 'Physical relationships'),
                                  ('plausibility', 'Plausibility'), ('numerical', 'Numerical checks')):
            group = [check for check in report.checks if check.category == category]
            if group:
                print(f'{heading}:')
                for check in group:
                    print(f'  {check.status.value:7} {check.name}: {check.reason}')
        print(f'Overall: {report.outcome.value}')
        if args.command == 'debug':
            investigation = investigate_timestep(args.script, args.max_experiments, args.timeout,
                                                 baseline=report)
            print('Agent investigation:')
            for hypothesis in investigation.state.hypotheses:
                print(f'  H  {hypothesis.claim}: {hypothesis.status}')
            for experiment in investigation.experiments:
                print(f'  dt={experiment.timestep:g}; energy drift={experiment.relative_drift:.3%}')
            print(f'  {investigation.explanation}')
        return 1 if report.outcome.value == 'FAILED' else 0
    except (OSError, SyntaxError, ValueError) as exc:
        parser.exit(2, f'epsilonmu: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
