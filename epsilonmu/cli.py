"""Command line entry point."""

import argparse
import json
from pathlib import Path

from .analysis import inspect_file
from .agent import investigate_timestep
from .pipeline import verify_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='epsilonmu')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'test', 'verify', 'debug'):
        command = sub.add_parser(name)
        command.add_argument('script', type=Path)
        if name != 'inspect':
            command.add_argument('--timeout', type=float, default=10)
        if name == 'debug':
            command.add_argument('--max-experiments', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.script.is_file():
        parser.error(f'not a file: {args.script}')
    try:
        if args.command == 'inspect':
            item = inspect_file(args.script)
            print(f'System: {item.domain}\nSolver: {item.solver}\nTimestep: {item.timestep}\nImports: {", ".join(item.imports)}')
            print(f'Confidence: {item.classification.confidence:.2f}')
            if item.classification.evidence:
                print('Evidence: ' + '; '.join(item.classification.evidence))
            return 0
        report = verify_file(args.script, args.timeout)
        print(f'EpsilonMu | {args.script.name} | {report.analysis.domain}')
        for check in report.checks:
            print(f'  {check.status.value:7} {check.name}: {check.reason}')
        print(f'Status: {report.status.value}')
        if args.command == 'debug':
            investigation = investigate_timestep(args.script, args.max_experiments, args.timeout,
                                                 baseline=report)
            print('Agent investigation:')
            for experiment in investigation.experiments:
                print(f'  dt={experiment.timestep:g}; energy drift={experiment.relative_drift:.3%}')
            print(f'  {investigation.explanation}')
        return 1 if report.status.value == 'FAIL' else 0
    except (OSError, SyntaxError, ValueError) as exc:
        parser.exit(2, f'epsilonmu: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
