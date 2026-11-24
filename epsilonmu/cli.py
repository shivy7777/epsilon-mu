"""Command line entry point."""

import argparse
from pathlib import Path

from .analysis import inspect_file
from .agent import investigate_timestep
from .pipeline import verify_file
from .physics_rules import verify_relationships, relationship_coverage


def show_coverage(analysis, checks):
    coverage=relationship_coverage(analysis,checks)
    print('Governing relationships checked: '+(', '.join(coverage['checked']) or 'none'))
    print('Unverified registry relationships: '+(', '.join(coverage['unverified']) or 'none listed'))


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
            command.add_argument('--agent', action='store_true')
            command.add_argument('--max-steps',type=int,default=4)
            command.add_argument('--max-attempts',type=int,default=4)
            command.add_argument('--max-tokens',type=int,default=16000)
            command.add_argument('--deadline',type=float,default=60)
            command.add_argument('--max-cost')
            command.add_argument('--rates',nargs=2,metavar=('INPUT','OUTPUT'))
    args = parser.parse_args(argv)
    if args.command in ('configure','config','providers'):
        return dispatch(args)
    if not args.script.is_file():
        parser.error(f'not a file: {args.script}')
    try:
        if getattr(args,'agent',False):
            import json
            from .ai.client import Client
            from .ai.configuration import load
            from .ai.budget import Budget
            from .investigation import investigate
            config=load()
            budget=Budget(args.max_attempts,args.max_tokens,args.max_cost,args.rates,args.deadline)
            def confirm(preview):
                print(preview)
                try:
                    return input('Approve this action? [y/N] ').strip().lower()=='y'
                except EOFError:
                    return False
            state=investigate(args.script,Client(config,budget=budget),confirm,confirm,
                              max_steps=args.max_steps,timeout=args.timeout)
            print(json.dumps(state,indent=2,ensure_ascii=True))
            return 1 if state['overall']=='FAILED' else 0 if state['overall'] else 2
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
            show_coverage(item,physical)
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
        show_coverage(report.analysis,report.checks)
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
