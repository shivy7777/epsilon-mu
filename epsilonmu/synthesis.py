"""Terminal evidence synthesis. Only host-derived statements may be highlighted."""
import json

from .ai.client import Evidence
from .ai.providers import ProposalError


def summary(state):
    records = []

    def add(ref, kind, category, statement, *related):
        records.append(dict(claim=f'{ref}.{kind}', evidence=[ref, *related],
                            category=category, statement=statement))

    for finding in state['findings']:
        ref = finding['id']
        add(ref, 'baseline', finding['category'],
            f"{finding['check']}: {finding['status']}")
        if ref in state['unresolved']:
            add(ref, 'open', 'UNRESOLVED_QUESTION',
                f"What accounts for the {finding['check']} finding?")
    for call in state['calls']:
        if not call['new_evidence']:
            continue
        ref = call['id']
        output = call['output']
        baseline = output['baseline_drift']
        for i, trial in enumerate(output['experiments'], 1):
            drift = trial['relative_drift']
            add(ref, f'measured{i}', 'MEASURED_RESULT',
                f"Relative energy drift was {baseline} at dt={output['baseline_dt']} "
                f"and {drift} at dt={trial['timestep']}.")
            comparison = 'lower' if drift < baseline else 'equal' if drift == baseline else 'higher'
            add(ref, f'{comparison}{i}', 'SUPPORTED_INTERPRETATION',
                f"The refined run had {comparison} relative drift at equal simulated duration.")
        add(ref, 'cause', 'UNRESOLVED_QUESTION',
            'What causes the baseline drift? This comparison alone cannot establish its cause.')
    for h in state['hypotheses']:
        add(h['id'], 'prediction', 'SPECULATIVE_HYPOTHESIS', h['explanation'])
        if h['evidence']:
            add(h['id'], 'assessment', 'HYPOTHESIS_ASSESSMENT',
                f"Prediction {h['assessment']} by {h['basis'] or 'insufficient evidence'}.",
                *h['evidence'])
    return dict(source='deterministic', records=records, highlights=[])


def synthesize(client, approve, result):
    records = result['records']
    allowed = {item['claim'] for item in records}
    schema = dict(type='object', additionalProperties=False, required=['highlights'],
                  properties=dict(highlights=dict(type='array', minItems=1, maxItems=6,
                                  uniqueItems=True, items=dict(type='string', enum=sorted(allowed)))))

    def decode(value):
        if not isinstance(value, dict) or set(value) != {'highlights'}:
            raise ProposalError('invalid_synthesis')
        choices = value['highlights']
        if (not isinstance(choices, list) or not 1 <= len(choices) <= 6
                or any(not isinstance(x, str) or x not in allowed for x in choices)
                or len(set(choices)) != len(choices)):
            raise ProposalError('unsupported_synthesis_claim')
        return choices

    def items(categories):
        return tuple(json.dumps([r['claim'], r['statement'], r['evidence']])
                     for r in records if r['category'] in categories)

    evidence = Evidence(measurements=items({'MEASURED_RESULT'}),
                        hypotheses=items({'SPECULATIVE_HYPOTHESIS', 'HYPOTHESIS_ASSESSMENT'}),
                        conclusions=items({'BASELINE_CHECK', 'CONTEXTUAL_REFERENCE',
                                          'SUPPORTED_INTERPRETATION', 'UNRESOLVED_QUESTION'}))
    instructions = ('Refinement completed. Select up to six existing claim identifiers that best '
                    'summarize the findings, contradictory evidence and uncertainty. '
                    'Return only {"highlights":["<claim identifier>"]}.')
    highlights = client.synthesize(evidence, approve, instructions, schema, decode)
    result.update(source='model selection', highlights=highlights)
