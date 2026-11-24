"""Bounded scientific investigation over one researcher-selected simulation."""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import investigate_timestep
from .ai.client import Evidence
from .ai.configuration import AIError
from .pipeline import verify_file
from .physics_rules import verify_relationships, relationship_coverage


@dataclass(frozen=True)
class Tool:
    argument: str | None
    output: str
    prerequisite: str
    limitation: str
    execution: str = 'read baseline'


TOOLS = {
    'inspection': Tool(None,'relationship list','parseable AST','supported expressions only'),
    'model': Tool(None,'classification','parseable AST','structural candidate only'),
    'observation': Tool(None,'numeric samples','successful observation','bounded samples'),
    'symbolic': Tool('check','check result','applicable relationship','stated assumptions only'),
    'parameter': Tool('check','check result','reference context','references can be environment dependent'),
    'conservation': Tool('check','check result','sampled invariant','drift alone cannot identify cause'),
    'stability': Tool('check','check result','sampled states','finiteness does not prove stability'),
    'refinement': Tool(None,'drift comparison','baseline energy drift','one equal-duration comparison','execute temporary copy'),
    'evidence': Tool('id','prior record','existing evidence ID','retrieval supplies no independent evidence'),
}


def group(check):
    if check.category=='physical relationship':
        return 'symbolic'
    if check.category=='plausibility':
        return 'parameter'
    if check.name in ('energy','angular momentum','linear momentum'):
        return 'conservation'
    if check.name=='finite states':
        return 'stability'
    return None


def proposal(text):
    p=json.loads(text)
    if type(p)==dict and set(p)=={'stop'} and p['stop'] is True:
        return None
    if not isinstance(p,dict) or set(p)!={'tool','prediction','arguments'}:
        raise ValueError
    tool=TOOLS[p['tool']]
    args=p['arguments']
    if not isinstance(args,dict) or set(args)!=({tool.argument} if tool.argument else set()):
        raise ValueError
    if any(not isinstance(v,str) or not 0<len(v)<=100 for v in args.values()):
        raise ValueError
    predictions=(('pass','fail','warning','inconclusive') if tool.argument=='check' else
                 ('pass','fail') if p['tool']=='refinement' else ('observe',))
    if isinstance(p['prediction'],str):
        p['prediction']=p['prediction'].lower()
    if p['prediction'] not in predictions:
        raise ValueError
    return p


def investigate(path,client,approve_send,approve_tool,*,max_steps=4,timeout=10):
    if type(max_steps)!=int or not 1<=max_steps<=20 or not 0<timeout<=120:
        raise ValueError('Invalid investigation limits.')
    path=Path(path).resolve()
    budget=client.budget
    fingerprint=lambda: hashlib.sha256(path.read_bytes()).hexdigest()
    initial=fingerprint()
    state=dict(simulation=path.name,sha256=initial,overall=None,measurements=[],findings=[],
               hypotheses=[],calls=[],unresolved=[],stop='step limit',tools={k:asdict(v) for k,v in TOOLS.items()})
    seen=set()
    try:
        if set(client.config.share)!={'measurements','hypotheses','conclusions'}:
            raise AIError('Agent mode requires sharing permissions for measurements, hypotheses and conclusions.')
        if approve_tool('Execute baseline observation of the selected trusted simulation?') is not True:
            state['stop']='cancelled'
            return state
        report=verify_file(path,min(timeout,budget.remaining()))
        if fingerprint()!=initial:
            state['stop']='simulation changed'
            return state
        state['overall']=report.outcome.value
        analysis=report.analysis
        state['model']=analysis.domain
        symbolic=verify_relationships(analysis)
        state['coverage']=relationship_coverage(analysis,report.checks)
        state['measurements']=[dict(category='BASELINE_OBSERVATION',name=o.name,provenance=o.provenance.value,count=len(o.values),
                                    last=o.values[-1]) for o in report.observations if o.values]
        for i,c in enumerate(report.checks):
            tool=group(c)
            if tool=='symbolic' and c not in symbolic:
                tool=None
            state['findings'].append(dict(id=f'B{i+1}',tool=tool,check=c.name,status=c.status.value,
                                          category='CONTEXTUAL_REFERENCE' if tool=='parameter' else 'BASELINE_CHECK',
                                          reason=c.reason,metrics=c.metrics))
        state['unresolved']=[x['id'] for x in state['findings'] if x['status'] in ('WARNING','UNKNOWN','FAIL')]
        if analysis.domain=='unknown':
            state['stop']='unsupported model'
            return state
        instructions=('Encode a JSON proposal inside the commentary string. Choose a testable hypothesis '
                      'by predicting a named check outcome. Format: {"tool":"conservation",'
                      '"arguments":{"check":"energy"},"prediction":"fail"}. Use pass/fail/warning/inconclusive for checks '
                      'and refinement (pass means drift falls); observe for data retrieval. '
                      'Use {"stop":true} for insufficient evidence. Tool argument names: '+
                      json.dumps({k:v.argument for k,v in TOOLS.items()}))
        for _ in range(max_steps):
            budget.remaining()
            # Keep full records locally. Avoid resending duplicate baseline records
            # when subsequent steps refer to the same observation.
            evidence=Evidence(measurements=tuple(json.dumps([x['name'],x['last'],x['provenance']])
                                                 for x in state['measurements'][:24]),
                              conclusions=(state['overall'],analysis.domain,
                                           *(json.dumps([x['id'],x['tool'],x['check'],x['status'],x['metrics']])
                                             for x in state['findings'][:32]),
                                           *(json.dumps([x['id'],x['tool'],x['output']])[:1400]
                                             for x in state['calls'][-2:])),
                              hypotheses=tuple(json.dumps([x['id'],x['explanation'],x['assessment'],x['evidence']])
                                               for x in state['hypotheses']))
            response=client.propose(evidence,approve_send,instructions)
            try:
                p=proposal(response.commentary)
            except (KeyError,ValueError,TypeError,RecursionError):
                state['stop']='invalid proposal'
                break
            if p is None:
                state['stop']='insufficient evidence or no further test proposed'
                break
            name,args=p['tool'],p['arguments']
            if TOOLS[name].argument=='check' and not any(x['tool']==name and x['check']==args['check']
                                                         for x in state['findings']):
                state['stop']='unavailable check'
                break
            if name=='evidence' and not any(x['id']==args['id'] for x in state['findings']+state['calls']):
                state['stop']='unknown evidence ID'
                break
            signature=(name,tuple(sorted(args.items())))
            if signature in seen:
                state['stop']='repeated action'
                break
            seen.add(signature)
            h=None
            if p['prediction']!='observe':
                h=dict(id=f'H{len(state["hypotheses"])+1}',origin='model proposal',category='MODEL_HYPOTHESIS',
                       explanation=(f'{args["check"]} will return {p["prediction"]}' if 'check' in args
                                    else 'Equal-duration timestep refinement will '+('reduce drift' if p['prediction']=='pass' else 'leave drift unchanged or greater')),
                       required_steps=[name],prediction=p['prediction'].upper(),assessment='proposed',
                       evidence=[],supporting=[],contradicting=[],uncertainty=TOOLS[name].limitation)
                state['hypotheses'].append(h)
            if approve_tool(json.dumps(p,ensure_ascii=True)) is not True:
                state['stop']='cancelled'
                break
            budget.remaining()
            record_id=f'T{len(seen)}'
            status=None
            category='BASELINE_CHECK'
            if TOOLS[name].argument=='check':
                matches=[x for x in state['findings'] if x['tool']==name and x['check']==args['check']]
                output=matches
                if len(matches)==1:
                    status=matches[0]['status']
                    category=matches[0]['category']
            elif name=='refinement':
                if fingerprint()!=initial:
                    state['stop']='simulation changed'
                    break
                exp=investigate_timestep(path,1,min(timeout,budget.remaining()),baseline=report)
                output=dict(baseline_dt=analysis.timestep,baseline_drift=exp.baseline_drift,
                            experiments=[asdict(x) for x in exp.experiments],reason=exp.explanation)
                if exp.experiments and exp.baseline_drift is not None:
                    category='NEW_EXPERIMENT'
                    status='PASS' if exp.experiments[0].relative_drift<exp.baseline_drift else 'FAIL'
                else:
                    category=None
            elif name=='observation':
                category='BASELINE_OBSERVATION'
                output=state['measurements']
            elif name=='inspection':
                output=[dict(variable=r.target,line=r.line,expression=str(r.expression)) for r in analysis.relationships[:32]]
            elif name=='model':
                output=asdict(analysis.classification)
            else:
                retrieved=next(x for x in state['findings']+state['calls'] if x['id']==args['id'])
                category=retrieved['category']
                output=retrieved.get('output',retrieved)
            budget.remaining()
            state['calls'].append(dict(id=record_id,tool=name,arguments=args,output=output,
                                       category=category,new_evidence=category=='NEW_EXPERIMENT' and name=='refinement'))
            if p['prediction']!='observe':
                predicted=p['prediction'].upper()
                comparable='INCONCLUSIVE' if status=='UNKNOWN' else status
                assessment=('supported' if comparable==predicted else
                            'unresolved' if comparable not in ('PASS','FAIL') else 'contradicted')
                h.update(assessment=assessment,evidence=[record_id],basis=category,
                    supporting=[record_id] if assessment=='supported' else [],
                    contradicting=[record_id] if assessment=='contradicted' else [],
                    uncertainty=TOOLS[name].limitation)
                if assessment=='unresolved':
                    state['unresolved'].append(record_id)
    except KeyboardInterrupt:
        state['stop']='cancelled'
    except AIError as exc:
        state['stop']=str(exc)
    finally:
        state['authorizations']=list(client.authorizations)
        state['budget']=dict(attempts=budget.attempts,input_tokens=budget.reported_input_tokens,
                             output_tokens=budget.reported_output_tokens,unknown_usage=budget.unknown_usage,
                             estimated_cost=None if budget.estimated_cost is None else str(budget.estimated_cost))
    return state
