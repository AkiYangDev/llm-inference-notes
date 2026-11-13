"""CPU fault examples for expert migration; no accelerator/graph implementation.

Each toy expert has two unsigned weight bytes and a separate scale byte.
Its output is scale * dot(weight, input), combined with fixed logical routes.
Captured references model storage identity only, not an actual NPU graph.
All arithmetic uses integers/Fraction. The canary is deliberately allowed to
move correctly while an individual data path is faulty, to show its scope.

Run from repository or extracted package root:
    python examples/eplb-weight-migration-correctness/migration_checks.py --output results
"""
import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

OLD=['A','B','C']
NEW=['B','C','A']
Q={'A':bytes([2,4]),'B':bytes([6,8]),'C':bytes([10,12])}
S={'A':bytes([1]),'B':bytes([2]),'C':bytes([3])}
INPUTS=[((1,2),['A','B']),((3,1),['B','C']),((2,3),['A','C'])]


def evaluate(q,scale,layout):
    result=[]
    for x,route in INPUTS:
        total=Fraction(0)
        for logical in route:
            slot=layout.index(logical)
            total += Fraction(scale[slot][0]*sum(a*b for a,b in zip(q[slot],x)),len(route))
        result.append(str(total))
    return result


def run_case(case):
    q=[bytearray(Q[e]) for e in OLD]
    scales=[bytearray(S[e]) for e in OLD]
    captured_q,captured_scales=list(q),list(scales)
    original_ids=tuple(map(id,q+scales))
    baseline=evaluate(q,scales,OLD)
    if case=='unsafe_cycle':
        # Deliberately wrong: later sources may already have been overwritten.
        for dst,logical in enumerate(NEW):
            src=OLD.index(logical)
            q[dst][:]=q[src]
            scales[dst][:]=scales[src]
    elif case=='replace_storage':
        q=[bytearray(Q[e]) for e in NEW]
        scales=[bytearray(S[e]) for e in NEW]
    else:
        # Q/S represent immutable snapshots indexed by logical expert.
        for dst,logical in enumerate(NEW):
            q[dst][:]=Q[logical]
            if case!='stale_scale':scales[dst][:]=S[logical]
    if case=='corrupt_weight':q[0][:]=q[0][::-1]
    layout=OLD if case=='stale_mapping' else NEW
    canary=list(NEW)
    result=dict(case=case,
                canary_matches=canary==NEW,
                main_bytes_match=all(bytes(q[i])==Q[e] for i,e in enumerate(NEW)),
                scale_bytes_match=all(bytes(scales[i])==S[e] for i,e in enumerate(NEW)),
                mapping_matches=layout==NEW,
                storage_identity_preserved=tuple(map(id,q+scales))==original_ids,
                eager_output_matches=evaluate(q,scales,layout)==baseline,
                captured_output_matches=evaluate(captured_q,captured_scales,layout)==baseline)
    if case=='correct':
        for dst,logical in enumerate(OLD):
            q[dst][:]=Q[logical]
            scales[dst][:]=S[logical]
        assert evaluate(captured_q,captured_scales,OLD)==baseline
        assert tuple(map(id,q+scales))==original_ids
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=['correct','stale_scale','stale_mapping','corrupt_weight','replace_storage','unsafe_cycle']
    rows=[run_case(c) for c in cases]
    by_name={r['case']:r for r in rows}
    assert all(v is True for k,v in rows[0].items() if k!='case')
    assert all(r['canary_matches'] for r in rows)
    assert by_name['stale_scale']['main_bytes_match'] and not by_name['stale_scale']['scale_bytes_match']
    assert by_name['stale_mapping']['main_bytes_match'] and not by_name['stale_mapping']['mapping_matches']
    assert by_name['replace_storage']['eager_output_matches'] and not by_name['replace_storage']['captured_output_matches']
    assert all(not by_name[c]['eager_output_matches'] for c in ['stale_scale','stale_mapping','corrupt_weight','unsafe_cycle'])
    with (args.output/'fault_checks.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    a,b=bytes([2,4]),bytes([4,2])
    reference_logits=[10,9,0];wrong_logits=[10,1,-50]
    facts=dict(kind='CPU teaching cases, not a device regression result',
               cyclic_layout={'old':OLD,'new':NEW},
               all_faults_return_finite_numbers=True,
               roundtrip_restore_A='PASS',
               byte_sum_collision={'sum_a':sum(a),'sum_b':sum(b),'bytes_equal':a==b,
                                   'sha256_equal':hashlib.sha256(a).digest()==hashlib.sha256(b).digest()},
               same_first_token={'argmax_reference':reference_logits.index(max(reference_logits)),
                                 'argmax_wrong':wrong_logits.index(max(wrong_logits)),
                                 'max_logit_error':max(abs(x-y) for x,y in zip(reference_logits,wrong_logits))})
    assert facts['byte_sum_collision']==dict(sum_a=6,sum_b=6,bytes_equal=False,sha256_equal=False)
    assert facts['same_first_token']==dict(argmax_reference=0,argmax_wrong=0,max_logit_error=50)
    (args.output/'checks.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for r in rows:
        print(r['case'], 'canary=',r['canary_matches'],'eager=',r['eager_output_matches'],'captured=',r['captured_output_matches'])
    print('Correct in-place move and restore A pass; all five injected faults are detected.')


if __name__=='__main__':main()
