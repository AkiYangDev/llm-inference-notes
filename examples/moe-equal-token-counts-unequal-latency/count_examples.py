"""Arithmetic examples for equal expert-route totals and different execution shapes.

This is not an NPU benchmark or a kernel simulator. A 64-row tile is a
teaching assumption; no Ascend or DeepEP kernel alignment is inferred from it.
The timeline assumes both ranks must arrive before a one-ms transfer begins.
Real communication can pipeline and overlap. Times below are invented for
the dependency example and are explicitly identified as such in the article.

Run from repository or extracted ZIP root:
    python examples/moe-equal-token-counts-unequal-latency/count_examples.py --output results
"""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = {'four_large': [256]*4 + [0]*12,
                'sixteen_equal': [64]*16,
                'one_large_fifteen_tails': [1009] + [1]*15}
    rows = []
    for name, counts in profiles.items():
        rows.append(dict(profile=name, logical_route_rows=sum(counts),
                         active_experts=sum(x > 0 for x in counts),
                         assumed_tile_m=64,
                         tile_rows=sum(((x+63)//64)*64 for x in counts),
                         expert_counts=';'.join(map(str, counts))))
    assert [r['logical_route_rows'] for r in rows] == [1024]*3
    assert [r['active_experts'] for r in rows] == [4,16,16]
    assert [r['tile_rows'] for r in rows] == [1024,1024,1984]
    write_csv(args.output/'count_profiles.csv', rows)
    # Hypothetical event dependencies, not sampled device data.
    events = []
    for rank, ready in [(0,2),(1,5)]:
        segments = [('prepare',0,ready)]
        if ready < 5:
            segments.append(('wait_for_peer',ready,5))
        segments.extend([('transfer',5,6),('expert_gemm',6,9),('combine',9,10)])
        events.extend(dict(rank=rank,phase=phase,start_ms=start,end_ms=end)
                      for phase,start,end in segments)
    assert sum(e['end_ms']-e['start_ms'] for e in events if e['rank']==0)==10
    assert sum(e['end_ms']-e['start_ms'] for e in events if e['rank']==1)==10
    write_csv(args.output/'dependency_timeline.csv', events)
    per_step = [[1024,0],[0,1024]]
    totals = [sum(step[r] for step in per_step) for r in range(2)]
    balance = lambda values: sum(values) / len(values) / max(values)
    window_balance = balance(totals)
    step_balance = sum(map(balance, per_step)) / len(per_step)
    assert window_balance == 1.0 and step_balance == 0.5
    communication = {f'rank{rank}':6-ready for rank,ready in [(0,2),(1,5)]}
    report = dict(kind='teaching arithmetic; no device speed measurements',
                  row_tile_overhead_ratio=1984/1024,
                  routes_per_step=per_step,
                  balance_of_window_totals=window_balance,
                  average_per_step_balance=step_balance,
                  communication_interval_ms=communication,
                  expert_gemm_ms={'rank0':3,'rank1':3},
                  full_dependency_chain_ms=10,
                  no_peer_delay_chain_ms=7)
    (args.output/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('All three profiles have 1024 routes; active groups and assumed tile rows differ.')
    print('Assumed 64-row tiling: 1024 / 1024 / 1984 rows; no latency prediction.')
    print('Window balance=1.0; mean per-step balance=0.5.')
    print('Teaching timeline: communication intervals 4/1 ms; expert GEMMs both 3 ms.')


if __name__ == '__main__':
    main()
