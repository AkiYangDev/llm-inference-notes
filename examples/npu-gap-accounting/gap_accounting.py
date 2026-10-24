"""Account for recorded task intervals in one kernel_details CSV.

This is a single-file interval calculator, not a device-utilization estimator
or a multi-rank critical-path analyzer. All timestamps use microseconds.
"""

import argparse
import csv
import json
from decimal import Decimal


def number(text):
    value = Decimal(str(text).strip())
    if not value.is_finite():
        raise ValueError('timestamps and durations must be finite')
    return value


def analyze_rows(rows, start_us=None, end_us=None):
    tasks, identities = [], set()
    duplicate_count = 0
    for row in rows:
        start = number(row['Start Time(us)'])
        duration = number(row['Duration(us)'])
        if duration < 0:
            raise ValueError('duration must be nonnegative')
        if not duration:
            continue
        # Do not deduplicate anonymous rows by time alone: separate streams
        # can legitimately contain identical intervals.
        if row.get('Stream ID') and row.get('Task ID'):
            identity = (row['Stream ID'], row['Task ID'], row.get('Model ID'),
                        start, duration, row.get('Name'))
            if identity in identities:
                duplicate_count += 1
                continue
            identities.add(identity)
        wait_text = row.get('Wait Time(us)', '').strip()
        wait = number(wait_text) if wait_text else None
        if wait is not None and wait < 0:
            raise ValueError('wait must be nonnegative or missing')
        tasks.append((start, start + duration, wait))
    if not tasks:
        raise ValueError('no positive-duration task records')

    lo = number(start_us) if start_us is not None else min(x[0] for x in tasks)
    hi = number(end_us) if end_us is not None else max(x[1] for x in tasks)
    if hi <= lo:
        raise ValueError('window end must be greater than window start')
    # Subtract Decimal origins before converting any timestamp to float.
    clipped = [(max(s, lo)-lo, min(e, hi)-lo, w)
               for s, e, w in tasks if s < hi and e > lo]
    merged = []
    for s, e, _ in sorted(clipped, key=lambda task: (task[0], task[1])):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    zero = Decimal(0)
    wall = hi-lo
    busy = sum((e-s for s, e in merged), zero)
    duration_sum = sum((e-s for s, e, _ in clipped), zero)
    waits_known = all(w is not None for _, _, w in clipped)
    wait_sum = sum((w for _, _, w in clipped), zero) if waits_known else None
    ms = lambda value: float(value / 1000) if value is not None else None
    return {
        'scope': 'one CSV, all supplied device-task records; not all-rank idle',
        'recorded_unique_tasks': len(tasks), 'included_tasks': len(clipped),
        'duplicate_rows_removed': duplicate_count,
        'wall_ms': ms(wall), 'duration_sum_ms': ms(duration_sum),
        'recorded_busy_union_ms': ms(busy), 'recorded_gap_ms': ms(wall-busy),
        'gap_ratio': float((wall-busy)/wall),
        'recorded_wait_sum_ms': ms(wait_sum),
        'duration_plus_wait_sum_ms': ms(duration_sum+wait_sum) if waits_known else None,
        'wait_scope': 'full recorded wait per overlapping task, not clipped elapsed time',
        'prelaunch_gap_ms': ms(merged[0][0]) if merged else ms(wall),
        'tail_gap_ms': ms(wall-merged[-1][1]) if merged else 0.0,
        'internal_gap_ms': ms(sum((b[0]-a[1] for a, b in zip(merged, merged[1:])), zero)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_path')
    parser.add_argument('--start-us', help='explicit absolute window start')
    parser.add_argument('--end-us', help='explicit absolute window end')
    args = parser.parse_args()
    with open(args.csv_path, encoding='utf-8-sig', newline='') as f:
        result = analyze_rows(csv.DictReader(f), args.start_us, args.end_us)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
