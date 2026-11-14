"""CPU teaching models for the EPLB reading series; no device benchmark."""
from collections import Counter
import argparse
import json
from pathlib import Path

MIB = 1024**2
GIB = 1024**3


def stable_slots(old, desired):
    remaining = Counter(desired)
    result = [None] * len(old)
    for i, expert in enumerate(old):
        if remaining[expert]:
            result[i] = expert
            remaining[expert] -= 1
    pending = list(remaining.elements())
    for i in range(len(result)):
        if result[i] is None:
            result[i] = pending.pop(0)
    assert not pending and Counter(result) == Counter(desired)
    return result


def summarize_moves(old, new, bytes_per_expert):
    assert len(old) == len(new)
    changed = 0
    missing = []
    for before, after in zip(old, new, strict=True):
        assert len(before) == len(after)
        changed += sum(a != b for a, b in zip(before, after, strict=True))
        missing.append(sorted(set(after) - set(before)))
    introduced = sum(map(len, missing))
    return dict(changed_slots=changed, remote_new_experts_per_rank=missing,
                remote_payload_bytes=introduced * bytes_per_expert)


def calculate():
    load = dict(A=120, B=50, C=20, D=10)
    old_load = [load['A'] + load['D'], load['B'] + load['C']]
    new_load = [50 + 40 + 10, 40 + 40 + 20]
    assert old_load == [130, 70] and new_load == [100, 100]

    old = [['A', 'B', 'C'], ['D', 'E', 'F']]
    local_permutation = [['C', 'A', 'B'], ['F', 'D', 'E']]
    remote_swap = [['D', 'B', 'C'], ['A', 'E', 'F']]
    stabilized = [stable_slots(a, b) for a, b in zip(old, local_permutation, strict=True)]
    assert stabilized == old
    local = summarize_moves(old, local_permutation, 40 * MIB)
    swapped = summarize_moves(old, remote_swap, 40 * MIB)
    assert local['changed_slots'] == 6 and local['remote_payload_bytes'] == 0
    assert swapped['changed_slots'] == 2 and swapped['remote_payload_bytes'] == 80 * MIB
    assert stable_slots(['A', 'B', 'C'], ['A', 'A', 'D']) == ['A', 'A', 'D']
    # A snapshot is required when an in-place permutation overwrites future sources.
    snapshot = ['A', 'B']
    correct = [snapshot[1], snapshot[0]]
    wrong = ['A', 'B']
    wrong[0] = wrong[1]
    wrong[1] = wrong[0]
    assert correct == ['B', 'A'] and wrong == ['B', 'B']
    sender_only = summarize_moves([['A', 'B'], ['C', 'D'], ['E', 'E']],
                                  [['A', 'B'], ['A', 'D'], ['C', 'E']], 40 * MIB)
    assert sender_only['remote_new_experts_per_rank'][1] == ['A']

    clocks = []
    for role, step_ms, interval, window in [('P',80,50,50),('D',35,1000,1000),
                                          ('D',35,4000,1000),('D',35,4000,4000)]:
        clocks.append(dict(role=role, step_ms=step_ms, I=interval, W=window,
                           nominal_check_s=interval*step_ms/1000,
                           nominal_history_s=window*step_ms/1000))

    bounds = []
    for p, speed in [(0.1,2),(0.3,2),(0.6,2),(0.3,None),(0.3,1.6)]:
        fraction = 1-p+(p/speed if speed is not None else 0)
        bounds.append(dict(affected_fraction=p, local_speedup=speed,
                           new_time_fraction=fraction, rate_gain_pct=(1/fraction-1)*100))
    assert abs(bounds[1]['new_time_fraction']-0.85) < 1e-12
    interval_ms = 4000 * 35
    net_saved_ms = 4000 * 0.2 - 600
    assert net_saved_ms == 200

    H,I = 4096,2048
    primary = 3*H*I
    scales = (2*I+H)*4
    expert = primary+scales
    payload = expert*8*20
    assert primary == 24*MIB and scales == 32*1024 and payload/MIB == 3845
    assert 17*64*1 == 17*16*4 == 1088
    return dict(
        scope='Teaching arithmetic only. No inferred accelerator latency or new service experiment.',
        primer=dict(logical_load=load, original_rank_load=old_load, replicated_rank_load=new_load),
        movement=dict(local_permutation=local, stable_slot_layout=stabilized,
                      remote_swap=swapped, sender_only_case=sender_only,
                      snapshot_permutation=correct, overwritten_source=wrong),
        uniform_clock_assumption=clocks,
        critical_path_bounds=bounds,
        amortization=dict(update_ms=600, saved_per_forward_ms=0.2,
                          break_even_forwards=3000, interval_forwards=4000,
                          original_interval_ms=interval_ms, net_saved_ms=net_saved_ms),
        byte_budget=dict(H=H,I=I,main_bytes_per_element=1,
                         teaching_scale_rule='One FP32 scale per output channel',
                         main_bytes=primary,scale_bytes=scales,expert_bytes=expert,
                         new_remote_experts_per_layer=8,layers=20,
                         global_sent_payload_bytes=payload,
                         shared_channel_assumption_GiB_s=25,
                         shared_channel_lower_bound_ms=payload/(25*GIB)*1000,
                         independent_senders_GiB=[1.5,0.5,0.5,0.5],
                         sender_rate_GiB_s=20,slowest_sender_lower_bound_ms=75,
                         aggregate_only_estimate_ms=37.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('models.json'))
    args=parser.parse_args()
    args.output.write_text(json.dumps(calculate(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified placement loads, slot reuse, source protection, clock scales, speedup bounds and byte budgets.')
