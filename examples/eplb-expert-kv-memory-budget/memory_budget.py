"""Idealized DeepSeek-V3 expert/KV byte accounting; not a serving benchmark.

Run from the repository or extracted archive root:
    python examples/eplb-expert-kv-memory-budget/memory_budget.py --output results

Model dimensions: deepseek-ai/DeepSeek-V3 @ e815299b0bcbac849fa540c768ef21845365c9eb.
Assumptions: PP1, EP16, unsharded expert matrices, 1 byte/weight,
BF16 compressed MLA on all 61 local layers, no sequence sharding.
Scales, padding, allocator rounding, prefix sharing, speculative reservations
and page tails are excluded. All other allocations have already been deducted
from the 40 GiB baseline KV budget. A row is a local cache token, not a
cluster-wide concurrency slot. No runtime or accelerator dependency is used.
"""
import argparse
import csv
import json
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20


def calculate():
    hidden, intermediate, moe_layers, ep = 7168, 2048, 58, 16
    one_expert_layer_bytes = 3 * hidden * intermediate
    kv_bytes_per_token = 61 * (512 + 64) * 2
    context_tokens = 16 * 1024 + 1024
    rows = []
    for redundant in (0, 16, 32):
        assert (256 + redundant) % ep == 0
        extra_bytes = redundant // ep * moe_layers * one_expert_layer_bytes
        kv_bytes = 40 * GIB - extra_bytes
        tokens = kv_bytes // kv_bytes_per_token
        rows.append(dict(
            redundant_per_layer_ep_group=redundant,
            physical_slots_per_layer_ep_group=256 + redundant,
            added_experts_per_layer_rank=redundant // ep,
            added_weight_bytes_per_rank=extra_bytes,
            kv_bytes_per_rank=kv_bytes,
            ideal_local_kv_tokens=tokens,
            full_17k_sequences=tokens // context_tokens,
        ))
    # Independent integer expectations for the published worked example.
    assert one_expert_layer_bytes == 44_040_192
    assert kv_bytes_per_token == 70_272
    assert [r['added_weight_bytes_per_rank'] for r in rows] == [0, 2_554_331_136, 5_108_662_272]
    assert [r['ideal_local_kv_tokens'] for r in rows] == [611191, 574842, 538493]
    assert [r['full_17k_sequences'] for r in rows] == [35, 33, 30]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = calculate()
    with (args.output / 'capacity.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        kind='idealized arithmetic, not device measurements',
        one_expert_one_moe_layer_bytes=44_040_192,
        bf16_mla_bytes_per_local_token=70_272,
        context_tokens=17_408,
        stat_buffer_61_layers_w1000_r16_bytes=1000 * 61 * 272 * 4,
        stat_buffer_r0_to_r16_delta_bytes=1000 * 61 * 16 * 4,
        capacity_limited_example=dict(
            baseline_batch=35, baseline_step_ms=35,
            candidate_batch=33, candidate_step_ms=33,
            baseline_tokens_s=35 / (35 / 1000),
            candidate_tokens_s=33 / (33 / 1000),
            note='Teaching assumptions: ordinary decode, one output per request per step; no DSpark or update cost.'),
        rows=rows,
    )
    (args.output / 'budget.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Arithmetic only; bytes exclude scales, padding and runtime allocations.')
    print('R/group  added MiB/rank  KV GiB/rank  local tokens  17K sequences')
    for r in rows:
        print(f"{r['redundant_per_layer_ep_group']:>7}  {r['added_weight_bytes_per_rank']/MIB:>14.0f}  "
              f"{r['kv_bytes_per_rank']/GIB:>11.3f}  {r['ideal_local_kv_tokens']:>12}  {r['full_17k_sequences']:>13}")


if __name__ == '__main__':
    main()
