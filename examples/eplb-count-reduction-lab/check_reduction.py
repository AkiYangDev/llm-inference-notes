"""Exact integer comparisons for a proposed statistics reduction, on CPU."""

import itertools
import json
import random


def make_counts(ranks, steps, layers, experts, seed):
    rng = random.Random(seed)
    maps = [list(range(experts)) + [layer % experts, (layer + 1) % experts]
            for layer in range(layers)]
    counts = [[[[rng.randrange(100) if step % 3 else 0
                 for _ in mapping] for mapping in maps]
               for step in range(steps)] for _ in range(ranks)]
    return counts, maps


def full_then_sum(counts, maps, experts):
    ranks, steps, layers = len(counts), len(counts[0]), len(maps)
    records = [[[0] * experts for _ in range(layers)] for _ in range(steps)]
    for r, s, l in itertools.product(range(ranks), range(steps), range(layers)):
        for physical, logical in enumerate(maps[l]):
            records[s][l][logical] += counts[r][s][l][physical]
    return [[sum(records[s][l][e] for s in range(steps))
             for e in range(experts)] for l in range(layers)]


def local_sum_then_reduce(counts, maps, experts):
    total = [[0] * experts for _ in maps]
    for rank_counts in counts:
        # Local sum on physical slots, then scatter to logical experts.
        local = [[sum(record[l][p] for record in rank_counts)
                  for p in range(len(mapping))] for l, mapping in enumerate(maps)]
        for l, mapping in enumerate(maps):
            for p, logical in enumerate(mapping):
                total[l][logical] += local[l][p]
    return total


def main():
    comparisons = 0
    for ranks, steps, layers, experts, seed in itertools.product(
        (1, 2, 4), (1, 2, 7, 50), (1, 3), (2, 5), (0, 1, 2)
    ):
        counts, maps = make_counts(ranks, steps, layers, experts, seed)
        a = full_then_sum(counts, maps, experts)
        b = local_sum_then_reduce(counts, maps, experts)
        # Independent scalar reference, using a different nesting order.
        oracle = [[sum(counts[r][s][l][p]
                       for p, logical in enumerate(maps[l]) if logical == e
                       for s in range(steps) for r in range(ranks))
                   for e in range(experts)] for l in range(layers)]
        assert a == b == oracle
        comparisons += 1

    same_totals = [[100, 0], [0, 100]]
    concurrent = [[50, 50], [50, 50]]
    assert list(map(sum, zip(*same_totals))) == list(map(sum, zip(*concurrent)))
    assert sum(max(row) for row in same_totals) == 200
    assert sum(max(row) for row in concurrent) == 100

    # The physical-to-logical map changes between these two records.
    records, per_record_maps = [[10, 0], [0, 10]], [[0, 1], [1, 0]]
    correct = [0, 0]
    for row, mapping in zip(records, per_record_maps):
        for p, e in enumerate(mapping):
            correct[e] += row[p]
    stale_map_result = [10, 10]
    assert correct == [20, 0] and correct != stale_map_result

    full = 1000 * 43 * 256 * 4
    compact = 43 * 256 * 8
    exact_sum = 3_000_000 * 1000
    wrapped_int32 = (exact_sum + 2**31) % 2**32 - 2**31
    assert exact_sum > 2**31 - 1 and wrapped_int32 < 0
    return dict(scope="CPU exact-integer study; no collective or device timing",
                equivalence_cases=comparisons, full_int32_payload_bytes=full,
                compact_int64_payload_bytes=compact,
                payload_ratio=full // compact,
                equal_totals_different_peak_sums=[200, 100],
                changing_map_correct=correct, changing_map_wrong=stale_map_result,
                overflow_example_exact=exact_sum, overflow_example_int32=wrapped_int32)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
