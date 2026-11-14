"""CPU byte arithmetic and metadata examples; no NPU operations are simulated."""

import json
from math import prod


def nd_block_bytes(shape, strides, itemsize, capacity, offset=0):
    """Teaching implementation of dense expert-row checks, in byte units."""
    if not shape or prod(shape) == 0 or offset != 0:
        raise ValueError("expected a nonempty expert prefix at offset zero")
    dense = 1
    for stride, size in sorted(zip(strides[1:], shape[1:])):
        if size > 1:
            if stride != dense:
                raise ValueError("gaps or overlapping elements inside an expert")
            dense *= size
    if strides[0] != dense:
        raise ValueError("expert rows must be independent contiguous blocks")
    block = strides[0] * itemsize
    if capacity % block or shape[0] * block > capacity:
        raise ValueError("capacity cannot contain the declared expert rows")
    return block, capacity // block


def nz_metadata_checks(weight, full, storage_elements):
    """Mirror NZ metadata conditions, without emulating a device descriptor."""
    if len(weight["shape"]) != 3 or len(full["shape"]) != 3:
        return False
    return (
        weight["itemsize"] == full["itemsize"]
        and weight["stride0"] == full["stride0"]
        and prod(full["shape"]) == full["shape"][0] * weight["stride0"]
        and full["shape"][0] > 0
        and storage_elements % full["shape"][0] == 0
    )


def main():
    capacity = 6 * 17 * 64
    # One byte array, two descriptions: int8 [6,17,64], int32 [6,17,16].
    original = nd_block_bytes((6, 17, 64), (1088, 64, 1), 1, capacity)
    packed = nd_block_bytes((6, 17, 16), (272, 16, 1), 4, capacity)
    assert original == packed == (1088, 6)
    assert nd_block_bytes((6, 17, 16), (272, 16, 1), 4, 102 * 16 * 4) == packed
    assert nd_block_bytes((5, 17, 16), (272, 16, 1), 4, capacity) == packed

    data = bytearray((i * 37 + 11) % 256 for i in range(capacity))
    before = bytes(data)
    blocks = [before[i * 1088 : (i + 1) * 1088] for i in range(6)]
    order = [2, 0, 1, 4, 3, 5]
    for dst, src in enumerate(order):
        data[dst * 1088 : (dst + 1) * 1088] = blocks[src]
    assert all(data[i * 1088 : (i + 1) * 1088] == blocks[src]
               for i, src in enumerate(order))
    assert data[-1088:] == before[-1088:]

    invalid = [
        ((6, 17, 16), (272, 16, 1), 4, capacity, 4),
        ((6, 17, 8), (272, 16, 2), 4, capacity, 0),
        ((7, 17, 16), (272, 16, 1), 4, capacity, 0),
        ((6, 17, 16), (272, 16, 1), 4, capacity + 1, 0),
    ]
    rejected = 0
    for args in invalid:
        try:
            nd_block_bytes(*args)
        except ValueError:
            rejected += 1
    assert rejected == len(invalid)

    # Shapes supplied by the reported Ascend 950PR FP4 probe.
    cases = []
    for name, exported, allocated, m, n in [
        ("w13_full", 80, 80, 2048, 4096),
        ("w13_shared_suffix", 80, 81, 2048, 4096),
        ("w2_full", 80, 80, 1024, 4096),
    ]:
        weight = dict(shape=(exported, m, n), stride0=m * n, itemsize=1)
        full = dict(shape=(allocated, n, m), stride0=m * n, itemsize=1)
        elements = allocated * m * n
        assert nz_metadata_checks(weight, full, elements)
        cases.append(dict(name=name, exported_rows=exported,
                          allocation_rows=allocated, block_bytes=m * n,
                          storage_bytes=elements,
                          exported_bytes=exported * m * n,
                          strict_metadata_pass=True))
    return dict(scope="CPU byte/metadata checks, not an NPU execution",
                nd_block_bytes=1088, nd_allocation_bytes=capacity,
                packed_and_reshape_equal=True, byte_permutation_pass=True,
                invalid_nd_cases_rejected=rejected, nz_probe_metadata=cases)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
