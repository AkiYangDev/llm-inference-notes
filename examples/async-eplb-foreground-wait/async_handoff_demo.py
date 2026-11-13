"""Teaching model for an EPLB handoff; standard library, no device execution.

Two integers stand in for one expert's weight blocks. The simulated forward
adds them. This is not an MoE kernel, a communication test, or a benchmark.
"""
import argparse
import csv
import json
from pathlib import Path

EXPERTS = {"A": [2, 3], "B": [5, 7]}


class Slot:
    def __init__(self):
        self.logical_id = "A"
        self.live_weights = list(EXPERTS["A"])
        self.buffer = None
        self.buffer_free = True
        self.committing = False

    def receive(self, expert):
        if not self.buffer_free:
            raise RuntimeError("Buffer is still reserved for the previous handoff")
        self.buffer = (expert, list(EXPERTS[expert]))
        self.buffer_free = False

    def forward(self):
        if self.committing:
            raise RuntimeError("Next use must be ordered after the commit")
        actual = sum(self.live_weights)
        expected = sum(EXPERTS[self.logical_id])
        return dict(logical_id=self.logical_id, actual=actual, expected=expected,
                    correct=actual == expected)

    def commit(self, ranks_ready):
        if not all(ranks_ready):
            return False
        assert self.buffer is not None
        self.committing = True
        address_identity = id(self.live_weights)
        expert, blocks = self.buffer
        self.live_weights[:] = blocks
        self.logical_id = expert
        assert id(self.live_weights) == address_identity
        self.committing = False
        # A real implementation additionally has stream/device dependencies.
        # The model keeps the buffer reserved until an explicit acknowledgement.
        return True

    def acknowledge_consumed(self):
        self.buffer = None
        self.buffer_free = True


def handoff_cases():
    partial = Slot()
    partial.live_weights[0] = EXPERTS["B"][0]
    partial_result = partial.forward()
    assert partial_result["actual"] == 8 and not partial_result["correct"]

    early_map = Slot()
    early_map.logical_id = "B"
    early_result = early_map.forward()
    assert early_result["actual"] == 5 and early_result["expected"] == 12
    assert not early_result["correct"]

    safe = Slot()
    safe.receive("B")
    before = safe.forward()
    assert before["correct"] and before["logical_id"] == "A"
    assert safe.commit([True, False]) is False
    assert safe.forward() == before
    assert safe.commit([True, True]) is True
    after = safe.forward()
    assert after["correct"] and after["logical_id"] == "B"
    premature_reuse_rejected = False
    try:
        safe.receive("A")
    except RuntimeError:
        premature_reuse_rejected = True
    assert premature_reuse_rejected
    safe.acknowledge_consumed()
    safe.receive("A")
    return dict(partial_live_overwrite=partial_result, early_mapping_switch=early_result,
                staged_but_not_committed=before, after_safe_commit=after,
                unready_peer_keeps_old_layout=True,
                premature_buffer_reuse_rejected=premature_reuse_rejected,
                scope="Sequential state model; no GPU/NPU ordering is simulated.")


def timeline(mode):
    assert mode in {"baseline", "sync", "async"}
    time = 0
    committed = mode == "baseline"
    rows = []
    for step in range(1, 6):
        if mode == "async" and not committed and time >= 35:
            rows.append(dict(mode=mode, lane="foreground", operation="commit", start_ms=time, end_ms=time + 2))
            time += 2
            committed = True
        rows.append(dict(mode=mode, lane="foreground", operation=f"forward_{step}", start_ms=time, end_ms=time + 10))
        time += 10
        if step == 1 and mode == "sync":
            rows.append(dict(mode=mode, lane="foreground", operation="transfer", start_ms=time, end_ms=time + 25))
            time += 25
            rows.append(dict(mode=mode, lane="foreground", operation="commit", start_ms=time, end_ms=time + 2))
            time += 2
            committed = True
        if step == 1 and mode == "async":
            rows.append(dict(mode=mode, lane="worker", operation="transfer", start_ms=10, end_ms=35))
    assert committed
    assert sum(r["end_ms"] - r["start_ms"] for r in rows if r["operation"].startswith("forward")) == 50
    return time, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = handoff_cases()
    rows, totals = [], []
    for mode, expected in (("baseline", 50), ("sync", 77), ("async", 52)):
        completed, events = timeline(mode)
        assert completed == expected
        rows.extend(events)
        totals.append(dict(mode=mode, forwards=5, update_committed=mode != "baseline", completion_ms=completed))
    for name, data in (("timeline.csv", rows), ("completion.csv", totals)):
        with (args.output / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    (args.output / "handoff_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(kind="teaching_model", checks="PASS", completion=totals,
                          device_measurement=False), indent=2))


if __name__ == "__main__":
    main()
