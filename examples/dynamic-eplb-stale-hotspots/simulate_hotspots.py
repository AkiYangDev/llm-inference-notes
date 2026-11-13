"""Reproduce the article's teaching examples. Python standard library only.

This models accumulated A/B routing signals, not SGLang's placement solver,
expert dispatch, speculative decoding, or device execution.
"""
import argparse
import csv
import json
from collections import deque
from pathlib import Path
from statistics import mean


def window_cases(interval, window, change_step, end=8000):
    assert 0 < window <= interval
    buffer = deque(maxlen=window)
    rows = []
    first_b = None
    for step in range(1, end + 1):
        buffer.append(step >= change_step)
        if step % interval:
            continue
        assert len(buffer) == window
        new = sum(buffer)
        old = window - new
        a = old * 90 + new * 10
        b = old * 10 + new * 90
        signal = "B" if b > a else "A" if a > b else "tie"
        if signal == "B" and first_b is None:
            first_b = step
        rows.append(dict(
            I=interval, W=window, change_step=change_step, check_step=step,
            old_records=old, new_records=new, A_routes=a, B_routes=b,
            B_route_share=b / (a + b), signal=signal,
        ))
        # A completed dump resets the history. I >= W refills it by next check.
        buffer.clear()
    for row in rows:
        row["first_B_check_step"] = first_b
    return rows


def phase_sweep(interval, window):
    """One persistent A->B change for each position of a steady check cycle.

    Count from the first B record through the first check with B>A, inclusive.
    Ties retain the previous A signal. Updates and wall-clock time are omitted.
    """
    delays = []
    for change_step in range(1, interval + 1):
        until_check = interval - change_step + 1
        new_records = min(window, until_check)
        delays.append(until_check + (interval if new_records <= window / 2 else 0))
    return dict(I=interval, W=window, phases=interval,
                mean_records_to_B_signal=mean(delays),
                max_records_to_B_signal=max(delays),
                min_records_to_B_signal=min(delays))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for window in (1000, 4000):
        for change_step in (3401, 3901, 4001):
            cases.extend(window_cases(4000, window, change_step))
    selected = {(r["W"], r["change_step"], r["check_step"]): r for r in cases}
    assert selected[1000, 3401, 4000]["B_routes"] == 58000
    assert selected[4000, 3401, 4000]["B_routes"] == 88000
    assert selected[1000, 3901, 4000]["B_route_share"] == .18
    assert selected[1000, 4001, 8000]["B_route_share"] == .9
    assert all(r["signal"] == "B" for r in cases if r["check_step"] == 8000)

    phases = [phase_sweep(i, w) for i, w in ((1000, 1000), (4000, 1000), (4000, 4000))]
    assert [r["mean_records_to_B_signal"] for r in phases] == [1000.5, 2500.5, 4000.5]
    assert [r["max_records_to_B_signal"] for r in phases] == [1500, 4500, 6000]
    # Cross-check a small phase sweep using explicit circular-window samples.
    for change_step in range(1, 9):
        actual = window_cases(8, 4, change_step, end=16)[0]["first_B_check_step"]
        delay = 8 - change_step + 1
        expected = change_step - 1 + delay + (8 if min(4, delay) <= 2 else 0)
        assert actual == expected

    ages = [dict(W=w, record_spacing_ms=35, update_ms=600,
                 youngest_age_ms=600, oldest_age_ms=(w - 1) * 35 + 600,
                 mean_age_ms=(w - 1) / 2 * 35 + 600)
            for w in (1000, 4000)]
    assert ages[0]["mean_age_ms"] == 18082.5
    assert ages[1]["mean_age_ms"] == 70582.5
    assert round(100 * 1000 / (100 * 1000 + 900 * 10) * 100, 1) == 91.7
    assert [90 + 90 + 10 + 10, 10 + 10 + 90 + 90] == [200, 200]
    assert 600 / .2 == 3000
    write_csv(args.output / "window_cases.csv", cases)
    write_csv(args.output / "phase_summary.csv", phases)
    write_csv(args.output / "record_age.csv", ages)
    print(json.dumps({"kind": "teaching_model", "device_measurement": False,
                      "checks": "PASS", "window_rows": len(cases),
                      "phase_summary": phases, "record_age": ages}, indent=2))


if __name__ == "__main__":
    main()
