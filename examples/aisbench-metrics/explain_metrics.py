"""CPU teaching examples. No HTTP, tokenizer or model invocation."""
import json
from statistics import mean

def request_metrics(points_ms, output_tokens):
    if len(points_ms) < 2 or output_tokens < 1:
        raise ValueError("Need a send point, a response event and positive output")
    return {
        "E2E_ms": points_ms[-1] - points_ms[0],
        "TTFT_ms": points_ms[1] - points_ms[0],
        "TPOT_ms": ((points_ms[-1] - points_ms[1]) / (output_tokens - 1)
                    if output_tokens > 1 else 0),
        "ITL_ms": [b-a for a, b in zip(points_ms[1:], points_ms[2:])],
    }

def native_selection(requests, concurrency_cap):
    # Same event ordering as the cited source: Python's stable sort by time.
    events = []
    for index, row in enumerate(requests):
        events.extend([(row["start_s"], "start", index),
                       (row["end_s"], "end", index)])
    running, hits, first, last = 0, 0, None, None
    for time, kind, _ in sorted(events, key=lambda event: event[0]):
        running += 1 if kind == "start" else -1
        if running == concurrency_cap:
            if kind == "start":
                hits += 1
                if hits == 2:
                    first = time
            last = time
    if first is None:
        raise ValueError("No native stable stage")
    selected = [row for row in requests if first <= row["start_s"] <= last]
    duration = last-first
    if duration <= 0:
        raise ValueError("Non-positive teaching window")
    lifetime_sum = sum(row["end_s"]-row["start_s"] for row in selected)
    return {
        "section_s": [first, last],
        "selected_ids": [row["id"] for row in selected],
        "duration_s": duration,
        "full_selected_outputs": sum(row["tokens"] for row in selected),
        "reported_output_tps": sum(row["tokens"] for row in selected)/duration,
        "uncapped_lifetime_ratio": lifetime_sum/duration,
        "reported_concurrency": min(lifetime_sum/duration, concurrency_cap),
    }

def main():
    first = request_metrics([0, 100, 160, 220], 13)
    assert first == {"E2E_ms":220, "TTFT_ms":100, "TPOT_ms":10,
                     "ITL_ms":[60, 60]}
    gaps = [[50, 50], [1]]
    weighting = {
        "pooled_event_mean_ms":mean([gap for row in gaps for gap in row]),
        "equal_request_mean_ms":mean([mean(row) for row in gaps]),
        "interval_count":sum(map(len, gaps)), "request_count":len(gaps)}
    assert abs(weighting["pooled_event_mean_ms"]-101/3) < 1e-12
    assert weighting["equal_request_mean_ms"] == 25.5
    requests = [
        {"id":"A", "start_s":0,   "end_s":10, "tokens":100},
        {"id":"B", "start_s":1,   "end_s":3,  "tokens":100},
        {"id":"C", "start_s":3.1, "end_s":5,  "tokens":100},
        {"id":"D", "start_s":5.1, "end_s":7,  "tokens":100},
        {"id":"E", "start_s":7.1, "end_s":12, "tokens":100},
    ]
    native = native_selection(requests, 2)
    assert native["section_s"] == [3.1, 7.1]
    assert native["selected_ids"] == ["C", "D", "E"]
    assert abs(native["reported_output_tps"]-75) < 1e-12
    assert abs(native["uncapped_lifetime_ratio"]-2.175) < 1e-12
    assert native["reported_concurrency"] == 2
    try:
        native_selection(requests, 99)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing stable stage should fail")
    print(json.dumps({"scope":"synthetic CPU arithmetic, not benchmark measurements",
                      "request":first,"weighting":weighting,"native":native},
                     ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

