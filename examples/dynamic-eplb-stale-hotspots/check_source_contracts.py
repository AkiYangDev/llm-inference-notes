"""Execute selected unmodified source methods with CPU-only fixtures.

The fixtures test Python control flow and buffer mutation. They do not test
tensor operations, distributed communication, or an end-to-end SGLang service.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

EXPECTED_COMMIT = "2914be608009b65099c4c52ba118a597bf1b1421"


def method(tree, class_name, method_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def compile_function(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<source-method>", "exec"), namespace)
    return namespace[node.name]


class Rows:
    def __init__(self, size):
        self.values = [0] * size

    def __len__(self):
        return len(self.values)

    def __setitem__(self, index, value):
        if index is Ellipsis:
            self.values[:] = [value] * len(self.values)
        else:
            self.values[index] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_COMMIT:
        raise SystemExit(f"Expected {EXPECTED_COMMIT}, got {head}; select the article snapshot.")
    paths = ["python/sglang/srt/eplb/eplb_manager.py",
             "python/sglang/srt/eplb/expert_distribution.py",
             "python/sglang/srt/eplb/eplb_algorithms/__init__.py"]
    trees, hashes = {}, {}
    for path in paths:
        raw = (args.repo / path).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(args.repo), "show", f"HEAD:{path}"])
        if raw.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
            raise SystemExit(f"Uncommitted source change: {path}")
        trees[path] = ast.parse(raw.decode("utf-8"))
        hashes[path] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()

    entry = compile_function(method(trees[paths[0]], "EPLBManager", "_entrypoint"), {})
    class Manager:
        _rebalance_num_iterations = 4
        current = 0
        checks = []

        def rebalance(self):
            self.checks.append(self.current)
            if False:
                yield

    manager = Manager()
    generator = entry(manager)
    for n in range(1, 14):
        manager.current = n
        next(generator)
    assert manager.checks == [5, 9, 13]

    circular = next(n for n in trees[paths[1]].body if isinstance(n, ast.ClassDef) and n.name == "_CircularBuffer")
    ns = {"_Buffer": object, "torch": SimpleNamespace(Tensor=list,
          zeros=lambda shape, **kw: Rows(shape[0]))}
    exec(compile(ast.Module(body=[circular], type_ignores=[]), "<source-buffer>", "exec"), ns)
    buffer = ns["_CircularBuffer"]((1,), 4, "int32", "cpu")
    for n in range(1, 7):
        buffer.append(n)
    overwritten = list(buffer.get_all().values)
    assert overwritten == [5, 6, 3, 4]
    buffer.reset()
    assert buffer.get_all().values == [0, 0, 0, 0]
    assert buffer._curr_index == 2
    buffer.append(7)
    assert buffer.get_all().values == [0, 0, 7, 0]

    dump = compile_function(method(trees[paths[1]], "_ExpertDistributionRecorderReal", "dump_record"), {"_OutputMode": str})
    events = []
    class Recorder:
        _accumulator = SimpleNamespace(dump=lambda **kw: events.append("dump") or {"logical_count": 7})

        def _reset(self):
            events.append("reset")
    assert dump(Recorder(), "object") == {"logical_count": 7}
    assert events == ["dump", "reset"]
    algorithm = ast.unparse(trees[paths[2]])
    assert "weight=tokens_per_expert.sum(dim=0)" in algorithm
    assert "tokens_per_expert=tokens_per_expert" in algorithm
    print(json.dumps({"commit": head, "source_hashes_LF": hashes,
                      "manager_checks_I4": manager.checks,
                      "circular_after_six_appends_W4": overwritten,
                      "dump_events": events, "algorithm_input_contracts": "PASS",
                      "scope": "Original Python methods; CPU fixtures only.",
                      "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
