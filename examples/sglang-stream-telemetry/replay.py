"""Replay synthetic SSE through pinned original AIS methods, with no HTTP.

Only method extraction and a fake response/clock are used. This does not build
the actual AIS registry, tokenizer, Output writer or a running SGLang server.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from telemetry import retain_metadata, summarize_counts

EXPECTED = {
    'models/api_models/base_api.py': '203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755',
    'models/api_models/vllm_custom_api_chat.py': '093e1aae05b49647c3dc0fb6ecf6e41812914ac2293874979cc9deaabc60018c',
}


def methods(path, class_name, names):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    definitions = [copy.deepcopy(n) for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in definitions} == set(names)
    for f in definitions:
        f.decorator_list = []
        f.returns = None
        for a in [*f.args.args, *f.args.kwonlyargs]:
            a.annotation = None
        if f.args.vararg:
            f.args.vararg.annotation = None
        if f.args.kwarg:
            f.args.kwarg.annotation = None
    ns = {'json': json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(path), 'exec'), ns)
    return {n.name: ns[n.name] for n in definitions}


def drive(coroutine):
    # Fake async operations complete synchronously; no event loop or sockets.
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError('Unexpected external await in offline fixture')


class Output:
    def __init__(self):
        self.content, self.reasoning_content = '', ''
        self.input_tokens, self.output_tokens = 0, 0
        self.extra_details_data, self.time_points = {}, []
        self.success = False

    async def record_time_point(self):
        self.time_points.append([0, .100, .140, .160, .180][len(self.time_points)])


def execute(cls, events, fragmented):
    data = b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events)
    data += b'data: [DONE]\n\n'
    parts = [data[i:i+7] for i in range(0, len(data), 7)] if fragmented else [data]

    async def content():
        for part in parts:
            yield part

    class Response:
        status = 200
        async def __aenter__(self):
            self.content = content()
            return self
        async def __aexit__(self, *args):
            return False

    obj = cls()
    obj.url, obj.headers = 'http://unused.invalid/v1/chat/completions', {}
    obj.session = SimpleNamespace(post=lambda *a, **k: Response())
    obj._accumulate_response_anomaly_payload = lambda *a: None
    out = Output()
    drive(obj.stream_infer({}, out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aisbench-root', required=True)
    args = parser.parse_args()
    root = Path(args.aisbench_root) / 'ais_bench/benchmark'
    for relative, expected in EXPECTED.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    base = type('PinnedBase', (), methods(root / 'models/api_models/base_api.py', 'BaseAPIModel', ['iter_lines', 'stream_infer']))
    plain = type('PinnedChat', (base,), methods(root / 'models/api_models/vllm_custom_api_chat.py', 'VLLMCustomAPIChat', ['_parse_usage', 'parse_stream_response']))
    adapter = ast.parse(Path(__file__).with_name('sglang_telemetry.py').read_text(encoding='utf-8'))
    cls = next(n for n in adapter.body if isinstance(n, ast.ClassDef))
    cls.decorator_list = []
    ns = {'VLLMCustomAPIChat': plain, 'retain_metadata': retain_metadata}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'adapter fixture', 'exec'), ns)
    observed = ns['SGLangTelemetryChat']
    details = dict(spec_verify_ct=2, spec_num_correct_drafts=6, spec_num_proposed_drafts=10, spec_correct_drafts_histogram=[0, 0, 1, 0, 1, 0])
    events = [
        {'choices': [{'delta': {'content': '示例甲'}}]},
        {'choices': [{'delta': {'content': '示例乙', 'reasoning_content': '示例推理'}}]},
        {'choices': [], 'sglext': {'spec_tokens_details': details}},
        {'choices': [], 'usage': {'prompt_tokens': 1000, 'completion_tokens': 8, 'prompt_tokens_details': {'cached_tokens': 900}}},
    ]
    for split in [False, True]:
        a, b = execute(plain, events, split), execute(observed, events, split)
        assert (a.content, a.reasoning_content, a.input_tokens, a.output_tokens, a.success, a.time_points) == (b.content, b.reasoning_content, b.input_tokens, b.output_tokens, b.success, b.time_points)
        assert b.time_points == [0, .100, .140, .160, .180]
        extra = json.loads(json.dumps(b.extra_details_data))
        assert extra['prefix_cache_usage']['cached_tokens'] == 900
        assert summarize_counts(extra)['strict_AR'] == 'NOT_OBSERVED'
        assert summarize_counts(extra, proposal_denominator_verified=True) == {'AL_output': 4, 'strict_AR': .6}
        missing_proposals = copy.deepcopy(extra)
        missing_proposals['sglang_spec_tokens_details'].pop('spec_num_proposed_drafts')
        assert summarize_counts(missing_proposals, proposal_denominator_verified=True) == {'AL_output': 4, 'strict_AR': 'NOT_OBSERVED'}
    # Missing is not zero, and repeated final metadata is not summed.
    bad = Output()
    retain_metadata({'usage': {'prompt_tokens': 1000, 'completion_tokens': 8}}, bad)
    assert bad.extra_details_data['prefix_cache_usage']['cached_tokens'] == 'NOT_OBSERVED'
    retain_metadata(events[2], bad)
    retain_metadata(events[2], bad)
    assert bad.extra_details_data['spec_capture_status'] == 'MULTIPLE_FINAL_METADATA'
    assert summarize_counts(bad.extra_details_data)['strict_AR'] == 'NOT_OBSERVED'
    invalid = copy.deepcopy(details)
    invalid['spec_correct_drafts_histogram'][2] += 1
    extra = {'spec_capture_status': 'RETAINED_UNVALIDATED', 'sglang_spec_tokens_details': invalid}
    try:
        summarize_counts(extra, proposal_denominator_verified=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid histogram accepted')
    print(json.dumps({'scope': 'SYNTHETIC_REPLAY_NOT_PERFORMANCE', 'original_clock_policy': 'PRESERVED', 'plain_and_observer_equal': True, 'fragmented_UTF8_frames': 'PASS', 'metadata_only_time_points': 2, 'missing_duplicate_histogram_checks': 'PASS', 'HTTP_requests': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
