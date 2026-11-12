"""Small device-graph examples; no model load, service, or distributed group.

Run on an otherwise available device:
    python graph_storage_demo.py --backend npu --device-index 0
    python graph_storage_demo.py --backend cuda --device-index 0

Expected values are assertions, not recorded device results. The set_ example
keeps the old storage alive deliberately; it is not a weight-update recipe.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('npu', 'cuda'), required=True)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()

    import torch

    if args.backend == 'npu':
        import torch_npu
        runtime_version = torch_npu.__version__
        runtime = torch.npu
        graph_type = runtime.NPUGraph
    else:
        runtime_version = None
        runtime = torch.cuda
        graph_type = runtime.CUDAGraph
    if not runtime.is_available():
        raise RuntimeError(f'{args.backend} is unavailable')
    runtime.set_device(args.device_index)
    device = f'{args.backend}:{args.device_index}'

    def make_graph():
        x = torch.tensor([2.0], device=device)
        weight = torch.tensor([3.0], device=device)
        calls = {'forward': 0}

        def forward():
            calls['forward'] += 1
            return x * weight

        main_stream = runtime.current_stream()
        warmup_stream = runtime.Stream()
        warmup_stream.wait_stream(main_stream)
        with runtime.stream(warmup_stream):
            for _ in range(3):
                warmup_output = forward()
        main_stream.wait_stream(warmup_stream)
        del warmup_output

        graph = graph_type()
        with runtime.graph(graph):
            output = forward()
        runtime.synchronize()
        return {
            'x': x, 'weight': weight, 'original_owner': weight.detach(),
            'output': output, 'graph': graph, 'calls': calls,
        }

    def replay(state):
        before = state['calls']['forward']
        state['graph'].replay()
        runtime.synchronize()
        assert state['calls']['forward'] == before
        return float(state['output'].item())

    results = []
    states = []  # Hold graphs and their external buffers until final sync.
    with torch.no_grad():
        state = make_graph()
        states.append(state)
        weight = state['weight']
        baseline = replay(state)
        original_pointer = weight.data_ptr()
        original_id = id(weight)
        weight.copy_(torch.tensor([5.0], device=device))
        after = replay(state)
        assert (baseline, after) == (6.0, 10.0)
        assert weight.data_ptr() == original_pointer and id(weight) == original_id
        results.append({'case': 'copy_', 'before': baseline, 'after': after,
                        'same_python_object': True, 'same_data_pointer': True})

        state = make_graph()
        states.append(state)
        weight = state['weight']
        baseline = replay(state)
        original_pointer = weight.data_ptr()
        original_id = id(weight)
        weight = torch.tensor([5.0], device=device)
        after = replay(state)
        eager = float((state['x'] * weight).item())
        assert (baseline, after, eager) == (6.0, 6.0, 10.0)
        assert id(weight) != original_id and weight.data_ptr() != original_pointer
        results.append({'case': 'python_rebind', 'before': baseline, 'after': after,
                        'eager_using_new_name': eager, 'same_python_object': False,
                        'same_data_pointer': False})

        state = make_graph()
        states.append(state)
        weight = state['weight']
        baseline = replay(state)
        original_pointer = weight.data_ptr()
        original_id = id(weight)
        replacement = torch.tensor([5.0], device=device)
        # original_owner still owns the allocation captured by the graph.
        # Removing that protection would make this an invalid-lifetime example.
        weight.set_(replacement)
        after = replay(state)
        eager = float((state['x'] * weight).item())
        assert (baseline, after, eager) == (6.0, 6.0, 10.0)
        assert id(weight) == original_id and weight.data_ptr() != original_pointer
        results.append({'case': 'set_storage', 'before': baseline, 'after': after,
                        'eager_using_changed_tensor': eager, 'same_python_object': True,
                        'same_data_pointer': False, 'old_storage_kept_alive': True})

        state = make_graph()
        states.append(state)
        assert replay(state) == 6.0
        reference = state['output']
        snapshot = reference.clone()
        state['x'].fill_(4.0)
        after = replay(state)
        assert (after, float(reference.item()), float(snapshot.item())) == (12.0, 12.0, 6.0)
        results.append({'case': 'output_reuse', 'new_output': after,
                        'saved_reference': float(reference.item()),
                        'saved_snapshot': float(snapshot.item())})

        x = torch.tensor([2.0], device=device)
        flag = True
        scale = 2.0
        calls = 0

        def python_branch():
            nonlocal calls
            calls += 1
            if flag:
                return x * scale
            return x * 7.0

        warmup_stream = runtime.Stream()
        warmup_stream.wait_stream(runtime.current_stream())
        with runtime.stream(warmup_stream):
            for _ in range(3):
                warmup_output = python_branch()
        runtime.current_stream().wait_stream(warmup_stream)
        del warmup_output
        graph = graph_type()
        with runtime.graph(graph):
            output = python_branch()
        runtime.synchronize()
        captured_calls = calls
        flag, scale = False, 5.0
        graph.replay()
        runtime.synchronize()
        assert calls == captured_calls and float(output.item()) == 4.0
        results.append({'case': 'python_control', 'graph_output': float(output.item()),
                        'python_forward_calls_unchanged': calls == captured_calls})

    runtime.synchronize()
    print(json.dumps({'torch': torch.__version__, 'torch_npu': runtime_version,
                      'backend': args.backend, 'device_name': runtime.get_device_name(),
                      'cases': results, 'passed': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
