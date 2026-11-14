"""Recalculate published observations and a separately labelled teaching example."""
import argparse
import json
from pathlib import Path

def calculate():
    data_path = Path(__file__).resolve().parents[2] / 'assets/tpot-worse-throughput-better/observed.json'
    source = json.loads(data_path.read_text(encoding='utf-8'))
    result = {'observed_commit': source['source'], 'observed': {}, 'teaching_only': {}}
    for mode in ('off', 'on'):
        arm = source['arms'][mode]
        w = arm['windows'][0]
        generation_ms = w['TPOT_ms']['mean'] * (source['output_tokens'] - 1)
        assert abs(generation_ms + w['TTFT_ms']['mean'] - w['E2E_ms']['mean']) < 1e-6
        assert abs(w['actual_received_output_tokens'] / w['duration_s'] - w['output_tokens_per_s']) < 1e-9
        assert arm['application_failures'] == 0
        result['observed'][mode] = {
            'output_tok_s': w['output_tokens_per_s'],
            'TTFT_s': w['TTFT_ms']['mean']/1000,
            'generation_s': generation_ms/1000,
            'E2E_s': w['E2E_ms']['mean']/1000,
            'TPOT_ms': w['TPOT_ms']['mean'],
            'D_running_sample_mean': arm['pipeline_metrics']['D'][0]['sglang:num_running_reqs']['mean'],
        }
    a,b=result['observed']['off'],result['observed']['on']
    result['observed']['change_pct']={k:(b[k]/a[k]-1)*100 for k in a}
    for batch,step_ms in ((32,10),(48,12)):
        result['teaching_only'][str(batch)]={'batch': batch, 'step_ms':step_ms, 'output_tok_s':batch*1000/step_ms}
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('calculations.json'))
    args=parser.parse_args()
    args.output.write_text(json.dumps(calculate(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified observed request decomposition, fixed-window token counts, and separate teaching arithmetic.')
