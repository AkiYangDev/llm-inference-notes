"""Validate data and actual MMEngine dump/fresh-process reload, without inference.

Run after exporting the same environment variables as the AISBench command.
This checks configuration serialization only, not AISBench registry construction,
tokenizer compatibility, HTTP connectivity, or inference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    from mmengine.config import Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('perf_config.py')))
    args = parser.parse_args()
    config = Config.fromfile(args.config)
    model = config.models[0]
    data = Path(config.datasets[0].path)
    assert data.is_absolute() and data.is_file(), 'AIS_DATASET must be an existing absolute file'
    rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
    assert rows and all(isinstance(x.get('question'), str) and isinstance(x.get('answer'), str) for x in rows)
    assert model.attr == 'service' and model.stream is True
    assert model.batch_size >= 1 and model.max_out_len >= 2
    assert model.request_rate >= 0 and model.retry == 1
    with tempfile.TemporaryDirectory(prefix='aisbench-config-') as folder:
        dumped = Path(folder) / 'resolved.py'
        config.dump(str(dumped))
        program = """import json,sys
from mmengine.config import Config
c=Config.fromfile(sys.argv[1])
assert isinstance(c.models[0].type,str)
assert c.models[0].type.endswith('.VLLMCustomAPIChat')
assert c.datasets[0].type.endswith('.CustomDataset')
print(json.dumps({'fresh_reload':'PASS','C':c.models[0].batch_size,'rate':c.models[0].request_rate,'output_tokens':c.models[0].max_out_len}))
"""
        subprocess.run([sys.executable, '-c', program, str(dumped)], check=True)
    print(json.dumps({'rows': len(rows), 'configuration_only': True, 'model_requests': 0}))


if __name__ == '__main__':
    main()
