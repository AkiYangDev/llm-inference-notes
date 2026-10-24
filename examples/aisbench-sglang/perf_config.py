"""Minimal HTTP performance config for AISBench commit e00d1f7.

String types deliberately survive MMEngine dump/reload without local classes.
Environment variables configure the already-running inference service.
"""
# Explicit non-lazy config mode: environment lookups execute while loading.
_base_ = []
import os

models = [dict(
    attr='service',
    type='ais_bench.benchmark.models.api_models.vllm_custom_api_chat.VLLMCustomAPIChat',
    abbr='sglang-chat-stream',
    path=os.environ['TOKENIZER_PATH'],
    model=os.environ['SERVED_MODEL'],
    host_ip=os.environ.get('HOST_IP', '127.0.0.1'),
    host_port=int(os.environ.get('HOST_PORT', '30000')),
    api_key=os.environ.get('API_KEY', ''),
    stream=True,
    request_rate=float(os.environ.get('REQUEST_RATE', '1')),
    batch_size=int(os.environ.get('CONCURRENCY', '1')),
    # e00d1f7 uses range(self.retry): 1 means one total attempt.
    retry=1,
    max_out_len=int(os.environ.get('OUTPUT_TOKENS', '32')),
    generation_kwargs=dict(temperature=0, ignore_eos=True),
)]

datasets = [dict(
    abbr='tutorial',
    type='ais_bench.benchmark.datasets.custom.CustomDataset',
    path=os.environ['AIS_DATASET'],
    meta_path='',
    reader_cfg=dict(input_columns=['question'], output_column='answer'),
    infer_cfg=dict(
        prompt_template=dict(
            type='ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template.PromptTemplate',
            template='{question}'),
        retriever=dict(
            type='ais_bench.benchmark.openicl.icl_retriever.icl_zero_retriever.ZeroRetriever'),
        inferencer=dict(
            type='ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer.GenInferencer'),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type='ais_bench.benchmark.openicl.icl_evaluator.AccEvaluator'),
        pred_role='BOT'),
)]

infer = dict(
    partitioner=dict(type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        type='ais_bench.benchmark.runners.local.LocalRunner',
        max_num_workers=1,
        task=dict(type='ais_bench.benchmark.tasks.openicl_api_infer.OpenICLApiInferTask'),
    ),
)

summarizer = dict(
    attr='performance',
    type='ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer',
    calculator=dict(
        type='ais_bench.benchmark.calculators.default_perf_metric_calculator.DefaultPerfMetricCalculator',
        stats_list=['Average', 'Min', 'Max', 'Median', 'P75', 'P90', 'P99'],
    ),
)
