"""Run-local adapter; make this directory importable by the real AIS worker."""
from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat
from ais_bench.benchmark.registry import MODELS
from telemetry import retain_metadata


@MODELS.register_module()
class SGLangTelemetryChat(VLLMCustomAPIChat):
    async def parse_stream_response(self, json_content, output):
        await super().parse_stream_response(json_content, output)
        retain_metadata(json_content, output)
