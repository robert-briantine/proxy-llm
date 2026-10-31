import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'proxy'))
from app import parse_metrics


def test_parse_metrics_labeled_series_and_comments():
    source = '''# HELP vllm:generation_tokens_total compteur
vllm:generation_tokens_total{model_name="x",engine="0"} 128.0
vllm:generation_tokens_total{model_name="x",engine="1"} 16
vllm:kv_cache_usage_perc{model_name="x"} 0.12
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 2
'''
    result = parse_metrics(source)
    assert result['vllm:generation_tokens_total'] == 144
    assert result['vllm:kv_cache_usage_perc'] == .12
    assert result['vllm:num_requests_running'] == 2


def test_empty_metrics():
    assert parse_metrics('# HELP ignored\ngarbage\n') == {}
