from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram


rag_requests_total = Counter('rag_requests_total', 'Total RAG requests')
rag_latency = Histogram('rag_latency_seconds', 'RAG request latency')
retrieval_latency = Histogram('retrieval_latency_seconds', 'Retrieval latency')
llm_latency = Histogram('llm_latency_seconds', 'LLM latency')
tokens_input = Counter('tokens_input_total', 'Total input tokens')
tokens_output = Counter('tokens_output_total', 'Total output tokens')


def setup_metrics(app):
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    )
    
    instrumentator.add(
        metrics.latency(),
        metrics.requests(),
    )
    
    instrumentator.instrument(app).expose(app)
    print("Prometheus metrics enabled at /metrics")