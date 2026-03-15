import logging
from contextlib import contextmanager
from threading import Lock

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from datapizza.tracing import console
from datapizza.tracing.memory_exporter import ContextSpanProcessor

tracer = trace.get_tracer(__name__)
log = logging.getLogger(__name__)


def get_total_spans(spans):
    return len(spans)


def get_seconds_span_duration(span):
    return round((span.end_time - span.start_time) / 1000000000, 2)


def get_token_usage(spans):
    model_tokens = {}
    for span in spans:
        if span.attributes.get("type") == "generation":
            model = span.attributes.get("model_name", "unknown")
            prompt_tokens = span.attributes.get("prompt_tokens_used", 0)
            completion_tokens = span.attributes.get("completion_tokens_used", 0)
            cached_tokens = span.attributes.get("cached_tokens_used", 0)
            thinking_tokens = span.attributes.get("thinking_tokens_used", 0)

            if model not in model_tokens:
                model_tokens[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "thinking_tokens": 0,
                }

            model_tokens[model]["prompt_tokens"] += prompt_tokens
            model_tokens[model]["completion_tokens"] += completion_tokens
            model_tokens[model]["cached_tokens"] += cached_tokens
            model_tokens[model]["thinking_tokens"] += thinking_tokens
    return model_tokens


class ContextTracing:
    _instance = None
    _lock = Lock()
    _context_processor: ContextSpanProcessor | None = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def _set_context_processor(self):
        if isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
            trace.set_tracer_provider(TracerProvider())

        tracer_provider = trace.get_tracer_provider()
        self._context_processor = ContextSpanProcessor()
        tracer_provider.add_span_processor(self._context_processor)  # type: ignore

    @contextmanager
    def trace(self, name: str | None = None):
        with self._lock:
            if not self._context_processor:
                self._set_context_processor()

        span = None
        trace_id = None

        class CurrentTrace:
            def __init__(self, trace_id, processor):
                self.trace_id = trace_id
                self.processor = processor

            def get_spans(self):
                return self.processor.get_spans_by_trace_id(self.trace_id)

        try:
            with tracer.start_as_current_span(name or "") as span:
                trace_id = span.get_span_context().trace_id
                self._context_processor.start_trace(trace_id)  # type: ignore
                yield CurrentTrace(trace_id, self._context_processor)
        except Exception as e:
            log.error(f"Error in trace collection: {e}")
            raise
        finally:
            if span and trace_id:
                try:
                    spans = self._context_processor.get_spans_by_trace_id(trace_id)  # type: ignore
                    total_span = get_total_spans(spans)
                    token_usage = get_token_usage(spans)
                    span_duration = get_seconds_span_duration(span)

                    # Create token usage table
                    table = Table(title="Token Usage")
                    table.add_column("Model")
                    table.add_column("Prompt Tokens")
                    table.add_column("Completion Tokens")
                    table.add_column("Cached Tokens")
                    table.add_column("Thinking Tokens")

                    for model, usage in token_usage.items():
                        table.add_row(
                            model,
                            str(usage["prompt_tokens"]),
                            str(usage["completion_tokens"]),
                            str(usage["cached_tokens"]),
                            str(usage["thinking_tokens"]),
                        )

                    panel = Panel(
                        Group(
                            f"Total Spans: {total_span}\nDuration: {span_duration}s",
                            table if token_usage else "No token usage",
                        ),
                        title=f"Trace Summary of [bold]{name}[/bold]",
                    )
                    console.print(panel)
                finally:
                    # Ensure cleanup even if display fails
                    self._context_processor.stop_trace(trace_id)  # type: ignore


@contextmanager
def generation_span(name: str | None = None):
    with tracer.start_as_current_span(name or "") as span:
        span.set_attribute("type", "generation")
        yield span


@contextmanager
def agent_span(name: str | None = None):
    with tracer.start_as_current_span(name or "") as span:
        span.set_attribute("type", "agent")
        yield span


@contextmanager
def tool_span(name: str | None = None):
    with tracer.start_as_current_span(name or "") as span:
        span.set_attribute("type", "tool")
        yield span
