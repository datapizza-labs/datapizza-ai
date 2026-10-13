import threading
import typing

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class InMemoryTraceExporter(SpanExporter):
    """Implementation of :class:`.SpanExporter` that stores spans in memory.
    This class can be used for testing purposes. It stores the exported spans
    in a list in memory that can be retrieved using the
    :func:`.get_finished_spans` method.
    """

    def __init__(self) -> None:
        self._finished_spans: dict[int, list[ReadableSpan]] = {}
        self._stopped = False
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Clear list of collected spans."""
        with self._lock:
            self._finished_spans.clear()

    def clear_trace(self, trace_id):
        with self._lock:
            if trace_id in self._finished_spans:
                del self._finished_spans[trace_id]

    def get_finished_spans(self) -> dict[int, list[ReadableSpan]]:
        """Get a snapshot of collected spans, grouped by trace ID."""
        with self._lock:
            return {
                trace_id: list(spans)
                for trace_id, spans in self._finished_spans.items()
            }

    def get_finished_spans_by_trace_id(self, trace_id: int) -> list[ReadableSpan]:
        """Get a snapshot of collected spans for a trace."""
        with self._lock:
            return list(self._finished_spans.get(trace_id, []))

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        """Stores a list of spans in memory."""
        if self._stopped:
            return SpanExportResult.FAILURE
        with self._lock:
            for span in spans:
                context = span.get_span_context()
                if context:
                    if context.trace_id not in self._finished_spans:
                        self._finished_spans[context.trace_id] = []

                    self._finished_spans[context.trace_id].append(span)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shut downs the exporter.

        Calls to export after the exporter has been shut down will fail.
        """
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class ContextSpanProcessor(SimpleSpanProcessor):
    """Simple SpanProcessor implementation.

    SimpleSpanProcessor is an implementation of `SpanProcessor` that
    passes ended spans directly to the configured `SpanExporter`.
    """

    def __init__(self):
        super().__init__(InMemoryTraceExporter())
        self.tracing_ids = set()

    def start_trace(self, trace_id: int) -> None:
        self.tracing_ids.add(trace_id)

    def stop_trace(self, trace_id: int) -> None:
        if trace_id in self.tracing_ids:
            self.span_exporter.clear_trace(trace_id)  # type: ignore
            self.tracing_ids.remove(trace_id)

    def on_end(self, span: ReadableSpan) -> None:
        if span.get_span_context().trace_id not in self.tracing_ids:  # type: ignore
            return

        super().on_end(span)

    def get_spans_by_trace_id(self, trace_id: int) -> list[ReadableSpan]:
        return self.span_exporter.get_finished_spans_by_trace_id(trace_id)  # type: ignore
