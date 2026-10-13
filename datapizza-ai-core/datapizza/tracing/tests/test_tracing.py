import threading
import time
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext

from datapizza.tracing.memory_exporter import (
    ContextSpanProcessor,
    InMemoryTraceExporter,
)
from datapizza.tracing.tracing import (
    ContextTracing,
    get_seconds_span_duration,
    get_total_spans,
)


class TestInMemoryTraceExporter:
    """Test suite for InMemoryTraceExporter class."""

    @pytest.fixture
    def exporter(self):
        """Create a fresh exporter for each test."""
        return InMemoryTraceExporter()

    @pytest.fixture
    def mock_span(self):
        """Create a mock ReadableSpan for testing."""
        span = Mock(spec=ReadableSpan)
        span_context = Mock(spec=SpanContext)
        span_context.trace_id = 12345
        span.get_span_context.return_value = span_context
        span.start_time = 1000000000  # 1 second in nanoseconds
        span.end_time = 2000000000  # 2 seconds in nanoseconds
        return span

    def test_initial_state(self, exporter):
        """Test exporter initial state."""
        assert len(exporter.get_finished_spans()) == 0
        assert not exporter._stopped
        assert exporter._finished_spans == {}

    def test_export_single_span(self, exporter, mock_span):
        """Test exporting a single span."""
        result = exporter.export([mock_span])

        assert result == SpanExportResult.SUCCESS
        spans = exporter.get_finished_spans_by_trace_id(12345)
        assert len(spans) == 1
        assert spans[0] == mock_span

    def test_export_multiple_spans_same_trace(self, exporter):
        """Test exporting multiple spans with the same trace ID."""
        span1 = Mock(spec=ReadableSpan)
        span2 = Mock(spec=ReadableSpan)

        for span in [span1, span2]:
            span_context = Mock(spec=SpanContext)
            span_context.trace_id = 12345
            span.get_span_context.return_value = span_context

        result = exporter.export([span1, span2])

        assert result == SpanExportResult.SUCCESS
        spans = exporter.get_finished_spans_by_trace_id(12345)
        assert len(spans) == 2
        assert span1 in spans
        assert span2 in spans

    def test_export_spans_different_traces(self, exporter):
        """Test exporting spans with different trace IDs."""
        span1 = Mock(spec=ReadableSpan)
        span2 = Mock(spec=ReadableSpan)

        span1_context = Mock(spec=SpanContext)
        span1_context.trace_id = 12345
        span1.get_span_context.return_value = span1_context

        span2_context = Mock(spec=SpanContext)
        span2_context.trace_id = 67890
        span2.get_span_context.return_value = span2_context

        result = exporter.export([span1, span2])

        assert result == SpanExportResult.SUCCESS
        assert len(exporter.get_finished_spans_by_trace_id(12345)) == 1
        assert len(exporter.get_finished_spans_by_trace_id(67890)) == 1

    def test_export_after_shutdown(self, exporter, mock_span):
        """Test that export fails after shutdown."""
        exporter.shutdown()

        result = exporter.export([mock_span])
        assert result == SpanExportResult.FAILURE
        assert len(exporter.get_finished_spans()) == 0

    @pytest.mark.parametrize("all_traces", [False, True])
    def test_returned_spans_cannot_clear_exported_trace(
        self, exporter, mock_span, all_traces
    ):
        exporter.export([mock_span])
        if all_traces:
            spans = exporter.get_finished_spans()[12345]
        else:
            spans = exporter.get_finished_spans_by_trace_id(12345)

        spans.clear()

        assert exporter.get_finished_spans_by_trace_id(12345) == [mock_span]

    @pytest.mark.parametrize("all_traces", [False, True])
    def test_returned_spans_do_not_change_after_later_exports(
        self, exporter, mock_span, all_traces
    ):
        exporter.export([mock_span])
        if all_traces:
            spans = exporter.get_finished_spans()[12345]
        else:
            spans = exporter.get_finished_spans_by_trace_id(12345)

        exporter.export([mock_span])

        assert spans == [mock_span]
        assert exporter.get_finished_spans_by_trace_id(12345) == [mock_span, mock_span]

    def test_clear_all_spans(self, exporter, mock_span):
        """Test clearing all spans."""
        exporter.export([mock_span])
        assert len(exporter.get_finished_spans()) > 0

        exporter.clear()
        assert len(exporter.get_finished_spans()) == 0

    def test_clear_specific_trace(self, exporter):
        """Test clearing spans for a specific trace."""
        span1 = Mock(spec=ReadableSpan)
        span2 = Mock(spec=ReadableSpan)

        span1_context = Mock(spec=SpanContext)
        span1_context.trace_id = 12345
        span1.get_span_context.return_value = span1_context

        span2_context = Mock(spec=SpanContext)
        span2_context.trace_id = 67890
        span2.get_span_context.return_value = span2_context

        exporter.export([span1, span2])

        exporter.clear_trace(12345)

        assert len(exporter.get_finished_spans_by_trace_id(12345)) == 0
        assert len(exporter.get_finished_spans_by_trace_id(67890)) == 1

    def test_clear_nonexistent_trace(self, exporter):
        """Test clearing a trace that doesn't exist."""
        # Should not raise an exception
        exporter.clear_trace(99999)
        assert len(exporter.get_finished_spans()) == 0

    def test_get_nonexistent_trace(self, exporter):
        """Test getting spans for a nonexistent trace."""
        spans = exporter.get_finished_spans_by_trace_id(99999)
        assert spans == []

    def test_force_flush(self, exporter):
        """Test force flush functionality."""
        result = exporter.force_flush()
        assert result is True

        result = exporter.force_flush(timeout_millis=1000)
        assert result is True


class TestContextSpanProcessor:
    """Test suite for ContextSpanProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a fresh processor for each test."""
        return ContextSpanProcessor()

    @pytest.fixture
    def mock_span(self):
        """Create a mock ReadableSpan for testing."""
        span = Mock(spec=ReadableSpan)
        span_context = Mock(spec=SpanContext)
        span_context.trace_id = 12345
        span.get_span_context.return_value = span_context
        return span

    def test_initial_state(self, processor):
        """Test processor initial state."""
        assert len(processor.tracing_ids) == 0
        assert processor.span_exporter is not None

    def test_start_trace(self, processor):
        """Test starting a trace."""
        trace_id = "test_trace_123"
        processor.start_trace(trace_id)

        assert trace_id in processor.tracing_ids

    def test_stop_trace(self, processor, mock_span):
        """Test stopping a trace."""
        trace_id = 12345
        processor.start_trace(trace_id)
        processor.span_exporter.export([mock_span])

        # Verify span is in exporter
        spans = processor.get_spans_by_trace_id(trace_id)
        assert len(spans) == 1

        processor.stop_trace(trace_id)

        # Verify trace is removed from tracking and exporter
        assert trace_id not in processor.tracing_ids
        spans = processor.get_spans_by_trace_id(trace_id)
        assert len(spans) == 0

    def test_stop_nonexistent_trace(self, processor):
        """Test stopping a trace that doesn't exist."""
        # Should not raise an exception
        processor.stop_trace("nonexistent_trace")
        assert len(processor.tracing_ids) == 0

    def test_on_end_untracked_span(self, processor, mock_span):
        """Test on_end with an untracked span."""
        # Don't start trace for this span
        processor.on_end(mock_span)
        assert len(processor.get_spans_by_trace_id(12345)) == 0


class TestTracingHelperFunctions:
    """Test suite for tracing helper functions."""

    def test_get_spans_metrics(self):
        """Test span count calculation."""
        spans = [Mock(), Mock(), Mock()]
        count = get_total_spans(spans)
        assert count == 3

        empty_spans = []
        count = get_total_spans(empty_spans)
        assert count == 0

    def test_get_seconds_span_duration(self):
        """Test span duration calculation."""
        mock_span = Mock()
        mock_span.start_time = 1000000000  # 1 second in nanoseconds
        mock_span.end_time = 3500000000  # 3.5 seconds in nanoseconds

        duration = get_seconds_span_duration(mock_span)
        assert duration == 2.5  # 3.5 - 1.0 = 2.5 seconds

    def test_get_seconds_span_duration_zero(self):
        """Test span duration with same start and end time."""
        mock_span = Mock()
        mock_span.start_time = 1000000000
        mock_span.end_time = 1000000000

        duration = get_seconds_span_duration(mock_span)
        assert duration == 0.0


class TestMetricCollector:
    """Test suite for metric_collector context manager."""

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer."""
        return Mock()

    @pytest.fixture
    def mock_span(self):
        """Create a mock span with context."""
        span = Mock()
        span_context = Mock()
        span_context.trace_id = 12345
        span.get_span_context.return_value = span_context
        span.start_time = 1000000000
        span.end_time = 2000000000
        return span

    @patch("datapizza.tracing.tracing.ContextTracing._context_processor")
    @patch("datapizza.tracing.tracing.tracer")
    @patch("datapizza.tracing.tracing.console")
    def test_metric_collector_success(
        self, mock_console, mock_tracer, mock_context_processor, mock_span
    ):
        """Test successful metric collection."""
        # Setup mock tracer context manager
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )

        # Setup mock context processor
        mock_context_processor.get_spans_by_trace_id.return_value = [
            mock_span,
            mock_span,
        ]

        # Use the context manager
        with ContextTracing().trace("test_operation"):
            pass

        # Verify interactions
        mock_context_processor.start_trace.assert_called_once_with(12345)
        mock_context_processor.stop_trace.assert_called_once_with(12345)
        mock_context_processor.get_spans_by_trace_id.assert_called_once_with(12345)
        mock_console.print.assert_called_once()

    @patch("datapizza.tracing.tracing.ContextTracing._context_processor")
    @patch("datapizza.tracing.tracing.tracer")
    def test_metric_collector_exception_handling(
        self, mock_tracer, mock_context_processor, mock_span
    ):
        """Test metric collector handles exceptions properly."""
        # Setup mock tracer context manager
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=False
        )

        # Setup exception in the context
        with pytest.raises(ValueError), ContextTracing().trace("test_operation"):
            raise ValueError("Test exception")

        # Verify cleanup still happens
        mock_context_processor.start_trace.assert_called_once_with(12345)
        mock_context_processor.stop_trace.assert_called_once_with(12345)


class TestThreadSafety:
    """Test suite for thread safety of the tracing components."""

    def test_concurrent_export(self):
        """Test that concurrent exports don't cause race conditions."""
        exporter = InMemoryTraceExporter()
        results = []
        errors = []

        def export_spans(thread_id):
            try:
                for i in range(10):
                    span = Mock(spec=ReadableSpan)
                    span_context = Mock(spec=SpanContext)
                    span_context.trace_id = thread_id * 1000 + i
                    span.get_span_context.return_value = span_context

                    result = exporter.export([span])
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=export_spans, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 spans each
        assert all(result == SpanExportResult.SUCCESS for result in results)

        # Verify all spans were stored
        all_spans = exporter.get_finished_spans()
        assert len(all_spans) == 50  # 50 different trace IDs

    def test_concurrent_clear_operations(self):
        """Test concurrent clear operations."""
        exporter = InMemoryTraceExporter()

        # Pre-populate with some spans
        for i in range(10):
            span = Mock(spec=ReadableSpan)
            span_context = Mock(spec=SpanContext)
            span_context.trace_id = i
            span.get_span_context.return_value = span_context
            exporter.export([span])

        errors = []

        def clear_operations():
            try:
                for i in range(5):
                    if i % 2 == 0:
                        exporter.clear()
                    else:
                        exporter.clear_trace(i)
                    time.sleep(
                        0.001
                    )  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple threads doing clear operations
        threads = []
        for _i in range(3):
            thread = threading.Thread(target=clear_operations)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

    def test_processor_thread_safety(self):
        """Test thread safety of ContextSpanProcessor."""
        processor = ContextSpanProcessor()
        errors = []

        def processor_operations(thread_id):
            try:
                trace_id = 11333
                processor.start_trace(trace_id)

                # Create and process some spans
                for _i in range(5):
                    span = Mock(spec=ReadableSpan)
                    span_context = Mock(spec=SpanContext)
                    span_context.trace_id = trace_id
                    span.get_span_context.return_value = span_context

                    processor.on_end(span)

                # Get spans and stop trace
                processor.get_spans_by_trace_id(trace_id)
                processor.stop_trace(trace_id)

            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=processor_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0

        # Verify all traces were cleaned up
        assert len(processor.tracing_ids) == 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_span_export(self):
        """Test exporting empty span list."""
        exporter = InMemoryTraceExporter()
        result = exporter.export([])

        assert result == SpanExportResult.SUCCESS
        assert len(exporter.get_finished_spans()) == 0

    def test_none_span_handling(self):
        """Test handling of None spans."""
        exporter = InMemoryTraceExporter()

        # This should raise an AttributeError when trying to call get_span_context()
        with pytest.raises(AttributeError):
            exporter.export([None])  # type: ignore

    def test_processor_with_invalid_trace_id_type(self):
        """Test processor with different trace ID types."""
        processor = ContextSpanProcessor()

        # Test with integer trace ID
        processor.start_trace(12345)
        assert 12345 in processor.tracing_ids

        processor.stop_trace(12345)
        assert 12345 not in processor.tracing_ids

    def test_multiple_shutdown_calls(self):
        """Test multiple shutdown calls don't cause issues."""
        exporter = InMemoryTraceExporter()

        exporter.shutdown()
        exporter.shutdown()  # Should not raise exception

        assert exporter._stopped is True
