from unittest.mock import Mock, patch

import pytest

from datapizza.tracing.instrumentor import (
    DatapizzaMonitoringInstrumentor,
    MissingDatapizzaConfigurationError,
)


@pytest.fixture
def required_env(monkeypatch):
    monkeypatch.setenv("DATAPIZZA_API_KEY", "test_api_key")
    monkeypatch.setenv("DATAPIZZA_PROJECT_ID", "test_project_id")
    monkeypatch.setenv("DATAPIZZA_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")


def test_instrumentor_reads_required_env(required_env):
    instrumentor = DatapizzaMonitoringInstrumentor()

    assert instrumentor.api_key == "test_api_key"
    assert instrumentor.project_id == "test_project_id"


def test_instrumentor_raises_when_required_env_missing(monkeypatch):
    monkeypatch.delenv("DATAPIZZA_API_KEY", raising=False)
    monkeypatch.delenv("DATAPIZZA_PROJECT_ID", raising=False)
    monkeypatch.delenv("DATAPIZZA_OTLP_ENDPOINT", raising=False)

    with pytest.raises(MissingDatapizzaConfigurationError):
        DatapizzaMonitoringInstrumentor()


def test_explicit_values_override_env(required_env):
    instrumentor = DatapizzaMonitoringInstrumentor(
        api_key="explicit_api_key",
        project_id="explicit_project_id",
        service_name="test-service",
        service_version="9.9.9",
        endpoint="http://localhost:4318/v1/traces",
    )

    assert instrumentor.api_key == "explicit_api_key"
    assert instrumentor.project_id == "explicit_project_id"
    assert instrumentor.service_name == "test-service"
    assert instrumentor.service_version == "9.9.9"
    assert instrumentor.endpoint == "http://localhost:4318/v1/traces"


def test_get_tracer_requires_instrument(required_env):
    instrumentor = DatapizzaMonitoringInstrumentor()

    with pytest.raises(RuntimeError, match=r"Call instrument\(\) first"):
        instrumentor.get_tracer(__name__)


def test_instrument_adds_span_processor_once(required_env):
    fake_provider = Mock()

    with (
        patch(
            "datapizza.tracing.instrumentor.trace.get_tracer_provider",
            return_value=fake_provider,
        ),
        patch("datapizza.tracing.instrumentor.OTLPSpanExporter") as mock_exporter,
        patch("datapizza.tracing.instrumentor.BatchSpanProcessor") as mock_processor,
    ):
        mock_exporter.return_value = Mock()
        mock_processor.return_value = Mock()

        instrumentor = DatapizzaMonitoringInstrumentor()
        instrumentor.instrument()
        instrumentor.instrument()

    assert fake_provider.add_span_processor.call_count == 1


def test_from_env_returns_ready_instrumentor(required_env):
    instrumentor = DatapizzaMonitoringInstrumentor.from_env()

    assert isinstance(instrumentor, DatapizzaMonitoringInstrumentor)
    assert instrumentor.api_key == "test_api_key"
