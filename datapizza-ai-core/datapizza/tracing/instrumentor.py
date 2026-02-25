import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import ProxyTracerProvider


class MissingDatapizzaConfigurationError(ValueError):
    pass


class DatapizzaMonitoringInstrumentor:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        endpoint: str | None = None,
    ):
        resolved_api_key = self._resolve_optional_value(api_key, "DATAPIZZA_API_KEY")
        resolved_project_id = self._resolve_optional_value(
            project_id, "DATAPIZZA_PROJECT_ID"
        )

        missing = []
        if resolved_api_key is None:
            missing.append("DATAPIZZA_API_KEY")
        if resolved_project_id is None:
            missing.append("DATAPIZZA_PROJECT_ID")
        resolved_endpoint = self._resolve_optional_value(
            endpoint, "DATAPIZZA_OTLP_ENDPOINT"
        )
        if resolved_endpoint is None:
            missing.append("DATAPIZZA_OTLP_ENDPOINT")
        if missing:
            missing_list = ", ".join(missing)
            raise MissingDatapizzaConfigurationError(
                f"Missing required configuration: {missing_list}"
            )

        self.api_key = resolved_api_key
        self.project_id = resolved_project_id
        self.service_name = (
            self._resolve_optional_value(service_name, "OTEL_SERVICE_NAME")
            or "datapizza"
        )
        self.service_version = (
            self._resolve_optional_value(service_version, "OTEL_SERVICE_VERSION")
            or "0.1.0"
        )
        self.endpoint = resolved_endpoint

        self._is_instrumented = False

    @classmethod
    def from_env(cls):
        return cls()

    @staticmethod
    def _clean_value(value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        if not cleaned:
            return None

        return cleaned

    def _resolve_optional_value(
        self, explicit_value: str | None, env_name: str
    ) -> str | None:
        value = self._clean_value(explicit_value)
        if value is not None:
            return value

        return self._clean_value(os.getenv(env_name))

    def instrument(self) -> None:
        if self._is_instrumented:
            return

        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": self.service_version,
            }
        )

        exporter = OTLPSpanExporter(
            endpoint=self.endpoint,
            headers={
                "x-project-id": self.project_id,
                "authorization": f"Bearer {self.api_key}",
            },
        )

        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, ProxyTracerProvider):
            trace.set_tracer_provider(TracerProvider(resource=resource))
            current_provider = trace.get_tracer_provider()

        if not hasattr(current_provider, "add_span_processor"):
            raise RuntimeError(
                "Tracer provider does not support span processors. "
                "Set an OpenTelemetry SDK TracerProvider before calling instrument()."
            )

        span_processor = BatchSpanProcessor(exporter)
        current_provider.add_span_processor(span_processor)
        self._is_instrumented = True

    def get_tracer(self, name: str, version: str | None = None):
        if not self._is_instrumented:
            raise RuntimeError(
                "Datapizza monitoring is not instrumented. Call instrument() first."
            )

        return trace.get_tracer(name, version)
