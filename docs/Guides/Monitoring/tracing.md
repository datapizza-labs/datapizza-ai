# Tracing

The tracing module provides an easy-to-use interface for collecting and displaying OpenTelemetry traces with rich console output. It's designed to help developers monitor performance and understand the execution flow of their applications.

## Features

- **In-memory trace collection** - Stores spans in memory for fast access
- **Context-aware tracking** - Only collects spans for explicitly tracked operations
- **Thread-safe operations** - Safe for use in multi-threaded applications
- **OpenTelemetry integration** - Works with standard OpenTelemetry instrumentation

## Quick Start


The simplest way to use tracing is with the `tracer` context manager:

```python
from datapizza.tracing import ContextTracing


# Basic tracing
with ContextTracing().trace("trace_name"):
    # Your code here
    result = some_datapizza_operations()

# Output will show:
# ╭─ Trace Summary of my_operation ────────────────────────────────── ╮
# │ Total Spans: 3                                                    │
# │ Duration: 2.45s                                                   │
# │ ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# │ ┃ Model       ┃ Prompt Tokens ┃ Completion Tokens ┃ Cached Tokens ┃
# │ ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ │ gpt-4o-mini │ 31            │ 27                │ 0             │
# │ └─────────────┴───────────────┴───────────────────┴───────────────┘
# ╰───────────────────────────────────────────────────────────────────╯
```

## Clients trace input/output/memory

If you want to log the input/output and the memory passed to client invoke you should set the env variable

`DATAPIZZA_TRACE_CLIENT_IO=TRUE`

default is `FALSE`



## Manual Span Creation

For more granular control, create spans manually:

```python
from opentelemetry import trace
from datapizza.tracing import ContextTracing

tracer = trace.get_tracer(__name__)

with ContextTracing().trace("trace_name"):
    with tracer.start_as_current_span("database_query"):
        # Database operation
        data = fetch_from_database()

    with tracer.start_as_current_span("data_validation"):
        # Validation logic
        validate_data(data)

    with tracer.start_as_current_span("business_logic"):
        # Core business logic
        result = process_business_rules(data)
```


## Adding External Exporters

The tracing module uses in-memory storage by default, but you can easily add external exporters to send traces to other systems.

### Create the resource

First of all you should set the trace provider


```python
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

resource = Resource.create(
   {
       ResourceAttributes.SERVICE_NAME: "your_service_name",
   }
)
trace.set_tracer_provider(TracerProvider(resource=resource))


```


### Zipkin Integration

Export traces to Zipkin for visualization and analysis:

`pip install opentelemetry-exporter-zipkin`

After setting the trace provider you can add the exporters

```python
from opentelemetry import trace
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

zipkin_url = "http://localhost:9411/api/v2/spans"

zipkin_exporter = ZipkinExporter(
    endpoint=zipkin_url,
)

tracer_provider = trace.get_tracer_provider()

span_processor = SimpleSpanProcessor(zipkin_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Now all traces will be sent to both in-memory storage and Zipkin
```


### OTLP (OpenTelemetry Protocol)

Export to any OTLP-compatible backend (Grafana, Datadog, etc.):

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    headers={"authorization": "Bearer your-token"}
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
```


## Performance Considerations

- Use `BatchSpanProcessor` for external exporters in production
- Set reasonable limits on span attributes and events
- Monitor memory usage with many active traces

```python
# Production configuration
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Batch spans for better performance
batch_processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    schedule_delay_millis=5000,
    max_export_batch_size=512,
)

trace.get_tracer_provider().add_span_processor(batch_processor)
```
