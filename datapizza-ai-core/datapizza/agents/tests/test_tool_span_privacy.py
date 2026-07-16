import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from datapizza.agents.agent import Agent
from datapizza.agents.runner import AgentRunner
from datapizza.clients import MockClient
from datapizza.tools import tool
from datapizza.type import FunctionCallBlock

TRACE_ENV = "DATAPIZZA_TRACE_CLIENT_IO"


@tool
def reveal(secret: str) -> str:
    return f"SENSITIVE:{secret}"


@tool
def empty_tool() -> str:
    return ""


def _make_call():
    return FunctionCallBlock(
        id="call-1",
        arguments={"secret": "hunter2"},
        name=reveal.name,
        tool=reveal,
    )


def _capture_span():
    span = MagicMock()

    @contextmanager
    def fake_tool_span(name=None):
        yield span

    return span, fake_tool_span


def _attribute_keys(span):
    return [call.args[0] for call in span.set_attribute.call_args_list]


def test_tool_span_omits_io_by_default(monkeypatch):
    monkeypatch.delenv(TRACE_ENV, raising=False)
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(_make_call(), agent)

    keys = _attribute_keys(span)
    assert "tool_arguments" not in keys
    assert "tool_result" not in keys


def test_tool_span_includes_io_when_enabled(monkeypatch):
    monkeypatch.setenv(TRACE_ENV, "true")
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(_make_call(), agent)

    keys = _attribute_keys(span)
    assert "tool_arguments" in keys
    assert "tool_result" in keys


def test_a_tool_span_omits_io_by_default(monkeypatch):
    monkeypatch.delenv(TRACE_ENV, raising=False)
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        asyncio.run(AgentRunner()._a_execute_tool(_make_call(), agent))

    keys = _attribute_keys(span)
    assert "tool_arguments" not in keys
    assert "tool_result" not in keys


def test_a_tool_span_includes_io_when_enabled(monkeypatch):
    monkeypatch.setenv(TRACE_ENV, "true")
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        asyncio.run(AgentRunner()._a_execute_tool(_make_call(), agent))

    keys = _attribute_keys(span)
    assert "tool_arguments" in keys
    assert "tool_result" in keys


def test_tool_span_includes_io_with_uppercase_env(monkeypatch):
    # The docs instruct DATAPIZZA_TRACE_CLIENT_IO=TRUE (uppercase); the gate must honor it.
    monkeypatch.setenv(TRACE_ENV, "TRUE")
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(_make_call(), agent)

    keys = _attribute_keys(span)
    assert "tool_arguments" in keys
    assert "tool_result" in keys


def test_tool_span_omits_io_for_non_true_value(monkeypatch):
    # Only the literal true/TRUE enables (parity with the client I/O gate); "1" must not.
    monkeypatch.setenv(TRACE_ENV, "1")
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(_make_call(), agent)

    keys = _attribute_keys(span)
    assert "tool_arguments" not in keys
    assert "tool_result" not in keys


def test_falsy_tool_result_never_sets_result_attribute(monkeypatch):
    # Even with tracing enabled, a falsy result must not produce a tool_result attribute.
    monkeypatch.setenv(TRACE_ENV, "true")
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[empty_tool])
    call = FunctionCallBlock(
        id="call-empty", arguments={}, name=empty_tool.name, tool=empty_tool
    )

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(call, agent)

    keys = _attribute_keys(span)
    assert "tool_arguments" in keys
    assert "tool_result" not in keys


def test_return_value_unchanged_regardless_of_flag(monkeypatch):
    _, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        monkeypatch.delenv(TRACE_ENV, raising=False)
        off = AgentRunner()._execute_tool(_make_call(), agent)
        monkeypatch.setenv(TRACE_ENV, "true")
        on = AgentRunner()._execute_tool(_make_call(), agent)

    assert off.result == "SENSITIVE:hunter2"
    assert on.result == off.result


def test_console_log_panel_called_even_when_tracing_disabled(monkeypatch):
    # The console channel is intentionally independent of the trace gate: it must
    # still fire while the span omits the sensitive attributes.
    monkeypatch.delenv(TRACE_ENV, raising=False)
    span, fake_tool_span = _capture_span()
    agent = Agent(name="t", client=MockClient(), tools=[reveal])
    agent._logger = MagicMock()

    with patch("datapizza.agents.runner.tool_span", fake_tool_span):
        AgentRunner()._execute_tool(_make_call(), agent)

    assert agent._logger.log_panel.called
    assert "tool_result" not in _attribute_keys(span)
