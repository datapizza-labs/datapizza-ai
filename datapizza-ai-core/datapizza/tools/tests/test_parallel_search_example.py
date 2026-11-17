"""Exercise the runnable example with local Streamable HTTP and a model double."""

import json
import runpy
import socket
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from datapizza.agents import Agent
from datapizza.clients import MockClient
from datapizza.core.clients import ClientResponse
from datapizza.tools.mcp_client import MCPClient
from datapizza.type import (
    FunctionCallBlock,
    FunctionCallResultBlock,
    StructuredBlock,
    TextBlock,
)

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "parallel_search.py"
SOURCE_URL = "https://docs.python.org/3/library/asyncio-task.html#task-groups"


@pytest.fixture
def mcp_server(request):
    mode = getattr(request, "param", "success")
    calls = []
    headers = []
    mcp = FastMCP(
        "documentation-search-fixture", stateless_http=True, json_response=True
    )

    @mcp.tool()
    def web_fetch(urls: list[str]) -> str:
        """An unrelated tool, deliberately listed before web_search."""
        pytest.fail("The example must expose only web_search")

    if mode != "missing":

        @mcp.tool()
        def web_search(objective: str, search_queries: list[str]) -> dict:
            """Search public documentation."""
            calls.append({"objective": objective, "search_queries": search_queries})
            if mode == "error":
                raise ValueError("Search unavailable")
            return {
                "results": []
                if mode == "empty"
                else [
                    {
                        "title": "Task groups",
                        "url": SOURCE_URL,
                        "excerpts": [
                            "TaskGroup is an asynchronous context manager that "
                            "waits for all tasks in the group to finish."
                        ],
                    }
                ]
            }

    app = mcp.streamable_http_app()

    async def record_headers(scope, receive, send):
        if scope["type"] == "http":
            headers.append(dict(scope["headers"]))
        await app(scope, receive, send)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(record_headers, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive():
                if time.monotonic() >= deadline:
                    pytest.fail("Local MCP fixture did not start within five seconds")
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}/mcp", calls, headers
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive(), "Local MCP fixture did not stop"


class DocumentationClient(MockClient):
    """Deterministic tool dispatch and structured output, with no provider key."""

    def __init__(self):
        super().__init__()
        self.choices = []
        self.tool_result = None
        self.formatted_input = None

    async def _a_invoke(
        self, input, tools=None, memory=None, tool_choice="auto", **kwargs
    ):
        self.choices.append(tool_choice)
        assert [tool.name for tool in tools] == ["web_search"]
        assert set(tools[0].required) == {"objective", "search_queries"}
        if len(self.choices) == 1:
            assert "TaskGroup" in input[0].content
            return ClientResponse(
                content=[
                    FunctionCallBlock(
                        id="docs-search",
                        name="web_search",
                        tool=tools[0],
                        arguments={
                            "objective": "Find the Python documentation for TaskGroup.",
                            "search_queries": [
                                "site:docs.python.org asyncio TaskGroup"
                            ],
                        },
                    )
                ]
            )

        block = memory[-1].blocks[-1]
        assert isinstance(block, FunctionCallResultBlock)
        self.tool_result = json.loads(block.result)
        if self.tool_result.get("isError"):
            results = []
        else:
            results = json.loads(self.tool_result["content"][0]["text"])["results"]
        answer = {
            "answer": results[0]["excerpts"][0] if results else "Answer unavailable.",
            "sources": [result["url"] for result in results],
        }
        return ClientResponse(content=[TextBlock(content=json.dumps(answer))])

    async def _a_structured_response(
        self,
        input,
        output_cls,
        memory=None,
        temperature=None,
        max_tokens=None,
        system_prompt=None,
        tools=None,
        tool_choice="auto",
        **kwargs,
    ):
        assert not tools
        self.formatted_input = input[0].content
        return ClientResponse(
            content=[
                StructuredBlock(
                    content=output_cls.model_validate_json(input[0].content)
                )
            ]
        )


@pytest.mark.asyncio
async def test_example_dispatches_search_and_preserves_original_url(mcp_server):
    example = runpy.run_path(str(EXAMPLE))
    url, calls, headers = mcp_server
    client = DocumentationClient()

    answer = await example["answer_question"](client, mcp_url=url)

    assert client.choices == ["required", "auto"]
    assert len(calls) == 1
    assert calls[0]["search_queries"] == ["site:docs.python.org asyncio TaskGroup"]
    assert answer.model_dump(mode="json")["sources"] == [SOURCE_URL]
    assert SOURCE_URL in client.formatted_input
    assert "waits for all tasks" in answer.answer
    assert headers
    assert all(h[b"user-agent"].decode() == example["USER_AGENT"] for h in headers)
    assert all(b"authorization" not in h and b"x-api-key" not in h for h in headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("mcp_server", ["empty", "error"], indirect=True)
async def test_example_can_return_unavailable_without_inventing_sources(mcp_server):
    example = runpy.run_path(str(EXAMPLE))
    url, calls, _ = mcp_server
    client = DocumentationClient()

    answer = await example["answer_question"](client, mcp_url=url)

    assert len(calls) == 1
    assert answer.sources == []
    assert answer.answer == "Answer unavailable."
    assert client.choices == ["required", "auto"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mcp_server", ["missing"], indirect=True)
async def test_example_reports_missing_search(mcp_server):
    example = runpy.run_path(str(EXAMPLE))
    client = DocumentationClient()

    with pytest.raises(RuntimeError, match="did not provide web_search"):
        await example["answer_question"](client, mcp_url=mcp_server[0])

    assert client.choices == []


def test_import_is_inactive_and_default_agent_still_runs(monkeypatch):
    def unexpected_connection(*args, **kwargs):
        pytest.fail("MCP must be opt-in")

    monkeypatch.setattr(
        "datapizza.tools.mcp_client.MCPClient.a_list_tools", unexpected_connection
    )
    runpy.run_path(str(EXAMPLE))
    assert Agent(name="default", client=MockClient()).run("Hello").text == "Hello"


def test_sources_validate_http_urls_without_rewriting_them():
    answer_cls = runpy.run_path(str(EXAMPLE))["DocsAnswer"]
    assert answer_cls(answer="Documented", sources=[SOURCE_URL]).sources == [SOURCE_URL]
    with pytest.raises(ValidationError):
        answer_cls(answer="Invalid", sources=["file:///private/input"])


def test_runnable_entry_point(mcp_server, monkeypatch, capsys):
    url, calls, _ = mcp_server
    client = DocumentationClient()

    def local_mcp(**kwargs):
        assert kwargs["url"] == "https://mcp.parallel.ai/mcp"
        return MCPClient(**{**kwargs, "url": url})

    def model_client(*, api_key, model):
        assert api_key == "local-fixture-only"
        assert model == "local-fixture-only"
        return client

    provider = ModuleType("datapizza.clients.openai")
    provider.OpenAIClient = model_client
    monkeypatch.setitem(sys.modules, "datapizza.clients.openai", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "local-fixture-only")
    monkeypatch.setenv("OPENAI_MODEL", "local-fixture-only")
    monkeypatch.setattr("datapizza.tools.mcp_client.MCPClient", local_mcp)

    runpy.run_path(str(EXAMPLE), run_name="__main__")

    output = capsys.readouterr().out
    answer = json.loads(output[output.rfind("\n{") :])
    assert answer["sources"] == [SOURCE_URL]
    assert len(calls) == 1
