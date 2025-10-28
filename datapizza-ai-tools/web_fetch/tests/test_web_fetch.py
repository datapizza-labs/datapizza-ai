import httpx
import pytest
from httpx import Request

from datapizza.tools.web_fetch import WebFetchTool


@pytest.fixture
def tool():
    """Provides a WebFetchTool instance."""
    return WebFetchTool()


def test_web_fetch_tool_success(tool, httpx_mock):
    """Test a successful web fetch."""
    url = "https://example.com"
    httpx_mock.add_response(url=url, text="Hello, world!")

    result = tool(url)

    assert result == "Hello, world!"
    request = httpx_mock.get_requests()[0]
    assert "user-agent" in request.headers


def test_web_fetch_tool_timeout(tool, httpx_mock):
    """Test the tool's timeout handling."""
    url = "https://example.com/timeout"
    httpx_mock.add_exception(
        httpx.TimeoutException("Timeout!", request=Request("GET", url))
    )

    result = tool(url)

    assert "Request timed out" in result


def test_web_fetch_tool_not_found_error(tool, httpx_mock):
    """Test the 404 Not Found error handling."""
    url = "https://example.com/404"
    httpx_mock.add_response(url=url, status_code=404)

    result = tool(url)

    assert "Resource not found" in result


def test_web_fetch_tool_server_error(tool, httpx_mock):
    """Test a 500 server error."""
    url = "https://example.com/server-error"
    httpx_mock.add_response(url=url, status_code=500)

    result = tool(url)

    assert "Server error 500" in result


def test_web_fetch_tool_request_error(tool, httpx_mock):
    """Test a generic request error."""
    url = "https://example.com/request-error"
    httpx_mock.add_exception(
        httpx.RequestError("Some error", request=Request("GET", url))
    )

    result = tool(url)

    assert "An error occurred" in result


def test_web_fetch_tool_custom_user_agent(httpx_mock):
    """Test that a custom user agent is correctly passed."""
    url = "https://example.com/custom-ua"
    custom_ua = "MyTestAgent/1.0"
    tool = WebFetchTool(user_agent=custom_ua)
    httpx_mock.add_response(url=url, text="Success")

    tool(url)

    request = httpx_mock.get_requests()[0]
    assert request.headers["user-agent"] == custom_ua


def test_web_fetch_tool_custom_timeout(httpx_mock):
    """Test that a custom timeout is correctly passed."""
    url = "https://example.com/custom-timeout"
    custom_timeout = 5.0
    tool = WebFetchTool(timeout=custom_timeout)
    httpx_mock.add_response(url=url, text="Success")

    tool(url)

    request = httpx_mock.get_requests()[0]
    assert request.extensions["timeout"]["connect"] == custom_timeout
