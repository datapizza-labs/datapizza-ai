import httpx

from datapizza.tools import Tool


class WebFetchTool(Tool):
    """
    The Web Fetch tool.
    It allows you to fetch the content of a given URL with configurable timeouts
    and specific error handling.
    """

    DEFAULT_TIMEOUT = 10.0
    DEFAULT_USER_AGENT = "DataPizza-AI-Tool/1.0"

    def __init__(self, timeout: float | None = None, user_agent: str | None = None):
        """Initializes the WebFetchTool.

        Args:
            timeout: The timeout for the request in seconds.
            user_agent: The User-Agent header to use for the request.
        """
        super().__init__(
            name="web_fetch",
            description="Fetches the content of a given URL.",
            func=self.__call__,
        )
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.user_agent = (
            user_agent if user_agent is not None else self.DEFAULT_USER_AGENT
        )

    def __call__(self, url: str) -> str:
        """Invoke the tool."""
        headers = {"User-Agent": self.user_agent}

        try:
            with httpx.Client(headers=headers) as client:
                response = client.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response.text
        except httpx.TimeoutException as e:
            return f"Request timed out while requesting {e.request.url!r}."
        except httpx.RequestError as e:
            return f"An error occurred while requesting {e.request.url!r}: {e}"
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            url_str = str(e.request.url)
            if status_code == 404:
                return f"Error: Resource not found at {url_str!r} (404)."
            elif status_code in (401, 403):
                return f"Error: Not authorized to access {url_str!r} ({status_code})."
            elif 400 <= status_code < 500:
                return (
                    f"Error: Client error {status_code} while requesting {url_str!r}."
                )
            elif 500 <= status_code < 600:
                return (
                    f"Error: Server error {status_code} while requesting {url_str!r}."
                )
            else:
                return f"Error response {status_code} while requesting {url_str!r}."
