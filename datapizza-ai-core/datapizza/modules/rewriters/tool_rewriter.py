from datapizza.core.clients import Client
from datapizza.core.modules.rewriter import Rewriter
from datapizza.memory.memory import Memory
from datapizza.tools import Tool
from datapizza.tools.tools import tool
from datapizza.type.type import FunctionCallBlock


class ToolRewriter(Rewriter):
    """
    A tool-based query rewriter that uses LLMs to transform user queries through structured tool interactions.
    """

    def __init__(
        self,
        client: Client,
        system_prompt: str | None = None,
        tool: Tool | None = None,
        tool_choice: str = "required",
        tool_output_name: str = "query",
        **invoke_args,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.invoke_args = invoke_args or {}

        if tool is None:
            # using a default tool
            self.tool = Tool(
                name="query_database",
                description="Query a database",
                func=self._search_vectorstore,
            )
        else:
            self.tool = tool

        self.tool_choice = tool_choice
        self.tool_output_name = tool_output_name

    def rewrite(self, user_prompt: str, memory: Memory | None = None) -> str:
        """
        Args:
            user_prompt: The user query to rewrite.
            memory: The memory to use for the rewrite.
            kwargs: Additional arguments to pass to the client.

        Returns:
            The rewritten query.
        """
        response = self.client.invoke(
            input=user_prompt,
            system_prompt=self.system_prompt,
            memory=memory,
            tool_choice=self.tool_choice,
            tools=[self.tool],
            **self.invoke_args,
        )

        if len(response.content) != 1:
            raise ValueError(
                "ToolRewriter supposed to return only one response, something bad occured"
            )
        else:
            if not isinstance(response.content[0], FunctionCallBlock):
                raise ValueError(
                    "ToolRewriter supposed to return a FunctionCallBlock, something bad occured"
                )

            return response.content[0].arguments[self.tool_output_name]

    async def a_rewrite(self, user_prompt: str, memory: Memory | None = None) -> str:
        """
        Args:
            user_prompt: The user query to rewrite.
            memory: The memory to use for the rewrite.
            kwargs: Additional arguments to pass to the client.

        Returns:
            The rewritten query.
        """
        response = await self.client.a_invoke(
            input=user_prompt,
            system_prompt=self.system_prompt,
            memory=memory,
            tool_choice=self.tool_choice,
            tools=[self.tool],
            **self.invoke_args,
        )
        if len(response.content) != 1:
            raise ValueError(
                "ToolRewriter supposed to return only one response, something bad occured"
            )
        else:
            if not isinstance(response.content[0], FunctionCallBlock):
                raise ValueError(
                    "ToolRewriter supposed to return a FunctionCallBlock, something bad occured"
                )
            return response.content[0].arguments[self.tool_output_name]

    @tool
    def _search_vectorstore(self, query: str):
        """
        Search the vectorstore for the most relevant chunks

        Args:
            query: The query to search the vectorstore for

        Returns:
            A list of Chunks that are most relevant to the query
        """
        pass
