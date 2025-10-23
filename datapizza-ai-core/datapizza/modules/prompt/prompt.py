import uuid

from jinja2.sandbox import SandboxedEnvironment

from datapizza.core.modules.prompt import Prompt
from datapizza.memory.memory import Memory
from datapizza.tools import Tool, tool
from datapizza.type import (
    ROLE,
    Chunk,
    FunctionCallBlock,
    FunctionCallResultBlock,
    TextBlock,
)


class ChatPromptTemplate(Prompt):
    """
    It takes as input a Memory, Chunks, Prompt and creates a Memory
    with all existing messages + the user's qry + function_call_retrieval +
    chunks retrieval.
    args:
        user_prompt_template: str # The user prompt jinja template
        retrieval_prompt_template: str # The retrieval prompt jinja template
    """

    def __init__(self, user_prompt_template, retrieval_prompt_template):
        env = SandboxedEnvironment()

        self.user_prompt_template = env.from_string(user_prompt_template)
        self.retrieval_prompt_template = env.from_string(retrieval_prompt_template)

    def format(
        self,
        memory: Memory | None = None,
        chunks: list[Chunk] | None = None,
        user_prompt: str = "",
        retrieval_query: str = "",
    ) -> Memory:
        """
        Creates a new memory object that includes:
        - Existing memory messages
        - User's query
        - Function call retrieval results
        - Chunks retrieval results

        Args:
            memory: The memory object to add the new messages to.
            chunks: The chunks to add to the memory.
            user_prompt: The user's query.
            retrieval_query: The query to search the vectorstore for.

        Returns:
            A new memory object with the new messages.
        """

        new_memory = Memory()

        # Add existing memory if any
        if memory:
            for turn in memory:
                new_memory.add_turn(turn.blocks, turn.role)

        # Add user's prompt
        formatted_user_prompt = self.user_prompt_template.render(
            user_prompt=user_prompt
        )
        new_memory.add_turn(
            blocks=[TextBlock(content=formatted_user_prompt)], role=ROLE.USER
        )

        tool_id = str(uuid.uuid4())

        if chunks is not None:
            new_memory.add_turn(
                blocks=FunctionCallBlock(
                    id=tool_id,
                    arguments={"query": retrieval_query},
                    name="search_vectorstore",
                    tool=Tool(func=self._search_vectorstore),
                ),
                role=ROLE.ASSISTANT,
            )

            formatted_retrieval = self.retrieval_prompt_template.render(chunks=chunks)
            new_memory.add_turn(
                blocks=FunctionCallResultBlock(
                    id=tool_id,
                    tool=Tool(func=self._search_vectorstore),
                    result=formatted_retrieval,
                ),
                role=ROLE.TOOL,
            )

        return new_memory

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
