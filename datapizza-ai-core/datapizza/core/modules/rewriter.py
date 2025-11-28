from abc import abstractmethod

from datapizza.core.models import PipelineComponent
from datapizza.memory import Memory


class Rewriter(PipelineComponent):
    """
    A rewriter that uses language models to transform text content with specific instructions and tools.
    """

    @abstractmethod
    def rewrite(self, user_prompt: str, memory: Memory | None = None) -> str:
        pass

    async def a_rewrite(self, user_prompt: str, memory: Memory | None = None) -> str:
        raise NotImplementedError

    def _run(self, user_prompt: str, memory: Memory | None = None, **kwargs) -> str:
        return self.rewrite(user_prompt, memory, **kwargs)

    async def _a_run(
        self, user_prompt: str, memory: Memory | None = None, **kwargs
    ) -> str:
        return await self.a_rewrite(user_prompt, memory, **kwargs)
