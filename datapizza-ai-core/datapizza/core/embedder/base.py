from abc import abstractmethod

from datapizza.core.models import PipelineComponent
from datapizza.type import SparseEmbedding
from typing import TypeVar, Generic

T = TypeVar('T')

class BaseEmbedder(PipelineComponent,Generic[T]):
    client: object
    a_client: object

    def _get_client(self):
        if not self.client:
            self._set_client()
        return self.client

    def _get_a_client(self):
        if not self.a_client:
            self._set_a_client()
        return self.a_client

    def _set_client(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _set_a_client(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _run(self, text: str):
        return self.embed(text)

    async def _a_run(self, text: str):
        return await self.a_embed(text)

    @abstractmethod
    def embed(self, text: str | list[str], **kwargs) -> T | list[T]:
        pass

    async def a_embed(self, text: str | list[str], **kwargs):
        raise NotImplementedError
