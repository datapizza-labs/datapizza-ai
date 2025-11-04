from datapizza.core.models import PipelineComponent
from datapizza.core.clients.client import Client
from datapizza.type import Model

class StructResponder(PipelineComponent):
    def __init__(self, client: Client, output_cls: type[Model]):
        self.client = client
        self.output_cls = output_cls

    def _run(self, *args, **kwargs) -> str:
        return self.client.structured_response(*args, output_cls=self.output_cls, **kwargs).structured_data[0]
    
    async def _a_run(self, *args, **kwargs) -> str:
        return (await self.client.a_structured_response(*args, output_cls=self.output_cls, **kwargs)).structured_data[0]
