from datapizza.core.models import PipelineComponent
from datapizza.core.clients.client import Client
from datapizza.type import Model

class StructResponder(PipelineComponent):
    def __init__(self, client: Client, output_cls: type[Model]):
        self.client = client
        self.output_cls = output_cls

    def _run(self, *args, **kwargs) -> str:
        response = self.client.structured_response(*args, output_cls=self.output_cls, **kwargs)
        return {
            "full_response": response,
            "structured_data": response.structured_data,
            "text": response.text,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
            "function_calls": response.function_calls,
            "thoughts": response.thoughts,
            "first_text": response.first_text,
        }
    
    async def _a_run(self, *args, **kwargs) -> str:
        response = await self.client.a_structured_response(*args, output_cls=self.output_cls, **kwargs)
        return {
            "full_response": response,
            "structured_data": response.structured_data,
            "text": response.text,
            "usage": response.usage,
            "stop_reason": response.stop_reason,
            "function_calls": response.function_calls,
            "thoughts": response.thoughts,
            "first_text": response.first_text,
        }
