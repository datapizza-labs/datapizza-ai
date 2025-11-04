from datapizza.core.models import PipelineComponent
from datapizza.core.clients.client import Client
from datapizza.type import Model
from pydantic import BaseModel
import importlib

class StructResponder(PipelineComponent):
    def __init__(self, client: Client, output_cls: type[Model | str], *args, **kwargs):
        self.client = client
        if isinstance(output_cls, str):
            try:
                # import module and class from string
                module_name, class_name = output_cls.rsplit(".", 1)
                module = importlib.import_module(module_name)
                self.output_cls = getattr(module, class_name)
            except Exception as e:
                raise ValueError(f"Error importing module {module_name} and class {class_name}: {e} \n If you are using this Module in a pipeline, make sure that output_cls is the module path to a Pydantic model.") from None
            assert issubclass(self.output_cls, BaseModel), f"Class {class_name} found, but it is not a Pydantic model (must be a subclass of BaseModel)."

        else:
            self.output_cls = output_cls

        # save args and kwargs, to allow override them in the pipeline definition
        self.args = args
        self.kwargs = kwargs

    def _run(self, *args, **kwargs) -> str:
        # merge args and kwargs
        args = self.args + args
        kwargs = self.kwargs | kwargs
        response = self.client.structured_response(output_cls=self.output_cls, *args, **kwargs)
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
        # merge args and kwargs
        args = self.args + args
        kwargs = self.kwargs | kwargs
        response = await self.client.a_structured_response(output_cls=self.output_cls, *args, **kwargs)
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
