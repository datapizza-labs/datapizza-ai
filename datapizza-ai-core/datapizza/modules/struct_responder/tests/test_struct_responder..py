from datapizza.clients import MockClient
from datapizza.modules.struct_responder import StructResponder
from pydantic import BaseModel
from datapizza.pipeline import DagPipeline
import os
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


def test_struct_responder():

    class Profile(BaseModel):
        name: str
        age: int
        occupation: str

    pipeline = DagPipeline()
    pipeline.add_module("struct_responder", StructResponder(
        client=client,
        output_cls=Profile,
    ))
    result = pipeline.run({"struct_responder": {"input": "test"}})
    assert result is not None

    print(result.get("struct_responder").get("structured_data")[0])



def test_struct_responder_with_yaml_pipeline():

    yaml_pipeline = """
    pipeline:
      - name: struct_responder
        type: StructResponder
        module: datapizza.modules.struct_responder
        params:
          client: openai
          output_cls: Profile
    """
    pipeline = DagPipeline.from_yaml(yaml_pipeline)
    result = pipeline.run({"struct_responder": {"input": "test"}})
    assert result is not None

    print(result.get("struct_responder").get("structured_data")[0])