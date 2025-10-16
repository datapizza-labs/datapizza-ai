
# Functional Pipeline

> **_WARNING:_**  This module is in beta. Signatures and interfaces may change in future releases.

The `FunctionalPipeline` module provides a flexible way to build data processing pipelines with complex dependency graphs. It allows you to define reusable processing nodes and connect them in various patterns including sequential execution, branching, parallel execution, and foreach loops.

## Core Components


### Dependency

Defines how data flows between [Nodes](../../API%20Reference/Type/node.md):

```python
@dataclass
class Dependency:
    node_name: str
    input_key: str | None = None
    target_key: str | None = None
```

- `node_name`: The name of the node to get data from
- `input_key`: Optional key for extracting a specific part of the node's output
- `target_key`: The key under which to store the data in the receiving node's input

### FunctionalPipeline

The main class for building and executing pipelines:

```python
class FunctionalPipeline:
    def __init__(self):
        self.nodes = []
```

## Building Pipelines

### Sequential Execution

```python
pipeline = FunctionalPipeline()
pipeline.run("load_data", DataLoader(), kwargs={"filepath": "data.csv"})
pipeline.then("transform", Transformer(), target_key="data")
pipeline.then("save", Saver(), target_key="transformed_data")
```

### Branching

```python
pipeline.branch(
    condition=is_valid_data,
    if_true=valid_data_pipeline,
    if_false=invalid_data_pipeline,
    dependencies=[Dependency(node_name="validate", target_key="validation_result")]
)
```

### Foreach Loop

```python
pipeline.foreach(
    name="process_items",
    do=item_processing_pipeline,
    dependencies=[Dependency(node_name="get_items")]
)
```

## Executing Pipelines

```python
result = pipeline.execute(
    initial_data={"load_data": {"filepath": "override.csv"}},
    context={"existing_data": {...}}
)
```

## YAML Configuration

You can define pipelines in YAML and load them at runtime:
This is useful for separating pipeline structure from code

```yaml
modules:
  - name: data_loader
    module: my_package.loaders
    type: CSVLoader
    params:
      encoding: "utf-8"

  - name: transformer
    module: my_package.transformers
    type: StandardTransformer

pipeline:
  - type: run
    name: load_data
    node: data_loader
    kwargs:
      filepath: "data.csv"

  - type: then
    name: transform
    node: transformer
    target_key: data
```

Load the pipeline:

```python
pipeline = FunctionalPipeline.from_yaml("pipeline_config.yaml")
result = pipeline.execute()
```


## Real-world Examples

### Question Answering Pipeline

Here's an example of a question answering pipeline that uses embeddings to retrieve relevant information and an LLM to generate a response:


Define the components:
```python
from datapizza.clients.google import GoogleClient
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import Dependency, FunctionalPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from dotenv import load_dotenv

load_dotenv()

rewriter = ToolRewriter(
    client=OpenAIClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt="Use only 1 time the tool to answer the user prompt.",
    )
)
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)

vector_store = QdrantVectorstore(host="localhost", port=6333)
vector_store.create_collection(collection_name="my_documents", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])
vector_store = vector_store.as_module_component() # required to use the vectorstore in the pipeline

prompt_template = ChatPromptTemplate(
    user_prompt_template="this is a user prompt: {{ user_prompt }}",
    retrieval_prompt_template="{% for chunk in chunks %} Relevant chunk: {{ chunk.text }} \n\n {% endfor %}",
)
generator = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    system_prompt="You are a senior Software Engineer. You are given a user prompt and you need to answer it given the context of the chunks.",
).as_module_component()

```

And now create and execute the pipeline

```python
pipeline = (FunctionalPipeline()
    .run(name="rewriter", node=rewriter, kwargs={"user_prompt": "tell me something about this document"})
    .then(name="embedder", node=embedder, target_key="text")
    .then(name="vector_store", node=vector_store, target_key="query_vector",
          kwargs={"collection_name": "my_documents", "k": 4})
    .then(name="prompt_template", node=prompt_template, target_key="chunks" , kwargs={"user_prompt": "tell me something about this document"})
    .then(name="generator", node=generator, target_key="memory", kwargs={"input": "tell me something about this document"})
    .get("generator")
)

result = pipeline.execute()
print(result)
```

When using `.then()`, the `target_key` parameter specifies the input parameter name for the current node's `run()` method that will receive the output from the previous node. In other words, `target_key` defines how the previous node's output gets mapped into the current node's `run()` method parameters.


This pipeline:

1. [Rewrites/processes](../../API%20Reference/Modules/rewriters.md) the user query
2. [Creates embeddings](../../API%20Reference/Embedders/chunk_embedder.md) from the processed query
3. Retrieves relevant chunks from a [vector database](../../API%20Reference/Vectorstore/qdrant_vectorstore.md)
4. [Creates a prompt template](../../API%20Reference/Modules/Prompt/ChatPromptTemplate.md) with the retrieved context
5. Generates a response using an LLM
6. Returns the generated response


### Branch and loop usage example

```python
from datapizza.core.models import PipelineComponent
from datapizza.pipeline import Dependency, FunctionalPipeline


class Scraper(PipelineComponent):
    def _run(self, number_of_links: int = 1):
        return ["example.com"] * number_of_links

class UpperComponent(PipelineComponent):
    def _run(self, item):
        return item.upper()

class SendNotification(PipelineComponent):
    def _run(self ):
        return "No Url found, Notification sent"

send_notification = FunctionalPipeline().run(name="send_notification", node=SendNotification())

upper_elements = FunctionalPipeline().foreach(
    name="loop_links",
    dependencies=[Dependency(node_name="get_link")],
    do=UpperComponent(),
)

pipeline = (
    FunctionalPipeline()
    .run(name="get_link", node=Scraper())
    .branch(
        condition=lambda pipeline_context: len(pipeline_context.get("get_link")) > 0,
        dependencies=[Dependency(node_name="get_link")],
        if_true=upper_elements,
        if_false=send_notification,
    )
)

results = pipeline.execute(initial_data={"get_link": {"number_of_links": 0}}) # put 1 to test the other branch
print(results)
```
