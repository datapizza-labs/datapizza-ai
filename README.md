<div align="center">

<img src="docs/assets/logo_bg_dark.png" alt="Datapizza AI Logo" width="200" height="200">

**Build reliable Gen AI solutions without overhead**

*Written in Python. Designed for speed. A no-fluff GenAI framework that gets your agents from dev to prod, fast*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/datapizza-ai.svg)](https://pypi.org/project/datapizza-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://img.shields.io/pypi/dm/datapizza-ai.svg)](https://pypi.org/project/datapizza-ai/)
[![GitHub stars](https://img.shields.io/github/stars/datapizza-labs/datapizza-ai.svg?style=social&label=Star)](https://github.com/datapizza-labs/datapizza-ai)

[🏠Homepage](https://datapizza.tech/en/ai-framework/) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](https://docs.datapizza.ai) • [🎯 Examples](#-examples) • [🤝 Community](#-community)

</div>

---

## 🌟 Why Datapizza AI?

A framework that keeps your agents predictable, your debugging fast, and your code trusted in production. Built by Engineers, trusted by Engineers.

<div align="center">

### ⚡ **Less abstraction, more control** | 🚀 **API-first design** | 🔧 **Observable by design**

</div>

## How to install
```sh
pip install datapizza-ai
```

## Client invoke

```python
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(api_key="YOUR_API_KEY")
result = client.invoke("Hi, how are u?")
print(result.text)
```

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🎯 **API-first**
- **Multi-Provider Support**: OpenAI, Google Gemini, Anthropic, Mistral, Azure
- **Tool Integration**: Built-in web search, document processing, custom tools
- **Memory Management**: Persistent conversations and context awareness

</td>
<td width="50%" valign="top">

### 🔍 **Composable**
- **Reusable blocks**: Declarative configuration, easy overrides
- **Document Processing**: PDF, DOCX, images with Azure AI & Docling
- **Smart Chunking**: Context-aware text splitting and embedding
- **Built-in reranking**: Add a reranker (e.g., Cohere) to boost relevance

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔧 **Observable**
- **OpenTelemetry tracing**: Standards-based instrumentation
- **Client I/O tracing**: Optional toggle to log inputs, outputs, and in-memory context
- **Custom spans**: Trace fine-grained phases and sub-steps to pinpoint bottlenecks

</td>
<td width="50%" valign="top">

### 🚀 **Vendor-Agnostic**
- **Swap models**: Change providers without rewiring business logic
- **Clear Interfaces**: Predictable APIs across all components
- **Rich Ecosystem**: Modular design with optional components
- **Migration-friendly**: Quick migration from other frameworks

</td>
</tr>
</table>

## 🚀 Quick Start

### Installation

```bash
# Core framework
pip install datapizza-ai

# With specific providers (optional)
pip install datapizza-ai-clients-openai
pip install datapizza-ai-clients-google
pip install datapizza-ai-clients-anthropic
```

### Start with Agent

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool

@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

client = OpenAIClient(api_key="YOUR_API_KEY")
agent = Agent(name="assistant", client=client, tools = [get_weather])

response = agent.run("What is the weather in Rome?")
# output: The weather in Rome is sunny
```


## 📊 Detailed Tracing


A key requirement for principled development of LLM applications over your data (RAG systems, agents) is being able to observe and debug.

Datapizza-ai provides built-in observability with OpenTelemetry tracing to help you monitor performance and understand execution flow.

<summary><b>🔍 Trace Your AI Operations</b></summary>

```sh
pip install datapizza-ai-tools-duckduckgo
```

```python
from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
from datapizza.tracing import ContextTracing

client = OpenAIClient(api_key="OPENAI_API_KEY")
agent = Agent(name="assistant", client=client, tools = [DuckDuckGoSearchTool()])

with ContextTracing().trace("my_ai_operation"):
    response = agent.run("Tell me some news about Bitcoin")

# Output shows:
# ╭─ Trace Summary of my_ai_operation ──────────────────────────────────╮
# │ Total Spans: 3                                                      │
# │ Duration: 2.45s                                                     │
# │ ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓ |
# │ ┃ Model       ┃ Prompt Tokens ┃ Completion Tokens ┃ Cached Tokens ┃ |
# │ ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩ |
# │ │ gpt-4o-mini │ 31            │ 27                │ 0             │ |
# │ └─────────────┴───────────────┴───────────────────┴───────────────┘ |
# ╰─────────────────────────────────────────────────────────────────────╯
```


![Demo](https://github.com/user-attachments/assets/02742e87-aa48-4308-94c8-6f362e3218b4)


## 🎯 Examples

### 🌐 Multi-Agent System

Build sophisticated AI systems where multiple specialized agents collaborate to solve complex tasks. This example shows how to create a trip planning system with dedicated agents for weather information, web search, and planning coordination.

```sh
# Install DuckDuckGo tool
pip install datapizza-ai-tools-duckduckgo
```


```python
from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4.1")

@tool
def get_weather(city: str) -> str:
    return f""" it's sunny all the week in {city}"""

weather_agent = Agent(
    name="weather_expert",
    client=client,
    system_prompt="You are a weather expert. Provide detailed weather information and forecasts.",
    tools=[get_weather]
)

web_search_agent = Agent(
    name="web_search_expert",
    client=client,
    system_prompt="You are a web search expert. You can search the web for information.",
    tools=[DuckDuckGoSearchTool()]
)

planner_agent = Agent(
    name="planner",
    client=client,
    system_prompt="You are a trip planner. You should provide a plan for the user. Make sure to provide a detailed plan with the best places to visit and the best time to visit them."
)

planner_agent.can_call([weather_agent, web_search_agent])

response = planner_agent.run(
    "I need to plan a hiking trip in Seattle next week. I want to see some waterfalls and a forest."
)
print(response.text)

```


### 📊 Document Ingestion

Process and index documents for retrieval-augmented generation (RAG). This pipeline automatically parses PDFs, splits them into chunks, generates embeddings, and stores them in a vector database for efficient similarity search.

```sh
pip install datapizza-ai-parsers-docling
```

```python
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")
embedder = ChunkEmbedder(client=OpenAIEmbedder(api_key="YOUR_API_KEY", model_name="text-embedding-3-small"))
vectorstore.create_collection("my_documents",vector_config=[VectorConfig(name="embedding", dimensions=1536)])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=1024),
        embedder,
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

pipeline.run("sample.pdf")

results = vectorstore.search(query_vector = [0.0] * 1536, collection_name="my_documents", k=5)
print(results)
```


### 📊 RAG (Retrieval-Augmented Generation)

Create a complete RAG pipeline that enhances AI responses with relevant document context. This example demonstrates query rewriting, embedding generation, document retrieval, and response generation in a connected workflow.

```python
from datapizza.clients.openai import OpenAIClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.modules.rewriters import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

openai_client = OpenAIClient(
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY"
)

dag_pipeline = DagPipeline()
dag_pipeline.add_module("rewriter", ToolRewriter(client=openai_client, system_prompt="Rewrite user queries to improve retrieval accuracy."))
dag_pipeline.add_module("embedder", OpenAIEmbedder(api_key= "YOUR_API_KEY", model_name="text-embedding-3-small"))
dag_pipeline.add_module("retriever", QdrantVectorstore(host="localhost", port=6333).as_retriever(collection_name="my_documents", k=5))
dag_pipeline.add_module("prompt", ChatPromptTemplate(user_prompt_template="User question: {{user_prompt}}\n:", retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"))
dag_pipeline.add_module("generator", openai_client)

dag_pipeline.connect("rewriter", "embedder", target_key="text")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="chunks")
dag_pipeline.connect("prompt", "generator", target_key="memory")

query = "tell me something about this document"
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"user_prompt": query},
    "retriever": {"collection_name": "my_documents", "k": 3},
    "generator":{"input": query}
})

print(f"Generated response: {result['generator']}")
```


## 🌐 Ecosystem

### 🤖 Supported AI Providers

<table >
<tr>
<td align="center"><img src="https://logosandtypes.com/wp-content/uploads/2022/07/OpenAI.png" width="32" style="border-radius: 50%"><br><b>OpenAI</b></td>
<td align="center"><img src="https://www.google.com/favicon.ico" width="32"><br><b>Google Gemini</b></td>
<td align="center"><img src="https://anthropic.com/favicon.ico" width="32"><br><b>Anthropic</b></td>
<td align="center"><img src="https://mistral.ai/favicon.ico" width="32"><br><b>Mistral</b></td>
<td align="center"><img src="https://azure.microsoft.com/favicon.ico" width="32"><br><b>Azure OpenAI</b></td>
</tr>
</table>

### 🔧 Tools & Integrations

| Category | Components |
|----------|------------|
| **📄 Document Parsers** | Azure AI Document Intelligence, Docling |
| **🔍 Vector Stores** | Qdrant |
| **🎯 Rerankers** | Cohere, Together AI |
| **🌐 Tools** | DuckDuckGo Search, Custom Tools |
| **💾 Caching** | Redis integration for performance optimization |
| **📊 Embedders** | OpenAI, Google, Cohere, FastEmbed |

## 🎓 Learning Resources

- 📖 **[Complete Documentation](https://docs.datapizza.ai)** - Comprehensive guides and API reference
- 🎯 **[RAG Tutorial](https://docs.datapizza.ai/latest/Guides/RAG/rag/)** - Build production RAG systems
- 🤖 **[Agent Examples](https://docs.datapizza.ai/latest/Guides/Agents/agent/)** - Real-world agent implementations

## 🤝 Community


- 💬 **[Discord Community](https://discord.gg/s5sJNHz2C8)**
- 📚 **[Documentation](https://docs.datapizza.ai)**
- 📧 **[GitHub Issues](https://github.com/datapizza-labs/datapizza-ai/issues)**
- 🐦 **[Twitter](https://x.com/datapizza_ai)**

### 🌟 Contributing

We love contributions! Whether it's:

- 🐛 **Bug Reports** - Help us improve
- 💡 **Feature Requests** - Share your ideas
- 📝 **Documentation** - Make it better for everyone
- 🔧 **Code Contributions** - Build the future together

Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built by Datapizza, the AI native company**

*A framework made to be easy to learn, easy to maintain and ready for production* 🍕

[⭐ Star us on GitHub](https://github.com/datapizza-labs/datapizza-ai) • [🚀 Get Started](https://docs.datapizza.ai) • [💬 Join Discord](https://discord.gg/s5sJNHz2C8)

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=datapizza-labs/datapizza-ai&type=Date)](https://star-history.dera.page/#datapizza-labs/datapizza-ai&Date)

</div>
