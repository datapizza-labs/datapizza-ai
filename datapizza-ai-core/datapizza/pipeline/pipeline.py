import importlib
import logging
from typing import Any

import yaml

from datapizza.clients import ClientFactory
from datapizza.core.models import PipelineComponent
from datapizza.core.utils import replace_env_vars
from datapizza.core.vectorstore import Vectorstore
from datapizza.type import Chunk

log = logging.getLogger(__name__)


def _replace_element_refs(value: Any, elements: dict[str, Any]) -> Any:
    """
    Replace element references (${element_name}) with actual element instances.

    Args:
        value: The value to process (can be string, dict, list, or any other type)
        elements: Dictionary mapping element names to their instances

    Returns:
        The value with element references replaced by actual instances
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        element_name = value[2:-1]
        if element_name in elements:
            return elements[element_name]
        # If not found in elements, return as-is (might be a constant or env var)
        return value
    elif isinstance(value, dict):
        return {k: _replace_element_refs(v, elements) for k, v in value.items()}
    elif isinstance(value, list):
        return [_replace_element_refs(item, elements) for item in value]
    else:
        return value


def _instantiate_element(element_config: dict[str, Any]) -> Any:
    """
    Instantiate an element from its configuration.

    Args:
        element_config: Dictionary with 'type', 'module', and optional 'params' keys

    Returns:
        The instantiated element
    """
    module_path = element_config["module"]
    class_name = element_config["type"]
    params = element_config.get("params", {})

    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return class_(**params)


class Pipeline:
    def __init__(self, components: list[PipelineComponent] | None = None):
        self.components = components or []

    def run(self, input_data=None):
        data = input_data
        for component in self.components:
            log.info(f"Running component {component.__class__.__name__}")
            data = component(data)
        return data

    async def a_run(self, input_data=None):
        data = input_data
        for component in self.components:
            log.info(f"Running component {component.__class__.__name__}")
            data = await component.a_run(data)
        return data


class IngestionPipeline:
    """
    A pipeline for ingesting data into a vector store.
    """

    def __init__(
        self,
        modules: list[PipelineComponent] | None = None,
        vector_store: Vectorstore | None = None,
        collection_name: str | None = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            modules (list[PipelineComponent], optional): List of pipeline components. Defaults to None.
            vector_store (Vectorstore, optional): Vector store to store the ingested data. Defaults to None.
            collection_name (str, optional): Name of the vector store collection to store the ingested data. Defaults to None.
        """
        self.pipeline = Pipeline(modules)
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.components = modules

        if self.vector_store and not self.collection_name:
            raise ValueError("Collection name must be set if vector store is provided")

    def run(
        self, file_path: str | list[str], metadata: dict | None = None
    ) -> list[Chunk] | None:
        """Run the ingestion pipeline.

        Args:
            file_path (str | list[str]): The file path or list of file paths to ingest.
            metadata (dict, optional): Metadata to add to the ingested chunks. Defaults to None.

        Returns:
            list[Chunk] | None: If vector_store is not set, returns all accumulated chunks from all files.
                                If vector_store is set, returns None after storing all chunks.
        """
        # Normalize to list for uniform processing
        if isinstance(file_path, str):
            file_paths = [file_path]
        elif isinstance(file_path, list):
            # Validate that all elements are strings
            if not all(isinstance(fp, str) for fp in file_path):
                raise ValueError("All elements in file_path list must be strings")
            file_paths = file_path
        else:
            raise ValueError("file_path must be a string or a list of strings")

        all_chunks = []

        # Process each file path
        for fp in file_paths:
            data = self.pipeline.run(fp)

            if not self.vector_store:
                # If no vector store, accumulate results
                if isinstance(data, list):
                    all_chunks.extend(data)
                else:
                    all_chunks.append(data)
            else:
                # Validate chunk data immediately
                if not isinstance(data, list) or not all(
                    isinstance(item, Chunk) for item in data
                ):
                    raise ValueError(
                        f"Data returned from pipeline for '{fp}' must be a list of Chunk objects"
                    )
                all_chunks.extend(data)

        if not self.vector_store:
            return all_chunks

        # Adding metadata to all accumulated chunks
        if metadata:
            for chunk in all_chunks:
                chunk.metadata.update(metadata)

        # Add all chunks to vector store at once (only if we have chunks)
        if all_chunks:
            self.vector_store.add(all_chunks, self.collection_name)

    async def a_run(
        self, file_path: str | list[str], metadata: dict | None = None
    ) -> list[Chunk] | None:
        """Run the ingestion pipeline asynchronously.

        Args:
            file_path (str | list[str]): The file path or list of file paths to ingest.
            metadata (dict, optional): Metadata to add to the ingested chunks. Defaults to None.

        Returns:
            list[Chunk] | None: If vector_store is not set, returns all accumulated chunks from all files.
                                If vector_store is set, returns None after storing all chunks.
        """
        # Normalize to list for uniform processing
        if isinstance(file_path, str):
            file_paths = [file_path]
        elif isinstance(file_path, list):
            # Validate that all elements are strings
            if not all(isinstance(fp, str) for fp in file_path):
                raise ValueError("All elements in file_path list must be strings")
            file_paths = file_path
        else:
            raise ValueError("file_path must be a string or a list of strings")

        all_chunks = []

        # Process each file path
        for fp in file_paths:
            data = await self.pipeline.a_run(fp)

            if not self.vector_store:
                # If no vector store, accumulate results
                if isinstance(data, list):
                    all_chunks.extend(data)
                else:
                    all_chunks.append(data)
            else:
                # Validate chunk data immediately
                if not isinstance(data, list) or not all(
                    isinstance(item, Chunk) for item in data
                ):
                    raise ValueError(
                        f"Data returned from pipeline for '{fp}' must be a list of Chunk objects"
                    )
                all_chunks.extend(data)

        if not self.vector_store:
            return all_chunks

        # Adding metadata to all accumulated chunks
        if metadata:
            for chunk in all_chunks:
                chunk.metadata.update(metadata)

        # Add all chunks to vector store at once (only if we have chunks)
        if all_chunks:
            await self.vector_store.a_add(all_chunks, self.collection_name)

    def from_yaml(self, config_path: str) -> "IngestionPipeline":
        """
        Load the ingestion pipeline from a YAML configuration file.

        The YAML configuration supports the following sections:
        - constants: Key-value pairs for string substitution using ${VAR_NAME} syntax
        - elements: Reusable component definitions that can be referenced in modules
        - ingestion_pipeline: The main pipeline configuration with clients, modules, vector_store, and collection_name

        Example elements section:
            elements:
                my_embedder:
                    type: GoogleEmbedder
                    module: datapizza.embedders.google
                    params:
                        max_char: 2000

        Elements can be referenced in module params using ${element_name} syntax:
            modules:
                - name: embedder
                  type: ChunkEmbedder
                  module: datapizza.embedders
                  params:
                      client: "${my_embedder}"

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            IngestionPipeline: The ingestion pipeline instance.
        """
        with open(config_path) as file:
            config = yaml.safe_load(file)

        constants = config.get("constants", {})
        # Use skip_unknown=True to allow element references (like ${my_embedder})
        # to pass through without being treated as missing env vars
        config = replace_env_vars(config, constants, skip_unknown=True)

        # Parse and instantiate elements
        elements = {}
        if "elements" in config:
            for element_name, element_config in config["elements"].items():
                try:
                    elements[element_name] = _instantiate_element(element_config)
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Could not load element '{element_name}' "
                        f"({element_config.get('type', 'N/A')}): {e!s}"
                    ) from e
                except KeyError as e:
                    raise ValueError(
                        f"Missing required key {e!s} in element configuration: {element_config}"
                    ) from e

        clients = {}
        ingestion_pipeline = config["ingestion_pipeline"]
        if "clients" in ingestion_pipeline:
            for client_name, client_config in ingestion_pipeline["clients"].items():
                provider = client_config.pop("provider")
                client = ClientFactory.create(
                    provider, client_config.get("api_key"), client_config.get("model")
                )
                clients[client_name] = client

        components = []
        if "modules" in ingestion_pipeline:
            for component_config in ingestion_pipeline["modules"]:
                try:
                    module_path = component_config["module"]
                    module = importlib.import_module(module_path)
                    class_ = getattr(module, component_config["type"])

                    params = component_config.get("params", {})

                    # Replace element references in params
                    params = _replace_element_refs(params, elements)

                    if "client" in params:
                        client_name = params["client"]
                        # Only do client lookup if it's still a string (not already replaced by element)
                        if isinstance(client_name, str):
                            if client_name not in clients:
                                raise ValueError(
                                    f"Client '{client_name}' not found in clients configuration"
                                )
                            params["client"] = clients[client_name]

                    component_instance = class_(**params)
                    components.append(component_instance)
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Could not load component {component_config.get('type', 'N/A')}: {e!s}"
                    ) from e
                except KeyError as e:
                    raise ValueError(
                        f"Missing required key {e!s} in module configuration: {component_config}"
                    ) from e

        vector_store = None
        if "vector_store" in ingestion_pipeline:
            vector_store_config = ingestion_pipeline["vector_store"]
            vector_store_type = vector_store_config["type"]
            vector_store_module = importlib.import_module(vector_store_config["module"])
            vector_store_class = getattr(vector_store_module, vector_store_type)
            vector_store_params = vector_store_config.get("params", {})
            vector_store = vector_store_class(**vector_store_params)
            self.vector_store = vector_store

        collection_name = None
        if "collection_name" in ingestion_pipeline:
            collection_name = ingestion_pipeline["collection_name"]
            self.collection_name = collection_name

        self.components = components
        self.pipeline = Pipeline(components)
        return self
