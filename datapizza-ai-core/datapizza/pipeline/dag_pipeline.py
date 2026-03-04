import logging
from copy import deepcopy
from dataclasses import dataclass

from datapizza.core.models import ChainableProducer, PipelineComponent
from datapizza.core.utils import replace_env_vars

log = logging.getLogger(__name__)


@dataclass
class Edge:
    from_node_name: str
    to_node_name: str
    src_key: str | None
    dst_key: str


class DagPipeline:
    """
    A pipeline that runs a graph of a dependency graph.
    """

    def __init__(self):
        self.nodes: dict[str, PipelineComponent] = {}
        self.edges = []

    # get the nodes that depend on the given node
    def _get_edges_from(self, node_name: str) -> list[Edge]:
        return [d for d in self.edges if d.from_node_name == node_name]

    # get the nodes that the given node depends on
    def _get_edges_to(self, node_name: str) -> list[Edge]:
        return [d for d in self.edges if d.to_node_name == node_name]

    def add_module(self, node_name: str, node: PipelineComponent):
        """
        Add a module to the pipeline.

        Args:
            node_name (str): The name of the module.
            node (PipelineComponent): The module to add.
        """
        # Nodes must be ChainableProducer or PipelineComponent or callable
        if isinstance(node, ChainableProducer):
            module_component = node.as_module_component()
            self.nodes[node_name] = module_component

        elif isinstance(node, PipelineComponent) or callable(node):
            self.nodes[node_name] = node
        else:
            raise ValueError(
                f"Node {node_name} must be a ChainableProducer, PipelineComponent, or callable."
            )

    def connect(
        self,
        source_node: str,
        target_node: str,
        target_key: str,
        source_key: str | None = None,
    ):
        """
        Connect two nodes in the pipeline.

        Args:
            source_node (str): The name of the source node.
            target_node (str): The name of the target node.
            target_key (str): The key to store the result of the target node in the source node.
            source_key (str, optional): The key to retrieve the result of the source node from the target node. Defaults to None.
        """
        self.edges.append(
            Edge(
                from_node_name=source_node,
                to_node_name=target_node,
                src_key=source_key,
                dst_key=target_key,
            )
        )

    def _get_nodes_ready_to_run(self, results: dict):
        ready_nodes = []
        for node_name in self.nodes:
            # continue if node is already processed
            if node_name in results:
                continue

            # If node has no dependencies, it's ready to be processed
            if not self._get_edges_to(node_name):
                ready_nodes.append(node_name)
                continue

            # Check if all dependencies are resolved
            dependencies = [d.from_node_name for d in self._get_edges_to(node_name)]
            if all(dep in results for dep in dependencies):
                ready_nodes.append(node_name)

        return ready_nodes

    def _get_args_for_node(self, node_name: str, _input: dict, results: dict) -> dict:
        # Get all the nodes that must be run before the given node
        # get the results from the previous nodes and return them as a dict

        previous_nodes = self._get_edges_to(node_name)
        # Start with the original input
        args = deepcopy(_input.get(node_name, {}))

        for edge in previous_nodes:
            from_node_result = results[edge.from_node_name]

            # Extract the correct value from source
            if edge.src_key:
                value = from_node_result.get(edge.src_key)
                if value is None and edge.src_key not in from_node_result:
                    log.warning(
                        f"Source key '{edge.src_key}' not found in result of node '{edge.from_node_name}'. Using None."
                    )

            else:
                value = from_node_result

            # Place it in the right location in args
            if edge.dst_key:
                args[edge.dst_key] = deepcopy(value)
            else:
                raise ValueError(
                    f"No destination key provided for node '{node_name}' from '{edge.from_node_name}'."
                )

        return args

    def run(self, data: dict) -> dict:
        """
        Run the pipeline.

        Args:
            data (dict): The input data to the pipeline.

        Returns:
            dict: The results of the pipeline.
        """
        processed_nodes = set()
        pipeline_results = {}
        queue = self._get_nodes_ready_to_run(pipeline_results)

        while queue:
            node_name = queue.pop(0)

            if node_name in processed_nodes:
                continue

            node = self.nodes[node_name]
            try:
                arguments = self._get_args_for_node(node_name, data, pipeline_results)
                log.debug(f"Arguments for node {node_name}: {list(arguments.keys())}")
                node_result = node(**arguments)
                pipeline_results[node_name] = node_result
                processed_nodes.add(node_name)

                newly_ready = self._get_nodes_ready_to_run(pipeline_results)
                for ready_node in newly_ready:
                    if ready_node not in processed_nodes and ready_node not in queue:
                        queue.append(ready_node)

            except Exception as e:
                log.error(f"Error running node {node_name}: {e!s}")
                raise

        if len(processed_nodes) < len(self.nodes):
            unprocessed = set(self.nodes.keys()) - processed_nodes
            log.warning(
                f"Not all nodes were processed. Unprocessed: {unprocessed}. Possible cycle or error."
            )
            # This could indicate a cycle in the graph or an earlier error

        return pipeline_results

    async def a_run(self, data: dict):
        """
        Run the pipeline asynchronously.

        Args:
            data (dict): The input data to the pipeline.

        Returns:
            dict: The results of the pipeline.
        """
        processed_nodes = set()
        pipeline_results = {}
        queue = self._get_nodes_ready_to_run(pipeline_results)

        while queue:
            node_name = queue.pop(0)

            if node_name in processed_nodes:
                continue

            node = self.nodes[node_name]
            try:
                arguments = self._get_args_for_node(node_name, data, pipeline_results)
                log.debug(f"Arguments for node {node_name}: {list(arguments.keys())}")
                node_result = await node.a_run(**arguments)
                pipeline_results[node_name] = node_result
                processed_nodes.add(node_name)

                newly_ready = self._get_nodes_ready_to_run(pipeline_results)
                for ready_node in newly_ready:
                    if ready_node not in processed_nodes and ready_node not in queue:
                        queue.append(ready_node)

            except Exception as e:
                log.error(f"Error running node {node_name}: {e!s}")
                raise

        if len(processed_nodes) < len(self.nodes):
            unprocessed = set(self.nodes.keys()) - processed_nodes
            log.warning(
                f"Not all nodes were processed. Unprocessed: {unprocessed}. Possible cycle or error."
            )

        return pipeline_results

    def from_yaml(self, config_path: str) -> "DagPipeline":
        """
        Load the pipeline from a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            DagPipeline: The pipeline instance.
        """
        import importlib

        import yaml

        from datapizza.clients import ClientFactory

        with open(config_path) as file:
            config = yaml.safe_load(file)

        constants = config.get("constants", [])
        config = replace_env_vars(config, constants)

        dag_pipeline = config["dag_pipeline"]
        clients = {}

        if "clients" in dag_pipeline:
            for client_name, client_config in dag_pipeline["clients"].items():
                provider = client_config.pop("provider")
                client = ClientFactory.create(provider, **client_config)
                clients[client_name] = client

        self.nodes = {}
        self.edges = []

        if "modules" in dag_pipeline:
            for module_config in dag_pipeline["modules"]:
                try:
                    module_name = module_config["name"]
                    module_path = module_config["module"]
                    module_import = importlib.import_module(module_path)
                    class_ = getattr(module_import, module_config["type"])

                    params = module_config.get("params", {})

                    if "client" in params:
                        client_name = params["client"]
                        if client_name not in clients:
                            raise ValueError(
                                f"Client '{client_name}' not found in clients configuration for node {module_name}"
                            )
                        params["client"] = clients[client_name]

                    component_instance = class_(**params)
                    self.add_module(module_name, component_instance)
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Could not load module {module_config.get('type', 'N/A')} for node {module_config.get('name', 'N/A')}: {e!s}"
                    ) from e
                except KeyError as e:
                    raise ValueError(
                        f"Missing required key {e!s} in module configuration: {module_config}"
                    ) from e

        if "connections" in dag_pipeline:
            for connection in dag_pipeline["connections"]:
                from_node = connection["from"]
                to_node = connection["to"]
                if from_node not in self.nodes:
                    raise ValueError(
                        f"Source node '{from_node}' for connection not found in loaded modules."
                    )
                if to_node not in self.nodes:
                    raise ValueError(
                        f"Target node '{to_node}' for connection not found in loaded modules."
                    )

                source_key = connection.get("source_key")
                target_key = connection.get("target_key")
                self.connect(
                    from_node,
                    to_node,
                    target_key=target_key,
                    source_key=source_key,
                )

        return self
