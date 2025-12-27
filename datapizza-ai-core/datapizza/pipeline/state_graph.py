import logging
import sys
from collections.abc import Callable
from dataclasses import Field, dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar

log = logging.getLogger(__name__)


START = sys.intern("__start__")
END = sys.intern("__end__")


@dataclass
class Node:
    component: Callable[..., Any]
    data: dict | None
    src_keys: dict[str, str] | None
    dst_key: str | None


@dataclass
class SimpleEdge:
    to_node_name: str


@dataclass
class ConditionalEdge:
    to_node_names: list[str]
    component: Callable[..., str]


class DataclassLike(Protocol):
    """Protocol to represent types that behave like dataclasses.

    Inspired by the private _DataclassT from dataclasses that uses a similar protocol as a bound."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


Edge: TypeAlias = SimpleEdge | ConditionalEdge

StateT = TypeVar("StateT", bound=DataclassLike)


class StateGraph(Generic[StateT]):
    """
    A pipeline that runs a graph of a dependency graph.
    """

    # TODO: from yaml
    # TODO: to diagram
    # TODO: validate src and dst keys

    nodes: dict[str, Node]
    edges: dict[str, Edge]

    state_schema: type[StateT]

    def __init__(self, state_schema: type[StateT]):
        self.nodes = {}
        self.edges = {}

        self.state_schema = state_schema

    def _validate_new_edge(self, source_node: str, target_nodes: list[str]):
        """
        Validates a new edge.

        Args:
            source_node (str): The source node.
            target_nodes (list[str]): The target nodes.

        Raises:
            ValueError: If the edge is invalid.
        """
        if source_node not in self.nodes and source_node not in (START, END):
            raise ValueError(f"Source node {source_node} does not exist in the graph.")

        for target_node in target_nodes:
            if target_node not in self.nodes and target_node not in (START, END):
                raise ValueError(
                    f"Target node {target_node} does not exist in the graph."
                )

        if source_node in self.edges:
            raise ValueError(f"Source node {source_node} already has an outgoing edge.")

    def _validate_graph(self):
        """
        Validates the graph.

        The following conditions must be met:
        - START and END nodes must be connected

        Raises:
            ValueError: If the graph is invalid.
        """
        if START not in self.edges:
            raise ValueError("Graph must have a START node connected.")

        connected_edges = set()
        for edge in self.edges.values():
            if isinstance(edge, SimpleEdge):
                connected_edges.add(edge.to_node_name)
            elif isinstance(edge, ConditionalEdge):
                connected_edges.update(edge.to_node_names)
            else:
                raise ValueError("Unknown edge type.")

        if END not in connected_edges:
            raise ValueError("Graph must have an END node connected.")

    def add_module(
        self,
        node_name: str,
        node: Callable[..., Any],
        data: dict | None = None,
        src_keys: dict[str, str] | None = None,
        dst_key: str | None = None,
    ):
        """
        Add a module to the pipeline.

        Args:
            node_name (str): The name of the module.
            node (Callable[..., Any]): The module to add.
            data (dict | None): Fixed arguments for the module.
            src_keys (dict[str, str] | None): Mapping of state keys to module input parameter names.
            dst_key (str | None): The destination key for the module output.

        Raises:
            ValueError: If the node already exists.
        """
        if node_name in self.nodes:
            raise ValueError(f"Node {node_name} already exists in the graph.")

        self.nodes[node_name] = Node(
            component=node,
            data=data,
            src_keys=src_keys,
            dst_key=dst_key,
        )

    def connect(
        self,
        source_node: str,
        target_node: str,
    ):
        """
        Connect two nodes in the pipeline.

        Args:
            source_node (str): The name of the source node.
            target_node (str): The name of the target node.
        """
        self._validate_new_edge(source_node, [target_node])

        self.edges[source_node] = SimpleEdge(
            to_node_name=target_node,
        )

    def branch(
        self,
        node_name: str,
        node: Callable[..., str],
        path_map: list[str],
    ):
        self._validate_new_edge(node_name, path_map)

        self.edges[node_name] = ConditionalEdge(
            to_node_names=path_map,
            component=node,
        )

    def run(self, initial_state: StateT | None = None) -> StateT:
        """
        Run the pipeline.

        Args:
            initial_state (StateT | None): The initial state of the pipeline. If None, an empty state will be used.

        Returns:
            StateT: The state of the pipeline.
        """
        self._validate_graph()

        state = self.state_schema() if initial_state is None else initial_state

        current_edge = self.edges[START]

        while True:
            # Evaluate the current edge to get the next node
            node_name = None
            if isinstance(current_edge, SimpleEdge):
                node_name = current_edge.to_node_name
            elif isinstance(current_edge, ConditionalEdge):
                node_name = current_edge.component(state)
                if node_name not in current_edge.to_node_names:
                    raise ValueError(
                        f"Branch node returned invalid path '{node_name}', expected one of {current_edge.to_node_names}."
                    )
            else:
                raise ValueError("Unknown edge type.")

            # Check for end node
            if node_name == END:
                break

            # Execute the node
            node = self.nodes[node_name]
            try:
                log.debug(f"State before node {node_name}: {state}")

                arguments = node.data or {}
                if node.src_keys:
                    for param_name, state_key in node.src_keys.items():
                        arguments[param_name] = getattr(state, state_key)
                else:
                    arguments.update(state.__dict__)

                node_output = node.component(**arguments)
                if node.dst_key:
                    setattr(state, node.dst_key, node_output)
                elif isinstance(node_output, dict):
                    state = self.state_schema(**node_output)
                else:
                    raise ValueError(
                        f"Node {node_name} did not return a dict and no dst_key was specified."
                    )

                # Get the next edge
                current_edge = self.edges[node_name]

            except Exception as e:
                log.error(f"Error running node {node_name}: {e!s}")
                raise

        return state

    async def a_run(self, initial_state: StateT | None = None) -> StateT:
        # TODO: add async support

        return self.run(initial_state)
