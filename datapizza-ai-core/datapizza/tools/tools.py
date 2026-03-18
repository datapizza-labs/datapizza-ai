import inspect
from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import Any

from .utils import (
    get_default_values,
    get_param_annotations,
    get_parameters,
    get_required_params,
)


class Tool:
    """Class that wraps a function while preserving its behavior and adding attributes."""

    def __init__(
        self,
        func: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        end: bool = False,
        properties: dict[str, dict[str, Any]] | None = None,
        required: list[str] | None = None,
        strict: bool = False,
    ):
        """
        Args:
            func (Callable | None): The function to wrap.
            name (str | None): The name of the tool.
            description (str | None): The description of the tool.
            end (bool): Whether the tool ends a chain of operations.
            properties (dict[str, dict[str, Any]] | None): The properties of the tool.
            required (list[str] | None): The required parameters of the tool.
            strict (bool): Whether the tool is strict.
        """
        self.func = func
        if not name and not func:
            raise ValueError("Must provide either name or function")

        self.name: str = name or func.__name__  # type: ignore
        self.strict: bool = strict
        self.description: str | None = description or (func.__doc__ if func else None)

        if func and not properties:
            self.required = get_required_params(inspect.signature(self.func))  # type: ignore
            param_annotations = get_param_annotations(inspect.signature(self.func))  # type: ignore
            default_values = get_default_values(inspect.signature(self.func))  # type: ignore
            self.properties = get_parameters(
                param_annotations, default_values=default_values
            )
        else:
            self.properties = properties
            self.required = required or []

        self.schema = self._get_function_schema()

        self.end_invoke = end
        if func:
            wraps(func)(self)

    def __call__(self, *args, **kwargs):
        if not self.func:
            raise ValueError("Function not set")

        return self.func(*args, **kwargs)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        bound_decorated = Tool(
            func=MethodType(self.func, instance) if self.func else None,
            name=self.name,
            description=self.description,
            end=self.end_invoke,
            properties=self.properties,
            required=self.required,
        )

        return bound_decorated

    @classmethod
    def tool_from_dict(cls, tool_dict):
        return Tool(
            func=None,
            name=tool_dict.get("name"),
            description=tool_dict.get("description"),
            end=tool_dict.get("end"),
            properties=tool_dict.get("properties"),
            required=tool_dict.get("required"),
            strict=tool_dict.get("strict"),
        )

    def _get_function_schema(self):
        return {
            "name": self.name,
            "description": self.description
            or self.func.__doc__
            or f"Function to {self.name}",
            "parameters": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
            },
        }

    def to_dict(self) -> dict:
        """Convert the tool to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "required": self.required,
            "end_invoke": self.end_invoke,
        }


def tool(func=None, name=None, description=None, end=False, strict=False):
    """
    Decorator to wrap a function in a DecoratedFunc instance.

    Can be used as @tool or @tool(name="custom_name", description="...")

    Args:
        func: The function to decorate
        **attributes: Additional attributes to attach to the function

    Returns:
        DecoratedFunc: A callable object wrapping the original function
    """

    def decorator(f):
        return Tool(f, name=name, description=description, end=end, strict=strict)

    # Handle both @tool and @tool(...)
    if func is None:
        return decorator
    return decorator(func)
