from enum import Enum

from datapizza.core.clients.client import Client


class Provider(str, Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    AZURE_OPENAI = "azure_openai"
    WATSONX = "watsonx"

class ClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create(
        provider: str | Provider,
        api_key: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        **kwargs,
    ) -> "Client":
        """
        Create a client instance based on the specified provider.

        Args:
            provider: The LLM provider to use (openai, google, or anthropic)
            api_key: API key for the provider
            model: Model name to use (provider-specific)
            system_prompt: System prompt to use
            temperature: Temperature for generation (0-2)
            **kwargs: Additional provider-specific arguments

        Returns:
            An instance of the appropriate client

        Raises:
            ValueError: If the provider is not supported
        """
        if isinstance(provider, str):
            provider = Provider(provider.lower())

        match provider:
            case Provider.OPENAI:
                try:
                    from datapizza.clients.openai import OpenAIClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAI client is not installed. Please install it with `pip install datapizza-ai-clients-openai`"
                    ) from e

                return OpenAIClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )

            case Provider.GOOGLE:
                try:
                    from datapizza.clients.google import GoogleClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Google client is not installed. Please install it with `pip install datapizza-ai-clients-google`"
                    ) from e

                return GoogleClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )

            case Provider.ANTHROPIC:
                try:
                    from datapizza.clients.anthropic import (  # type: ignore
                        AnthropicClient,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Anthropic client is not installed. Please install it with `pip install datapizza-ai-clients-anthropic`"
                    ) from e

                return AnthropicClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )

            case Provider.MISTRAL:
                try:
                    from datapizza.clients.mistral import MistralClient  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Mistral client is not installed. Please install it with `pip install datapizza-ai-clients-mistral`"
                    ) from e

                return MistralClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )

            case Provider.AZURE_OPENAI:
                try:
                    from datapizza.clients.azure_openai_client import (  # type: ignore
                        AzureOpenAIClient,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Azure OpenAI client is not installed. Please install it with `pip install datapizza-ai-clients-azure-openai`"
                    ) from e

                return AzureOpenAIClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )

            case Provider.WATSONX:
                try:
                    from datapizza.clients.watsonx import (  # type: ignore
                        WatsonXClient,
                    )
                except ImportError as e:
                    raise ImportError(
                        "IBM WatsonX client is not installed. Please install it with `pip install datapizza-ai-clients-watsonx`"
                    ) from e

                return WatsonXClient(
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Unsupported provider: {provider}")
