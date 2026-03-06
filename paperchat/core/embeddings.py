"""Load text embedding models from Milvus"""

from typing import Any

from pymilvus.model import DefaultEmbeddingFunction, dense

from paperchat.utils.api_keys import get_api_key
from paperchat.utils.logging import get_component_logger

logger = get_component_logger("embeddings")

PROVIDERS_REQUIRING_KEYS = {"openai", "gemini"}
SUPPORTED_EMBEDDING_MODELS = {
    "openai/text-embedding-3-small": {
        "class": dense.OpenAIEmbeddingFunction,
        "init_args": {"model_name": "text-embedding-3-small"},
        "dimensions": 1536,
    },
    "openai/text-embedding-3-large": {
        "class": dense.OpenAIEmbeddingFunction,
        "init_args": {"model_name": "text-embedding-3-large"},
        "dimensions": 3072,
    },
    "gemini/gemini-embedding-exp-03-07": {
        "class": dense.GeminiEmbeddingFunction,
        "init_args": {"model_name": "gemini-embedding-exp-03-07"},
        "dimensions": 3072,
    },
    "gemini/models/text-embedding-004": {
        "class": dense.GeminiEmbeddingFunction,
        "init_args": {"model_name": "models/text-embedding-004"},
        "dimensions": 768,
    },
}


def list_embedding_models() -> list[dict[str, Any]]:
    """List all supported embedding models, indicating if they require an API key."""
    available_models = []
    for identifier, config in SUPPORTED_EMBEDDING_MODELS.items():
        try:
            provider = identifier.split("/")[0]
        except IndexError:
            logger.warning(f"Skipping invalid embedding model identifier: {identifier}")
            continue

        api_key_present_or_not_required = True
        if provider in PROVIDERS_REQUIRING_KEYS and get_api_key(provider) is None:
            api_key_present_or_not_required = False

        if api_key_present_or_not_required:
            available_models.append(
                {
                    "id": identifier,
                    "name": identifier,
                    "dimensions": config.get("dimensions"),
                    "provider": provider,
                }
            )
    return available_models


def get_embedding_model(model_identifier: str) -> Any:
    """
    Get an initialized embedding function instance from Milvus.

    Args:
        model_identifier: The unique identifier ('provider/model_name').

    Returns:
        An initialized embedding function instance from pymilvus.model.
    """
    if model_identifier == "pymilvus/default":
        try:
            logger.info(
                "[get_embedding_model] Using generic default. Instantiating DefaultEmbeddingFunction()"
            )
            embedding_function = DefaultEmbeddingFunction()
            if not (
                hasattr(embedding_function, "dim")
                and isinstance(embedding_function.dim, int)
                and embedding_function.dim > 0
            ):
                raise AttributeError(
                    "DefaultEmbeddingFunction instance lacks a valid .dim attribute."
                )
            logger.info(
                f"[get_embedding_model] Successfully instantiated pymilvus default: {type(embedding_function)} with dim {embedding_function.dim}"
            )
            return embedding_function
        except Exception as e:
            logger.error(f"Error instantiating DefaultEmbeddingFunction: {e!s}", exc_info=True)
            raise RuntimeError("Failed to instantiate the pymilvus default embedding model.") from e

    if "/" not in model_identifier:
        raise ValueError(
            f"Invalid model identifier format: '{model_identifier}'. Expected 'provider/model_name'."
        )

    if model_identifier not in SUPPORTED_EMBEDDING_MODELS:
        available = [m["id"] for m in list_embedding_models()]
        raise ValueError(
            f"Unsupported embedding model identifier: '{model_identifier}'. "
            f"Available models (with keys): {available}"
        )

    config = SUPPORTED_EMBEDDING_MODELS[model_identifier]
    provider = model_identifier.split("/")[0]
    model_class = config["class"]
    init_args = config["init_args"].copy()

    if provider in PROVIDERS_REQUIRING_KEYS:
        api_key = get_api_key(provider)
        if not api_key:
            raise ValueError(
                f"API key for provider '{provider}' not found, required by '{model_identifier}'"
            )
        if provider in ["openai", "gemini"]:
            init_args["api_key"] = api_key

    try:
        embedding_function = model_class(**init_args)

        return embedding_function

    except Exception as e:
        logger.error(
            f"Error instantiating {model_class.__name__} for {model_identifier}: {e!s}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to instantiate embedding model '{model_identifier}' for provider '{provider}'. "
            f"Check API keys, model availability, and pymilvus installation."
        ) from e
