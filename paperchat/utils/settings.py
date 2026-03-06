"""Handles loading and saving application settings."""

import json
from pathlib import Path
from typing import Any

from paperchat.core.embeddings import SUPPORTED_EMBEDDING_MODELS

from .logging import get_component_logger

DEFAULT_EMBEDDING_MODEL = "pymilvus/default"
SETTINGS_DIR = Path.home() / ".paperchat"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
EMBEDDING_MODEL_SETTING_KEY = "embedding_model_identifier"

logger = get_component_logger("Settings")


def _ensure_settings_dir_exists() -> None:
    """Creates the settings directory if it doesn't exist."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    """Loads settings from the JSON file.

    Returns:
        A dictionary containing the settings, or an empty dictionary
        if the file doesn't exist or is invalid.
    """
    _ensure_settings_dir_exists()
    if not SETTINGS_FILE.is_file():
        logger.info(f"Settings file not found ({SETTINGS_FILE}), returning empty settings.")
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)
            if not isinstance(settings, dict):
                logger.warning(
                    f"Settings file ({SETTINGS_FILE}) does not contain a valid JSON object, returning empty settings."
                )
                return {}
            logger.info(f"Loaded settings from {SETTINGS_FILE}")
            return settings
    except json.JSONDecodeError:
        logger.exception(f"Error decoding JSON from {SETTINGS_FILE}, returning empty settings.")
        return {}
    except Exception:
        logger.exception(f"Failed to load settings from {SETTINGS_FILE}, returning empty settings.")
        return {}


def save_settings(settings: dict[str, Any]) -> bool:
    """Saves the provided settings dictionary to the JSON file."""
    _ensure_settings_dir_exists()
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        logger.info(f"Saved settings to {SETTINGS_FILE}")
        return True
    except Exception:
        logger.exception(f"Failed to save settings to {SETTINGS_FILE}")
        return False


def get_active_embedding_model() -> str:
    """Determines the active embedding model identifier.

    Checks user settings first, then falls back to the application default.
    Ensures the chosen model is currently supported.
    """
    settings = load_settings()
    user_choice = settings.get(EMBEDDING_MODEL_SETTING_KEY)

    if user_choice and isinstance(user_choice, str):
        if user_choice in SUPPORTED_EMBEDDING_MODELS:
            logger.info(f"Using user-selected embedding model: {user_choice}")
            return user_choice
        else:
            logger.warning(
                f"User-selected embedding model '{user_choice}' is not currently supported. "
                f"Falling back to default: {DEFAULT_EMBEDDING_MODEL}"
            )
    else:
        logger.info("No valid user setting found for embedding model.")

    # fallback to default
    logger.info(f"Using default embedding model: {DEFAULT_EMBEDDING_MODEL}")
    if DEFAULT_EMBEDDING_MODEL not in SUPPORTED_EMBEDDING_MODELS:
        logger.critical(
            f"Default embedding model '{DEFAULT_EMBEDDING_MODEL}' from config is not in SUPPORTED_EMBEDDING_MODELS! "
            "Please check core/embeddings.py and utils/config.py."
        )
        # last resort, try to find *any* supported model
        fallback = next(iter(SUPPORTED_EMBEDDING_MODELS), None)
        if fallback:
            logger.warning(f"Falling back to first available supported model: {fallback}")
            return fallback
        else:
            # fatal configuration error
            raise RuntimeError("Configuration Error: No supported embedding models found!")

    return DEFAULT_EMBEDDING_MODEL


def set_active_embedding_model(model_identifier: str) -> bool:
    """Saves the user's chosen embedding model to settings."""
    if model_identifier not in SUPPORTED_EMBEDDING_MODELS:
        logger.error(
            f"Attempted to set unsupported embedding model '{model_identifier}' in settings."
        )
        return False

    settings = load_settings()
    settings[EMBEDDING_MODEL_SETTING_KEY] = model_identifier
    return save_settings(settings)
