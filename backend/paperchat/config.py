import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import URL

DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_NAME = "paperchat"
DEFAULT_DB_PASSWORD = "paperchat"
DEFAULT_DB_PORT = 5433
DEFAULT_DB_USER = "paperchat"
DEFAULT_EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "paperchat"
DOCKER_BINARY = "docker"
DOCKER_SERVICE = "postgres"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = BACKEND_ROOT / "alembic.ini"
ALEMBIC_DIR = BACKEND_ROOT / "alembic"
COMPOSE_FILE = REPO_ROOT / "compose.yaml"


def _read_port(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as error:
        msg = f"{name} must be an integer, got {raw_value!r}."
        raise ValueError(msg) from error


@lru_cache
def get_database_url() -> str:
    if database_url := os.getenv("PAPERCHAT_DATABASE_URL"):
        return database_url

    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("PAPERCHAT_DB_USER", DEFAULT_DB_USER),
        password=os.getenv("PAPERCHAT_DB_PASSWORD", DEFAULT_DB_PASSWORD),
        host=os.getenv("PAPERCHAT_DB_HOST", DEFAULT_DB_HOST),
        port=_read_port("PAPERCHAT_DB_PORT", DEFAULT_DB_PORT),
        database=os.getenv("PAPERCHAT_DB_NAME", DEFAULT_DB_NAME),
    ).render_as_string(hide_password=False)


@lru_cache
def get_embedding_model_name() -> str:
    return os.getenv("PAPERCHAT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


@lru_cache
def get_cache_dir() -> Path:
    raw_cache_dir = os.getenv("PAPERCHAT_CACHE_DIR")
    if raw_cache_dir:
        return Path(raw_cache_dir).expanduser()
    return DEFAULT_CACHE_DIR


@lru_cache
def get_huggingface_cache_dir() -> Path:
    raw_hf_cache_dir = os.getenv("PAPERCHAT_HF_CACHE_DIR")
    if raw_hf_cache_dir:
        return Path(raw_hf_cache_dir).expanduser()
    return get_cache_dir() / "huggingface"


@lru_cache
def get_model_cache_dir() -> Path:
    raw_model_cache_dir = os.getenv("PAPERCHAT_MODEL_CACHE_DIR")
    if raw_model_cache_dir:
        return Path(raw_model_cache_dir).expanduser()
    return get_cache_dir() / "models"
