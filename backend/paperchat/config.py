import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy.engine import make_url

DEFAULT_DB_PATH = Path.home() / ".paperchat" / "paperchat.db"
DEFAULT_EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
DEFAULT_CACHE_DIR = Path.home() / ".paperchat" / "cache"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI_PATH = BACKEND_ROOT / "alembic.ini"
ALEMBIC_DIR = BACKEND_ROOT / "alembic"


@lru_cache
def get_database_path() -> Path:
    if database_url := os.getenv("PAPERCHAT_DATABASE_URL"):
        return _sqlite_database_path_from_url(database_url)
    raw = os.getenv("PAPERCHAT_DB_PATH")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_DB_PATH


@lru_cache
def get_database_url() -> str:
    if database_url := os.getenv("PAPERCHAT_DATABASE_URL"):
        get_database_path()
        return database_url
    db_path = get_database_path()
    return f"sqlite:///{db_path}"


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


def _sqlite_database_path_from_url(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        raise ValueError("PaperChat only supports sqlite URLs for PAPERCHAT_DATABASE_URL.")

    database = url.database
    if database in {None, "", ":memory:"}:
        raise ValueError("PAPERCHAT_DATABASE_URL must point to a SQLite database file.")

    return Path(database).expanduser()
