from pydantic import BaseModel


class RuntimeResponse(BaseModel):
    app_version: str
    data_dir: str
    database_path: str
    cache_dir: str
    embedding_model: str
