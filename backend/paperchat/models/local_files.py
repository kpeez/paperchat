from pydantic import BaseModel


class PickDocumentsResponse(BaseModel):
    paths: list[str]
