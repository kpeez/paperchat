import os
from typing import Any, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paperchat_backend.api.bootstrap import router


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        cast(Any, CORSMiddleware),
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


def run() -> None:
    port = int(os.environ.get("PAPERCHAT_PORT", "9712"))
    uvicorn.run("paperchat_backend.main:create_app", host="127.0.0.1", port=port, factory=True)
