from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.api import router


def create_app():
    new_app = FastAPI()
    new_app.include_router(router)
    return new_app


app = create_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
