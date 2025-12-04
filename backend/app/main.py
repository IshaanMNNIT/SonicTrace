from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import audio, health, media

app = FastAPI(title="SonicTrace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during dev; restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(media.router, prefix="/api")