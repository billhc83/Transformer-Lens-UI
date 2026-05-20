from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import models, inference, activations, patching, circuits, hooks, explain
from backend.ws import generation as ws_generation

app = FastAPI(title="TransformerLens UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(inference.router)
app.include_router(activations.router)
app.include_router(patching.router)
app.include_router(circuits.router)
app.include_router(hooks.router)
app.include_router(explain.router)
app.include_router(ws_generation.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
