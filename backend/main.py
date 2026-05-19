from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import models, inference, activations

app = FastAPI(title="TransformerLens UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(inference.router)
app.include_router(activations.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
