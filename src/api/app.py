from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto (ruta absoluta para evitar problemas con reloader)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.game import Game
from engine.parameters_engine import ScoreFilters
from pipeline.pipeline_recommendation import PipelineRecommendation
from scripts.profile_extractor import build_user_profile, SteamUserProfile

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Steam RPG Recommender Agent",
    description="Agente híbrido de recomendación de RPGs en Steam (embeddings + reglas).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PipelineRecommendation()

# ── Schemas ───────────────────────────────────────────────────────────────────

class SemmanticRequest(BaseModel):
    query: list[str]
    top_k: int


class RecommendationRequest(BaseModel):
    query: list[str]
    disliked_tags: list[str] | None=None
    top_k: int = 5
    # Filtros ScoreFilters
    isPrice: bool = False
    MinPrice: float = 0.0
    MaxPrice: float = 999999.0
    isPositiveRate: bool = False
    MinPositiveRate: float = 0.0
    isDate: bool = False
    MinYear: int = 0
    MaxYear: int = 9999
    isRecommendations: bool = False
    MinRecommendations: float = 0.0


class ProfileRequest(BaseModel):
    username: str


class ProfileResponse(BaseModel):
    user_id: str
    name: str
    played_app_ids: list[int]
    preferred_tags: list[str]
    amount_playing: list[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    filters = ScoreFilters(
        isPrice            = request.isPrice,
        MinPrice           = request.MinPrice,
        MaxPrice           = request.MaxPrice,
        isPositiveRate     = request.isPositiveRate,
        MinPositiveRate    = request.MinPositiveRate,
        isDate             = request.isDate,
        MinYear            = request.MinYear,
        MaxYear            = request.MaxYear,
        isRecommendations  = request.isRecommendations,
        MinRecommendations = request.MinRecommendations,
    )

    results = pipeline.recommend(
        query         = request.query,
        disliked_tags = request.disliked_tags,
        top_k         = request.top_k,
        filters       = filters,
        max_price     = request.MaxPrice,
    )
    return {"query": request.query, "results": results}


@app.post("/profile", response_model=ProfileResponse)
def get_profile(request: ProfileRequest):
    """
    Extrae el perfil de un usuario de Steam dado su nombre, URL o SteamID64.
    Devuelve sus juegos jugados, horas y tags preferidos.
    """
    try:
        profile: SteamUserProfile = build_user_profile(request.username)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ProfileResponse(
        user_id        = profile.user_id,
        name           = profile.name,
        played_app_ids = profile.played_app_ids,
        preferred_tags = profile.preferred_tags,
        amount_playing = profile.amount_playing,
    )


@app.post("/semantic_engine/search")
def search_semantic(request: SemmanticRequest):
    tags = [tag.strip() for tag in request.query.split(",")]
    results = pipeline.semantic_engine.search(tags=tags, top_k=request.top_k)
    return {"query": request.query, "results": results}


@app.post("/train")
def train_model():
    pipeline.knowledge_engine.build(force=True)
    return {"games_loaded": len(pipeline.knowledge_engine.games)}


@app.get("/games", response_model=list[Game])
def list_games(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tag: Optional[str] = Query(None, description="Filtrar por tag (parcial, case-insensitive)"),
):
    games = pipeline.knowledge_engine.games
    if tag:
        tag_lower = tag.lower()
        games = [g for g in games if any(tag_lower in t.lower() for t in g.tags)]
    start = (page - 1) * page_size
    return games[start: start + page_size]


@app.get("/games/{app_id}", response_model=Game)
def get_game(app_id: int):
    game = pipeline.knowledge_engine.id_to_game.get(app_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Juego {app_id} no encontrado.")
    return game

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)