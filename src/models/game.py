from pydantic import BaseModel, Field
from typing import Optional


class Game(BaseModel):
    app_id: int
    name: str
    price: float = 0.0                
    positive_reviews: float = 0.0
    negative_reviews: float = 0.0
    total_reviews: int = 0
    recommendations_ratio: float = 0.0        # porcentaje reseñas positivas (0-1)
    rating: float = 0.0                       # 0-10 evaluacion de reseñas especializadas
    release_year: Optional[int] = None
    platforms: Optional[list[str]] = None        # ← campo faltante (usado en score.py y explanation.py)
    short_description: Optional[str] = None
    categories: list[str] = []
    genres: list[str] = []
    tags: list[str] = []

    class Config:
        populate_by_name = True