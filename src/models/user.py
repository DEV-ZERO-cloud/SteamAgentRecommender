from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):

    user_id: str
    name: Optional[str] = "Anónimo" 
    played_app_ids: list[int] = Field(default_factory=list)
    preferred_tags: list[str] = Field(default_factory=list)

    amount_playing: list[int] = Field(default_factory=list)
    reviewed_app_ids: Optional[list[int]] = Field(default_factory=list)
    review_sentiment: Optional[list[int]] = Field(default_factory=list)


class UserPreferences(BaseModel):

    preferred_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    preferred_platforms: list[str] = Field(default_factory=list)
    free_to_play_only: bool = False
    max_price: Optional[float] = None
    min_rating: float = 0.0


class UserQuery(BaseModel):

    max_price: Optional[float] = None          # None = sin límite
    min_rating: Optional[float] = 0.0          # mínimo rating aceptable (0-10)
    min_recommendations: Optional[float] = 0.0
    min_date: Optional[str] = None             # None = sin límite
    free_query: str = ""                       # búsqueda en lenguaje natural