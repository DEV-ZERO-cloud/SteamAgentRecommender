from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):
    user_id: str
    name: str = "Anónimo"
    played_app_ids: list[int] = Field(default_factory=list)    # juegos ya jugados (excluir)
    preferred_tags: list[str] = Field(default_factory=list)

class UserQuery(BaseModel):
    max_price: Optional[float] = None          # None = sin límite
    min_rating: Optional[float] = 0.0          # mínimo rating aceptable (0-10)
    min_recommendations: Optional[float] = 0.0
    min_date: Optional[str] = None             # None = sin límite
    free_query: str = ""                       # búsqueda en lenguaje natural
