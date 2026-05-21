"""
Responsabilidad:
  1. Leer el CSV de juegos Steam
  2. Parsear cada fila → objeto Game (tipado, validado)
  3. Generar embeddings de texto para cada juego
  4. Construir y persistir el índice FAISS en disco
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from models.game import Game  # ajusta el import a tu estructura: src.models.game
from dotenv import load_dotenv

load_dotenv()


# ── Configuración ─────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_FAISS = 100   # candidatos semánticos que devuelve semantic_search por defecto


# ── Helpers internos ──────────────────────────────────────────────────────────

def _parse_list(value) -> list[str]:
    """Convierte 'RPG, Action, +' o NaN → ['RPG', 'Action']"""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or value.strip().lower() in ("nan", ""):
        return []
    return [t.strip() for t in value.split(",") if t.strip() and t.strip() != "+"]


def _parse_year(value: str) -> int | None:
    s = str(value or "").strip()
    # Formato "Apr 03, 2007"  → toma los últimos 4 dígitos
    # Formato "2007-04-03"    → toma los primeros 4
    for part in reversed(s.replace("-", " ").split()):
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def _calc_rating(positive: float, negative: float) -> float:
    """Score 0-10 con confianza logarítmica (más reseñas = más peso)."""
    total = positive + negative
    if total == 0:
        return 0.0
    confidence = min(1.0, np.log10(total + 1) / 4)
    raw = (positive / total) * 10
    return round(raw * confidence + 5.0 * (1 - confidence), 2)


def _game_to_text(g: Game) -> str:
    """Texto que representa un juego para el embedding."""
    tags_str  = ", ".join(g.tags[:15])
    genre_str = ", ".join(g.genres)
    desc      = (g.short_description or "")[:300]
    return f"{g.name}. {desc} Géneros: {genre_str}. Tags: {tags_str}."


def _row_to_game(row: pd.Series) -> Game:
    """Convierte una fila del DataFrame en un objeto Game validado."""
    positive = float(row.get("positive_reviews") or 0)
    negative = float(row.get("negative_reviews") or 0)
    total    = int(positive + negative)

    # Positive ratio (usado en Paso 4: parameters.json)
    recommendations_ratio = round(positive / total, 4) if total > 0 else 0.0

    return Game(
        app_id                   = int(row["appid"]),
        name                     = str(row.get("name") or "Unknown"),
        price                    = float(row.get("price") or 0.0),
        genres                   = _parse_list(row.get("genres")),
        categories               = _parse_list(row.get("categories")),
        tags                     = _parse_list(row.get("tags")),
        positive_reviews         = positive,
        negative_reviews         = negative,
        total_reviews            = total,
        # ── campos derivados ──────────────────────────────────────────
        rating                   = _calc_rating(positive, negative),
        recommendations_ratio    = recommendations_ratio,
        release_year             = _parse_year(str(row.get("release_date") or "")),
        # ── sin datos en CSV ──────────────────────────────────────────
        platforms                = None,
        short_description        = str(row.get("short_description") or "") or None,
    )


# ── Motor de conocimiento ─────────────────────────────────────────────────────

class KnowledgeEngine:
    """
    Construye y mantiene la base de conocimiento del agente.

    Atributos públicos (disponibles tras build() o load()):
        games       → lista ordenada de objetos Game
        id_to_game  → dict {app_id: Game} para lookups O(1)
    """

    def __init__(
        self,
        csv_path: str | Path,
        index_dir: str | Path = "src/embeddings",
    ):
        self.csv_path  = Path(csv_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.index_dir / "games.index"
        self._meta_path  = self.index_dir / "games_meta.pkl"

        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None

        self.games: list[Game] = []
        self.id_to_game: dict[int, Game] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def build(self, force: bool = False) -> None:
        """
        Carga el CSV y construye el índice FAISS.
        Si ya existe un índice válido en disco, lo carga (más rápido).
        Usa force=True para reconstruir aunque ya exista.
        """
        self._load_games_from_csv()

        if not force and self._index_exists_and_valid():
            print("[KnowledgeEngine] Índice FAISS encontrado en disco, cargando…")
            self._load_index_from_disk()
        else:
            print("[KnowledgeEngine] Construyendo índice FAISS…")
            self._build_faiss_index()

    def semantic_search(self, query: str, k: int = TOP_K_FAISS) -> dict[int, float]:
        """
        Paso 2 del pipeline: búsqueda semántica.
        Retorna {app_id: similarity_score} para los k juegos más relevantes.
        """
        self._ensure_ready()
        model = self._get_model()

        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, min(k, len(self.games)))

        result: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.games):
                result[self.games[idx].app_id] = float(score)
        return result

    def get_game(self, app_id: int) -> Game | None:
        """Lookup O(1) por app_id. Útil en Paso 3 (Prolog) y Paso 4 (scorer)."""
        return self.id_to_game.get(app_id)

    def get_games(self, app_ids: list[int]) -> list[Game]:
        """Convierte lista de app_ids → lista de Games. Ignora IDs no encontrados."""
        return [self.id_to_game[aid] for aid in app_ids if aid in self.id_to_game]

    # ── Internos: carga CSV ───────────────────────────────────────────────────

    def _load_games_from_csv(self) -> None:
        print(f"[KnowledgeEngine] Leyendo CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path, sep="|")
        self.games = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parseando juegos"):
            try:
                game = _row_to_game(row)
                self.games.append(game)
            except Exception as e:
                print(f"Error en fila (appid={row.get('appid', '?')}): {e}")

        self.id_to_game = {g.app_id: g for g in self.games}
        print(f"[KnowledgeEngine] {len(self.games)} juegos cargados.")

    # ── Internos: FAISS ───────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[KnowledgeEngine] Cargando modelo de embeddings: {EMBED_MODEL}")
            self._model = SentenceTransformer(EMBED_MODEL)
        return self._model

    def _build_faiss_index(self) -> None:
        model  = self._get_model()
        texts  = [_game_to_text(g) for g in self.games]

        print(f"[KnowledgeEngine] Generando embeddings para {len(texts)} juegos…")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatIP(dim)   # Inner Product ≡ coseno (con normalización)
        self._index.add(embeddings)

        # Persistir en disco
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump([g.app_id for g in self.games], f)

        print(f"[KnowledgeEngine] Índice guardado en {self._index_path}")

    def _index_exists_and_valid(self) -> bool:
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        with open(self._meta_path, "rb") as f:
            saved_ids = pickle.load(f)
        return len(saved_ids) == len(self.games)

    def _load_index_from_disk(self) -> None:
        self._index = faiss.read_index(str(self._index_path))
        print(f"[KnowledgeEngine] Índice FAISS listo ({self._index.ntotal} vectores).")

    def _ensure_ready(self) -> None:
        if self._index is None or not self.games:
            raise RuntimeError(
                "KnowledgeEngine no está listo. Llama a build() primero."
            )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ke = KnowledgeEngine(
        csv_path  = "/mnt/user-data/uploads/steam_rpg_games.csv",
        index_dir = "/tmp/embeddings_test",
    )
    ke.build()

    print("\n── Lookup por ID ──")
    game = ke.get_game(1700)
    if game:
        print(f"  {game.name} | ${game.price} | {game.release_year} | rating={game.rating}")
        print(f"  tags[:5]       : {game.tags[:5]}")
        print(f"  recommendations_ratio : {game.recommendations_ratio:.1%}")

    print("\n── Búsqueda semántica ──")
    results = ke.semantic_search("dark fantasy dungeon crawler", k=5)
    for app_id, score in sorted(results.items(), key=lambda x: -x[1]):
        g = ke.get_game(app_id)
        print(f"  [{score:.3f}] {g.name if g else app_id}")