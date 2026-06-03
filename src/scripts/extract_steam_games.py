import requests
import pandas as pd
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

# =====================================================
# CONFIGURACIÓN
# =====================================================

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

def _get_steam_api_key() -> str:
    key = os.getenv("STEAM_API_KEY")
    if not key:
        raise Exception(
            "STEAM_API_KEY no encontrada en .env. "
            "Obtén una en https://steamcommunity.com/dev/apikey "
            "y crea un archivo .env con: STEAM_API_KEY=tu_clave"
        )
    return key

NUM_GAMES = 350000          # cantidad de juegos a descargar
MAX_WORKERS = 50          # hilos concurrentes

OUTPUT_FILE = "./src/data/steam_games_details.csv"

# =====================================================
# SESIÓN HTTP REUTILIZABLE
# =====================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/136.0 Safari/537.36"
    )
})

# =====================================================
# OBTENER LISTA DE JUEGOS
# =====================================================

def get_games(limit=5000):

    url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"

    params = {
        "key": _get_steam_api_key(),
        "include_games": True,
        "include_dlc": False,
        "include_software": False,
        "include_hardware": False,
        "include_extended_app_info": False,
        "max_results": NUM_GAMES
    }

    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()

    apps = response.json()["response"]["apps"]

    print(f"Total juegos encontrados: {len(apps):,}")

    return apps[:limit]

# =====================================================
# DETALLES DEL JUEGO
# =====================================================

def get_game_details(appid):

    try:

        url = (
            f"https://store.steampowered.com/api/appdetails"
            f"?appids={appid}"
        )

        response = session.get(
            url,
            timeout=20
        )

        data = response.json()

        game = data.get(str(appid))

        if not game:
            return None

        if not game.get("success"):
            return None

        info = game["data"]

        return {
            "appid": appid,
            "name": info.get("name"),
            "release_date": (
                info.get("release_date", {})
                .get("date")
            ),
            "price": (
                info.get("price_overview", {})
                .get("initial", 0)
            ) / 100,
            "discount_percent": (
                info.get("price_overview", {})
                .get("discount_percent", 0)
            ),
            "genres": ", ".join(
                g["description"]
                for g in info.get("genres", [])
            ),
            "categories": ", ".join(
                c["description"]
                for c in info.get("categories", [])
            ),
            "developers": ", ".join(
                info.get("developers", [])
            ),
            "recommendations": (
                info.get("recommendations", {})
                .get("total", 0)
            ),
            "achievements": (
                info.get("achievements", {})
                .get("total", 0)
            )
        }

    except Exception:
        return None

# =====================================================
# REVIEWS
# =====================================================

def get_reviews_summary(appid):

    try:

        url = (
            f"https://store.steampowered.com/"
            f"appreviews/{appid}"
            "?json=1&filter=summary"
        )

        summary = (
            session
            .get(url, timeout=20)
            .json()
            .get("query_summary", {})
        )

        return {
            "positive_reviews": summary.get(
                "total_positive", 0
            ),
            "negative_reviews": summary.get(
                "total_negative", 0
            ),
            "total_reviews": summary.get(
                "total_reviews", 0
            )
        }

    except Exception:

        return {
            "positive_reviews": None,
            "negative_reviews": None,
            "total_reviews": None
        }

# =====================================================
# TAGS
# =====================================================

def get_tags(appid):

    try:

        url = (
            f"https://store.steampowered.com/app/{appid}"
        )

        html = session.get(
            url,
            timeout=20
        ).text

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        tags = [
            tag.text.strip()
            for tag in soup.select(".app_tag")
        ]

        return ", ".join(tags)

    except Exception:
        return None

# =====================================================
# DATOS COMPLETOS DE UN JUEGO
# =====================================================

def get_full_data(appid):

    details = get_game_details(appid)

    if details is None:
        return None

    reviews = get_reviews_summary(appid)

    tags = get_tags(appid)

    return {
        **details,
        **reviews,
        "tags": tags
    }

# =====================================================
# PROCESAMIENTO PARALELO
# =====================================================

def build_dataset_parallel(
    games,
    max_workers=50
):

    dataset = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = {
            executor.submit(
                get_full_data,
                game["appid"]
            ): game["appid"]
            for game in games
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Procesando juegos"
        ):

            try:

                result = future.result()

                if result:
                    dataset.append(result)

            except Exception:
                pass

    return pd.DataFrame(dataset)

# =====================================================
# MAIN
# =====================================================

def main():

    games = get_games(NUM_GAMES)

    print(
        f"Juegos seleccionados: {len(games):,}"
    )

    df = build_dataset_parallel(
        games,
        max_workers=MAX_WORKERS
    )

    print(
        f"Juegos procesados: {len(df):,}"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(
        f"Archivo guardado: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()