import requests
import pandas as pd
import time
import re

from bs4 import BeautifulSoup
from difflib import SequenceMatcher

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =========================================================
# 1. FUENTES EXTERNAS CURADAS - BEST RPGS 2023
# =========================================================

CURATED_SOURCES = [
    "https://as.com/meristation/plataformas/computadora-personal/top/videojuegos-rol/",
    "https://vandal.elespanol.com/rankings/pc/rol"
]

# =========================================================
# FILTROS
# =========================================================

BLACKLIST = {
    "platform",
    "platforms",
    "release date",
    "developer",
    "publisher",
    "genre",
    "vibe",
    "rpg",
    "best rpgs",
    "best games",
}

MIN_SIMILARITY = 0.65

# =========================================================
# EXTRAER NOMBRES
# =========================================================

def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = text.strip()

    return text

def extract_game_names(url):
    game_names = set()

    EXTRA_BLACKLIST = {
        "pc",
        "ps5",
        "xbox",
        "switch",
        "analisis",
        "análisis",
        "imagenes",
        "imágenes",
        "videos",
        "video",
        "nota",
        "avance",
        "review",
    }

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup.find_all(["h2", "h3", "a"]):

            text = clean_text(
                tag.get_text(" ", strip=True)
            )

            text_lower = text.lower()

            # =========================================
            # FILTROS
            # =========================================

            if (
                len(text) < 3
                or len(text) > 80
                or text_lower in BLACKLIST
                or text_lower in EXTRA_BLACKLIST
            ):
                continue

            if any(word in text_lower for word in EXTRA_BLACKLIST):
                continue

            # Debe tener letras
            if not re.search(r"[A-Za-z]", text):
                continue

            # Evitar textos basura
            if text.isnumeric():
                continue

            game_names.add(text)

    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return sorted(game_names)

# =========================================================
# SIMILITUD
# =========================================================

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# =========================================================
# BUSCAR EN STEAM
# =========================================================

def search_steam_app(game_name):
    try:
        url = "https://store.steampowered.com/api/storesearch/"

        response = requests.get(
            url,
            params={
                "term": game_name,
                "l": "english",
                "cc": "us"
            },
            headers=HEADERS,
            timeout=15
        )

        data = response.json()

        if not data["items"]:
            return None

        best_match = None
        best_score = 0

        for item in data["items"]:
            steam_name = item["name"]

            score = similarity(game_name, steam_name)

            if score > best_score:
                best_score = score
                best_match = item

        if best_score >= MIN_SIMILARITY:
            return best_match["id"]

    except Exception as e:
        print(f"Search error {game_name}: {e}")

    return None

# =========================================================
# DETALLES STEAM
# =========================================================

def get_steam_details(app_id):
    try:
        url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&l=english"

        response = requests.get(url, headers=HEADERS, timeout=20)
        data = response.json()

        if not data[str(app_id)]["success"]:
            return None

        game = data[str(app_id)]["data"]

        genres = [
            g["description"]
            for g in game.get("genres", [])
        ]

        # FILTRAR SOLO RPGS
        if "RPG" not in genres:
            return None

        # ============================
        # PRECIO NORMALIZADO A USD
        # ============================
        price_info = game.get("price_overview")

        if game.get("is_free"):
            price_usd = 0.0
        elif price_info:
            price_usd = price_info.get("final", 0) / 100
        else:
            price_usd = None  # DLCs o casos sin precio

        return {
            "app_id": app_id,
            "name": game.get("name"),
            "release_date": game.get("release_date", {}).get("date"),

            # 🔥 ahora es numérico en USD
            "price_usd": price_usd,

            # opcional: mantener display
            "price_display": (
                "Free"
                if game.get("is_free")
                else price_info.get("final_formatted") if price_info else None
            ),

            "genres": ", ".join(genres),
            "categories": ", ".join([
                c["description"]
                for c in game.get("categories", [])
            ])
        }

    except Exception as e:
        print(f"Details error {app_id}: {e}")

    return None

# =========================================================
# STEAMSPY
# =========================================================

def get_steamspy_data(app_id):
    try:
        url = f"https://steamspy.com/api.php?request=appdetails&appid={app_id}"

        response = requests.get(url, headers=HEADERS, timeout=20)
        data = response.json()

        return {
            "positive_reviews": data.get("positive", 0),
            "negative_reviews": data.get("negative", 0),
            "recommendation_score": data.get("score_rank", ""),
            "tags": ", ".join(
                list(data.get("tags", {}).keys())
            ) if isinstance(data.get("tags"), dict) else ""
        }

    except:
        pass

    return {
        "positive_reviews": 0,
        "negative_reviews": 0,
        "recommendation_score": "",
        "tags": ""
    }

# =========================================================
# PIPELINE
# =========================================================

all_games = set()

print("Extrayendo listas curadas...\n")

for source in CURATED_SOURCES:
    names = extract_game_names(source)

    print(f"{source} -> {len(names)} candidatos")

    all_games.update(names)

print(f"\nTotal candidatos: {len(all_games)}")

results = []

for game_name in sorted(all_games):

    print(f"\nBuscando: {game_name}")

    app_id = search_steam_app(game_name)

    if not app_id:
        continue

    steam_data = get_steam_details(app_id)

    if not steam_data:
        continue

    steamspy_data = get_steamspy_data(app_id)

    row = {
        **steam_data,
        **steamspy_data
    }

    results.append(row)

    print(f"OK -> {steam_data['name']}")

    time.sleep(1)

# =========================================================
# EXPORTAR
# =========================================================

df = pd.DataFrame(results)

df.drop_duplicates(subset=["app_id"], inplace=True)

df.to_csv(
    "./src/data/best_rpgs_steam.csv",
    index=False,
    sep="|",
    encoding="utf-8-sig"
)

print("\n================================")
print("CSV generado correctamente")
print("================================")

print(df.head())