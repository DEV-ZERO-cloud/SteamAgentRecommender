# Batch Steam Profile Extractor → CSV
from typing import List
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError

from dotenv import load_dotenv

import pandas as pd
import requests
import argparse
import os
import re


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

HEADERS = {"User-Agent": "Mozilla/5.0"}

OUTPUT_PATH = "./src/data/steam_users.csv"

# Max parallel workers for tag fetching and user processing
TAG_WORKERS = 10
USER_WORKERS = 5


# =========================================================
# MODEL
# =========================================================

class SteamUserProfile(BaseModel):

    user_id: str
    name: str = ""
    played_app_ids: List[int] = Field(default_factory=list)
    preferred_tags: List[str] = Field(default_factory=list)
    amount_playing: List[int] = Field(default_factory=list)


# =========================================================
# SEARCH PROFILE
# =========================================================

def search_steam_profile(username: str):

    with sync_playwright() as p:

        try:
            browser = p.chromium.launch(headless=True)

        except PlaywrightError:
            raise Exception(
                "Playwright no tiene navegadores instalados. "
                "Ejecuta: playwright install"
            )

        page = browser.new_page()
        url = f"https://steamcommunity.com/search/users/#text={username}"
        page.goto(url)
        page.wait_for_timeout(5000)
        html = page.content()
        browser.close()

        match = re.search(r"steamcommunity\.com/profiles/(\d+)", html)
        if match:
            return match.group(1)

        match = re.search(r"steamcommunity\.com/id/([^\"/]+)", html)
        if match:
            return match.group(1)

        return None


# =========================================================
# RESOLVE VANITY
# =========================================================

def resolve_vanity_url(vanity_name: str) -> str:

    url = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
    params = {"key": _get_steam_api_key(), "vanityurl": vanity_name}

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise Exception(f"Steam API devolvió {response.status_code}")

    data = response.json()

    if data["response"]["success"] != 1:
        raise Exception(f"No se encontró vanity url: {vanity_name}")

    return data["response"]["steamid"]


# =========================================================
# EXTRACT STEAM ID
# =========================================================

_STEAMID64_OFFSET = 76561197960265728

def extract_steam_id(user_input: str) -> str:

    if user_input.isdigit():
        num = int(user_input)
        if num < _STEAMID64_OFFSET:
            return str(_STEAMID64_OFFSET + num)
        return user_input

    profile_match = re.search(
        r"steamcommunity\.com/profiles/(\d+)", user_input
    )
    if profile_match:
        return profile_match.group(1)

    vanity_match = re.search(
        r"steamcommunity\.com/id/([^/]+)", user_input
    )
    if vanity_match:
        return resolve_vanity_url(vanity_match.group(1))

    # Intentar resolver el texto como vanity name directamente (sin URL completa)
    # Si no es un vanity válido, la API lanza excepción → seguir a Playwright
    try:
        return resolve_vanity_url(user_input)
    except Exception:
        pass

    found_profile = search_steam_profile(user_input)

    if not found_profile:
        raise Exception(f"No se pudo encontrar el usuario: {user_input}")

    if found_profile.isdigit():
        return found_profile

    return resolve_vanity_url(found_profile)


# =========================================================
# PLAYER SUMMARY
# =========================================================

def get_player_summary(steam_id: str) -> dict:

    url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
    params = {"key": _get_steam_api_key(), "steamids": steam_id}

    response = requests.get(url, params=params)
    data = response.json()
    players = data["response"]["players"]

    if not players:
        raise Exception("Usuario no encontrado")

    return players[0]


# =========================================================
# OWNED GAMES
# =========================================================

def get_owned_games(steam_id: str) -> list:

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {
        "key": _get_steam_api_key(),
        "steamid": steam_id,
        "include_appinfo": True,
        "include_played_free_games": True,
    }

    response = requests.get(url, params=params)
    data = response.json()

    return data.get("response", {}).get("games", [])


# =========================================================
# GAME TAGS  (unchanged logic, used inside thread pool)
# =========================================================

def get_game_tags(app_id: int) -> list[str]:

    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        app_data = data[str(app_id)]

        if not app_data["success"]:
            return []

        genres = app_data["data"].get("genres", [])
        return [genre["description"] for genre in genres]

    except Exception:
        return []


# =========================================================
# BUILD PROFILE  (parallelized tag fetching)
# =========================================================

def build_user_profile(user_input: str) -> SteamUserProfile:

    steam_id = extract_steam_id(user_input)

    # Fetch summary and games concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_summary = pool.submit(get_player_summary, steam_id)
        future_games   = pool.submit(get_owned_games, steam_id)
        summary = future_summary.result()
        games   = future_games.result()

    # Keep only games with playtime
    played_games = [
        (game["appid"], round(game.get("playtime_forever", 0) / 60))
        for game in games
        if round(game.get("playtime_forever", 0) / 60) > 0
    ]

    played_app_ids  = [g[0] for g in played_games]
    amount_playing  = [g[1] for g in played_games]

    # Fetch tags for all played games in parallel
    tag_counter: Counter = Counter()

    with ThreadPoolExecutor(max_workers=TAG_WORKERS) as pool:
        future_to_app = {
            pool.submit(get_game_tags, app_id): app_id
            for app_id in played_app_ids
        }
        for future in as_completed(future_to_app):
            tags = future.result()
            tag_counter.update(tags)

    preferred_tags = [tag for tag, _ in tag_counter.most_common(10)]

    return SteamUserProfile(
        user_id=steam_id,
        name=summary.get("personaname", ""),
        played_app_ids=played_app_ids,
        preferred_tags=preferred_tags,
        amount_playing=amount_playing,
    )


# =========================================================
# PROCESS USERS  (parallelized across users)
# =========================================================

def _process_single(username: str) -> dict | None:
    """Worker function for a single username. Returns dict or None on error."""
    print(f"Procesando: {username}")
    try:
        profile = build_user_profile(username)
        print(f"OK -> {profile.name}")
        return {
            "user_id":        profile.user_id,
            "name":           profile.name,
            "played_app_ids": profile.played_app_ids,
            "preferred_tags": profile.preferred_tags,
            "amount_playing": profile.amount_playing,
        }
    except Exception as e:
        print(f"ERROR -> {username}: {e}")
        return None


def process_users(usernames: list[str]) -> list[dict]:

    profiles: list[dict] = []

    with ThreadPoolExecutor(max_workers=USER_WORKERS) as pool:
        futures = {pool.submit(_process_single, u): u for u in usernames}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                profiles.append(result)

    return profiles


# =========================================================
# SAVE CSV
# =========================================================

def save_profiles_csv(profiles: list[dict]):

    os.makedirs("src/data", exist_ok=True)

    df = pd.DataFrame(profiles)
    df.to_csv(OUTPUT_PATH, index=False, sep="|")

    print(f"CSV guardado en: {OUTPUT_PATH}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Steam Batch Profile Extractor")
    parser.add_argument("users", nargs="+", help="Lista de usuarios Steam")
    args = parser.parse_args()

    profiles = process_users(args.users)
    save_profiles_csv(profiles)
    print("Proceso finalizado")