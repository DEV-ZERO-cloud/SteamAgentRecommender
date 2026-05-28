# Batch Steam Profile Extractor → CSV  (optimized: full async I/O)
from __future__ import annotations

import asyncio
import os
import re
import argparse
from collections import Counter
from typing import Optional
from urllib.parse import quote

import httpx
import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError

from models.user import User

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

STEAM_API_KEY = os.getenv("STEAM_API_KEY")

if not STEAM_API_KEY:
    raise Exception("STEAM_API_KEY no encontrada en .env")

HEADERS = {"User-Agent": "Mozilla/5.0"}

OUTPUT_PATH = "./src/data/steam_users.csv"

TAG_CONCURRENCY   = 15
USER_WORKERS      = 5
HTTP_TIMEOUT      = 15


# =========================================================
# HTTP CLIENT
# =========================================================

def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=HEADERS,
        timeout=HTTP_TIMEOUT,
        limits=httpx.Limits(max_connections=60, max_keepalive_connections=30),
    )


# =========================================================
# STEAM ID RESOLUTION
# =========================================================

def search_steam_profile(username: str) -> Optional[str]:
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            raise Exception(
                "Playwright no tiene navegadores instalados. "
                "Ejecuta: playwright install"
            )
        page = browser.new_page()
        page.goto(f"https://steamcommunity.com/search/users/#text={username}")
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


async def resolve_vanity_url(vanity_name: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(
        "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/",
        params={"key": STEAM_API_KEY, "vanityurl": vanity_name},
    )
    resp.raise_for_status()
    data = resp.json()
    if data["response"]["success"] != 1:
        raise Exception(f"No se encontró vanity url: {vanity_name}")
    return data["response"]["steamid"]


async def extract_steam_id(user_input: str, client: httpx.AsyncClient) -> str:
    if user_input.isdigit():
        return user_input
    m = re.search(r"steamcommunity\.com/profiles/(\d+)", user_input)
    if m:
        return m.group(1)
    m = re.search(r"steamcommunity\.com/id/([^/]+)", user_input)
    if m:
        return await resolve_vanity_url(m.group(1), client)
    loop = asyncio.get_running_loop()
    found = await loop.run_in_executor(None, search_steam_profile, user_input)
    if not found:
        raise Exception(f"No se pudo encontrar el usuario: {user_input}")
    if found.isdigit():
        return found
    return await resolve_vanity_url(found, client)


# =========================================================
# STEAM API CALLS
# =========================================================

async def get_player_summary(steam_id: str, client: httpx.AsyncClient) -> dict:
    resp = await client.get(
        "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/",
        params={"key": STEAM_API_KEY, "steamids": steam_id},
    )
    resp.raise_for_status()
    players = resp.json()["response"]["players"]
    if not players:
        raise Exception("Usuario no encontrado")
    return players[0]


async def get_owned_games(steam_id: str, client: httpx.AsyncClient) -> list:
    resp = await client.get(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        params={
            "key": STEAM_API_KEY,
            "steamid": steam_id,
            "include_appinfo": True,
            "include_played_free_games": True,
        },
    )
    resp.raise_for_status()
    return resp.json().get("response", {}).get("games", [])


async def get_game_tags(
    app_id: int,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> list[str]:
    async with sem:
        try:
            resp = await client.get(
                f"https://store.steampowered.com/api/appdetails?appids={app_id}",
            )
            data = resp.json()
            app_data = data.get(str(app_id), {})
            if not app_data.get("success"):
                return []
            return [g["description"] for g in app_data["data"].get("genres", [])]
        except Exception:
            return []


# =========================================================
# CORE PROFILE BUILDER
# =========================================================

async def _build_user_profile_async(user_input: str) -> User:
    async with _make_client() as client:

        # 1. Resolve Steam ID
        steam_id = await extract_steam_id(user_input, client)

        # 2. summary + games en paralelo (reviews se omiten aquí;
        #    usa build_review_dataset() si las necesitas por separado)
        summary, games = await asyncio.gather(
            get_player_summary(steam_id, client),
            get_owned_games(steam_id, client),
        )

        # 3. Filter to games with playtime
        played_games = [
            (game["appid"], round(game.get("playtime_forever", 0) / 60))
            for game in games
            if round(game.get("playtime_forever", 0) / 60) > 0
        ]
        played_app_ids = [g[0] for g in played_games]
        amount_playing = [g[1] for g in played_games]

        # 4. Fetch tags for all played games in parallel
        tag_sem   = asyncio.Semaphore(TAG_CONCURRENCY)
        tag_lists = await asyncio.gather(
            *[get_game_tags(app_id, client, tag_sem) for app_id in played_app_ids]
        )

    # Aggregate tags
    tag_counter: Counter = Counter()
    for tags in tag_lists:
        tag_counter.update(tags)
    preferred_tags = [tag for tag, _ in tag_counter.most_common(10)]

    # Build review arrays (vacíos; usar build_review_dataset() para poblarlos)
    reviewed_app_ids: list[int] = []
    review_sentiment: list[int] = []

    return User(
        user_id=steam_id,
        name=summary.get("personaname", ""),
        played_app_ids=played_app_ids,
        preferred_tags=preferred_tags,
        amount_playing=amount_playing,
        reviewed_app_ids=reviewed_app_ids,
        review_sentiment=review_sentiment,
    )


def build_user_profile(user_input: str) -> User:
    """Sync wrapper for CLI / batch compatibility."""
    return asyncio.run(_build_user_profile_async(user_input))


# =========================================================
# BATCH PROCESSING
# =========================================================

async def _process_single_async(username: str) -> dict | None:
    print(f"Procesando: {username}")
    try:
        profile = await _build_user_profile_async(username)
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


async def _process_users_async(usernames: list[str]) -> list[dict]:
    results = await asyncio.gather(*[_process_single_async(u) for u in usernames])
    return [r for r in results if r is not None]


def process_users(usernames: list[str]) -> list[dict]:
    return asyncio.run(_process_users_async(usernames))


# =========================================================
# REVIEW DATASET HELPER
# =========================================================

async def _build_review_dataset_async(steam_id: str) -> list[dict]:
    async with _make_client() as client:
        reviews = await get_user_reviews_from_profile(steam_id, client)
    return [
        {"user_id": steam_id, "app_id": r["app_id"], "liked": int(r["voted_up"])}
        for r in reviews
    ]


def build_review_dataset(steam_id: str) -> list[dict]:
    return asyncio.run(_build_review_dataset_async(steam_id))


# =========================================================
# CSV EXPORT
# =========================================================

def save_profiles_csv(profiles: list[dict]):
    os.makedirs("src/data", exist_ok=True)
    df = pd.DataFrame(profiles)
    str_cols = {"preferred_tags"}
    int_cols = {"played_app_ids", "amount_playing"}
    for col in str_cols | int_cols:
        if col in df.columns:
            if col in str_cols:
                df[col] = df[col].apply(lambda x: "[" + ", ".join(f'"{v}"' for v in x) + "]")
            else:
                df[col] = df[col].apply(lambda x: "[" + ", ".join(str(v) for v in x) + "]")
    df.to_csv(OUTPUT_PATH, index=False, sep="|", quoting=3)  # quoting=3 → QUOTE_NONE
    print(f"CSV guardado en: {OUTPUT_PATH}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steam Batch Profile Extractor")
    parser.add_argument("users", nargs="+", help="Lista de usuarios Steam")
    args = parser.parse_args()

    profiles = process_users(args.users)
    save_profiles_csv(profiles)
    print("Proceso finalizado")