from __future__ import annotations

import base64
import csv
import datetime as dt
import json
import math
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path("data")
DEFAULT_QUERY_URL = (
    "https://www.airbnb.com/s/Istria--Croatia/homes?"
    "refinement_paths%5B%5D=/homes&query=Istria,%20Croatia&check_in=2026-06-14"
    "&check_out=2026-06-28&adults=2&search_type=filter_change&search_mode=regular_search"
    "&price_filter_input_type=2&price_filter_num_nights=14&price_max=1680"
)
NIGHTS_TARGET = 14
NIGHTS_TOLERANCE = 2
MAX_EUR_PER_NIGHT = 120.0
MAX_WALK_MINUTES_TO_BEACH = 15

# Hardcoded city allowlist for routes typically under 7h from Zagreb by FlixBus/Arriva.
REACHABLE_CITIES_UNDER_7H = {
    "buzet",
    "fazana",
    "labin",
    "lovran",
    "medulin",
    "novigrad",
    "opatija",
    "pazin",
    "porec",
    "pula",
    "rabac",
    "rijeka",
    "rovinj",
    "umag",
    "vodnjan",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en",
}


@dataclass
class Listing:
    listing_id: str
    title: str
    subtitle: str
    rating_text: str | None
    total_price_eur: float | None
    nightly_price_eur: float | None
    latitude: float
    longitude: float
    inferred_city: str | None
    city_reachable_under_7h_bus: bool
    beach_distance_km_estimate: float | None
    beach_walk_minutes_estimate: float | None
    satisfies_filters: bool
    room_url: str | None


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def extract_state_payload(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        script_id = script.get("id", "")
        if script_id.startswith("data-deferred-state-") and script.string:
            parsed = json.loads(script.string)
            if "niobeClientData" in parsed and parsed["niobeClientData"]:
                return parsed["niobeClientData"][0][1]
    raise RuntimeError("Could not find Airbnb deferred state payload in page.")


def parse_eur(value: str | None) -> float | None:
    if not value:
        return None
    clean = value.replace("\xa0", " ")
    match = re.search(r"€\s*([\d\.,]+)", clean)
    if not match:
        return None
    number = match.group(1)
    if "," in number and "." in number:
        # Right-most separator is most likely decimal, the other thousands.
        if number.rfind(",") > number.rfind("."):
            number = number.replace(".", "").replace(",", ".")
        else:
            number = number.replace(",", "")
    elif "," in number:
        whole, frac = number.rsplit(",", 1)
        if len(frac) == 3:
            number = whole + frac
        else:
            number = whole + "." + frac
    elif "." in number:
        whole, frac = number.rsplit(".", 1)
        if len(frac) == 3:
            number = whole + frac
    try:
        return float(number)
    except ValueError:
        return None


def parse_total_price(item: dict[str, Any]) -> float | None:
    primary = item.get("structuredDisplayPrice", {}).get("primaryLine", {})
    if "discountedPrice" in primary:
        return parse_eur(primary.get("discountedPrice"))
    if "price" in primary:
        return parse_eur(primary.get("price"))
    if "accessibilityLabel" in primary:
        return parse_eur(primary.get("accessibilityLabel"))
    return None


def decode_room_id(encoded_id: str) -> str | None:
    try:
        decoded = base64.b64decode(encoded_id).decode()
        _, room_id = decoded.split(":", 1)
        return room_id
    except Exception:
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def overpass_beach_points(lat: float, lon: float, radius_m: int = 4000) -> list[tuple[float, float]]:
    query = f"""
[out:json][timeout:25];
(
  node(around:{radius_m},{lat},{lon})[natural=beach];
  way(around:{radius_m},{lat},{lon})[natural=beach];
  relation(around:{radius_m},{lat},{lon})[natural=beach];
  node(around:{radius_m},{lat},{lon})[tourism=beach];
  way(around:{radius_m},{lat},{lon})[leisure=beach_resort];
);
out center 60;
"""
    response = requests.post(
        "https://overpass.kumi.systems/api/interpreter",
        data=query,
        headers=HEADERS,
        timeout=40,
    )
    response.raise_for_status()
    payload = response.json()
    points: list[tuple[float, float]] = []
    for element in payload.get("elements", []):
        if "lat" in element and "lon" in element:
            points.append((element["lat"], element["lon"]))
            continue
        center = element.get("center", {})
        if "lat" in center and "lon" in center:
            points.append((center["lat"], center["lon"]))
    return points


def estimate_beach_walk_minutes(lat: float, lon: float) -> tuple[float | None, float | None]:
    points = overpass_beach_points(lat, lon)
    if not points:
        return None, None
    direct_distance_km = min(haversine_km(lat, lon, p_lat, p_lon) for p_lat, p_lon in points)
    walking_path_km = direct_distance_km * 1.35
    walking_minutes = walking_path_km / 4.5 * 60
    return direct_distance_km, walking_minutes


def infer_city(title: str, subtitle: str) -> str | None:
    match = re.search(r"\bin\s+([^,]+)$", title, flags=re.IGNORECASE)
    if match:
        city_raw = normalize_text(match.group(1)).split()
        if city_raw:
            return city_raw[0]

    searchable = f"{normalize_text(title)} {normalize_text(subtitle)}"
    for city in sorted(REACHABLE_CITIES_UNDER_7H):
        if re.search(rf"\b{re.escape(city)}\b", searchable):
            return city
    return None


def nights_from_url(search_url: str) -> int:
    parsed = urlparse(search_url)
    params = parse_qs(parsed.query)
    check_in = params.get("check_in", [None])[0]
    check_out = params.get("check_out", [None])[0]
    if not check_in or not check_out:
        raise ValueError("URL must contain check_in and check_out parameters.")
    in_date = dt.date.fromisoformat(check_in)
    out_date = dt.date.fromisoformat(check_out)
    return (out_date - in_date).days


def build_page_url(search_url: str, page: int) -> str:
    parsed = urlparse(search_url)
    params = parse_qs(parsed.query)
    params["items_offset"] = [str(page * 20)]
    params["section_offset"] = [str(page)]
    params["pagination_search"] = ["true"]
    encoded = urlencode(params, doseq=True)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{encoded}"


def fetch_listings(search_url: str, max_pages: int = 5) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for page in range(max_pages):
        page_url = build_page_url(search_url, page)
        response = requests.get(page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        payload = extract_state_payload(response.text)
        search_results = (
            payload.get("data", {})
            .get("presentation", {})
            .get("staysSearch", {})
            .get("results", {})
            .get("searchResults", [])
        )
        new_items = 0
        for item in search_results:
            listing_id = item.get("demandStayListing", {}).get("id")
            if not listing_id or listing_id in seen_ids:
                continue
            seen_ids.add(listing_id)
            collected.append(item)
            new_items += 1
        if new_items == 0:
            break
    return collected


def to_listing(item: dict[str, Any], nights: int) -> Listing | None:
    demand = item.get("demandStayListing", {})
    encoded_id = demand.get("id")
    location = demand.get("location", {}).get("coordinate", {})
    lat = location.get("latitude")
    lon = location.get("longitude")
    if encoded_id is None or lat is None or lon is None:
        return None

    title = item.get("title", "")
    subtitle = item.get("subtitle", "")
    total_price = parse_total_price(item)
    nightly_price = (total_price / nights) if total_price and nights > 0 else None
    city = infer_city(title=title, subtitle=subtitle)
    city_allowed = city in REACHABLE_CITIES_UNDER_7H if city else False

    beach_distance_km = None
    beach_walk_minutes = None
    try:
        beach_distance_km, beach_walk_minutes = estimate_beach_walk_minutes(lat, lon)
    except Exception:
        pass

    within_price = nightly_price is not None and nightly_price <= MAX_EUR_PER_NIGHT
    near_beach = (
        beach_walk_minutes is not None
        and beach_walk_minutes <= MAX_WALK_MINUTES_TO_BEACH
    )
    is_match = within_price and near_beach and city_allowed
    room_id = decode_room_id(encoded_id)
    return Listing(
        listing_id=encoded_id,
        title=title,
        subtitle=subtitle,
        rating_text=item.get("avgRatingLocalized"),
        total_price_eur=round(total_price, 2) if total_price is not None else None,
        nightly_price_eur=round(nightly_price, 2) if nightly_price is not None else None,
        latitude=lat,
        longitude=lon,
        inferred_city=city,
        city_reachable_under_7h_bus=city_allowed,
        beach_distance_km_estimate=(
            round(beach_distance_km, 3) if beach_distance_km is not None else None
        ),
        beach_walk_minutes_estimate=(
            round(beach_walk_minutes, 1) if beach_walk_minutes is not None else None
        ),
        satisfies_filters=is_match,
        room_url=f"https://www.airbnb.com/rooms/{room_id}" if room_id else None,
    )


def persist(listings: list[Listing], search_url: str, nights: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "airbnb_istria_all.json"
    filtered_path = DATA_DIR / "airbnb_istria_filtered.json"
    csv_path = DATA_DIR / "airbnb_istria_filtered.csv"
    run_meta_path = DATA_DIR / "airbnb_istria_run_metadata.json"

    rows = [asdict(x) for x in listings]
    filtered = [row for row in rows if row["satisfies_filters"]]

    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    filtered_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    fieldnames = list(filtered[0].keys()) if filtered else list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(filtered)

    metadata = {
        "source_url": search_url,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "criteria": {
            "nights_range": [NIGHTS_TARGET - NIGHTS_TOLERANCE, NIGHTS_TARGET + NIGHTS_TOLERANCE],
            "max_eur_per_night": MAX_EUR_PER_NIGHT,
            "max_beach_walk_minutes": MAX_WALK_MINUTES_TO_BEACH,
            "bus_constraint": "Hardcoded reachable city allowlist from Zagreb (<7h).",
            "reachable_cities": sorted(REACHABLE_CITIES_UNDER_7H),
        },
        "resolved_nights_from_url": nights,
        "total_scraped": len(rows),
        "total_matching": len(filtered),
        "outputs": {
            "all_json": str(raw_path),
            "filtered_json": str(filtered_path),
            "filtered_csv": str(csv_path),
        },
    }
    run_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    search_url = DEFAULT_QUERY_URL
    nights = nights_from_url(search_url)
    if not (NIGHTS_TARGET - NIGHTS_TOLERANCE <= nights <= NIGHTS_TARGET + NIGHTS_TOLERANCE):
        raise ValueError(
            f"Search URL resolves to {nights} nights. Required range is "
            f"{NIGHTS_TARGET - NIGHTS_TOLERANCE}..{NIGHTS_TARGET + NIGHTS_TOLERANCE}."
        )
    raw_items = fetch_listings(search_url, max_pages=5)
    listings = [listing for item in raw_items if (listing := to_listing(item, nights))]
    persist(listings=listings, search_url=search_url, nights=nights)
    total_matches = sum(1 for l in listings if l.satisfies_filters)
    print(f"Processed {len(listings)} listings. Matching criteria: {total_matches}.")
    print(f"Saved outputs to {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
