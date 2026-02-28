import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Any

from apify_client import ApifyClient

ACTOR_ID = "GsNzxEKzE2vQ5d9HN"
DEFAULT_LOCATIONS = [
    "Zadar County, Croatia",
    "Sibenik, Croatia",
    "Split, Croatia",
    "Makarska, Croatia",
    "Rijeka, Croatia",
]
BUS_OK_CITY_HINTS = [
    "zadar",
    "sibenik",
    "šibenik",
    "vodice",
    "bilice",
    "bibinje",
    "split",
    "makarska",
    "rijeka",
]
BEACH_HINTS = [
    "beach",
    "beachfront",
    "waterfront",
    "seafront",
    "sea view",
    "by the sea",
    "near the sea",
]


def load_env_file(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape and filter Airbnb listings for your Croatia use case."
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=DEFAULT_LOCATIONS,
        help="Location queries passed to the actor.",
    )
    parser.add_argument("--check-in", default="2026-06-14", help="YYYY-MM-DD")
    parser.add_argument("--check-out", default="2026-06-28", help="YYYY-MM-DD")
    parser.add_argument("--adults", type=int, default=2)
    parser.add_argument("--currency", default="EUR")
    parser.add_argument(
        "--max-total-price",
        type=float,
        default=1680.0,
        help="Max total stay price to send to actor.",
    )
    parser.add_argument(
        "--max-nightly",
        type=float,
        default=120.0,
        help="Max nightly price after post-filtering.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where all output files are written.",
    )
    return parser.parse_args()


def stay_nights(check_in: str, check_out: str) -> int:
    start = datetime.strptime(check_in, "%Y-%m-%d")
    end = datetime.strptime(check_out, "%Y-%m-%d")
    return (end - start).days


def extract_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.,-]", "", value).replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("amount", "value", "price", "total", "nightly"):
            if key in value:
                parsed = extract_price(value.get(key))
                if parsed is not None:
                    return parsed
    return None


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def text_blob(item: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("title", "name", "description", "summary", "city", "location"):
        val = item.get(key)
        if isinstance(val, str):
            chunks.append(val)
    amenities = item.get("amenities")
    if isinstance(amenities, list):
        chunks.extend(str(a) for a in amenities)
    return " ".join(chunks).lower()


def is_beach_friendly(item: dict[str, Any]) -> bool:
    blob = text_blob(item)
    return any(hint in blob for hint in BEACH_HINTS)


def is_bus_ok(item: dict[str, Any]) -> bool:
    city_candidates = [
        item.get("city"),
        item.get("location"),
        item.get("locality"),
        item.get("address"),
        item.get("title"),
        item.get("name"),
    ]
    combined = " ".join(str(c) for c in city_candidates if c).lower()
    return any(hint in combined for hint in BUS_OK_CITY_HINTS)


def normalize_item(item: dict[str, Any], nights: int) -> dict[str, Any]:
    listing_url = first_present(item, ["url", "listingUrl", "link", "listing_url"])
    title = first_present(item, ["title", "name"]) or "Untitled listing"
    city = first_present(item, ["city", "locality", "location"]) or "Unknown"

    total_raw = first_present(
        item,
        ["totalPrice", "total_price", "price", "finalPrice", "priceTotal"],
    )
    nightly_raw = first_present(
        item,
        ["nightlyPrice", "nightly_price", "pricePerNight", "price_per_night"],
    )
    total_price = extract_price(total_raw)
    nightly_price = extract_price(nightly_raw)

    if nightly_price is None and total_price is not None and nights > 0:
        nightly_price = total_price / nights
    if total_price is None and nightly_price is not None and nights > 0:
        total_price = nightly_price * nights

    return {
        "title": title,
        "city": city,
        "url": listing_url,
        "totalPrice": total_price,
        "nightlyPrice": nightly_price,
        "beachHintMatch": is_beach_friendly(item),
        "busTimeRegionMatch": is_bus_ok(item),
        "raw": item,
    }


def make_output_paths(data_dir: str) -> dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "raw_jsonl": os.path.join(data_dir, f"raw_listings_{timestamp}.jsonl"),
        "normalized_jsonl": os.path.join(data_dir, f"normalized_listings_{timestamp}.jsonl"),
        "filtered_json": os.path.join(data_dir, f"filtered_listings_{timestamp}.json"),
        "shortlist_csv": os.path.join(data_dir, f"shortlist_{timestamp}.csv"),
        "run_summary_json": os.path.join(data_dir, f"run_summary_{timestamp}.json"),
    }


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_shortlist_csv(path: str, rows: list[dict[str, Any]]) -> None:
    headers = [
        "title",
        "city",
        "nightlyPrice",
        "totalPrice",
        "beachHintMatch",
        "busTimeRegionMatch",
        "url",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h) for h in headers})


def main() -> None:
    load_env_file()
    args = parse_args()

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise RuntimeError(
            "APIFY_API_TOKEN is missing. Add it to environment or .env file."
        )

    nights = stay_nights(args.check_in, args.check_out)
    if nights not in {12, 14, 16}:
        print(
            f"Warning: your date window is {nights} nights (expected 14 +/- 2 nights)."
        )

    client = ApifyClient(token)

    run_input = {
        "locationQueries": args.locations,
        "startUrls": [],
        "enrichUserProfiles": False,
        "checkIn": args.check_in,
        "checkOut": args.check_out,
        "locale": "en-US",
        "currency": args.currency,
        "priceMax": args.max_total_price,
        "adults": args.adults,
        "children": 0,
        "infants": 0,
        "pets": 0,
    }

    run = client.actor(ACTOR_ID).call(run_input=run_input)

    raw_items: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        raw_items.append(item)
        normalized.append(normalize_item(item, nights))

    filtered = [
        x
        for x in normalized
        if x["nightlyPrice"] is not None
        and x["nightlyPrice"] <= args.max_nightly
        and x["beachHintMatch"]
        and x["busTimeRegionMatch"]
    ]
    filtered.sort(key=lambda x: (x["nightlyPrice"], x["totalPrice"] or 0))

    output_paths = make_output_paths(args.data_dir)
    write_jsonl(output_paths["raw_jsonl"], raw_items)
    write_jsonl(output_paths["normalized_jsonl"], normalized)
    with open(output_paths["filtered_json"], "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    write_shortlist_csv(output_paths["shortlist_csv"], filtered)

    summary = {
        "actorId": ACTOR_ID,
        "apifyRunId": run.get("id"),
        "query": {
            "locations": args.locations,
            "checkIn": args.check_in,
            "checkOut": args.check_out,
            "nights": nights,
            "adults": args.adults,
            "currency": args.currency,
            "maxTotalPrice": args.max_total_price,
            "maxNightly": args.max_nightly,
        },
        "counts": {
            "raw": len(raw_items),
            "normalized": len(normalized),
            "filtered": len(filtered),
        },
        "files": output_paths,
    }
    with open(output_paths["run_summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Total scraped: {len(normalized)}")
    print(f"Matched your filters: {len(filtered)}")
    print("Saved output files:")
    for path in output_paths.values():
        print(f"- {path}")

    for idx, listing in enumerate(filtered[:15], start=1):
        nightly = listing["nightlyPrice"]
        total = listing["totalPrice"]
        print(
            f"{idx}. {listing['title']} | {listing['city']} | "
            f"nightly≈{nightly:.2f} {args.currency} | total≈{total:.2f} {args.currency} | "
            f"{listing['url']}"
        )


if __name__ == "__main__":
    main()