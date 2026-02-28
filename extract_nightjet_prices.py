#!/usr/bin/env python3
"""Extract Nightjet connection prices from a saved HTML page into CSV."""

from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

CSV_COLUMNS = [
    "date",
    "start_station",
    "arrival_station",
    "couchette_price",
    "sleeper_price",
]


def normalize_price(raw_price: str) -> str:
    """Extract a compact price string such as '69,80' from text like 'ab € 69,80'."""
    if not raw_price:
        return ""
    match = re.search(r"(\d{1,4},\d{2})", raw_price)
    return match.group(1) if match else raw_price.strip()


def map_type_to_column(type_text: str) -> str | None:
    """Map Nightjet seat type labels to CSV column names."""
    lowered = type_text.lower()
    if "liege" in lowered or "mini cabin" in lowered or "couchette" in lowered:
        return "couchette_price"
    if "schlaf" in lowered or "sleeper" in lowered:
        return "sleeper_price"
    return None


def extract_rows(html_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    connection_pattern = re.compile(r"<li><nj-connection\b.*?</nj-connection></li>", re.DOTALL)
    date_pattern = re.compile(
        r'connection-overview__data--dep">.*?connection-overview__data-date">(.*?)</span>',
        re.DOTALL,
    )
    dep_station_pattern = re.compile(
        r'connection-overview__data--dep">.*?connection-overview__data-station">(.*?)</span>',
        re.DOTALL,
    )
    arr_station_pattern = re.compile(
        r'connection-overview__data--arr">.*?connection-overview__data-station">(.*?)</span>',
        re.DOTALL,
    )
    best_price_pattern = re.compile(
        r'screen-reader-text">(.*?)</span><span class="best-price__price">([^<]+)</span>',
        re.DOTALL,
    )

    for connection_html in connection_pattern.findall(html_text):
        dep_date_match = date_pattern.search(connection_html)
        dep_station_match = dep_station_pattern.search(connection_html)
        arr_station_match = arr_station_pattern.search(connection_html)

        row = {
            "date": html.unescape(dep_date_match.group(1)).strip() if dep_date_match else "",
            "start_station": html.unescape(dep_station_match.group(1)).strip() if dep_station_match else "",
            "arrival_station": html.unescape(arr_station_match.group(1)).strip() if arr_station_match else "",
            "couchette_price": "",
            "sleeper_price": "",
        }

        for seat_type_raw, price_raw in best_price_pattern.findall(connection_html):
            seat_type = html.unescape(seat_type_raw).strip()
            price = html.unescape(price_raw).strip()
            column = map_type_to_column(seat_type)
            if column is None:
                continue

            row[column] = normalize_price(price)

        if row["date"] or row["start_station"] or row["arrival_station"]:
            rows.append(row)

    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract date, stations, and seat-type prices from a saved Nightjet HTML page."
    )
    parser.add_argument("html_file", type=Path, help="Path to the saved Nightjet HTML file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same path as input with .csv extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path: Path = args.html_file
    default_output = Path(__file__).resolve().parent / "data" / f"{html_path.stem}.csv"
    output_path: Path = args.output or default_output

    html_text = html_path.read_text(encoding="utf-8", errors="ignore")
    rows = extract_rows(html_text)
    write_csv(rows, output_path)
    print(f"Extracted {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
