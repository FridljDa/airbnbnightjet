from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_DIR = Path("data")
ALL_LISTINGS_PATH = DATA_DIR / "airbnb_istria_all.json"
FILTERED_LISTINGS_PATH = DATA_DIR / "airbnb_istria_filtered.json"
RUN_META_PATH = DATA_DIR / "airbnb_istria_run_metadata.json"
NIGHTJET_DEFAULT_PATH = DATA_DIR / "nightjet_prices.csv"


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if not RUN_META_PATH.exists():
        return {}
    return json.loads(RUN_META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_nightjet_prices() -> pd.DataFrame:
    candidates = [NIGHTJET_DEFAULT_PATH] + sorted(
        [
            path
            for path in DATA_DIR.glob("*.csv")
            if "nightjet" in path.name.lower() and path != NIGHTJET_DEFAULT_PATH
        ]
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        return pd.DataFrame()

    df = pd.read_csv(source_path)
    expected = {
        "date",
        "start_station",
        "arrival_station",
        "couchette_price",
        "sleeper_price",
    }
    if not expected.issubset(df.columns):
        return pd.DataFrame()

    for col in ["couchette_price", "sleeper_price"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(r"[^0-9.]", "", regex=True)
            .replace("", pd.NA)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep parsed datetime for sorting/plotting while showing original date text in the table.
    df["date_parsed"] = pd.to_datetime(df["date"], format="%a, %d. %b %Y", errors="coerce")
    return df


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in [
        "average_rating",
        "total_price_eur",
        "nightly_price_eur",
        "latitude",
        "longitude",
        "beach_distance_km_estimate",
        "beach_walk_minutes_estimate",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "inferred_city" in df.columns:
        df["inferred_city"] = df["inferred_city"].fillna("unknown").astype(str)

    if "city_reachable_under_7h_bus" in df.columns:
        df["city_reachable_under_7h_bus"] = df["city_reachable_under_7h_bus"].fillna(False)

    return df


def filtered_view(df: pd.DataFrame, reachable_cities: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    st.sidebar.header("Filters")
    only_reachable = st.sidebar.checkbox("Only reachable cities (<7h bus)", value=True)

    nightly_series = df["nightly_price_eur"].dropna()
    nightly_min = float(nightly_series.min()) if not nightly_series.empty else 0.0
    nightly_max = float(nightly_series.max()) if not nightly_series.empty else 200.0
    max_price = st.sidebar.slider(
        "Max price / night (€)",
        min_value=float(int(nightly_min)),
        max_value=float(int(max(nightly_max, nightly_min + 1))),
        value=float(int(min(120, max(nightly_max, nightly_min + 1)))),
        step=1.0,
    )

    walk_series = df["beach_walk_minutes_estimate"].dropna()
    walk_max_data = float(walk_series.max()) if not walk_series.empty else 120.0
    max_walk = st.sidebar.slider(
        "Max beach walk (minutes)",
        min_value=0.0,
        max_value=float(int(max(15, walk_max_data))),
        value=float(int(min(15, max(15, walk_max_data)))),
        step=1.0,
    )
    include_missing_beach = st.sidebar.checkbox(
        "Include listings without beach estimate",
        value=False,
    )

    rating_series = df["average_rating"].dropna()
    rating_min_data = float(rating_series.min()) if not rating_series.empty else 0.0
    min_rating = st.sidebar.slider(
        "Min rating",
        min_value=0.0,
        max_value=5.0,
        value=float(max(4.5, rating_min_data)),
        step=0.05,
    )

    all_cities = sorted(set(df["inferred_city"].dropna().astype(str).tolist()))
    default_cities = reachable_cities or all_cities
    selected_cities = st.sidebar.multiselect(
        "Reachable cities",
        options=all_cities,
        default=[city for city in default_cities if city in all_cities],
    )
    search_text = st.sidebar.text_input("Search title/subtitle", "").strip().lower()

    mask = pd.Series(True, index=df.index)
    mask &= df["nightly_price_eur"].le(max_price) | df["nightly_price_eur"].isna()
    mask &= df["average_rating"].ge(min_rating) | df["average_rating"].isna()

    if include_missing_beach:
        mask &= df["beach_walk_minutes_estimate"].le(max_walk) | df["beach_walk_minutes_estimate"].isna()
    else:
        mask &= df["beach_walk_minutes_estimate"].le(max_walk)

    if only_reachable and "city_reachable_under_7h_bus" in df.columns:
        mask &= df["city_reachable_under_7h_bus"].fillna(False)

    if selected_cities:
        mask &= df["inferred_city"].isin(selected_cities)

    if search_text:
        titles = df["title"].fillna("").str.lower()
        subtitles = df["subtitle"].fillna("").str.lower()
        mask &= titles.str.contains(search_text) | subtitles.str.contains(search_text)

    return df.loc[mask].copy()


def render_map(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No listings to show on map with current filters.")
        return

    map_df = df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        st.info("Listings are missing coordinates.")
        return

    map_df["dot_color"] = map_df["satisfies_filters"].map(
        lambda x: [16, 185, 129, 200] if bool(x) else [59, 130, 246, 180]
    )
    map_df["dot_radius"] = map_df["nightly_price_eur"].fillna(80).clip(lower=40, upper=180) * 15

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_fill_color="dot_color",
        get_radius="dot_radius",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df["latitude"].mean()),
        longitude=float(map_df["longitude"].mean()),
        zoom=8.2,
        pitch=0,
    )

    tooltip = {
        "html": (
            "<b>{title}</b><br/>"
            "City: {inferred_city}<br/>"
            "Nightly: €{nightly_price_eur}<br/>"
            "Rating: {average_rating}<br/>"
            "Walk to beach: {beach_walk_minutes_estimate} min"
        )
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


def render_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    ordered = df.sort_values(["nightly_price_eur", "average_rating"], ascending=[True, False])
    table = ordered[
        [
            "title",
            "inferred_city",
            "nightly_price_eur",
            "average_rating",
            "beach_walk_minutes_estimate",
            "city_reachable_under_7h_bus",
            "room_url",
        ]
    ].rename(
        columns={
            "title": "Title",
            "inferred_city": "City",
            "nightly_price_eur": "Nightly (€)",
            "average_rating": "Rating",
            "beach_walk_minutes_estimate": "Beach walk (min)",
            "city_reachable_under_7h_bus": "Reachable <7h bus",
            "room_url": "Listing URL",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current results as CSV",
        data=csv_bytes,
        file_name="airbnb_filtered_view.csv",
        mime="text/csv",
    )


def render_metrics(df: pd.DataFrame, total_count: int) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Visible listings", f"{len(df)}")
    col2.metric("Total in dataset", f"{total_count}")
    col3.metric("Median nightly", f"€{df['nightly_price_eur'].median():.0f}" if not df.empty else "N/A")
    col4.metric("Median rating", f"{df['average_rating'].median():.2f}" if not df.empty else "N/A")


def render_nightjet_view() -> None:
    st.caption("Couchette and sleeper prices extracted from saved Nightjet HTML.")
    df = load_nightjet_prices()
    if df.empty:
        st.error(
            "No Nightjet CSV found with expected columns in `data/`. "
            "Run `extract_nightjet_prices.py` first."
        )
        return

    st.sidebar.header("Nightjet Filters")
    stations_from = sorted(df["start_station"].dropna().astype(str).unique().tolist())
    stations_to = sorted(df["arrival_station"].dropna().astype(str).unique().tolist())

    selected_from = st.sidebar.multiselect(
        "Start station",
        options=stations_from,
        default=stations_from,
    )
    selected_to = st.sidebar.multiselect(
        "Arrival station",
        options=stations_to,
        default=stations_to,
    )
    only_with_any_price = st.sidebar.checkbox("Only rows with at least one price", value=True)

    mask = pd.Series(True, index=df.index)
    if selected_from:
        mask &= df["start_station"].isin(selected_from)
    if selected_to:
        mask &= df["arrival_station"].isin(selected_to)
    if only_with_any_price:
        mask &= df[["couchette_price", "sleeper_price"]].notna().any(axis=1)

    view = df.loc[mask].copy()
    view = view.sort_values(["date_parsed", "date"], ascending=[True, True])

    c1, c2, c3 = st.columns(3)
    c1.metric("Visible rows", f"{len(view)}")
    c2.metric(
        "Couchette median",
        f"€{view['couchette_price'].median():.2f}" if not view["couchette_price"].dropna().empty else "N/A",
    )
    c3.metric("Sleeper median", f"€{view['sleeper_price'].median():.2f}" if not view["sleeper_price"].dropna().empty else "N/A")

    st.subheader("Price trends")
    trend = view.dropna(subset=["date_parsed"]).set_index("date_parsed")[
        ["couchette_price", "sleeper_price"]
    ]
    if trend.empty:
        st.info("No parsable dates available for trend chart.")
    else:
        st.line_chart(trend, height=280)

    st.subheader("Nightjet prices table")
    table = view[
        ["date", "start_station", "arrival_station", "couchette_price", "sleeper_price"]
    ].rename(
        columns={
            "date": "Date",
            "start_station": "Start",
            "arrival_station": "Arrival",
            "couchette_price": "Couchette (€)",
            "sleeper_price": "Sleeper (€)",
        }
    )
    price_columns = ["Couchette (€)", "Sleeper (€)"]

    def color_by_relative_price(column: pd.Series) -> list[str]:
        values = pd.to_numeric(column, errors="coerce")
        non_null = values.dropna()
        if non_null.empty:
            return [""] * len(column)

        min_val = float(non_null.min())
        max_val = float(non_null.max())

        styles: list[str] = []
        for raw in values:
            if pd.isna(raw):
                styles.append("")
                continue
            if max_val == min_val:
                ratio = 0.5
            else:
                ratio = (float(raw) - min_val) / (max_val - min_val)

            # Low prices are greener, high prices are redder.
            red = int(226 * ratio + 46 * (1 - ratio))
            green = int(80 * ratio + 204 * (1 - ratio))
            blue = int(75 * ratio + 113 * (1 - ratio))
            styles.append(f"background-color: rgb({red}, {green}, {blue}); color: #111111;")
        return styles

    styled_table = table.style.format({col: "€{:.2f}" for col in price_columns}).apply(
        color_by_relative_price,
        axis=0,
        subset=price_columns,
    )
    st.dataframe(styled_table, width="stretch", hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Nightjet view as CSV",
        data=csv_bytes,
        file_name="nightjet_prices_view.csv",
        mime="text/csv",
    )


def render_airbnb_view() -> None:
    all_rows = load_json(ALL_LISTINGS_PATH)
    filtered_rows = load_json(FILTERED_LISTINGS_PATH)
    metadata = load_metadata()

    if not all_rows and not filtered_rows:
        st.error(
            "No listing data found. Run the scraper first so `data/airbnb_istria_all.json` exists."
        )
        return

    source = st.radio(
        "Dataset",
        ["All scraped listings", "Only pre-filtered matches"],
        horizontal=True,
    )
    base_rows = all_rows if source == "All scraped listings" else filtered_rows
    base_df = build_dataframe(base_rows)

    if base_df.empty:
        st.warning("Selected dataset is empty.")
        return

    criteria = metadata.get("criteria", {})
    reachable_cities = criteria.get("reachable_cities", [])

    if metadata:
        with st.expander("Run metadata", expanded=False):
            st.json(metadata)

    df = filtered_view(base_df, reachable_cities=reachable_cities)
    render_metrics(df, total_count=len(base_df))

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Map view")
        render_map(df)
    with right:
        st.subheader("Quick picks")
        if df.empty:
            st.info("No listings match your filters.")
        else:
            picks = df.sort_values(
                ["nightly_price_eur", "beach_walk_minutes_estimate", "average_rating"],
                ascending=[True, True, False],
            ).head(5)
            for _, row in picks.iterrows():
                st.markdown(
                    f"- **{row['title']}** ({row['inferred_city']})  \n"
                    f"  €{row['nightly_price_eur']:.0f}/night · ⭐ {row['average_rating']:.2f} · "
                    f"{row['beach_walk_minutes_estimate'] if pd.notna(row['beach_walk_minutes_estimate']) else 'N/A'} min walk  \n"
                    f"  [Open listing]({row['room_url']})"
                )

    st.subheader("Listings table")
    render_table(df)


def main() -> None:
    st.set_page_config(page_title="Airbnb Finder Friend", page_icon="🏡", layout="wide")
    st.title("🏡 Airbnb Finder Friend")
    view = st.radio("View", ["Airbnb listings", "Nightjet prices"], horizontal=True)
    if view == "Airbnb listings":
        st.caption("Interactive view for your scraped Istria Airbnb listings.")
        render_airbnb_view()
    else:
        st.caption("Interactive view for extracted Nightjet prices.")
        render_nightjet_view()


if __name__ == "__main__":
    main()
