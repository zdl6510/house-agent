"""Deterministic LangGraph used by the homepage to load real housing data."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.agent.common.database import DatabaseUnavailable, mysql_connection, quote_identifier


class CatalogState(TypedDict, total=False):
    keyword: str
    location: str
    budget_min: float | int | str | None
    budget_max: float | int | str | None
    layout: str
    orientation: str
    sort: str
    limit: int
    listings: list[dict[str, Any]]
    total: int
    stale: bool
    error: str | None


ALIASES = {
    "id": ("id", "house_id", "listing_id", "room_id"),
    "title": ("title", "house_title", "name", "house_name"),
    "price": ("price", "rent", "monthly_rent", "rent_price"),
    "city": ("city_name", "city", "cityname"),
    "region": ("region_name", "district", "region", "area_name", "area"),
    "address": ("address", "location", "community", "community_name"),
    "layout": ("room_type", "layout", "house_type", "type"),
    "orientation": ("orientation", "direction", "toward"),
    "intro": ("intro", "description", "house_desc", "detail"),
    "tags": ("tags", "features", "labels"),
    "image": ("image_url", "image", "cover", "cover_url", "photo_url"),
    "status": ("status", "house_status", "available"),
}

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "listings": []}
CACHE_TTL_SECONDS = 60


def _find_column(columns: list[str], field: str) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    return next((lower_map[name] for name in ALIASES[field] if name in lower_map), None)


def _select_table(connection) -> tuple[str, dict[str, str | None]]:
    with connection.cursor() as cursor:
        cursor.execute("SHOW TABLES")
        tables = [next(iter(row.values())) for row in cursor.fetchall()]

        best: tuple[int, str, dict[str, str | None]] | None = None
        for table in tables:
            cursor.execute(f"SHOW COLUMNS FROM {quote_identifier(table)}")
            columns = [row["Field"] for row in cursor.fetchall()]
            mapping = {field: _find_column(columns, field) for field in ALIASES}
            score = sum(mapping[field] is not None for field in ("title", "price", "city", "region", "layout"))
            if mapping["title"] and mapping["price"] and (best is None or score > best[0]):
                best = (score, table, mapping)

    if best is None:
        raise DatabaseUnavailable("数据库中未找到可展示的房源表")
    return best[1], best[2]


def _serialize_tags(value: Any, orientation: Any, layout: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(value, list):
        tags.extend(str(item).strip() for item in value)
    elif value:
        text = str(value).strip()
        try:
            decoded = json.loads(text)
            tags.extend(str(item).strip() for item in decoded) if isinstance(decoded, list) else tags.append(text)
        except (TypeError, ValueError):
            tags.extend(part.strip() for part in re_split_tags(text))
    tags.extend(str(item).strip() for item in (layout, orientation) if item)
    return list(dict.fromkeys(tag for tag in tags if tag))[:4]


def re_split_tags(value: str) -> list[str]:
    for separator in ("，", "、", "|"):
        value = value.replace(separator, ",")
    return value.split(",")


def _query_catalog(state: CatalogState) -> list[dict[str, Any]]:
    limit = max(1, min(int(state.get("limit") or 9), 24))
    keyword = str(state.get("keyword") or "").strip()
    location = str(state.get("location") or "").strip()
    layout_filter = str(state.get("layout") or "").strip()
    orientation_filter = str(state.get("orientation") or "").strip()
    sort = str(state.get("sort") or "price_asc")
    try:
        budget_min = float(state["budget_min"]) if state.get("budget_min") not in (None, "") else None
    except (TypeError, ValueError):
        budget_min = None
    try:
        budget_max = float(state["budget_max"]) if state.get("budget_max") not in (None, "") else None
    except (TypeError, ValueError):
        budget_max = None

    with mysql_connection() as connection:
        table, mapping = _select_table(connection)
        selected = {field: column for field, column in mapping.items() if column}
        projections = ", ".join(
            f"{quote_identifier(column)} AS {quote_identifier(field)}" for field, column in selected.items()
        )
        where: list[str] = []
        params: list[Any] = []
        if keyword:
            searchable = [
                mapping[key]
                for key in ("title", "city", "region", "address", "layout", "orientation", "intro", "tags")
                if mapping[key]
            ]
            for token in keyword.split():
                where.append("(" + " OR ".join(f"{quote_identifier(col)} LIKE %s" for col in searchable) + ")")
                params.extend([f"%{token}%"] * len(searchable))
        if location:
            location_columns = [mapping[key] for key in ("city", "region", "address") if mapping[key]]
            if location_columns:
                for token in location.split():
                    where.append("(" + " OR ".join(f"{quote_identifier(col)} LIKE %s" for col in location_columns) + ")")
                    params.extend([f"%{token}%"] * len(location_columns))
        if layout_filter and mapping["layout"]:
            where.append(f"{quote_identifier(mapping['layout'])} LIKE %s")
            params.append(f"%{layout_filter}%")
        if orientation_filter and mapping["orientation"]:
            where.append(f"{quote_identifier(mapping['orientation'])} LIKE %s")
            params.append(f"%{orientation_filter}%")
        if budget_min is not None:
            where.append(f"CAST({quote_identifier(mapping['price'])} AS DECIMAL(12,2)) >= %s")
            params.append(budget_min)
        if budget_max is not None:
            where.append(f"CAST({quote_identifier(mapping['price'])} AS DECIMAL(12,2)) <= %s")
            params.append(budget_max)
        if mapping["status"]:
            where.append(f"({quote_identifier(mapping['status'])} IS NULL OR LOWER(CAST({quote_identifier(mapping['status'])} AS CHAR)) NOT IN ('0','offline','deleted','已下架'))")

        sql = f"SELECT {projections} FROM {quote_identifier(table)}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        direction = "DESC" if sort == "price_desc" else "ASC"
        sql += f" ORDER BY CAST({quote_identifier(mapping['price'])} AS DECIMAL(12,2)) {direction} LIMIT %s"
        params.append(limit)
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = list(cursor.fetchall())

    listings = []
    for index, row in enumerate(rows):
        price = row.get("price")
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        listings.append({
            "id": str(row.get("id") or f"listing-{index + 1}"),
            "title": str(row.get("title") or "优选房源"),
            "price": price,
            "city": str(row.get("city") or ""),
            "region": str(row.get("region") or ""),
            "address": str(row.get("address") or ""),
            "layout": str(row.get("layout") or ""),
            "intro": str(row.get("intro") or ""),
            "image": str(row.get("image") or ""),
            "tags": _serialize_tags(row.get("tags"), row.get("orientation"), row.get("layout")),
        })
    return listings


def lookup_listing_detail(listing_id: str = "", title: str = "") -> dict[str, Any] | None:
    """Look up one listing by its database identifier, with an exact-title fallback."""

    listing_id = str(listing_id or "").strip()
    title = str(title or "").strip()
    if not listing_id and not title:
        return None

    with mysql_connection() as connection:
        table, mapping = _select_table(connection)
        selected = {field: column for field, column in mapping.items() if column}
        projections = ", ".join(
            f"{quote_identifier(column)} AS {quote_identifier(field)}" for field, column in selected.items()
        )
        where: list[str] = []
        params: list[Any] = []
        if listing_id and mapping["id"]:
            where.append(f"CAST({quote_identifier(mapping['id'])} AS CHAR) = %s")
            params.append(listing_id)
        elif title and mapping["title"]:
            where.append(f"{quote_identifier(mapping['title'])} = %s")
            params.append(title)
        if not where:
            return None

        sql = f"SELECT {projections} FROM {quote_identifier(table)} WHERE " + " OR ".join(where) + " LIMIT 1"
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()

    if not row:
        return None
    price = row.get("price")
    try:
        price = float(price) if price is not None else None
    except (TypeError, ValueError):
        price = None
    return {
        "id": str(row.get("id") or listing_id),
        "title": str(row.get("title") or title or "优选房源"),
        "price": price,
        "city": str(row.get("city") or ""),
        "region": str(row.get("region") or ""),
        "address": str(row.get("address") or ""),
        "layout": str(row.get("layout") or ""),
        "orientation": str(row.get("orientation") or ""),
        "intro": str(row.get("intro") or ""),
        "status": str(row.get("status") or ""),
        "image": str(row.get("image") or ""),
        "tags": _serialize_tags(row.get("tags"), row.get("orientation"), row.get("layout")),
    }


def load_catalog(state: CatalogState) -> CatalogState:
    """Load real listings, serving the latest successful snapshot on outages."""

    unfiltered = not any(
        state.get(key) not in (None, "", "price_asc")
        for key in ("keyword", "location", "budget_min", "budget_max", "layout", "orientation", "sort")
    )
    with _cache_lock:
        if unfiltered and _cache["listings"] and time.monotonic() - _cache["at"] < CACHE_TTL_SECONDS:
            listings = list(_cache["listings"])
            return {"listings": listings, "total": len(listings), "stale": False, "error": None}

    try:
        listings = _query_catalog(state)
        if unfiltered and listings:
            with _cache_lock:
                _cache.update(at=time.monotonic(), listings=list(listings))
        return {"listings": listings, "total": len(listings), "stale": False, "error": None}
    except Exception as exc:  # graph nodes must degrade gracefully during DB outages
        with _cache_lock:
            cached = list(_cache["listings"])
        return {
            "listings": cached,
            "total": len(cached),
            "stale": bool(cached),
            "error": str(exc) if isinstance(exc, DatabaseUnavailable) else "房源数据暂时不可用，请稍后重试",
        }


catalog_builder = StateGraph(CatalogState)
catalog_builder.add_node("load_catalog", load_catalog)
catalog_builder.add_edge(START, "load_catalog")
catalog_builder.add_edge("load_catalog", END)
catalog_graph = catalog_builder.compile()
