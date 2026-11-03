import sqlite3
import tempfile
from contextlib import closing
from datetime import timedelta
from pathlib import Path

from delivery_dashboard.db import build_database
from delivery_dashboard.queries import (
    Filters,
    date_bounds,
    heatmap_table,
    list_neighborhoods,
    render_sql,
    run_query,
)

_CACHE: dict = {}


def db_path() -> Path:
    """Banco de teste criado uma vez por sessão (45 dias de dados fictícios)."""
    if "path" not in _CACHE:
        _CACHE["dir"] = tempfile.TemporaryDirectory()
        _CACHE["path"] = build_database(Path(_CACHE["dir"].name) / "t.db", days=45, seed=11)
    return _CACHE["path"]


def full_filters(path, hoods=None) -> Filters:
    first, last = date_bounds(path)
    return Filters(first.isoformat(), last.isoformat(), hoods)


def test_totals_agree_across_queries():
    path = db_path()
    f = full_filters(path)
    kpis = run_query(path, "kpis", f).iloc[0]
    by_hour = run_query(path, "orders_by_hour", f)
    daily = run_query(path, "daily_orders", f)
    heat = run_query(path, "heatmap", f)
    assert kpis["orders"] == by_hour["orders"].sum() == daily["orders"].sum() == heat["orders"].sum()
    assert 0 <= kpis["late_pct"] <= 100 and 0 <= kpis["cancel_pct"] <= 100
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == kpis["orders"]
        revenue = conn.execute("SELECT SUM(total) FROM orders WHERE status = 'entregue'").fetchone()[0]
    assert abs(kpis["revenue"] - revenue) < 0.01


def test_filters_match_direct_counts_and_block_sql_injection():
    path = db_path()
    first, _ = date_bounds(path)
    start, end = (first + timedelta(days=7)).isoformat(), (first + timedelta(days=13)).isoformat()
    got = run_query(path, "kpis", Filters(start, end, ("Centro", "Vila Nova"))).iloc[0]["orders"]
    with closing(sqlite3.connect(path)) as conn:
        expected = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE date(created_at) BETWEEN ? AND ? "
            "AND neighborhood IN (?, ?)",
            (start, end, "Centro", "Vila Nova"),
        ).fetchone()[0]
    assert got == expected and expected > 0
    evil = Filters(start, end, ("x' OR '1'='1",))
    assert run_query(path, "kpis", evil).iloc[0]["orders"] == 0


def test_top_products_and_ranking_per_category():
    path = db_path()
    f = full_filters(path)
    top = run_query(path, "top_products", f, limit=5)
    assert len(top) == 5 and top["units"].is_monotonic_decreasing
    ranked = run_query(path, "top_per_category", f)
    assert ranked["position"].between(1, 3).all()
    for _, group in ranked.groupby("category"):
        assert group["units"].is_monotonic_decreasing
    assert run_query(path, "revenue_by_category", f)["revenue"].is_monotonic_decreasing


def test_delay_queries():
    path = db_path()
    f = full_filters(path)
    by_hood = run_query(path, "late_by_neighborhood", f)
    assert by_hood["late_pct"].between(0, 100).all() and by_hood["late_pct"].is_monotonic_decreasing
    worst = run_query(path, "worst_delays", f, limit=10)
    assert len(worst) == 10 and (worst["delay_minutes"] > 0).all()
    assert worst["delay_minutes"].is_monotonic_decreasing


def test_heatmap_table_and_helpers():
    path = db_path()
    f = full_filters(path)
    table = heatmap_table(run_query(path, "heatmap", f))
    assert list(table.index) == ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    assert table.to_numpy().sum() == run_query(path, "kpis", f).iloc[0]["orders"]

    first, last = date_bounds(path)
    assert first < last
    hoods = list_neighborhoods(path)
    assert len(hoods) == 8 and hoods == sorted(hoods)

    sql = render_sql("top_products", Filters("2026-01-01", "2026-01-31", ("Centro",)), limit=7)
    assert "o.neighborhood IN (?)" in sql and "LIMIT 7" in sql and "BETWEEN ? AND ?" in sql
