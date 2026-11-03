import sqlite3
from contextlib import closing

import pandas as pd

from delivery_dashboard.db import build_database, ensure_database
from delivery_dashboard.generate_data import make_dataset


def test_dataset_is_reproducible_and_internally_consistent():
    first, second = make_dataset(days=30, seed=3), make_dataset(days=30, seed=3)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])

    orders, items, products = first["orders"], first["order_items"], first["products"]
    assert orders["order_id"].is_unique
    assert set(items["order_id"]) <= set(orders["order_id"])
    assert set(items["product_id"]) <= set(products["product_id"])
    assert orders["created_at"].str[11:13].astype(int).between(10, 23).all()

    delivered = orders[orders["status"] == "entregue"]
    canceled = orders[orders["status"] == "cancelado"]
    assert len(canceled) > 0
    assert delivered["delivered_at"].notna().all() and canceled["delivered_at"].isna().all()
    assert (pd.to_datetime(delivered["delivered_at"]) > pd.to_datetime(delivered["created_at"])).all()

    per_order = items.assign(value=items["quantity"] * items["unit_price"]).groupby("order_id")["value"].sum()
    merged = orders.set_index("order_id").join(per_order)
    assert ((merged["subtotal"] - merged["value"]).abs() < 0.011).all()
    assert ((merged["total"] - (merged["subtotal"] + merged["delivery_fee"])).abs() < 0.011).all()


def test_database_schema_view_and_idempotent_ensure(tmp_path):
    path = tmp_path / "d.db"
    ensure_database(path, days=21, seed=1)
    modified = path.stat().st_mtime_ns
    ensure_database(path, days=21, seed=1)
    assert path.stat().st_mtime_ns == modified  # não recria o banco

    with closing(sqlite3.connect(path)) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        assert {"products", "orders", "order_items", "v_orders"} <= names
        columns = {r[1] for r in conn.execute("PRAGMA table_info(v_orders)")}
        assert {"day", "hour", "weekday", "delivery_minutes", "is_late"} <= columns
        wrongly_late = conn.execute(
            "SELECT COUNT(*) FROM v_orders WHERE is_late = 1 AND delivery_minutes < promised_minutes - 0.1"
        ).fetchone()[0]
        canceled_with_delivery = conn.execute(
            "SELECT COUNT(*) FROM v_orders WHERE status = 'cancelado' AND (is_late = 1 OR delivery_minutes IS NOT NULL)"
        ).fetchone()[0]
    assert wrongly_late == 0 and canceled_with_delivery == 0


def test_generated_patterns_peaks_weekends_and_overload(tmp_path):
    path = build_database(tmp_path / "d.db", days=60, seed=7)
    with closing(sqlite3.connect(path)) as conn:
        by_hour = pd.read_sql_query(
            "SELECT hour, COUNT(*) AS n, AVG(CASE WHEN status = 'entregue' THEN is_late END) AS late "
            "FROM v_orders GROUP BY hour",
            conn,
        )
        by_weekday = pd.read_sql_query(
            "SELECT weekday, COUNT(*) AS n FROM v_orders GROUP BY weekday", conn
        ).set_index("weekday")["n"]
    assert by_hour.nlargest(3, "n")["hour"].isin([12, 13, 19, 20, 21]).all()
    busiest = by_hour.loc[by_hour["n"].idxmax(), "late"]
    quiet = by_hour[by_hour["hour"].between(14, 17)]["late"].mean()
    assert busiest > quiet * 2
    assert by_weekday[6] > by_weekday[1]  # sábado > segunda
