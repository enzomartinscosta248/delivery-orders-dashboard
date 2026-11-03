"""Consultas SQL do dashboard. Todas usam a view v_orders (ver db.py) e filtros parametrizados."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

WEEKDAY_ORDER = [1, 2, 3, 4, 5, 6, 0]  # segunda..domingo (no SQLite, %w: 0 = domingo)
WEEKDAY_LABELS = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sáb", 0: "Dom"}


@dataclass(frozen=True)
class Filters:
    start: str  # 'AAAA-MM-DD'
    end: str
    neighborhoods: tuple[str, ...] | None = None

    def where(self) -> tuple[str, list]:
        """Cláusula WHERE com placeholders (?): os valores nunca são colados no SQL."""
        clauses, params = ["o.day BETWEEN ? AND ?"], [self.start, self.end]
        if self.neighborhoods:
            marks = ", ".join("?" for _ in self.neighborhoods)
            clauses.append(f"o.neighborhood IN ({marks})")
            params += list(self.neighborhoods)
        return " AND ".join(clauses), params


QUERIES: dict[str, str] = {
    # Indicadores gerais. AVG ignora NULL, então CASE sem ELSE filtra só os pedidos entregues.
    "kpis": """
SELECT
    COUNT(*)                                                         AS orders,
    SUM(CASE WHEN o.status = 'entregue' THEN o.total ELSE 0 END)     AS revenue,
    AVG(CASE WHEN o.status = 'entregue' THEN o.total END)            AS avg_ticket,
    AVG(CASE WHEN o.status = 'entregue' THEN o.delivery_minutes END) AS avg_delivery_min,
    100.0 * AVG(CASE WHEN o.status = 'entregue' THEN o.is_late END)  AS late_pct,
    100.0 * AVG(CASE WHEN o.status = 'cancelado' THEN 1.0 ELSE 0.0 END) AS cancel_pct
FROM v_orders o
WHERE {where}
""",
    # Volume e taxa de atraso por hora do dia.
    "orders_by_hour": """
SELECT
    o.hour,
    COUNT(*) AS orders,
    100.0 * AVG(CASE WHEN o.status = 'entregue' THEN o.is_late END) AS late_pct
FROM v_orders o
WHERE {where}
GROUP BY o.hour
ORDER BY o.hour
""",
    # Dia da semana x hora (vira o mapa de calor).
    "heatmap": """
SELECT o.weekday, o.hour, COUNT(*) AS orders
FROM v_orders o
WHERE {where}
GROUP BY o.weekday, o.hour
""",
    "daily_orders": """
SELECT
    o.day,
    COUNT(*) AS orders,
    SUM(CASE WHEN o.status = 'entregue' THEN o.total ELSE 0 END) AS revenue
FROM v_orders o
WHERE {where}
GROUP BY o.day
ORDER BY o.day
""",
    "revenue_by_category": """
SELECT
    p.category,
    ROUND(SUM(i.quantity * i.unit_price), 2) AS revenue,
    SUM(i.quantity)                          AS units
FROM order_items i
JOIN products  p ON p.product_id = i.product_id
JOIN v_orders  o ON o.order_id   = i.order_id
WHERE o.status = 'entregue' AND {where}
GROUP BY p.category
ORDER BY revenue DESC
""",
    "top_products": """
SELECT
    p.name     AS product,
    p.category AS category,
    SUM(i.quantity)                          AS units,
    ROUND(SUM(i.quantity * i.unit_price), 2) AS revenue
FROM order_items i
JOIN products  p ON p.product_id = i.product_id
JOIN v_orders  o ON o.order_id   = i.order_id
WHERE o.status = 'entregue' AND {where}
GROUP BY p.product_id
ORDER BY units DESC
LIMIT {limit}
""",
    # CTE + função de janela: os 3 mais vendidos de cada categoria.
    "top_per_category": """
WITH sales AS (
    SELECT p.category, p.name AS product, SUM(i.quantity) AS units
    FROM order_items i
    JOIN products p ON p.product_id = i.product_id
    JOIN v_orders o ON o.order_id   = i.order_id
    WHERE o.status = 'entregue' AND {where}
    GROUP BY p.product_id
),
ranked AS (
    SELECT category, product, units,
           RANK() OVER (PARTITION BY category ORDER BY units DESC) AS position
    FROM sales
)
SELECT category, position, product, units
FROM ranked
WHERE position <= 3
ORDER BY category, position
""",
    "late_by_neighborhood": """
SELECT
    o.neighborhood,
    COUNT(*)                          AS delivered,
    ROUND(AVG(o.delivery_minutes), 1) AS avg_minutes,
    ROUND(100.0 * AVG(o.is_late), 1)  AS late_pct
FROM v_orders o
WHERE o.status = 'entregue' AND {where}
GROUP BY o.neighborhood
ORDER BY late_pct DESC
""",
    "worst_delays": """
SELECT
    o.order_id,
    o.created_at,
    o.neighborhood,
    o.promised_minutes,
    o.delivery_minutes,
    ROUND(o.delivery_minutes - o.promised_minutes, 1) AS delay_minutes
FROM v_orders o
WHERE o.status = 'entregue' AND o.is_late = 1 AND {where}
ORDER BY delay_minutes DESC
LIMIT {limit}
""",
}


def render_sql(name: str, filters: Filters, limit: int = 10) -> str:
    """SQL pronto para exibir (com placeholders `?` no lugar dos valores)."""
    return QUERIES[name].format(where=filters.where()[0], limit=int(limit)).strip()


def run_query(db_path: str | Path, name: str, filters: Filters, limit: int = 10) -> pd.DataFrame:
    where, params = filters.where()
    sql = QUERIES[name].format(where=where, limit=int(limit))
    with closing(sqlite3.connect(db_path)) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def date_bounds(db_path: str | Path) -> tuple[date, date]:
    with closing(sqlite3.connect(db_path)) as conn:
        first, last = conn.execute("SELECT MIN(day), MAX(day) FROM v_orders").fetchone()
    return date.fromisoformat(first), date.fromisoformat(last)


def list_neighborhoods(db_path: str | Path) -> list[str]:
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT DISTINCT neighborhood FROM orders ORDER BY 1").fetchall()
    return [row[0] for row in rows]


def heatmap_table(heat: pd.DataFrame) -> pd.DataFrame:
    """Pivota (dia da semana x hora), com segunda-feira primeiro e rótulos em português."""
    table = heat.pivot_table(
        index="weekday", columns="hour", values="orders", aggfunc="sum", fill_value=0
    )
    table = table.reindex([d for d in WEEKDAY_ORDER if d in table.index])
    table.index = [WEEKDAY_LABELS[d] for d in table.index]
    return table
