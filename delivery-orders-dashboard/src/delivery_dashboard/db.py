"""Banco SQLite: esquema, índices, view analítica e construção a partir dos dados fictícios."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from .generate_data import make_dataset

SCHEMA = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      REAL NOT NULL
);
CREATE TABLE orders (
    order_id         INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL,
    created_at       TEXT NOT NULL,   -- 'AAAA-MM-DD HH:MM:SS'
    delivered_at     TEXT,            -- nulo quando o pedido foi cancelado
    promised_minutes INTEGER NOT NULL,
    status           TEXT NOT NULL,   -- 'entregue' | 'cancelado'
    neighborhood     TEXT NOT NULL,
    payment_method   TEXT NOT NULL,
    subtotal         REAL NOT NULL,
    delivery_fee     REAL NOT NULL,
    total            REAL NOT NULL
);
CREATE TABLE order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
"""

INDEXES_AND_VIEWS = """
CREATE INDEX idx_orders_created ON orders(created_at);
CREATE INDEX idx_orders_neighborhood ON orders(neighborhood);
CREATE INDEX idx_items_product ON order_items(product_id);

-- View analítica: acrescenta dia, hora, dia da semana (0 = domingo), tempo de entrega e atraso.
CREATE VIEW v_orders AS
SELECT
    o.*,
    date(o.created_at)                                         AS day,
    CAST(strftime('%H', o.created_at) AS INTEGER)              AS hour,
    CAST(strftime('%w', o.created_at) AS INTEGER)              AS weekday,
    ROUND((julianday(o.delivered_at) - julianday(o.created_at)) * 1440.0, 1) AS delivery_minutes,
    CASE
        WHEN o.status = 'entregue'
         AND (julianday(o.delivered_at) - julianday(o.created_at)) * 1440.0 > o.promised_minutes
        THEN 1 ELSE 0
    END                                                        AS is_late
FROM orders o;
"""


def build_database(path: str | Path, **dataset_kwargs) -> Path:
    """Cria o banco do zero (de forma atômica: grava em arquivo temporário e renomeia)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    tables = make_dataset(**dataset_kwargs)
    with closing(sqlite3.connect(tmp)) as conn:
        conn.executescript(SCHEMA)
        for name in ("products", "orders", "order_items"):
            tables[name].to_sql(name, conn, if_exists="append", index=False)
        conn.executescript(INDEXES_AND_VIEWS)
        conn.commit()
    os.replace(tmp, path)
    return path


def ensure_database(path: str | Path, **dataset_kwargs) -> Path:
    """Garante que o banco existe (gera na primeira execução)."""
    path = Path(path)
    if not path.exists():
        build_database(path, **dataset_kwargs)
    return path


if __name__ == "__main__":
    target = build_database("data/delivery.db")
    print(f"Banco criado em {target}")
