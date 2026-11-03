"""Gerador de dados FICTÍCIOS de pedidos de delivery (mesma seed gera os mesmos dados).

Os padrões que aparecem nas análises (picos de almoço e jantar, mais pedidos nos fins de semana,
atrasos quando a cozinha está sobrecarregada) foram embutidos aqui de propósito. O objetivo do
projeto é demonstrar a análise, não "descobrir" fatos sobre o mundo real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# bairro -> (minutos médios de deslocamento, peso na demanda)
NEIGHBORHOODS = {
    "Centro": (10, 0.24),
    "Jardim das Flores": (14, 0.16),
    "Vila Nova": (17, 0.15),
    "Boa Vista": (20, 0.14),
    "Santa Luzia": (23, 0.12),
    "Parque Industrial": (27, 0.08),
    "Alto da Serra": (30, 0.06),
    "Beira Rio": (34, 0.05),
}
PAYMENTS = {"Pix": 0.45, "Cartão de crédito": 0.33, "Cartão de débito": 0.12, "Dinheiro": 0.10}

PRODUCTS = [  # (nome, categoria, preço)
    ("Pizza Margherita", "Pizzas", 42.90), ("Pizza Calabresa", "Pizzas", 44.90),
    ("Pizza Quatro Queijos", "Pizzas", 49.90), ("Pizza Frango com Catupiry", "Pizzas", 47.90),
    ("Hambúrguer Clássico", "Hambúrgueres", 29.90), ("Hambúrguer Bacon", "Hambúrgueres", 34.90),
    ("Duplo Cheddar", "Hambúrgueres", 39.90), ("Hambúrguer Veggie", "Hambúrgueres", 32.90),
    ("Combo 20 peças", "Japonesa", 79.90), ("Temaki de Salmão", "Japonesa", 32.90),
    ("Hot Roll", "Japonesa", 28.90), ("Yakisoba", "Japonesa", 36.90),
    ("Salada Caesar", "Saudável", 27.90), ("Bowl de Frango", "Saudável", 31.90),
    ("Wrap Integral", "Saudável", 26.90),
    ("Batata Frita", "Porções", 19.90), ("Onion Rings", "Porções", 21.90),
    ("Refrigerante lata", "Bebidas", 6.50), ("Suco natural", "Bebidas", 9.90),
    ("Água mineral", "Bebidas", 4.50), ("Cerveja long neck", "Bebidas", 11.90),
    ("Brownie", "Sobremesas", 14.90), ("Pudim", "Sobremesas", 12.90),
    ("Sorvete 2 bolas", "Sobremesas", 15.90),
]

# Prato principal por período do dia (almoço, jantar, outros horários)
MAIN_WEIGHTS = [
    {"Saudável": 0.30, "Japonesa": 0.22, "Hambúrgueres": 0.28, "Pizzas": 0.20},
    {"Pizzas": 0.36, "Hambúrgueres": 0.30, "Japonesa": 0.24, "Saudável": 0.10},
    {"Hambúrgueres": 0.35, "Pizzas": 0.35, "Japonesa": 0.20, "Saudável": 0.10},
]
EXTRA_WEIGHTS = {"Bebidas": 0.5, "Porções": 0.3, "Sobremesas": 0.2}

HOURS = np.arange(10, 24)
HOUR_WEIGHTS = np.array(
    [0.01, 0.05, 0.12, 0.10, 0.045, 0.02, 0.02, 0.03, 0.07, 0.12, 0.15, 0.12, 0.07, 0.025]
)
WEEKDAY_MULTIPLIER = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.30, 1.15])  # seg..dom
CAPACITY_PER_HOUR = 30.0  # pedidos/hora que a operação absorve sem atrasar


def _pick(rng: np.random.Generator, ids: np.ndarray, size: int) -> np.ndarray:
    """Escolhe produtos de uma categoria; os primeiros da lista são mais populares."""
    weights = np.linspace(1.0, 0.35, len(ids))
    return rng.choice(ids, size=size, p=weights / weights.sum())


def make_dataset(
    days: int = 240, base_orders: int = 180, seed: int = 7, start: str = "2026-01-05"
) -> dict[str, pd.DataFrame]:
    """Retorna {'products', 'orders', 'order_items'} prontos para ir ao banco."""
    rng = np.random.default_rng(seed)

    products = pd.DataFrame(PRODUCTS, columns=["name", "category", "price"])
    products.insert(0, "product_id", np.arange(1, len(products) + 1))
    ids_by_category = {c: g["product_id"].to_numpy() for c, g in products.groupby("category")}
    price_of = products.set_index("product_id")["price"]

    # --- demanda diária: dia da semana + tendência + dias de promoção + ruído ---
    day_index = pd.date_range(start, periods=days, freq="D")
    trend = 1 + 0.25 * np.arange(days) / days
    promo = np.where(rng.random(days) < 0.04, rng.uniform(1.25, 1.45, days), 1.0)
    noise = rng.lognormal(0.0, 0.05, days)
    expected = base_orders * WEEKDAY_MULTIPLIER[day_index.weekday] * trend * promo * noise
    orders_per_day = rng.poisson(expected)

    # --- horário dentro do dia: picos no almoço e no jantar (mais forte sex/sáb) ---
    day_of_order = np.repeat(np.arange(days), orders_per_day)
    n = len(day_of_order)
    hour = np.empty(n, dtype=int)
    for d in range(days):
        weights = HOUR_WEIGHTS.copy()
        if day_index[d].weekday() in (4, 5):
            weights[(HOURS >= 19) & (HOURS <= 22)] *= 1.2
        idx = np.where(day_of_order == d)[0]
        hour[idx] = rng.choice(HOURS, size=len(idx), p=weights / weights.sum())
    seconds = hour * 3600 + rng.integers(0, 3600, n)
    created = day_index.values.astype("datetime64[s]")[day_of_order] + seconds.astype("timedelta64[s]")
    order = np.argsort(created, kind="stable")
    created, day_of_order, hour = created[order], day_of_order[order], hour[order]

    # --- bairro, pagamento e cliente ---
    names = list(NEIGHBORHOODS)
    neighborhood = rng.choice(names, size=n, p=[NEIGHBORHOODS[k][1] for k in names])
    travel_base = np.array([NEIGHBORHOODS[k][0] for k in neighborhood], dtype=float)
    payment = rng.choice(list(PAYMENTS), size=n, p=list(PAYMENTS.values()))
    customer = rng.integers(1, 4000, n)

    # --- itens: um prato principal (conforme o período) + 0 a 3 extras ---
    period = np.where((hour >= 11) & (hour <= 15), 0, np.where(hour >= 18, 1, 2))
    main_category = np.empty(n, dtype=object)
    for p, weights in enumerate(MAIN_WEIGHTS):
        idx = np.where(period == p)[0]
        main_category[idx] = rng.choice(list(weights), size=len(idx), p=list(weights.values()))
    main_product = np.zeros(n, dtype=int)
    for category, ids in ids_by_category.items():
        idx = np.where(main_category == category)[0]
        if len(idx):
            main_product[idx] = _pick(rng, ids, len(idx))
    main_qty = rng.choice([1, 2, 3], size=n, p=[0.85, 0.13, 0.02])

    n_extra = np.minimum(rng.poisson(1.0, n), 3)
    extra_order = np.repeat(np.arange(n), n_extra)
    extra_category = rng.choice(list(EXTRA_WEIGHTS), size=len(extra_order), p=list(EXTRA_WEIGHTS.values()))
    extra_product = np.zeros(len(extra_order), dtype=int)
    for category in EXTRA_WEIGHTS:
        idx = np.where(extra_category == category)[0]
        if len(idx):
            extra_product[idx] = _pick(rng, ids_by_category[category], len(idx))
    extra_qty = rng.choice([1, 2], size=len(extra_order), p=[0.9, 0.1])

    items = pd.DataFrame(
        {
            "order_id": np.concatenate([np.arange(1, n + 1), extra_order + 1]),
            "product_id": np.concatenate([main_product, extra_product]),
            "quantity": np.concatenate([main_qty, extra_qty]),
        }
    )
    items = items.groupby(["order_id", "product_id"], as_index=False)["quantity"].sum()
    items["unit_price"] = items["product_id"].map(price_of).to_numpy()
    subtotal = np.bincount(
        items["order_id"] - 1, weights=(items["quantity"] * items["unit_price"]).to_numpy(), minlength=n
    ).round(2)

    # --- entrega: preparo + deslocamento + fila quando a hora está sobrecarregada ---
    slot = day_of_order * 24 + hour
    load = np.bincount(slot)[slot] / CAPACITY_PER_HOUR
    congestion = np.clip(load - 0.85, 0.0, None)
    prep = rng.lognormal(np.log(13.0), 0.28, n)
    travel = travel_base * rng.lognormal(0.0, 0.18, n)
    queue = congestion * rng.uniform(10, 30, n)
    delivery_minutes = prep + travel + queue + rng.normal(0, 1.5, n)
    promised = travel_base + 25  # prazo prometido: deslocamento esperado + 25 min

    canceled = rng.random(n) < 0.02 + 0.03 * np.clip(load - 0.85, 0.0, 1.5)
    delivered = created + (delivery_minutes * 60).astype("timedelta64[s]")
    fee = np.round(3.5 + 0.12 * travel_base, 2)

    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "customer_id": customer,
            "created_at": pd.Series(created).dt.strftime("%Y-%m-%d %H:%M:%S"),
            "delivered_at": pd.Series(delivered).dt.strftime("%Y-%m-%d %H:%M:%S").where(~canceled, None),
            "promised_minutes": promised.astype(int),
            "status": np.where(canceled, "cancelado", "entregue"),
            "neighborhood": neighborhood,
            "payment_method": payment,
            "subtotal": subtotal,
            "delivery_fee": fee,
            "total": np.round(subtotal + fee, 2),
        }
    )
    return {"products": products, "orders": orders, "order_items": items}
