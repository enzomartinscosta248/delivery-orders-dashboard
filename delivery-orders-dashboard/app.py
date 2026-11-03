"""Dashboard inteligente de pedidos de delivery (Streamlit)."""

import sys
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parent
except NameError:  # sem __file__ (execução interativa)
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from delivery_dashboard import queries as q  # noqa: E402
from delivery_dashboard.db import ensure_database  # noqa: E402
from delivery_dashboard.forecast import (  # noqa: E402
    FORECASTERS,
    daily_series,
    forecast_future,
    rolling_backtest,
    score_forecasts,
)
from delivery_dashboard.insights import delay_summary, peak_summary  # noqa: E402

DB_PATH = ROOT / "data" / "delivery.db"
TEAL, RED = "#0F6E6E", "#B5382A"

st.set_page_config(page_title="Pedidos de delivery", page_icon="🛵", layout="wide")


def number(value, decimals: int = 0, prefix: str = "", suffix: str = "") -> str:
    """Formata no padrão brasileiro (1.234,56); valores ausentes viram '—'."""
    if value is None or pd.isna(value):
        return "—"
    text = f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefix}{text}{suffix}"


@st.cache_resource(show_spinner="Gerando os dados fictícios (só na primeira execução)...")
def database() -> str:
    return str(ensure_database(DB_PATH))


DB = database()


@st.cache_data(show_spinner=False)
def run(name: str, start: str, end: str, hoods: tuple, limit: int = 10) -> pd.DataFrame:
    return q.run_query(DB, name, q.Filters(start, end, hoods or None), limit)


# ---------------------------------------------------------------- filtros
first_day, last_day = q.date_bounds(DB)
all_hoods = q.list_neighborhoods(DB)

st.sidebar.header("Filtros")
period = st.sidebar.date_input(
    "Período", value=(first_day, last_day), min_value=first_day, max_value=last_day
)
if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("Escolha a data inicial e a final para continuar.")
    st.stop()
start, end = period[0].isoformat(), period[1].isoformat()

selected = st.sidebar.multiselect("Bairros", all_hoods, default=all_hoods)
if not selected:
    st.warning("Selecione ao menos um bairro.")
    st.stop()
hoods = () if len(selected) == len(all_hoods) else tuple(selected)
filters = q.Filters(start, end, hoods or None)

st.title("Pedidos de delivery")
st.caption(
    "Dados fictícios gerados por script. Os padrões (picos, atrasos) foram embutidos no gerador: "
    "o objetivo aqui é mostrar a análise, não descobrir fatos."
)

kpis = run("kpis", start, end, hoods).iloc[0]
if int(kpis["orders"]) == 0:
    st.warning("Não há pedidos nesse período e nesses bairros.")
    st.stop()

tab_overview, tab_peaks, tab_products, tab_delays, tab_forecast = st.tabs(
    ["Resumo", "Horários de pico", "Produtos", "Atrasos", "Previsão de demanda"]
)

# ---------------------------------------------------------------- resumo
with tab_overview:
    cols = st.columns(5)
    cols[0].metric("Pedidos", number(kpis["orders"]))
    cols[1].metric("Faturamento", number(kpis["revenue"], 2, "R$ "))
    cols[2].metric("Ticket médio", number(kpis["avg_ticket"], 2, "R$ "))
    cols[3].metric("Tempo médio de entrega", number(kpis["avg_delivery_min"], 0, suffix=" min"))
    cols[4].metric("Entregas atrasadas", number(kpis["late_pct"], 1, suffix="%"))
    st.caption(f"Cancelamentos: {number(kpis['cancel_pct'], 1, suffix='%')} dos pedidos.")

    daily = run("daily_orders", start, end, hoods)
    st.subheader("Pedidos por dia")
    st.line_chart(daily.assign(day=pd.to_datetime(daily["day"])), x="day", y="orders", color=TEAL)
    st.subheader("Faturamento por categoria")
    st.bar_chart(run("revenue_by_category", start, end, hoods), x="category", y="revenue", color=TEAL)
    with st.expander("SQL desta aba"):
        st.code(q.render_sql("kpis", filters), language="sql")
        st.code(q.render_sql("revenue_by_category", filters), language="sql")

# ---------------------------------------------------------------- picos
with tab_peaks:
    by_hour = run("orders_by_hour", start, end, hoods)
    heat = q.heatmap_table(run("heatmap", start, end, hoods))
    st.subheader("Pedidos por hora do dia")
    st.bar_chart(by_hour, x="hour", y="orders", color=TEAL)
    st.subheader("Dia da semana e hora")
    st.dataframe(heat.style.background_gradient(cmap="YlGnBu", axis=None))
    st.markdown("\n".join(f"- {message}" for message in peak_summary(by_hour, heat)))
    with st.expander("SQL desta aba"):
        st.code(q.render_sql("orders_by_hour", filters), language="sql")
        st.code(q.render_sql("heatmap", filters), language="sql")

# ---------------------------------------------------------------- produtos
with tab_products:
    top_n = st.slider("Quantos produtos mostrar", 5, 20, 10)
    top = run("top_products", start, end, hoods, top_n)
    st.subheader("Mais vendidos (unidades)")
    st.bar_chart(top, x="product", y="units", color=TEAL)
    st.dataframe(top)
    st.subheader("Os 3 mais vendidos de cada categoria")
    st.dataframe(run("top_per_category", start, end, hoods))
    with st.expander("SQL desta aba"):
        st.code(q.render_sql("top_products", filters, top_n), language="sql")
        st.code(q.render_sql("top_per_category", filters), language="sql")

# ---------------------------------------------------------------- atrasos
with tab_delays:
    by_hour = run("orders_by_hour", start, end, hoods)
    by_hood = run("late_by_neighborhood", start, end, hoods)
    left, right = st.columns(2)
    with left:
        st.subheader("Atraso por hora (%)")
        st.bar_chart(by_hour, x="hour", y="late_pct", color=RED)
    with right:
        st.subheader("Atraso por bairro (%)")
        st.bar_chart(by_hood, x="neighborhood", y="late_pct", color=RED)
    st.markdown("\n".join(f"- {message}" for message in delay_summary(by_hour, by_hood)))
    st.subheader("As entregas mais atrasadas")
    st.dataframe(run("worst_delays", start, end, hoods, 15))
    with st.expander("SQL desta aba"):
        st.code(q.render_sql("late_by_neighborhood", filters), language="sql")
        st.code(q.render_sql("worst_delays", filters, 15), language="sql")

# ---------------------------------------------------------------- previsão
with tab_forecast:
    st.markdown(
        "Compara três métodos de previsão com os pedidos reais. A cada semana, cada método "
        "enxerga só o passado, como aconteceria na prática."
    )
    weeks = st.slider("Semanas de teste", 2, 8, 4)
    horizon = st.slider("Dias a prever daqui para frente", 7, 28, 14)

    history = run("daily_orders", first_day.isoformat(), last_day.isoformat(), hoods)
    series = daily_series(history)
    try:
        backtest = rolling_backtest(series, horizon=7, n_folds=weeks)
    except ValueError as error:
        st.warning(str(error))
        st.stop()

    scores = score_forecasts(backtest)
    best = scores.iloc[0]["modelo"]
    st.subheader("Previsão e pedidos reais")
    st.line_chart(backtest.drop(columns="fold"))
    st.dataframe(scores.round(2))
    st.caption(
        f"Melhor método no teste: {best}. MAE é o erro médio em pedidos por dia; "
        "MAPE é o erro percentual médio."
    )

    sigma = float((backtest["real"] - backtest[best]).std())
    future = forecast_future(series, FORECASTERS[best], horizon, sigma)
    st.subheader(f"Próximos {horizon} dias")
    st.line_chart(pd.concat([series.iloc[-28:], future], axis=1))
    st.caption(
        "A faixa é uma aproximação: previsão ± 1,28 desvio do erro observado no teste "
        "(cerca de 80% de chance, se os erros forem parecidos com os do teste). "
        "Promoções e feriados não entram no modelo."
    )
    with st.expander("Como cada método funciona"):
        st.markdown(
            "- **Mesmo dia da semana passada:** repete o último valor observado naquele dia.\n"
            "- **Média do dia da semana:** média das últimas 4 semanas naquele dia, o que suaviza o ruído.\n"
            "- **Regressão:** ajusta uma tendência linear e um efeito para cada dia da semana."
        )
    with st.expander("SQL desta aba"):
        st.code(q.render_sql("daily_orders", filters), language="sql")
