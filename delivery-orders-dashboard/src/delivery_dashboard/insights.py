"""Frases de interpretação geradas a partir dos números (nada é escrito à mão)."""

from __future__ import annotations

import pandas as pd


def peak_summary(by_hour: pd.DataFrame, heat: pd.DataFrame) -> list[str]:
    if by_hour.empty or heat.empty:
        return []
    top = by_hour.nlargest(3, "orders")
    hours = ", ".join(f"{int(h)}h" for h in sorted(top["hour"]))
    share = top["orders"].sum() / by_hour["orders"].sum()
    weekday, hour = heat.stack().idxmax()
    return [
        f"As 3 horas mais movimentadas ({hours}) concentram {share:.0%} dos pedidos.",
        f"O maior pico é {weekday} às {int(hour)}h, com {int(heat.loc[weekday, hour])} pedidos no período.",
    ]


def delay_summary(by_hour: pd.DataFrame, by_neighborhood: pd.DataFrame) -> list[str]:
    messages = []
    hours = by_hour.dropna(subset=["late_pct"])
    if not hours.empty:
        worst = hours.loc[hours["late_pct"].idxmax()]
        messages.append(
            f"A maior taxa de atraso é às {int(worst['hour'])}h: "
            f"{worst['late_pct']:.0f}% das entregas passam do prazo prometido."
        )
        corr = hours["orders"].corr(hours["late_pct"]) if len(hours) >= 4 else float("nan")
        if pd.notna(corr):
            if corr >= 0.5:
                reading = "os atrasos crescem junto com o volume de pedidos (sinal de sobrecarga)."
            elif corr <= -0.5:
                reading = "os atrasos são maiores nas horas mais vazias."
            else:
                reading = "não há relação clara entre volume e atraso."
            messages.append(f"Correlação entre pedidos e atraso por hora: {corr:.2f}. Em resumo, {reading}")
    if not by_neighborhood.empty:  # a consulta já vem ordenada do maior para o menor atraso
        worst, best = by_neighborhood.iloc[0], by_neighborhood.iloc[-1]
        messages.append(
            f"{worst['neighborhood']} tem a maior taxa de atraso ({worst['late_pct']:.0f}%) "
            f"e {best['neighborhood']} a menor ({best['late_pct']:.0f}%)."
        )
    return messages
