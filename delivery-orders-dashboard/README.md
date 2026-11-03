## Como funciona

```
gerador (numpy) → SQLite (tabelas + view) → consultas SQL → pandas → Streamlit
```

- **`generate_data.py`**: cria ~50 mil pedidos em 240 dias (semente fixa, mesmos dados sempre).
- **`db.py`**: esquema com chaves e índices, mais a view `v_orders`, que calcula dia, hora, dia da semana, tempo de entrega e atraso direto em SQL.
- **`queries.py`**: consultas parametrizadas (nenhum valor é concatenado no SQL), incluindo `JOIN`s, agregações, **CTE** e **função de janela** (`RANK() OVER`) para o top 3 de cada categoria.
- **`forecast.py`**: previsão de demanda diária e backtest.
- **`insights.py`**: frases de interpretação geradas a partir dos números.

## Previsão de demanda

Três métodos são comparados com os pedidos reais:

1. **Mesmo dia da semana passada** (referência simples).
2. **Média do dia da semana** nas últimas 4 semanas.
3. **Regressão** (Ridge) com tendência linear e efeito de cada dia da semana.

O **backtest é semanal**: a cada semana do período de teste, cada método enxerga só o passado e prevê os 7 dias seguintes, como numa operação real. O app mostra previsto x real, MAE, RMSE e MAPE, e projeta os próximos dias com o melhor método e uma faixa aproximada (± 1,28 desvio do erro do teste). Promoções e feriados não entram no modelo.

## Testes

```bash
pip install -e ".[dev]"
pytest
```

Os testes cobrem o gerador, o esquema do banco, as consultas (totais batendo entre si, filtros, proteção contra injeção de SQL), a previsão e a execução do app com o `AppTest` oficial do Streamlit.

## Estrutura

```
delivery-orders-dashboard/
├── app.py                      # interface Streamlit (fina: só monta as abas)
├── src/delivery_dashboard/     # dados, SQL, previsão e insights
├── tests/
├── .streamlit/config.toml      # tema
└── data/                       # delivery.db (gerado, não versionado)
```
## Licença

MIT.
