# btc-trader — BTC/JPY market-data platform (bitFlyer)

A modular, self-hostable platform for collecting, storing and visualising
BTC/JPY market data from bitFlyer, evaluating selectable algorithms, and
simulating paper trades.

**Live exchange execution is disabled.** Paper orders are stored locally; the
bitFlyer adapter still refuses real order submission, and the default trading
mode is `data_only`. See [Scope](#scope).

---

## Linux deployment

Docker deployment needs Docker Engine and the Docker Compose plugin. The host
does not need Python or PostgreSQL; the image installs ca-certificates, curl,
and tzdata itself. For a local install, provide Python 3.12 with venv,
ca-certificates, and tzdata; build-essential is only needed if a platform has
no compatible binary wheels.

## Quick start (Docker Compose)

```bash
cp .env.example .env      # edit the password and operational settings
chmod 600 .env
docker compose config --quiet
docker compose up -d --build
```

Then open:

| Service   | URL                          |
|-----------|------------------------------|
| Dashboard | <http://localhost:8501>      |
| API docs  | <http://localhost:8000/docs> |
| Health    | <http://localhost:8000/health> |

The `migrate` service applies Alembic migrations before `api`, `worker` and
`dashboard` start. PostgreSQL data lives in the `postgres_data` volume, so
`docker compose down && docker compose up -d` keeps your history.

For a production host, set APP_ENV=production and LOG_FORMAT=json in .env,
replace the example PostgreSQL password, and inject exchange credentials only
if required. The API and dashboard bind to all container interfaces, logs go
to stderr/stdout, and Compose gives each service 30 seconds for graceful
shutdown. Use docker compose logs -f worker api to inspect runtime failures.

Dependencies are pinned by uv.lock. After changing pyproject.toml, run uv lock
and commit both files before rebuilding the image.

## Quick start (local, no Docker)

```bash
uv venv --python 3.12 .venv
uv sync --locked --extra dev --extra dashboard
source .venv/bin/activate

cp .env.example .env

python -m btc_trader.scripts.init_db          # alembic upgrade head
python -m btc_trader.scripts.collect_once     # one collection, now
uvicorn btc_trader.main:app --port 8000       # API
python -m btc_trader.worker                   # scheduler (separate terminal)
streamlit run dashboard/app.py                # dashboard
```

---

## Architecture

```
bitFlyer public API
        │
        ▼
Exchange adapter  (exchange/bitflyer.py behind exchange/base.py)
        │            rate limiter · retries · backoff · error mapping
        ▼
Market-data collector  (market_data/collector.py)
        │            one implementation for scheduler + API + CLI
        ▼
PostgreSQL  (db/models.py, Alembic migrations)
        │
        ├──► FastAPI read API  (api/)
        └──► Streamlit dashboard  (dashboard/app.py)
```

Everything downstream of the adapter is exchange-agnostic. Adding an exchange
means implementing `ExchangeClient` and adding one entry to
`exchange/registry.py`.

### Layout

```
src/btc_trader/
├── config.py            all configuration, Pydantic Settings
├── enums.py             shared domain enums
├── errors.py            domain exception hierarchy
├── main.py              FastAPI app
├── worker.py            APScheduler process
├── api/                 routes + response schemas + dependencies
├── exchange/            base interface, models, HTTP transport, bitFlyer adapter
├── market_data/         collector, read service, repository, candle aggregation
├── scheduler/           job registration
├── db/                  models, session, portable column types, repositories
├── monitoring/          structured logging, health aggregation
├── utils/               UTC time helpers, JSON-safe serialisation
└── scripts/             collect_once, backfill, healthcheck, init_db
dashboard/app.py         Streamlit + Plotly UI
migrations/              Alembic
tests/                   unit + integration
```

---

## Configuration

Every operational value comes from the environment (see `.env.example` for the
complete annotated list). The ones you are most likely to change:

| Variable | Default | Effect |
|---|---|---|
| `MARKET_DATA_INTERVAL_SECONDS` | `300` | Collection frequency. Set to `60` for one collection per minute — no code change. |
| `DASHBOARD_REFRESH_SECONDS` | `300` | Dashboard auto-refresh period. |
| `TRADING_MODE` | `data_only` | `data_only` \| `paper` \| `live`. |
| `PRODUCT_CODE` | `BTC_JPY` | bitFlyer product code. |
| `DATABASE_URL` | SQLite when unset; PostgreSQL in Compose | `sqlite+aiosqlite:///./data/btc_trader.db` for local dev. |
| `MIN_PROFIT_PERCENT` | `1.0` | Minimum net profit for ordinary paper strategy sells. |
| `MAX_ORDER_JPY` / `MAX_POSITION_JPY` / `MAX_DAILY_LOSS_JPY` | `10000` / `100000` / `5000` | Paper risk limits. |
| `COLLECT_ORDER_BOOK` / `COLLECT_EXECUTIONS` | `true` | Optional extras per collection. |
| `RUN_SCHEDULER_IN_API` | `false` | `true` hosts the scheduler inside the API process (single container). |
| `LOG_FORMAT` | `console` | Use `json` in deployed environments. |

Strategy sensitivity and sizing are also runtime-editable from the dashboard:
window lengths, EMA/momentum thresholds, mean-reversion z-score threshold,
volatility window, risk fraction, and target volatility. Lower thresholds create
more signals but can increase noise and turnover; validate changes with the
backtest first.

`STRATEGY_NAME=deeplob` enables the CPU-only DeepLOB-style order-book model. It
requires reliable stored order-book snapshots (`COLLECT_ORDER_BOOK=true`) and
waits for enough history before emitting a directional signal. The model uses
temporal convolution features and a rolling ridge head trained only on past
forward-return labels; it is a deployable baseline, not a pretrained CNN/LSTM
artifact.

### Secrets

`.env` is git-ignored; `.env.example` contains placeholders only. API
credentials are optional — the collector uses public endpoints exclusively and
`/health` never requires them. Credentials are stored as `SecretStr`, excluded
from `/api/v1/system/config`, and the logging pipeline redacts any field whose
name looks like a credential. In production prefer Docker secrets or
environment injection over a file.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Database, scheduler and public-exchange status. 503 only if the database is down. |
| GET | `/api/v1/market/latest` | Newest stored ticker. |
| GET | `/api/v1/market/overview` | Price, bid/ask, spread, period change, staleness. |
| GET | `/api/v1/market/history?period=24h` | Stored ticks (`1h`, `6h`, `24h`, `7d`, `30d`, `all`). |
| GET | `/api/v1/market/order-book` | Latest depth snapshot. |
| GET | `/api/v1/market/candles?interval=5m` | OHLCV with explicit provenance (see below). |
| POST | `/api/v1/market/collect` | Trigger one immediate collection. |
| GET | `/api/v1/system/status` | Aggregated operational status. |
| GET | `/api/v1/system/events` | Recent operational events. |
| GET | `/api/v1/system/collector` | Recent collector runs. |
| GET | `/api/v1/system/config` | Effective configuration, secrets redacted. |
| GET | `/api/v1/signals`, `/orders`, `/trades`, `/portfolio` | Read-only projections; populated by paper mode after enough market data is available. |
| POST | `/api/v1/trading/paper/reset` | Reset the configured paper account to starting JPY and zero BTC; requires `{"confirm": true}`. |
| GET | `/api/v1/backtest` | Fee/slippage-aware replay of stored ticks. |
| GET | `/api/v1/backtest/walk-forward` | Train-window sensitivity selection followed by unseen test-window scoring. |
| GET | `/api/v1/trading/status` | Trading mode and every safety switch. |

`Decimal` fields serialise as JSON **strings** — a JSON number would be read
back as a binary float and silently lose precision on JPY prices.

---

## Data model notes

- **Money is never a float.** All prices, sizes, fees and P&L are `Decimal`, stored
  as `NUMERIC(30,12)` on PostgreSQL and as lossless text on SQLite.
- **Timestamps are always timezone-aware UTC.** The dashboard renders in
  `DASHBOARD_TIMEZONE` (default `Asia/Tokyo`).
- **Inserts are idempotent.** `market_ticks` is unique on
  `(exchange, product_code, exchange_timestamp)` and `executions` on
  `(exchange, product_code, exchange_execution_id)`, both written with
  `ON CONFLICT DO NOTHING`. Re-running the collector or a backfill can never
  duplicate history.
- **Candles carry provenance.** Bars built from trade prints are labelled
  `executions` (true OHLCV, real volume). Bars derived from periodic LTP samples
  are labelled `sampled_ticks`, always carry zero volume, and are never
  presented as exchange OHLC — with a 5-minute interval a "1m sampled candle"
  holds at most one observation.

Paper strategy evaluation writes `strategy_signals`, simulated `orders` and
`fills`, weighted-average `positions`, and `portfolio_snapshots`. The live
execution adapter remains intentionally unavailable.

BUY/SELL/HOLD decisions use configurable windows and thresholds. EMA and
momentum thresholds now widen against robust recent noise, and the optional
regime filter suppresses trend entries in mean-reverting tapes (and the reverse
for mean reversion). Each signal stores its effective threshold, strength, and
return autocorrelation so a HOLD is explainable rather than opaque. Paper and
backtest exits share the hard stop-loss and fee-aware profit gate. Paper and
backtest entries also enforce the configured BTC lot step/minimum; a blocked
entry is recorded on the signal with its calculated size and minimum so a BUY
cannot silently become “no transaction.”

The dashboard exposes **Reset paper trading** only in paper mode. After the
confirmation checkbox is selected, it removes paper orders, fills, the paper
position, and paper snapshots, then immediately creates a fresh snapshot with
`PAPER_INITIAL_JPY` and `0` BTC. Market data and strategy signals are retained
for analysis, and the reset is recorded in `system_events`.

---

## Reliability

- **Non-overlapping jobs.** APScheduler runs with `max_instances=1` and
  `coalesce=True`, so a collection that outlives its interval never runs
  concurrently with itself.
- **A failed request never kills the scheduler.** Job bodies are wrapped; every
  failure is logged, recorded in `collector_runs` and `system_events`, and the
  next tick proceeds.
- **Partial degradation is preserved.** The ticker is mandatory; if the order book
  or executions call fails the tick is still stored and the run is marked
  degraded.
- **Centralised rate limiting.** All exchange traffic passes through one
  sliding-window limiter (default 400 requests / 5 min) with exponential backoff
  plus full jitter on timeouts, 429s and 5xx. Non-idempotent requests are never
  retried automatically.

---

## Testing

```bash
uv lock --check
pytest                    # unit + integration, no network
pytest -m live            # opt-in: hits the real bitFlyer public API
ruff check . && ruff format --check .
mypy src
```

Tests run against SQLite via the same portable column types as production, and
bitFlyer is mocked with `respx` using recorded response shapes. No test contacts
a private endpoint.

---

## Scope

Implemented: project foundation, bitFlyer public market data, persistent
storage with deduplication, scheduling with a configurable interval and manual
trigger, the dashboard, selectable EMA crossover/mean-reversion/momentum
algorithms plus a DeepLOB-style order-book model, configurable sensitivity and
volatility-aware sizing, fee/slippage-aware paper execution with portfolio
accounting, and historical backtesting with walk-forward validation.

Deliberately **not** implemented: private-API trading, live execution,
exchange-order reconciliation, and partial-fill handling. The live safety gates
remain fail-closed until those components are implemented and tested.

The unused pre-restructure prototype was removed because it was not imported by
the application.

### Test checklists that depend on later phases

PROJECT.md §47 (profitability) and §48 (trading safety) enumerate cases for code
this version does not contain. What is covered today, and what is not:

| §48 case | Status |
| --- | --- |
| live mode disabled, automatic trading disabled, kill switch enabled | covered — `tests/unit/test_trading_safety.py` |
| missing exchange credentials, exchange unavailable | covered — no request is signed or sent |
| stale market data | paper execution rejects stale ticks |
| maximum order / position exceeded, daily loss limit | enforced for paper entries |
| duplicate order, partial fill | deferred — paper fills are immediate and live reconciliation does not exist |

Paper cost basis, fees, slippage, minimum-profit exits, trailing-stop exits, and
volatility sizing are covered by `tests/integration/test_paper.py` and
`tests/unit/test_backtesting.py`. DeepLOB signal quality, paper fills, and
order-book backtesting are covered by `tests/unit/test_deeplob.py`. **No code
path can submit a real exchange order**: the bitFlyer adapter refuses.
