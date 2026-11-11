# btc-trader — automated BTC/JPY trend trader for bitFlyer

A self-hostable system that collects BTC/JPY market data from bitFlyer, runs a
research-backed daily trend-following strategy, and executes it **on its own**,
either on a paper ledger or with real orders on bitFlyer Lightning spot.

The default mode is `data_only`. Real orders need four explicit switches, a
capital limit and API credentials (see [Live trading](#live-trading)). Read
[docs/strategy_research.md](docs/strategy_research.md) before risking money.
The strategy reduced drawdowns sharply in the past, but it has shown **no
return edge** since its publication.

---

## Quick start (Docker Compose)

```bash
cp .env.example .env      # set the PostgreSQL password and operational settings
chmod 600 .env
docker compose build
docker compose up -d
```

For a Linux server, paper-then-live rollout, backups and the emergency stop,
follow **[docs/deployment.md](docs/deployment.md)**.

| Service   | URL                            |
|-----------|--------------------------------|
| Dashboard | <http://localhost:8501>        |
| API docs  | <http://localhost:8000/docs>   |
| Health    | <http://localhost:8000/health> |

Ports are published on `127.0.0.1` only (`BIND_ADDRESS`). From another machine,
use an SSH tunnel or a VPN.

`migrate` applies Alembic migrations before `api`, `worker` and `dashboard`
start. PostgreSQL data lives in the `postgres_data` volume. The worker runs
unattended from there:

- it collects a ticker every 5 minutes;
- it backfills a year or more of daily history on first start;
- it builds its own daily bars every day;
- in paper or live mode, it evaluates the strategy and rebalances.

## Quick start (local)

```bash
uv venv --python 3.12 .venv            # 3.12–3.14 (psycopg-binary has no 3.15 wheels)
uv sync --locked --extra dev --extra dashboard
cp .env.example .env
python -m btc_trader.scripts.init_db          # alembic upgrade head
python -m btc_trader.scripts.backfill_daily   # optional: the worker does this itself
uvicorn btc_trader.main:app --port 8000       # API
python -m btc_trader.worker                   # scheduler, strategy and execution
streamlit run dashboard/app.py                # dashboard
```

---

## How it works

```
bitFlyer public API ──► collector ──► market_ticks ──┐
bitbank daily candles ──► daily history ──► candles ─┤ (own bars preferred, basis-adjusted)
                                                     ▼
                       strategy.py  daily Donchian trend ensemble → target BTC exposure
                                                     ▼
                       rebalance.py band · lot size · limits · fees → at most one order
                          ├── paper.py      simulated fills (TRADING_MODE=paper)
                          ├── live.py       limit IOC orders on bitFlyer (TRADING_MODE=live)
                          └── backtesting.py the same planner over history
```

**Strategy** (`trend_ensemble`, after Zarattini, Pagani & Barbon 2025): nine
Donchian sleeves run on completed UTC daily closes, with lookbacks of
5–360 days.

- A sleeve goes long on a new N-day closing high.
- It exits on a close below a trailing midpoint stop that only ratchets up.
- Each long sleeve holds `target vol / realised vol` of its share of equity
  (default target 25% a year).

The output is a target BTC share of equity between 0% and 100%. A volatility-only
resize is skipped unless it moves the holding by at least 20%.

**Why daily:** a bitFlyer round trip costs about 0.35–0.40%. Strategies that
trade on 5-minute signals lost 60–85% a year on real data; see the research note.

**Execution cadence:** the strategy decides once per completed day; the result
is one signal row per day. The worker re-checks every collection interval, so
it retries after a blocked attempt, an unfilled order, maintenance or an
outage, and resizes only if the holding drifts outside the band. In 2018–2026
the backtest averaged about 2 small orders a week.

### Modules

```
src/btc_trader/
├── strategy.py          trend ensemble (pure, deterministic, look-ahead tested)
├── rebalance.py         target exposure → one order, shared by paper/live/backtest
├── strategy_service.py  daily decision → signal row → paper or live
├── paper.py             fee-aware paper ledger
├── live.py              live execution, reconciliation, circuit breaker
├── backtesting.py       daily replay vs buy & hold, calendar years, cost stress
├── market_data/         collector, daily bars, history provider, chart candles
├── exchange/            interface, bitFlyer adapter (signed private API), HTTP
├── scheduler/           APScheduler jobs (collection, daily history, strategy, snapshots)
├── monitoring/          logging, health, webhook alerts
├── api/, db/, utils/    FastAPI routes, models, helpers
└── scripts/             init_db, collect_once, backfill_daily, live_check, healthcheck
```

---

## Live trading

The worker places real orders only when **all** of these are true:

| Setting | Value |
|---|---|
| `TRADING_MODE` | `live` |
| `ENABLE_AUTOMATIC_TRADING` | `true` |
| `LIVE_TRADING_ACKNOWLEDGED` | `true` |
| `TRADING_KILL_SWITCH` | `false` |
| `BITFLYER_API_KEY` / `BITFLYER_API_SECRET` | set (trade + read permissions; **no withdrawal**) |
| `LIVE_CAPITAL_JPY` | > 0 — the JPY equity the bot may manage |
| `STRATEGY_NAME` | `trend_ensemble` |

`LIVE_CAPITAL_JPY` caps the JPY the bot spends. **All BTC in the account counts
as the bot's** and may be sold, so keep other BTC in a separate account.

### What happens on each live evaluation

1. **Reconcile** every unresolved order against bitFlyer before anything else.
2. **Gates:** switches, capital, no pending order, the circuit breaker, the
   daily maintenance window (03:55–04:15 JST), and board state. The board must
   be `RUNNING` with health `NORMAL`, `BUSY` or `VERY BUSY`.
3. **Read the truth** from bitFlyer: fresh bid/ask, balances, and your actual
   fee tier (`gettradingcommission`).
4. **Plan** with the same rules as paper and backtest: band, lot step,
   `MAX_ORDER_JPY`, `MAX_POSITION_JPY`, and the daily loss limit (which pauses
   buys only).
5. **Persist** the order as `SUBMITTING`, **then** submit it once.
   - The order is a LIMIT IOC with a 1-minute expiry, priced
     `LIVE_MAX_SLIPPAGE_BPS` (default 0.30%) through the touch. It fills
     against the book immediately, never at a worse price, and never rests.
   - Buys are capped by available JPY at the limit price.
   - Sells leave room for the commission bitFlyer charges in BTC.
6. **Poll** to a final state and record the fills (commission converted to JPY).
   Then update the position and snapshot the account.

**No automatic retries of an order.** A timeout makes the order `UNKNOWN`.
Reconciliation then either finds it among recent exchange orders or, after
10 minutes, marks it `FAILED` (not executed). An accepted order that bitFlyer
stops listing is settled from its fills after 3 minutes. While any order is
unresolved, no new order is sent. If `LIVE_MAX_FAILED_ORDERS_PER_DAY` (default
3) orders fail or are rejected within 24 hours, trading pauses and a critical
alert is sent. Only the worker places orders; `POST /api/v1/strategy/evaluate`
refuses in live mode.

### Going live safely

```bash
python -m btc_trader.scripts.live_check
```

This is read-only. It checks key permissions (and warns if the key can
withdraw), balances, fee tier, exchange state, and every gate. To confirm the
whole order path once with real money:

```bash
python -m btc_trader.scripts.live_check --test-order buy --confirm-real-money
```

This places a single 0.001 BTC order (about ¥13,000 at current prices). Then
paper-trade for weeks, start with a small `LIVE_CAPITAL_JPY`, and set
`ALERT_WEBHOOK_URL` (Slack or Discord). Alerts cover:

- fills, rejections and unknown outcomes;
- trading halts;
- jobs that start failing, and when they recover.

**Minimum lot:** bitFlyer's 0.001 BTC minimum is about ¥13k. With less than
roughly ¥100k of capital, many of the strategy's small rebalances cannot be
placed. The backtest with ¥100k still tracked the uncapped strategy closely
(fewer trades, similar returns), but the deviation grows as capital falls.

---

## Configuration

Everything comes from the environment (`.env.example` is annotated). Most
operational values can also be changed at runtime from the dashboard, and the
worker picks them up without a restart. Once saved from the dashboard, a value
overrides `.env`. The exception is `TRADING_KILL_SWITCH=true` in `.env`, which
always wins.

| Variable | Default | Effect |
|---|---|---|
| `TRADING_MODE` | `data_only` | `data_only` \| `paper` \| `live` |
| `STRATEGY_NAME` | `none` | `trend_ensemble` to trade |
| `STRATEGY_TARGET_ANNUAL_VOLATILITY_PERCENT` | `25` | Risk dial: higher means more BTC and deeper drawdowns |
| `STRATEGY_MAX_EXPOSURE_PERCENT` | `100` | Cap on BTC share of equity (spot, no leverage) |
| `STRATEGY_REBALANCE_BAND_PERCENT` | `20` | Minimum volatility-only resize, as % of the holding |
| `LIVE_CAPITAL_JPY` | `0` | Equity the bot manages live; `0` blocks live orders |
| `LIVE_MAX_SLIPPAGE_BPS` | `30` | Limit-price distance through the touch |
| `MAX_ORDER_JPY` / `MAX_POSITION_JPY` / `MAX_DAILY_LOSS_JPY` | `300000` / `500000` / `30000` | Buys only; risk-reducing sells are never blocked |
| `MIN_PROFIT_PERCENT` | `1.0` | Applies to `STRATEGY_EXIT` sells; stop and volatility exits are exempt (§21) |
| `DAILY_HISTORY_PROVIDER` | `bitbank` | Public daily BTC/JPY history; `none` uses own ticks only |
| `ALERT_WEBHOOK_URL` | empty | Slack/Discord webhook |
| `MARKET_DATA_INTERVAL_SECONDS` | `300` | Collection and evaluation frequency |
| `BIND_ADDRESS` | `127.0.0.1` | Host address Docker publishes the API and dashboard on |

Secrets (`BITFLYER_API_*`, `ALERT_WEBHOOK_URL`) are `SecretStr`. They are never
logged and never served by the API.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Database, scheduler and exchange status (503 only if the database is down) |
| GET | `/api/v1/market/latest`, `/overview`, `/history`, `/candles` | Stored market data |
| POST | `/api/v1/market/collect` | One immediate collection |
| GET / POST | `/api/v1/market/daily-history`, `/daily-history/sync` | Daily-bar coverage / load history |
| GET | `/api/v1/strategy/preview` | Today's target exposure, sleeves and stops |
| POST | `/api/v1/strategy/evaluate` | One paper evaluation (refused in live mode) |
| GET | `/api/v1/backtest` | Strategy vs buy & hold, calendar years, cost stress |
| GET | `/api/v1/signals`, `/orders`, `/trades`, `/portfolio` | Ledger projections for the current mode |
| GET | `/api/v1/trading/status` | Mode and every safety switch |
| POST | `/api/v1/trading/paper/reset` | Reset the paper ledger (`{"confirm": true}`) |
| GET / PATCH | `/api/v1/system/settings` | Runtime settings (secrets redacted) |
| GET | `/api/v1/system/status`, `/events`, `/collector` | Operations |

No endpoint places a live order. `Decimal` fields are JSON strings. Do not
expose the API or dashboard to the internet without authentication: use a VPN,
Tailscale or an authenticating reverse proxy (§56).

---

## Data notes

- **Money is `Decimal`**, stored as `NUMERIC(30,12)` on PostgreSQL. Timestamps
  are UTC-aware.
- **Candle provenance:** daily bars from ticks are `sampled_ticks`
  (exchange `bitflyer`). Provider bars are `exchange_ohlc`, with the provider
  name as the exchange.
- **Signals** store the full decision (sleeves, stops, volatility, data
  provenance). `metadata_json.execution` records what was done: `filled`,
  `no_trade`, `submitted`, `rejected` or `blocked_*`. Orders store their
  decision context; live orders also store the limit price, the fee rate and
  the bid/ask at decision time.

---

## Testing

```bash
uv lock --check
pytest                          # no network; bitFlyer is mocked with respx
ruff check . && ruff format --check . && mypy src
```

The live engine is tested against a simulated account that behaves like
bitFlyer: IOC fills, BTC-denominated commission, unlisted unfilled orders, and
responses lost after acceptance. The tests cover every gate, rejections, the
circuit breaker, lost-response reconciliation without a resend, unfilled-IOC
retries, partial fills, and an end-to-end daily decision. Signing is verified
against the exact bytes on the wire.
