## 1. Project Overview

Build a modular, self-hostable Bitcoin algorithmic trading platform using the bitFlyer exchange.

The initial trading pair is:

```text
BTC/JPY
bitFlyer product code: BTC_JPY
```

The system must:

1. Collect BTC/JPY market data from bitFlyer.
2. Store market data incrementally in a persistent database.
3. Fetch new data periodically.
4. Default to a 5-minute collection interval.
5. Make the collection interval configurable.
6. Allow market-data collection to be triggered manually.
7. Provide a web dashboard showing BTC/JPY price movement.
8. Automatically refresh the visualization.
9. Provide a modular strategy framework.
10. Allow strategies to produce:
   - BUY
   - SELL
   - HOLD
11. Support automated order execution later.
12. Support configurable minimum-profit requirements before selling.
13. Default the minimum profit target to 1%.
14. Account for fees and preferably estimated slippage when deciding whether a trade is profitable.
15. Support paper trading before enabling live trading.
16. Store orders, executions, positions, signals, and P&L.
17. Run reliably on:
   - a home server,
   - a VPS,
   - or a cloud VM.
18. Be deployable using Docker Compose.
19. Keep exchange-specific code isolated so another exchange could be added later.
20. Prioritize safety and observability over execution speed.

This project initially targets spot BTC/JPY trading.

Do not implement margin, leverage, derivatives, or short selling unless explicitly requested later.

---

# 2. Guiding Design Principles

The system must follow these principles.

## 2.1 Separation of concerns

Do NOT write one large trading script.

Separate:

```text
Exchange API
    ↓
Market Data Collector
    ↓
Database
    ↓
Strategy Engine
    ↓
Risk / Portfolio Engine
    ↓
Execution Engine
    ↓
Exchange
```

Visualization operates primarily from the database rather than repeatedly querying bitFlyer.

---

## 2.2 Strategy-independent infrastructure

The trading strategy has not yet been selected.

Therefore:

- infrastructure must not contain strategy-specific assumptions;
- strategies must implement a standard interface;
- replacing one strategy with another must not require modifications to market-data, storage, execution, or dashboard code.

---

## 2.3 Paper trading first

Live trading must NOT be the default.

Default:

```text
TRADING_MODE=paper
```

Supported modes:

```text
data_only
paper
live
```

Meaning:

### data_only

Collect and visualize data only.

No strategy execution and no trades.

### paper

Run strategies and simulate trades.

No real orders sent to bitFlyer.

### live

Real orders may be submitted.

Live mode must require explicit configuration.

---

# 3. Recommended Technology Stack

Use Python unless there is a compelling reason not to.

Recommended stack:

```text
Python 3.12+
FastAPI
SQLAlchemy 2.x
Alembic
PostgreSQL
Pydantic / pydantic-settings
APScheduler
Plotly
Streamlit OR a FastAPI + lightweight frontend
httpx
pytest
Docker
Docker Compose
```

For the first version, prefer:

```text
Backend: FastAPI
Database: PostgreSQL
Dashboard: Streamlit + Plotly
Scheduler: APScheduler
Deployment: Docker Compose
```

This keeps the system simple enough for a home server while preserving a clean upgrade path.

SQLite may be supported for local development, but PostgreSQL should be the production default.

---

# 4. High-Level Architecture

```text
                   ┌───────────────────────┐
                   │      bitFlyer API     │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Exchange Adapter      │
                   │ bitFlyerClient        │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │ Market Data Collector │
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │      PostgreSQL       │
                   │                       │
                   │ market_data           │
                   │ trades                │
                   │ orders                │
                   │ signals               │
                   │ positions             │
                   │ portfolio_snapshots   │
                   └───────┬────────┬──────┘
                           │        │
                    ┌──────▼───┐ ┌──▼────────────┐
                    │Dashboard │ │ Strategy Engine│
                    └──────────┘ └──────┬────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │ Portfolio / Risk │
                               │ Engine           │
                               └────────┬─────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │ Execution Engine │
                               └────────┬─────────┘
                                        │
                                        ▼
                                   bitFlyer
```

---

# 5. Project Directory Structure

Use approximately this structure:

```text
bitcoin-trader/
│
├── PROJECT.md
├── README.md
├── pyproject.toml
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
│
├── migrations/
│
├── src/
│   └── btc_trader/
│       │
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       │
│       ├── api/
│       │   ├── routes_market.py
│       │   ├── routes_trading.py
│       │   └── routes_system.py
│       │
│       ├── exchange/
│       │   ├── base.py
│       │   ├── bitflyer.py
│       │   └── models.py
│       │
│       ├── market_data/
│       │   ├── collector.py
│       │   ├── service.py
│       │   └── repository.py
│       │
│       ├── strategy/
│       │   ├── base.py
│       │   ├── engine.py
│       │   └── strategies/
│       │       └── example.py
│       │
│       ├── portfolio/
│       │   ├── service.py
│       │   ├── pnl.py
│       │   └── position_manager.py
│       │
│       ├── risk/
│       │   ├── engine.py
│       │   └── rules.py
│       │
│       ├── execution/
│       │   ├── engine.py
│       │   ├── paper.py
│       │   └── live.py
│       │
│       ├── scheduler/
│       │   └── jobs.py
│       │
│       ├── db/
│       │   ├── models.py
│       │   ├── session.py
│       │   └── repositories/
│       │
│       ├── monitoring/
│       │   ├── logging.py
│       │   └── metrics.py
│       │
│       └── utils/
│
├── dashboard/
│   └── app.py
│
├── scripts/
│   ├── collect_once.py
│   ├── backfill.py
│   └── healthcheck.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

Do not create large generic files such as:

```text
utils.py
trading.py
helpers.py
```

containing unrelated functionality.

---

# 6. Configuration

All configurable behavior must come through configuration rather than hardcoded constants.

Use environment variables with Pydantic Settings.

Example:

```env
APP_ENV=development

TRADING_MODE=data_only

EXCHANGE=bitflyer
PRODUCT_CODE=BTC_JPY

MARKET_DATA_INTERVAL_SECONDS=300

DATABASE_URL=postgresql+psycopg://btc:btc@db:5432/btc_trader

BITFLYER_API_KEY=
BITFLYER_API_SECRET=

STRATEGY_NAME=none

MIN_PROFIT_PERCENT=1.0

MAX_POSITION_JPY=100000
MAX_ORDER_JPY=10000
MAX_DAILY_LOSS_JPY=5000

ENABLE_AUTOMATIC_TRADING=false

DASHBOARD_REFRESH_SECONDS=300

LOG_LEVEL=INFO
```

The important parameters must include:

```text
MARKET_DATA_INTERVAL_SECONDS
MIN_PROFIT_PERCENT
MAX_POSITION_JPY
MAX_ORDER_JPY
MAX_DAILY_LOSS_JPY
TRADING_MODE
ENABLE_AUTOMATIC_TRADING
STRATEGY_NAME
```

---

# 7. Secret Management

Never put the actual bitFlyer API key or secret inside source code.

Never commit credentials to Git.

Do not log credentials.

Production secrets should preferably come from:

- Docker secrets,
- environment injection,
- a secrets manager,
- or another secure runtime mechanism.

For local development `.env` may be supported, but `.env` must be included in `.gitignore`.

Provide:

```text
.env.example
```

containing placeholders only.

Live trading credentials should use the minimum permissions required.

bitFlyer's private API requires API-key/secret authentication, and bitFlyer explicitly warns against embedding or storing API secrets insecurely. Treat exchange credentials as high-value production secrets.

---

# 8. Exchange Abstraction

Create an exchange interface independent of bitFlyer.

Example:

```python
class ExchangeClient(ABC):

    @abstractmethod
    async def get_ticker(self, product_code: str) -> Ticker:
        ...

    @abstractmethod
    async def get_order_book(self, product_code: str) -> OrderBook:
        ...

    @abstractmethod
    async def get_recent_executions(
        self,
        product_code: str,
        limit: int = 100,
    ) -> list[Execution]:
        ...

    @abstractmethod
    async def get_balances(self) -> list[Balance]:
        ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        ...
```

Implement:

```text
BitflyerClient
```

behind this interface.

This allows future additions such as:

```text
KrakenClient
BinanceClient
CoinbaseClient
```

without changing strategy logic.

---

# 9. bitFlyer Market Data

Initially collect public market information for:

```text
BTC_JPY
```

Important sources include:

```text
ticker
order book
recent executions
exchange health / state where useful
```

At minimum, each scheduled collection must retrieve ticker data.

Store enough information to reconstruct historical BTC/JPY movement.

Suggested ticker fields:

```text
timestamp
product_code
ltp
best_bid
best_ask
best_bid_size
best_ask_size
total_bid_depth
total_ask_depth
volume
volume_by_product
```

The precise fields should follow the actual bitFlyer API response.

Do not fabricate unavailable fields.

---

# 10. Market Data Collection

Implement:

```python
async def collect_market_data() -> None:
    ...
```

Responsibilities:

1. Query current BTC/JPY market information.
2. Normalize exchange responses.
3. Store the observation.
4. Avoid duplicate inserts.
5. Log execution duration.
6. Record failures.
7. Retry transient network failures.
8. Never crash the scheduler because of one failed request.

Default frequency:

```text
300 seconds
```

The interval must be configurable.

Example:

```env
MARKET_DATA_INTERVAL_SECONDS=300
```

Changing this to:

```env
MARKET_DATA_INTERVAL_SECONDS=60
```

must result in one collection per minute without source-code modification.

---

# 11. Manual Data Collection

In addition to scheduled polling, support manual collection.

API:

```text
POST /api/v1/market-data/collect
```

The endpoint should trigger one immediate collection.

Alternatively provide CLI:

```bash
python -m btc_trader.scripts.collect_once
```

Both scheduled and manual collection must use the same collection service.

Do not duplicate collection logic.

---

# 12. Database Design

Use PostgreSQL.

Use SQLAlchemy models and Alembic migrations.

Suggested tables follow.

---

## 12.1 market_ticks

```text
id
exchange
product_code
exchange_timestamp
received_at

ltp
best_bid
best_ask

best_bid_size
best_ask_size

total_bid_depth
total_ask_depth

volume
volume_by_product

raw_payload
```

Indexes:

```text
(exchange, product_code, exchange_timestamp)
product_code + exchange_timestamp
```

Add a uniqueness constraint where appropriate.

---

## 12.2 executions

Store individual exchange market executions if execution-level history is collected.

```text
id
exchange
product_code
exchange_execution_id
side
price
size
executed_at
received_at
```

Unique:

```text
exchange + product_code + exchange_execution_id
```

This permits incremental ingestion without duplicates.

---

## 12.3 strategy_signals

```text
id
strategy_name
strategy_version
product_code
timestamp

signal
confidence

reference_price

metadata_json
```

Signal values:

```text
BUY
SELL
HOLD
```

---

## 12.4 orders

```text
id
client_order_id
exchange_order_id

exchange
product_code

side
order_type

requested_price
requested_size

status

strategy_name
signal_id

created_at
submitted_at
completed_at
updated_at

trading_mode

raw_response
```

---

## 12.5 fills

```text
id
order_id
exchange_execution_id

price
size
commission

executed_at
```

---

## 12.6 positions

For the initial spot system, maintain the effective BTC inventory and cost basis.

Suggested fields:

```text
id
product_code

btc_quantity
average_entry_price

cost_basis_jpy

realized_pnl_jpy
unrealized_pnl_jpy

updated_at
```

---

## 12.7 portfolio_snapshots

```text
id
timestamp

jpy_balance
btc_balance

btc_market_price

btc_value_jpy
total_value_jpy

realized_pnl_jpy
unrealized_pnl_jpy
```

These snapshots are useful for dashboard performance charts.

---

## 12.8 system_events

Store important operational events.

```text
id
timestamp
severity
component
event_type
message
metadata
```

Examples:

```text
collector_failed
api_rate_limited
order_rejected
risk_limit_triggered
trading_disabled
```

---

# 13. Numeric Precision

Never use binary floating point for monetary accounting.

Use:

```python
Decimal
```

for:

```text
JPY amounts
BTC quantities
prices
fees
P&L
```

Database numeric fields should use appropriate `NUMERIC`/`DECIMAL` types.

---

# 14. Scheduler

Use APScheduler initially.

Jobs:

```text
market_data_collection_job
strategy_evaluation_job
portfolio_snapshot_job
exchange_reconciliation_job
```

The market-data collection interval must be configurable.

Avoid overlapping executions.

For example, if a collection takes longer than its interval, a second copy of the same job should not start concurrently.

Implement:

```text
max_instances=1
coalesce=true
```

or equivalent behavior.

---

# 15. Dashboard

Provide a browser-based dashboard.

The first version should prioritize function over visual complexity.

Use:

```text
Streamlit + Plotly
```

unless another architecture is explicitly selected.

Dashboard sections:

## Market Overview

Display:

```text
BTC/JPY current price
best bid
best ask
spread
24-hour / stored-period change
latest data timestamp
collector status
```

---

## Price Chart

Interactive BTC/JPY price chart.

Support periods:

```text
1 hour
6 hours
24 hours
7 days
30 days
all
```

Initially plot:

```text
LTP vs time
```

Later add:

```text
bid
ask
volume
moving averages
strategy indicators
BUY markers
SELL markers
```

---

## Portfolio

Display:

```text
BTC balance
JPY balance
BTC market value
average BTC cost basis
realized P&L
unrealized P&L
total portfolio value
```

---

## Strategy

Display:

```text
active strategy
strategy version
latest signal
signal timestamp
signal confidence
automatic trading enabled/disabled
```

---

## Trading

Display:

```text
recent orders
recent executions
open orders
trade mode
paper/live status
```

---

## System

Display:

```text
collector last run
collector last successful run
API health
database health
scheduler status
last error
```

---

# 16. Dashboard Refresh

Default:

```text
300 seconds
```

Configuration:

```env
DASHBOARD_REFRESH_SECONDS=300
```

The UI should display:

```text
Last market update: YYYY-MM-DD HH:MM:SS
Next refresh: ...
```

The dashboard should read historical data from PostgreSQL instead of querying bitFlyer separately.

This prevents unnecessary exchange API requests.

---

# 17. Strategy Interface

Do not implement a real trading strategy yet.

Create the framework only.

Example:

```python
class TradingSignal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
```

Input:

```python
@dataclass
class StrategyContext:
    product_code: str
    market_data: pd.DataFrame
    position: Position
    portfolio: Portfolio
    timestamp: datetime
```

Output:

```python
@dataclass
class StrategyDecision:
    signal: TradingSignal
    confidence: Decimal | None
    reason: str
    metadata: dict
```

Interface:

```python
class TradingStrategy(ABC):

    name: str
    version: str

    @abstractmethod
    def evaluate(
        self,
        context: StrategyContext,
    ) -> StrategyDecision:
        ...
```

---

# 18. Example Placeholder Strategy

A placeholder strategy may exist solely for testing.

Example:

```text
AlwaysHoldStrategy
```

It must always return:

```text
HOLD
```

Do NOT silently introduce moving-average, RSI, momentum, ML, or other trading logic as the production strategy.

Those will be added later.

---

# 19. Strategy Registry

Allow strategies to be selected by configuration.

Example:

```env
STRATEGY_NAME=always_hold
```

Registry:

```python
STRATEGIES = {
    "always_hold": AlwaysHoldStrategy,
}
```

Future:

```text
moving_average
momentum
mean_reversion
RSI
ML classifier
reinforcement learning
custom strategy
```

---

# 20. Decision Pipeline

A strategy signal must NEVER directly submit an order.

Required pipeline:

```text
Market Data
    ↓
Strategy
    ↓
Signal
    ↓
Portfolio Manager
    ↓
Risk Engine
    ↓
Profitability Check
    ↓
Execution Engine
    ↓
Exchange
```

A SELL signal does not necessarily mean that the system may sell.

Likewise, a BUY signal does not necessarily mean the requested order is permitted.

---

# 21. Minimum Profit / Break-Even Requirement

The system must support a configurable minimum required profit before normal profit-taking sells.

Default:

```env
MIN_PROFIT_PERCENT=1.0
```

Meaning approximately:

```text
Do not execute a normal strategy SELL intended to realize profit
unless estimated net profit >= 1%.
```

However, this rule must NOT block emergency risk-management exits.

There must be a distinction between:

```text
STRATEGY_EXIT
RISK_EXIT
MANUAL_EXIT
```

The minimum-profit rule applies to:

```text
STRATEGY_EXIT
```

unless explicitly configured otherwise.

It does not prevent stop-loss, kill-switch, emergency liquidation, or manual overrides.

---

# 22. Profit Calculation

Do not calculate profit simply as:

```text
current_price / buy_price - 1
```

Use net economics.

Approximate calculation:

```text
entry_cost
    = BTC quantity × entry price
    + entry fees

estimated_exit_value
    = BTC quantity × expected exit price
    - estimated exit fees
    - estimated slippage

net_profit
    = estimated_exit_value - entry_cost

net_profit_percentage
    = net_profit / entry_cost × 100
```

A strategy SELL may execute only when:

```text
net_profit_percentage >= MIN_PROFIT_PERCENT
```

for ordinary profit-taking sells.

---

# 23. Spread Awareness

For estimated execution prices:

BUY should generally evaluate against:

```text
ask
```

SELL should generally evaluate against:

```text
bid
```

Do not evaluate profitability using LTP alone.

For example:

```text
entry estimate = ask
exit estimate = bid
```

This provides a more realistic estimate.

---

# 24. Position Cost Basis

Implement a well-defined cost basis.

For multiple BTC purchases, initially use:

```text
weighted average cost
```

Example:

```text
0.01 BTC @ ¥10,000,000
0.02 BTC @ ¥11,000,000
```

Average cost:

```text
(0.01 × 10,000,000 + 0.02 × 11,000,000)
------------------------------------------------
                    0.03
```

Include fees in cost basis where practical.

Later alternative accounting methods can be introduced if required.

---

# 25. Paper Trading

Paper trading should use the exact same pipeline as live trading except for the final exchange execution adapter.

Architecture:

```text
ExecutionEngine
       │
       ├── PaperExecutionAdapter
       │
       └── BitflyerExecutionAdapter
```

Paper trading should simulate:

```text
orders
fills
BTC balances
JPY balances
fees
P&L
```

Optional simulated slippage:

```env
PAPER_SLIPPAGE_BPS=5
```

---

# 26. Live Trading Safety

Live trading must require BOTH:

```env
TRADING_MODE=live
ENABLE_AUTOMATIC_TRADING=true
```

If either condition is missing:

```text
NO LIVE ORDER MAY BE SENT
```

Add another explicit safety mechanism if appropriate:

```env
LIVE_TRADING_ACKNOWLEDGED=true
```

---

# 27. Risk Management

Risk management must be independent of strategy code.

Implement configurable limits.

At minimum:

```env
MAX_ORDER_JPY=10000
MAX_POSITION_JPY=100000
MAX_DAILY_LOSS_JPY=5000
```

Potential future controls:

```text
MAX_TRADES_PER_DAY
MAX_TRADES_PER_HOUR
MAX_BTC_POSITION
MAX_DRAWDOWN_PERCENT
COOLDOWN_AFTER_TRADE_SECONDS
MAX_CONSECUTIVE_LOSSES
```

---

# 28. Kill Switch

Implement a global trading kill switch.

Possible configuration:

```env
TRADING_KILL_SWITCH=false
```

If true:

```text
new orders = forbidden
```

Market-data collection and dashboard operation should continue.

Expose status prominently in the dashboard.

---

# 29. Stale Data Protection

Never trade using stale data.

Example configuration:

```env
MAX_MARKET_DATA_AGE_SECONDS=600
```

Before any order:

```python
if now - latest_market_timestamp > max_age:
    reject_trade()
```

Record:

```text
reason = stale_market_data
```

---

# 30. Duplicate Order Protection

Prevent duplicate order submission.

Every intended order should have an internal unique:

```text
client_order_id
```

Before submitting:

1. verify that an identical pending order does not already exist;
2. persist the intended order;
3. submit;
4. persist the exchange acceptance ID;
5. reconcile status.

Retries must not accidentally create duplicate live orders.

---

# 31. Order State Machine

Use explicit order states.

Example:

```text
CREATED
VALIDATED
REJECTED
SUBMITTING
SUBMITTED
PARTIALLY_FILLED
FILLED
CANCEL_PENDING
CANCELLED
FAILED
UNKNOWN
```

Never assume that an HTTP response alone means an order has completed.

Reconcile against exchange order/execution information.

---

# 32. Exchange Reconciliation

Periodically reconcile local state with bitFlyer.

Check:

```text
balances
orders
executions
```

Purpose:

```text
detect partial fills
detect external/manual trades
recover after restart
correct stale order status
```

The database should ultimately reflect exchange truth for live trading.

---

# 33. API Rate Limiting

All bitFlyer requests must pass through a centralized API client.

Do not allow individual modules to make uncontrolled HTTP requests.

Implement:

```text
rate limiter
retry logic
backoff
timeout
```

Handle:

```text
429
5xx
network timeout
connection failure
```

Retries must use exponential backoff with jitter.

Trading requests need stricter idempotency handling than read requests.

---

# 34. API Endpoints

FastAPI should expose internal application endpoints.

Suggested endpoints:

```text
GET  /health

GET  /api/v1/market/latest
GET  /api/v1/market/history

POST /api/v1/market/collect

GET  /api/v1/portfolio

POST /api/v1/trading/paper/reset  (body: {"confirm": true})

GET  /api/v1/signals

GET  /api/v1/orders

GET  /api/v1/trades

GET  /api/v1/system/status
```

Trading-related manual endpoints may later include:

```text
POST /api/v1/trading/enable
POST /api/v1/trading/disable
```

Do not expose live-order endpoints publicly without authentication.

---

# 35. Logging

Use structured logging.

Prefer JSON logs in deployed environments.

Every important event should include relevant context.

Example:

```json
{
  "event": "market_data_collected",
  "product_code": "BTC_JPY",
  "ltp": "...",
  "duration_ms": 125,
  "timestamp": "..."
}
```

Never log:

```text
API secret
authorization signatures
private credentials
```

---

# 36. Trading Audit Trail

Every trading decision should be explainable later.

Store:

```text
market state
strategy
strategy version
strategy decision
reason
risk checks
profitability calculation
requested order
exchange response
fills
resulting position
```

The system should make questions such as the following answerable:

```text
Why did the bot buy BTC at this time?

Why did it refuse a SELL signal?

Which strategy version generated this order?

What market price was observed?

What profit percentage was calculated?

What risk rule permitted/rejected the order?
```

---

# 37. Error Handling

Do not use:

```python
except Exception:
    pass
```

Errors must:

1. be logged;
2. be categorized;
3. preserve context;
4. avoid crashing unrelated components.

Create domain exceptions such as:

```text
ExchangeError
ExchangeAuthenticationError
ExchangeRateLimitError
MarketDataError
RiskViolation
OrderSubmissionError
StaleMarketDataError
```

---

# 38. Time Handling

All backend timestamps must be stored in:

```text
UTC
```

Dashboard may display:

```text
Asia/Tokyo
```

Never store naïve datetimes.

---

# 39. Historical Data

The periodic collector will build historical data incrementally.

Support an optional backfill mechanism.

Command:

```bash
python -m btc_trader.scripts.backfill
```

However, because exchange historical availability may vary, the backfill implementation must depend only on data actually available from bitFlyer.

Do not fabricate candle history.

If OHLCV candles are required later, build them from execution/tick data or explicitly introduce another data provider.

---

# 40. Candle Aggregation

Provide an internal service capable of aggregating stored observations into OHLCV-like intervals where adequate input data exists.

Possible intervals:

```text
1m
5m
15m
1h
4h
1d
```

For proper OHLCV, use trade execution data where possible.

Do not pretend sampled LTP observations represent true exchange OHLC unless clearly marked as sampled candles.

---

# 41. Docker Deployment

Provide:

```text
Dockerfile
docker-compose.yml
```

Initial services:

```yaml
services:
  api:
  collector:
  dashboard:
  db:
```

Optionally the API and collector may initially share one application container if operationally simpler.

Recommended logical layout:

```text
btc-api
btc-worker
btc-dashboard
postgres
```

---

# 42. Docker Volumes

Database must use persistent storage.

Example:

```text
postgres_data
```

Restarting containers must not erase historical trading data.

---

# 43. Health Checks

Docker Compose should define health checks.

Application endpoint:

```text
GET /health
```

Expected data:

```json
{
  "status": "ok",
  "database": "ok",
  "scheduler": "ok",
  "exchange_public_api": "ok"
}
```

Do not require private API credentials merely to consider the data collector healthy.

---

# 44. Home Server Deployment

The system must run with approximately:

```bash
git clone ...
cp .env.example .env

docker compose up -d
```

After configuration, the user should be able to access:

```text
Dashboard
API
```

through configured ports.

Do not require Kubernetes.

---

# 45. Cloud Deployment

The same containers should work on:

```text
AWS
GCP
Azure
Oracle Cloud
DigitalOcean
generic Linux VPS
```

Cloud-specific dependencies should be optional.

---

# 46. Testing Requirements

Use pytest.

Testing should include:

## Unit tests

```text
profit calculations
fee calculations
cost basis
strategy interface
risk rules
stale-data checks
position accounting
config validation
```

## Integration tests

```text
database
repositories
scheduler
FastAPI endpoints
exchange response parsing
```

## Exchange tests

Mock bitFlyer.

Live API calls must not occur during normal tests.

---

# 47. Critical Profitability Tests

Test at least:

```text
price below cost basis
price exactly at cost basis
gross profit < 1%
gross profit = 1%
gross profit > 1%
net profit < 1% after fees
net profit > 1% after fees
spread causes profit to fall below threshold
```

Use `Decimal`.

---

# 48. Trading Safety Tests

Test:

```text
live mode disabled
automatic trading disabled
kill switch enabled
stale market data
maximum order exceeded
maximum position exceeded
daily loss limit exceeded
duplicate order
missing exchange credentials
exchange unavailable
partial fill
```

Every test must prove that an unsafe order is NOT submitted.

---

# 49. Code Quality

Use:

```text
ruff
mypy
pytest
```

Prefer type hints throughout the business logic.

Use Pydantic models at external boundaries.

Avoid unnecessary design-pattern complexity.

Prefer small services with explicit responsibilities.

---

# 50. Development Phases

Development must proceed incrementally.

Do NOT implement everything at once.

---

## Phase 1 — Project Foundation

Implement:

```text
Python project
configuration
Docker
PostgreSQL
SQLAlchemy
Alembic
logging
health endpoint
```

Acceptance:

```text
docker compose up
```

starts application and database successfully.

---

## Phase 2 — bitFlyer Public Market Data

Implement:

```text
BitflyerClient
ticker retrieval
order book retrieval
recent executions retrieval
normalization
```

Initially no private API is necessary.

Acceptance:

```text
BTC_JPY current market data can be fetched manually.
```

---

## Phase 3 — Persistent Market Data

Implement:

```text
market_ticks table
executions table
repositories
incremental inserts
deduplication
```

Acceptance:

Repeated execution stores only new observations where appropriate.

---

## Phase 4 — Scheduling

Implement:

```text
default 5-minute polling
configurable interval
manual collection endpoint
retries
failure logging
```

Acceptance:

With:

```env
MARKET_DATA_INTERVAL_SECONDS=300
```

data is collected approximately every five minutes.

Changing the value changes the frequency without code modification.

---

## Phase 5 — Dashboard

Implement:

```text
current BTC/JPY
historical price chart
bid/ask
spread
volume where available
collector status
```

Acceptance:

Dashboard reflects newly collected database data automatically.

---

## Phase 6 — Strategy Framework

Implement only:

```text
TradingStrategy interface
StrategyContext
StrategyDecision
AlwaysHoldStrategy
strategy registry
signal persistence
```

No production strategy yet.

Acceptance:

The strategy engine evaluates stored data and generates HOLD.

---

## Phase 7 — Portfolio / P&L

Implement:

```text
position tracking
weighted average cost
realized P&L
unrealized P&L
fee-aware calculations
minimum-profit validation
```

Acceptance:

Unit tests validate calculations.

---

## Phase 8 — Paper Trading

Implement:

```text
BUY
SELL
HOLD
simulated balance
simulated orders
simulated fills
fees
slippage
P&L
```

Acceptance:

The complete trading pipeline runs without contacting bitFlyer private trading endpoints.

---

## Phase 9 — Risk Engine

Implement:

```text
max order
max position
daily loss
kill switch
stale-data protection
duplicate-order protection
```

Acceptance:

Risk engine can block proposed orders.

---

## Phase 10 — bitFlyer Private API

Only after the previous phases are stable.

Implement:

```text
authentication
balances
private executions
orders
commission lookup where useful
order submission
cancellation
reconciliation
```

Acceptance:

Private read-only functionality works first.

Do not enable real order execution yet.

---

## Phase 11 — Controlled Live Trading

Add live execution.

Initially require:

```text
very small configurable trade size
explicit live mode
explicit automatic-trading enablement
kill switch
full audit logging
```

Acceptance:

Only one deliberately initiated controlled trade should be tested first.

---

# 51. Initial Coding-Agent Instructions

When implementing this project, follow these rules:

1. Read this entire PROJECT.md before modifying code.

2. Implement the project phase-by-phase.

3. Do not jump directly to live trading.

4. Do not add a trading strategy unless explicitly requested.

5. Use `AlwaysHoldStrategy` until a real strategy is provided.

6. Never hardcode API credentials.

7. Never commit `.env`.

8. Never log secrets.

9. Use `Decimal` for prices, quantities, fees, and P&L.

10. Store timestamps in UTC.

11. Keep exchange access behind `ExchangeClient`.

12. Keep strategy logic independent from bitFlyer.

13. Keep risk logic independent from strategy logic.

14. A strategy signal must never directly send an exchange order.

15. Every order must pass risk validation.

16. Ordinary strategy-driven SELL orders must pass the configured net-profit requirement.

17. Emergency/risk exits must be able to override the minimum-profit rule.

18. Default to `data_only` or `paper` mode.

19. Never enable live trading automatically.

20. Write tests whenever implementing financial calculations or risk rules.

21. Use database migrations rather than manually changing database schemas.

22. Make background jobs idempotent.

23. Prevent scheduler jobs from overlapping.

24. Handle API rate limiting.

25. Use exponential backoff for transient read failures.

26. Do not blindly retry order submissions.

27. Reconcile live-order state against bitFlyer.

28. Maintain an audit trail for every automated trading decision.

29. Prefer simple, explicit architecture over premature microservices.

30. Keep everything runnable through Docker Compose.

---

# 52. First Implementation Task

The first coding task should NOT involve trading.

Implement only:

```text
Phase 1
+
Phase 2
+
Phase 3
+
Phase 4
```

Specifically:

```text
1. Initialize Python project.
2. Create Docker environment.
3. Create PostgreSQL database.
4. Implement SQLAlchemy/Alembic.
5. Implement configuration.
6. Implement BitflyerClient public methods.
7. Fetch BTC_JPY ticker.
8. Fetch BTC_JPY order book.
9. Fetch recent BTC_JPY executions.
10. Store market observations.
11. Prevent duplicates.
12. Add scheduler.
13. Default scheduler interval to 300 seconds.
14. Make interval configurable.
15. Add manual collection command/API.
16. Add structured logging.
17. Add tests.
```

Stop there.

Do not build automated trading yet.

---

# 53. Definition of Done for Initial Version

The first useful version is complete when:

1. `docker compose up -d` successfully starts the application.

2. PostgreSQL is persistent.

3. BTC/JPY data can be fetched from bitFlyer.

4. Data is automatically fetched every 5 minutes by default.

5. The interval can be changed through configuration.

6. Collection can be manually triggered.

7. Historical data accumulates without duplicate corruption.

8. A dashboard shows BTC/JPY movement.

9. Dashboard refresh interval is configurable.

10. Exchange errors do not terminate the service.

11. No API secrets exist in source code.

12. No real trades can accidentally occur.

---

# 54. Future Strategy Work

The trading strategy will be designed separately.

Possible future research areas include:

```text
moving-average crossover
momentum
mean reversion
RSI / technical signals
breakout strategies
volatility strategies
order-book signals
statistical models
supervised ML
time-series models
reinforcement learning
ensemble strategies
```

Do not select one automatically.

Before enabling any strategy for live trading, implement:

```text
historical backtesting
transaction fees
spread
slippage
walk-forward evaluation
paper trading
risk constraints
```

---

# 55. Future Architecture Extensions

The architecture should make these extensions possible without major redesign:

```text
multiple trading pairs
multiple exchanges
WebSocket market data
Redis caching
Celery / distributed workers
Prometheus metrics
Grafana
Telegram/Slack alerts
backtesting engine
strategy experiment tracking
multiple strategies
portfolio optimization
WebSocket dashboard updates
Kubernetes deployment
```

None of these are required initially.

---

# 56. Security Boundary

The dashboard and API may eventually have the ability to affect real money.

Therefore, when live trading exists:

```text
Do not expose it directly to the public Internet without authentication.
```

For home-server use, prefer:

```text
local network
VPN
Tailscale
reverse proxy with authentication
```

over unrestricted public exposure.

---

# 57. Core Philosophy

Build this system in this order:

```text
Observe
    ↓
Store
    ↓
Visualize
    ↓
Analyze
    ↓
Simulate
    ↓
Risk-control
    ↓
Execute
```

Never reverse that sequence.

The first objective is a reliable BTC/JPY market-data platform.

Automated trading is a later capability built on top of that reliable foundation.
