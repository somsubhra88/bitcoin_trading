# Strategy research: why the platform trades daily trends

This note records the evidence behind the strategy change. Every number here
comes from real BTC/JPY prices with bitFlyer's cost structure. No result was
used to tune parameters after the fact.

## 1. The cost hurdle

bitFlyer Lightning spot charges 0.15% per side for accounts with less than
¥100k of 30-day volume. The fee drops with volume, and maker and taker pay the
same rate ([bitFlyer fees](https://bitflyer.com/en-jp/s/commission)). Add a
quoted BTC_JPY spread of about 3–4 bps and some slippage, and **one round trip
costs about 0.35–0.40%**. Any signal has to predict moves larger than that.

## 2. The retired 5-minute strategies lost money on real data

The previous strategies (`ema_crossover`, `mean_reversion`,
`momentum_regime`) were replayed with their own code and default parameters.
The data was one year of real BTC/JPY 5-minute bars (bitbank, 2025-11 →
2026-11, 105,016 bars). Each replay used the paper exit rules (1% minimum
profit, 0.8% trailing stop, 1.5% hard stop), 25% allocation, 0.15% fee, 5 bps
slippage and a 1.8 bps half spread.

| Strategy        | Trades | Win rate | Avg trade | Total return | Max drawdown |
|-----------------|-------:|---------:|----------:|-------------:|-------------:|
| ema_crossover   |    791 |   21.7%  |  −0.46%   |     −59.6%   |     60.0%    |
| mean_reversion  |  1,723 |   23.0%  |  −0.44%   |     −84.9%   |     85.1%    |
| momentum_regime |  1,310 |   22.2%  |  −0.44%   |     −76.1%   |     76.5%    |
| buy & hold      |      1 |     —    |     —     |     −17.6%   |       —      |

The average loss per trade is almost exactly one round-trip cost. The signals
carry no edge at this horizon, so every trade pays the spread and fees. This
matches the literature: order-flow and short-horizon technical signals in
BTC do not survive realistic fees
([Svogun & Bazán-Palomino 2022](https://www.sciencedirect.com/science/article/abs/pii/S1042443122000816);
a cost-aware walk-forward study of hourly BTC models, [arXiv 2606.00060](https://arxiv.org/html/2606.00060v1)).
The DeepLOB-style model also forecast a 15-minute horizon, and the same
argument applies to it. It was **not** replayed: there is no public historical
order-book data for bitFlyer.

## 3. What the evidence supports

* **Slow time-series momentum** works in BTC and survives retail costs
  ([Hudson & Urquhart 2021](https://link.springer.com/article/10.1007/s10479-019-03357-1);
  the Monash trend-following study for crypto investors). Its main benefit is
  much smaller drawdowns. Hudson & Urquhart also warn that BTC's
  out-of-sample *return* edge is weaker than its history suggests.
* **Volatility scaling** improves risk-adjusted returns because volatility is
  predictable and expected returns do not rise to compensate for it
  ([Moreira & Muir 2017](https://www.nber.org/papers/w22208)).
* **Zarattini, Pagani & Barbon (2025)**, *Catching Crypto Trends* (SFI
  25-80), combine both ideas. Nine Donchian-channel sleeves run on daily
  closes (5…360 days). A sleeve enters at a new N-day high, starts its
  trailing stop at the channel midpoint, ratchets that stop up only, and
  exits on a close below it. Each sleeve is sized to a 25% volatility target
  with a 20% no-trade band. An [independent replication](https://github.com/LineGM/catching_crypto_trends)
  reproduces the paper's BTC results.

The platform implements that model (`src/btc_trader/strategy.py`) with the
published parameters, adapted to a spot account: long or cash only, total
exposure at most 100%.

## 4. Results on BTC/JPY

The data was bitbank daily BTC/JPY from 2018-01-01 to 2026-11-04, simulated
through the platform's own backtester. Decisions use a day's close and fill at
the next day's open. Costs were 0.15% fee plus 5 bps slippage per side.

| 2018-04 → 2026-11       | CAGR   | Volatility | Sharpe | Max drawdown | Avg exposure |
|-------------------------|-------:|-----------:|-------:|-------------:|-------------:|
| Trend ensemble, 25% vol | +16.4% |   14.5%    |  1.12  |    19.1%     |     20%      |
| Trend ensemble, 50% vol | +30.2% |   26.1%    |  1.14  |    35.2%     |     35%      |
| Buy & hold              | +39.6% |   61.3%    |  0.85  |    71.5%     |    100%      |

Calendar years at the default 25% target:

| Year | Strategy | Buy & hold | Strategy max DD | Buy & hold max DD |
|-----:|---------:|-----------:|----------------:|------------------:|
| 2018 |   −2.3%  |   −44.3%   |       4.9%      |       66.5%       |
| 2019 |  +34.6%  |   +91.7%   |       9.8%      |       48.1%       |
| 2020 |  +54.5%  |  +280.8%   |       7.6%      |       53.9%       |
| 2021 |  +17.6%  |   +79.3%   |      10.3%      |       52.7%       |
| 2022 |  −10.2%  |   −59.1%   |      10.5%      |       62.8%       |
| 2023 |  +31.1%  |  +175.4%   |       9.8%      |       18.1%       |
| 2024 |  +39.9%  |  +145.9%   |      11.2%      |       30.7%       |
| 2025 |   −7.9%  |    −7.1%   |      11.8%      |       32.6%       |
| 2026 |   +1.8%  |    −6.1%   |       5.3%      |       38.0%       |

Cost stress at the 25% target: 0.5x costs gives Sharpe 1.17, 1x gives 1.12,
2x gives 1.02 and 4x gives 0.81. The edge does not depend on optimistic fills.
Dropping any single lookback changes the Sharpe ratio by at most ±0.02, so no
single sleeve carries the result.

**Out-of-sample check.** The paper's data ends on 2025-03-19. From 2025-03-20
to 2026-11-04 the ensemble returned about 0% (Sharpe 0.04) with a 15% max
drawdown. Buy & hold returned −0.5% with a 49% drawdown. Since publication,
the return edge is **zero**; only the risk reduction has held up.

## 5. Plain conclusions

* The strategy is a **risk-managed way to hold Bitcoin**, not a source of
  excess return. It gave up much of BTC's upside and cut drawdowns by about
  70%. The most recent 19 months produced no return advantage.
* Expect about 100 small rebalances a year, typically 1–3 orders a week.
  Fees were about 1.5% of equity per year at the default settings.
* With ¥1,000,000 of paper capital, single orders sit near bitFlyer's 0.001 BTC
  minimum lot. Much smaller accounts will see entries blocked; the paper ledger
  records why.
* Paper-trade it for weeks before going live, and start with a small
  `LIVE_CAPITAL_JPY`.

## 6. Execution research for unattended live trading

These facts come from bitFlyer's API documentation and a read-only check
against a real account (November 2026). They drive `live.py`:

| Fact | Consequence in the code |
|---|---|
| Spot commission is 0.15% for small accounts (`gettradingcommission` returned `0.0015`). It falls with 30-day volume, and BTC/JPY commission is charged **in BTC** | The live planner reads the real rate each time. Sells leave `available / (1 + fee)` so a BTC-denominated commission cannot fail the order. Fills store commission × price in JPY |
| Minimum order 0.001 BTC, size step 1e-8, 1 JPY price tick | Lot rounding is shared with paper and backtest. Limit prices are rounded away from the touch |
| No client order id. `sendchildorder` returns an acceptance id | The order is committed as `SUBMITTING` before sending and never retried. A lost reply is reconciled by matching side, size and time among recent orders |
| Orders that never executed are **not listed** by `getchildorders` | An accepted order that is unlisted after its 1-minute expiry plus margin is settled from its fills, normally none |
| `getboardstate` reports health (`NORMAL`, `BUSY`, `VERY BUSY`, `SUPER BUSY`, `NO ORDER`, `STOP`) and board state (`RUNNING`, `CLOSED`, `CIRCUIT BREAK`, ...) | Orders are sent only while `RUNNING` and at most `VERY BUSY`. (The old parser mapped `VERY BUSY` to `UNKNOWN`.) |
| Daily maintenance 04:00–04:10 JST, followed by a 2-minute itayose auction | No orders from 03:55 to 04:15 JST |
| Limits: 500 private requests and 300 order requests per 5 minutes | One evaluation uses about 5 requests and at most one order, every 5 minutes |
| Observed book: 1–4 bps spread. A 0.5 BTC market order costs ≤ 6 bps of impact | A LIMIT IOC 30 bps through the touch fills normal-size orders in full and caps flash-crash slippage |

**Timing of the daily close.** Re-cutting the same 4-hour bitbank data so that
"days" end at different UTC hours gave Sharpe ratios between 1.06 and 1.27.
Every variant beat buy & hold (0.84–0.87). The best hour (16:00 UTC) is almost
certainly noise, so the platform keeps the conventional UTC midnight rather
than tuning it.

## Reproduce

```bash
python -m btc_trader.scripts.backfill_daily --years 9      # bitbank daily candles
curl "localhost:8000/api/v1/backtest?strategy_name=trend_ensemble"
curl "localhost:8000/api/v1/backtest?strategy_name=trend_ensemble&start=2025-03-20"
curl "localhost:8000/api/v1/backtest?strategy_name=trend_ensemble&cost_multiplier=2"
```
