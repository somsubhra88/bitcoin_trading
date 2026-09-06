You are a senior quantitative researcher and ML systems engineer. Build a reproducible, research-grade Bitcoin trading system inspired by ArchetypeTrader: learn reusable strategy archetypes from hindsight-optimal training trajectories, train a high-level policy to select an archetype for each market horizon, then train a constrained low-level adapter that can refine the selected policy online. The result is for research and paper trading, not a claim of profitability.

OBJECTIVE
Implement an end-to-end repository for BTC/USDT using 10-minute data. Support:
1. Spot mode: target position {0, +1}.
2. Perpetual-futures mode: target position {-1, 0, +1}, configurable leverage and funding.
3. OHLCV-only input as a baseline.
4. Optional level-2 order-book snapshots, preferably 25 levels, for the full model.

Use Python 3.11+, PyTorch, Gymnasium, NumPy, Polars or pandas, Hydra or typed YAML configuration, Typer/Click CLI, pytest, and deterministic seeds. Do not make the project notebook-only.

NON-NEGOTIABLE RESEARCH RULES
- Every feature, label, reward, fill, and action at time t must use only information available by t.
- Decide on a completed bar and execute no earlier than the next bar/open or the next observable bid/ask.
- Fit scalers, feature selectors, codebooks, and models on training data only.
- Use chronological, purged walk-forward splits with an embargo at least as long as the maximum lookback plus trading horizon.
- Never tune on the final test set.
- Include fees, spread, slippage, funding, latency, partial fills, and rejected or stale orders where applicable.
- Add assertions and tests that deliberately fail on look-ahead leakage, timestamp misalignment, impossible fills, or broken portfolio accounting.

DATA PIPELINE
Create adapters for local CSV/Parquet files and an optional exchange downloader for public historical data. Store UTC timestamps, validate monotonicity, deduplicate records, identify gaps, and produce a data-quality report. Never hard-code credentials.

Create causal features:
- OHLCV: log returns, ranges, gaps, volume changes, rolling realized volatility, ATR, RSI, MACD, moving-average distances, momentum, downside volatility, and time-of-day/day-of-week encodings.
- Order book: spread, microprice, imbalance at several depths, depth slope, weighted midprice, order-flow imbalance, and liquidity/volatility measures.
- Portfolio state: current target and filled position, cash/equity, unrealized PnL, drawdown, recent turnover, elapsed holding time, and remaining risk budget.

Make lookbacks and feature sets configurable. Save feature definitions, fitted transformations, data hashes, and split boundaries.

TRADING ENVIRONMENT
Implement a Gymnasium environment with:
- Observation: causal market window plus portfolio state.
- Action: target position, not a direct buy/sell instruction.
- Execution simulator: next-observation execution, configurable maker/taker fees, bid-ask spread, square-root or volume-participation slippage, latency, minimum quantity/notional, funding, and optional order-book walking.
- Reward: net log-equity change minus configurable turnover, downside-risk, drawdown, and inventory penalties. Record unshaped net PnL separately so reward shaping cannot disguise poor economics.
- Bankruptcy, liquidation, stale-data, and end-of-episode handling.
- A reconciled ledger where equity equals cash plus marked inventory minus liabilities and costs.

ARCHETYPE PHASE 1: DEMONSTRATIONS AND VECTOR-QUANTIZED POLICY LEARNING
Use only training intervals. Sample roughly 30,000 configurable trajectories of horizon h=72 ten-minute steps. Implement a dynamic-programming planner that finds the best fee-aware action sequence under a paper-style “at most one position change” constraint. Also implement a configurable two-change entry/exit version as an ablation. The planner is training-only and must never run during validation, testing, paper trading, or live inference.

Train an LSTM/GRU encoder, vector-quantization codebook, and causal autoregressive decoder:
- Default K=10 archetypes, code dimension 16, hidden dimension 128.
- Encoder maps a demonstration trajectory to a discrete code.
- Decoder receives the selected code and only currently available observations, then predicts the action sequence autoregressively.
- Loss includes action reconstruction, codebook, and commitment terms.
- Detect codebook collapse using utilization, perplexity, dead-code counts, and action diversity.
- Save representative trajectories and human-readable statistics for every archetype: return distribution, trade direction, turnover, volatility regime, drawdown, holding pattern, and feature profile.

ARCHETYPE PHASE 2: HIGH-LEVEL SELECTOR
Freeze or optionally fine-tune the Phase-1 decoder. Train a categorical recurrent PPO selector that, at the start of each horizon, chooses one archetype from the codebook using only a pre-horizon market lookback and portfolio state. Keep the selector algorithm behind an interface so recurrent DQN or another discrete actor-critic can be substituted.

The horizon reward is realized net log return, including all costs and risk penalties. Add a configurable auxiliary cross-entropy/KL term toward the training demonstration’s archetype label. Include an explicit cash/no-trade archetype or abstention action so the agent is never forced to trade. Log archetype probabilities, realized outcomes, regime statistics, and switching frequency.

ARCHETYPE PHASE 3: CONSTRAINED POLICY ADAPTER
Train a step-level adapter that observes current causal state, selected archetype, decoder action, realized-versus-base cumulative return, and steps remaining. It may retain the base action or apply one override per horizon; make the override budget configurable.

During training only, use dynamic programming to calculate the best hindsight override and a regret-aware target. Optimize:
- Actual improvement over the unmodified archetype.
- Penalty for regret relative to the best training-only hindsight adaptation.
- Transaction-cost, turnover, drawdown, and downside-risk penalties.
- Optional imitation loss toward the best adaptation.

At inference, remove all hindsight labels and planners. Add tests proving the deployed path cannot access future arrays.

RISK SHIELD
Place a deterministic shield outside the learned policy:
- Maximum gross/net exposure and leverage.
- Volatility targeting and position scaling.
- Maximum daily loss and portfolio drawdown.
- Cooldown after a stop or repeated losses.
- No new order when data are stale, spread is extreme, liquidity is inadequate, or estimated slippage exceeds a threshold.
- Liquidation-distance checks for perpetuals.
- Kill switch and forced flattening.

Log both proposed and executed actions so shield intervention rates are measurable.

TRAINING AND MODEL SELECTION
Use at least 10 random seeds for serious experiments, mixed precision where safe, gradient clipping, checkpointing, and early stopping. Select hyperparameters only on purged validation folds using a predeclared objective such as median validation Calmar or a weighted combination of net return, Sharpe, CVaR, and turnover. Save exact configurations, package versions, seeds, git commit, model hashes, and training curves.

EVALUATION
Run anchored or rolling walk-forward evaluation. Compare against:
- Cash and buy-and-hold.
- EMA/MACD and volatility-targeted trend baselines.
- Supervised return forecast plus threshold policy.
- Recurrent PPO.
- DQN for discrete actions.
- SAC or TQC for a continuous-position ablation.
- Phase 1 only, Phase 1+2, full Phase 1+2+3, and all major loss/risk-shield ablations.

Report total and annualized return, annualized volatility, Sharpe, Sortino, Calmar, maximum drawdown, 5% CVaR, turnover, fees, funding, trade count, hit rate, average win/loss, exposure, leverage, liquidation events, and shield intervention rate. Use stationary/block bootstrap confidence intervals and report seed distributions, not only the best seed.

Stress-test fees, slippage, latency, spread, and funding at 0.5x, 1x, 2x, and 4x assumptions. Plot equity, underwater curve, rolling risk metrics, positions, gross versus net PnL, cost attribution, archetype usage, codebook health, and regime-conditioned results.

Reject the strategy as non-robust when performance depends on one seed, one test segment, unrealistic fills, omitted costs, excessive turnover, or collapses under modest friction increases. State negative findings plainly.

DELIVERABLES
Produce:
1. A clear repository tree.
2. Complete source files, not pseudocode.
3. Configurations for a fast synthetic smoke test and a full BTC experiment.
4. CLI commands for ingest, features, Phase 1, Phase 2, Phase 3, evaluation, and paper trading.
5. Unit, property, integration, leakage, accounting, and determinism tests.
6. A README describing assumptions, data schema, commands, expected outputs, and known limitations.
7. A Dockerfile and lockfile.
8. A paper-trading adapter with idempotent order handling, reconciliation, retries, structured logs, and metrics.

Keep live execution disabled by default. Require a separate explicit --enable-live flag, environment-based secrets, exchange-specific safety checks, and a prominent warning. Begin by stating architecture decisions and assumptions, then show the repository tree, then provide the implementation file by file, followed by commands, tests, and limitations. Do not claim or imply guaranteed returns.
