"""Strategy selection/evaluation without granting any order authority."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from btc_trader.config import Settings
from btc_trader.db.models import StrategySignal
from btc_trader.deeplob import evaluate_order_book, parameters_from_settings
from btc_trader.market_data.repository import MarketDataRepository
from btc_trader.paper import execute_signal
from btc_trader.strategy import STRATEGY_VERSION, StrategyParameters, evaluate
from btc_trader.utils.timeutils import utcnow


async def evaluate_and_store(session: AsyncSession, settings: Settings) -> StrategySignal | None:
    # The database-backed effective settings are the cross-process source of
    # truth across API and worker processes.
    name = settings.strategy_name.strip().lower()
    if name == "none" or not settings.strategies_enabled:
        return None
    repository = MarketDataRepository(session)
    if name == "deeplob":
        snapshots = await repository.order_books_between(
            settings.product_code,
            limit=max(
                settings.strategy_deeplob_training_window + settings.strategy_deeplob_window,
                500,
            ),
        )
        decision = evaluate_order_book(snapshots, parameters=parameters_from_settings(settings))
        ticks = await repository.ticks_between(settings.product_code, limit=10)
    else:
        history_limit = max(
            settings.strategy_slow_window,
            settings.strategy_mean_reversion_window,
            settings.strategy_momentum_window,
            settings.strategy_volatility_window,
        ) * 2
        ticks = await repository.ticks_between(settings.product_code, limit=history_limit)
        decision = evaluate(
            name,
            ticks,
            parameters=StrategyParameters(
                fast_window=settings.strategy_fast_window,
                slow_window=settings.strategy_slow_window,
                mean_reversion_window=settings.strategy_mean_reversion_window,
                momentum_window=settings.strategy_momentum_window,
                volatility_window=settings.strategy_volatility_window,
                ema_threshold_percent=settings.strategy_ema_threshold_percent,
                mean_reversion_zscore_threshold=settings.strategy_mean_reversion_zscore_threshold,
                momentum_threshold_percent=settings.strategy_momentum_threshold_percent,
                edge_multiple=settings.strategy_edge_multiple,
                regime_filter=settings.strategy_regime_filter,
            ),
        )
    if decision is None:
        return None
    signal = StrategySignal(
        strategy_name=name,
        strategy_version=STRATEGY_VERSION,
        product_code=settings.product_code,
        timestamp=utcnow(),
        signal=decision.signal,
        confidence=decision.confidence,
        reference_bid=next(
            (tick.best_bid for tick in reversed(ticks) if tick.best_bid is not None), None
        ),
        reference_ask=next(
            (tick.best_ask for tick in reversed(ticks) if tick.best_ask is not None), None
        ),
        reference_price=decision.price,
        reason=decision.reason,
        metadata_json=decision.metadata,
    )
    session.add(signal)
    await session.flush()
    await execute_signal(session, settings, signal)
    return signal
