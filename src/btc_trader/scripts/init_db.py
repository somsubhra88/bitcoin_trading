"""Apply database migrations (PROJECT.md §51.21).

    python -m btc_trader.scripts.init_db

Runs ``alembic upgrade head`` programmatically so containers can migrate on
start-up without needing the Alembic CLI on PATH.  Schema changes always go
through migrations — never ``create_all`` in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic import command
from alembic.config import Config

from btc_trader.config import get_settings
from btc_trader.monitoring.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _project_root() -> Path:
    """Repository root: the directory containing ``alembic.ini``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "alembic.ini").is_file():
            return candidate
    raise FileNotFoundError("alembic.ini not found in any parent directory")


def alembic_config() -> Config:
    root = _project_root()
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    # env.py reads the URL from Settings; this keeps `alembic.ini` credential-free.
    config.set_main_option("sqlalchemy.url", get_settings().sync_database_url)
    return config


def upgrade(revision: str = "head") -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    logger.info("running_migrations", revision=revision)
    command.upgrade(alembic_config(), revision)
    logger.info("migrations_applied", revision=revision)


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    upgrade(arguments[0] if arguments else "head")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
