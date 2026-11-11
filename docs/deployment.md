# Deploying on a Linux server

This guide runs the full stack (PostgreSQL, migrations, API, worker and
dashboard) with Docker Compose on a Linux machine. It follows the intended
rollout: **paper trading first, live trading later**. It was written for
Ubuntu 22.04/24.04 and Debian 12; any distribution with Docker Engine works.
Both x86_64 and ARM64 (for example a Raspberry Pi 5 or a Graviton VM) are
supported.

The examples use `~/btct` as the install directory.

---

## 1. Prepare the server

```bash
# Docker Engine + the Compose plugin (official convenience script)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"      # log out and back in afterwards
sudo systemctl enable --now docker   # start Docker (and the bot) on boot

# Accurate time matters: bitFlyer rejects signed requests with a skewed clock.
timedatectl status                   # expect "System clock synchronized: yes"
sudo timedatectl set-ntp true        # if it is not
```

Recommended minimum: 1 vCPU, 1 GB RAM, 10 GB disk. Prefer a machine that stays
on around the clock, such as a VPS, a cloud VM or an always-on home server.

## 2. Copy the project

From the development machine, copy the source **without** the local virtualenv,
caches, local database or `.env`:

```bash
rsync -av --delete \
  --exclude .venv --exclude .env --exclude data \
  --exclude '__pycache__' --exclude .mypy_cache --exclude .ruff_cache \
  --exclude .pytest_cache --exclude .claude --exclude .DS_Store \
  ./ user@server:~/btct/
```

You can use `git clone` instead if the project is in a repository.

## 3. Configure `.env` for paper trading

On the server:

```bash
cd ~/btct
cp .env.example .env
chmod 600 .env
openssl rand -hex 24                 # use the output as POSTGRES_PASSWORD
nano .env
```

Change these values. Leave everything else at its default.

```dotenv
APP_ENV=production
LOG_FORMAT=json

TRADING_MODE=paper
STRATEGY_NAME=trend_ensemble

POSTGRES_PASSWORD=<the hex string from openssl>

# Size the paper account like the capital you plan to trade live, so the paper
# results (and the effect of the 0.001 BTC minimum lot) are representative.
PAPER_INITIAL_JPY=200000

# Strongly recommended: Slack or Discord webhook for fills, failures and halts.
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...

# Paper trading needs no exchange credentials: leave these EMPTY for now.
BITFLYER_API_KEY=
BITFLYER_API_SECRET=
```

## 4. Build and start

```bash
docker compose build
docker compose up -d
docker compose ps                    # all services "running"/"healthy"; migrate "exited (0)"
```

Within about two minutes, check that these work:

```bash
curl -s localhost:8000/health                       # "status":"ok"
curl -s localhost:8000/api/v1/market/daily-history  # "bars" > 360
curl -s localhost:8000/api/v1/strategy/preview      # "ready":true
curl -s localhost:8000/api/v1/trading/status        # mode paper
docker compose logs -f worker                       # Ctrl-C to stop following
```

The worker collects a ticker every 5 minutes. It backfills about 5 years of
daily bars from bitbank and evaluates the strategy every 5 minutes, which
produces one decision per completed UTC day. It may place its first paper order
right away if the strategy wants exposure.

## 5. Open the dashboard securely

The API and dashboard listen on `127.0.0.1` only. Neither has authentication,
and the dashboard can change trading settings, so do not publish them. From
your laptop, open an SSH tunnel:

```bash
ssh -N -L 8501:localhost:8501 -L 8000:localhost:8000 user@server
```

Then browse to <http://localhost:8501> (dashboard) and
<http://localhost:8000/docs> (API).

[Tailscale](https://tailscale.com) is a good permanent alternative. Set
`BIND_ADDRESS` in `.env` to the server's Tailscale IP, then run
`docker compose up -d`. Never use `0.0.0.0` on a machine with a public IP:
Docker bypasses `ufw`/`firewalld` rules.

## 6. During the paper period

Once a day, check:

- **Dashboard → Portfolio / Trading:** equity, orders and the daily decision.
- **`/api/v1/signals`:** one row per day. `metadata_json.execution.status`
  records what happened: `filled`, `no_trade`, or `blocked_*` with a reason.
- **`/api/v1/system/events`:** warnings and errors. Job failures also go to
  the webhook.
- **`docker compose ps`:** every service is healthy.

Expect few trades. The strategy decides once per day and often holds, and in
the backtest it averaged about two small orders a week. A few days of paper
trading proves the plumbing works, not the strategy's edge.

> **Settings precedence.** Clicking **Apply** in the dashboard sidebar saves
> *every* sidebar value to the database, and from then on those values override
> `.env`. The one exception is `TRADING_KILL_SWITCH=true` in `.env`, which always
> wins. If an `.env` change seems to have no effect, check
> `/api/v1/trading/status`, then either change it in the dashboard or clear the
> saved overrides:
>
> ```bash
> docker compose exec db psql -U btc -d btc_trader -c "DELETE FROM runtime_settings;"
> ```

## 7. Back up the database

The ledger (orders, fills, positions) lives in the `btc_trader_postgres_data`
volume, which survives restarts and rebuilds. Back it up daily anyway, for
example with `crontab -e`:

```cron
15 5 * * * cd ~/btct && docker compose exec -T db pg_dump -U btc -Fc btc_trader > ~/btc_backup_$(date +\%a).dump
```

This keeps a rolling week of backups. To restore one:
`docker compose exec -T db pg_restore -U btc -d btc_trader --clean < ~/btc_backup_Mon.dump`.

Never run `docker compose down -v`: `-v` deletes the database volume.

## 8. Going live

Work through this list in order.

1. **Create a bitFlyer API key** with only these permissions:
   balance, trading commission, new order, order list, and executions.
   **Do not enable withdrawal.** If bitFlyer offers an IP allow-list, restrict
   the key to the server's public IP.
2. **Use a dedicated account (or sub-account) for the bot.** All BTC in the
   account counts as the bot's and may be sold. JPY is capped by
   `LIVE_CAPITAL_JPY`.
3. **Add the key and run the read-only check.** It places no order:

   ```bash
   nano .env      # set BITFLYER_API_KEY and BITFLYER_API_SECRET
   docker compose up -d
   docker compose exec worker python -m btc_trader.scripts.live_check
   ```

   Expect `API key permissions: OK`, your fee tier, balances and exchange state.
   The live gates will show unchecked until step 5.
4. **Optional: one real test order.** This sends a 0.001 BTC IOC order, about
   ¥12–13k:

   ```bash
   docker compose exec worker python -m btc_trader.scripts.live_check --test-order buy --confirm-real-money
   ```

5. **Switch to live with small capital.** In `.env`:

   ```dotenv
   TRADING_MODE=live
   ENABLE_AUTOMATIC_TRADING=true
   LIVE_TRADING_ACKNOWLEDGED=true
   TRADING_KILL_SWITCH=false
   LIVE_CAPITAL_JPY=100000          # start small; raise later
   MAX_ORDER_JPY=100000
   MAX_POSITION_JPY=100000
   MAX_DAILY_LOSS_JPY=10000
   ```

   Then run:

   ```bash
   docker compose up -d                                   # recreates containers with the new .env
   docker compose exec worker python -m btc_trader.scripts.live_check   # every gate [x]
   curl -s localhost:8000/api/v1/trading/status           # mode live, live orders allowed
   ```

   If the dashboard settings were applied during the paper period, they still
   say `paper`. Switch the mode in the dashboard, or clear the overrides as
   described in section 6.

6. **Watch the first live day closely:** the webhook alerts, `/api/v1/orders`,
   and the bitFlyer web UI side by side.

Paper and live keep separate ledgers. Switching modes neither touches nor
resets the paper history.

## 9. Emergency stop

Use the first option that's available to you:

- **Dashboard:** sidebar → *Trading kill switch* → **Apply**. It takes effect
  within one collection interval.
- **`.env`:** `TRADING_KILL_SWITCH=true`, then `docker compose up -d`. This
  overrides anything saved in the dashboard.
- **Stop the worker** (the only process that places orders):
  `docker compose stop worker`.

None of these cancel or close an existing position. Orders are IOC, so nothing
rests on the book. To exit a position, sell in the bitFlyer UI after stopping
the bot.

## 10. Updating

```bash
# from the development machine: the same rsync command as in step 2, then on the server:
cd ~/btct
docker compose build
docker compose up -d               # migrate runs first, then the services restart
docker image prune -f
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `migrate` exits non-zero | `docker compose logs migrate`. Usually a wrong `POSTGRES_PASSWORD`, or a password with URL-special characters. The database password is set when the volume is first created; changing `.env` later does not change it. |
| `daily-history` stays at 0 bars | `docker compose logs worker \| grep daily_history`. An HTTPS or proxy problem reaching `public.bitbank.cc`. |
| `CERTIFICATE_VERIFY_FAILED` | A TLS-intercepting proxy or corporate VPN is on the network path. Run the bot on a network without it. |
| Strategy shows `blocked_stale_*` | The worker or exchange connectivity stopped. See `docker compose logs worker`. |
| Live: `blocked_circuit_breaker` | 3 or more failed or rejected orders in 24 h. Read `/api/v1/system/events` and fix the cause. Trading resumes when the failures age out of the 24 h window. |
| Signed requests fail with auth errors | The server clock is off (`timedatectl`), or the key lacks a permission (`live_check`). |
