# flip-options-bot

A 0-14 DTE equity options paper-trading bot for Alpaca. Built fresh in 2026-08 from lessons learned in `flip-alpaca-bot`'s diagnostic findings (the funnel collapse, the phantom close ledger, the invalid OCC symbols).

## What this is

A paper-first trading bot. Defaults to single-strategy (long-call, ATM-ish, 0-14 DTE) with a diagonal strategy registered but disabled by default. The registry pattern (from `go-trader-prior-art` and `lean-algorithm-python-prior-art`) means adding a strategy is a single new file under `src/flip_options_bot/strategies/`.

## Status

**Scaffold only.** This repo is the result of session N=1 of an N-session build. The daemon does NOT auto-start. There is NO live trading wired up. The structural fixes (idempotent close writes, position_id UUIDs, real-fill reconciliation, broker-resident TP/SL) are designed for but not yet wired to the broker layer — that's session N=2.

What's shipped in this scaffold:
- `config.py` — Settings dataclass, frozen, sourced from env (double-gated live mode)
- `risk/engine.py` — RiskEngine with the structural-fix tests (tick_rollover doesn't zero fresh state; loss caps fire BEFORE kill switch; record_close is idempotent)
- `journal/journal.py` — SQLite append-only journal keyed on `event_id` (the broker's `client_order_id`); INSERT OR IGNORE means a stuck reconcile loop cannot create duplicate close writes
- `signal/funnel.py` — FunnelRecorder with `scan_id` UUIDs; the diagnostic instrument that lets us see WHERE the candidate count collapses to zero
- `strategies/long_call.py` — first strategy: 0-14 DTE ATM-ish directional long call, with the funnel-collapse bug fixed structurally (conviction computed before funnel emit)
- `daemon.py` — entry point with `--once` smoke-test mode and `--config-check`
- `cli.py` — status/heartbeat inspection
- 30+ tests covering config, risk gates, journal idempotency, funnel recorder, strategy pure functions

## Install

```bash
cd /root/flip/projects/flip-options-bot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example /root/.config/flip-options-bot/.env
chmod 600 /root/.config/flip-options-bot/.env
# Edit and fill in Alpaca paper creds
```

## Test

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Smoke-test the daemon (does NOT trade)

```bash
flip-options-cli heartbeat
flip-options-bot --config-check
flip-options-bot --once
```

## What this bot does NOT do (yet)

- **Does NOT place live orders.** The execution layer is the next session.
- **Does NOT auto-start.** The systemd unit is in `deploy/` but NOT installed.
- **Does NOT use real-fill reconciliation yet.** The journal has the right shape; the WebSocket trade-update stream integration comes next.
- **Does NOT handle FOMC/earnings blackouts yet.** The settings have the flags; the scanner integration comes next.

## What this bot FIXES structurally (vs. flip-alpaca-bot)

| Issue in flip-alpaca-bot | Fix in flip-options-bot |
|---|---|
| Tick rollover zeroed daily P&L on fresh state | `tick_rollover` only zeros if `last_reset_day != ""` AND day changed |
| Loss cap fires AFTER kill switch short-circuit | `_check_loss_caps` runs FIRST, escalates to KILL on breach |
| Duplicate close events write phantom P&L | Journal uses `INSERT OR IGNORE` on `event_id` (broker's `client_order_id`) |
| `realized_pnl_original` (conservative-loss) used for reconciliation | Journal never reads that field; canonical ledger will use broker's actual fill price |
| Invalid OCC symbols (e.g. `SPY000000C00000000`) accepted as positions | Position_id is a UUID; OCC symbols are validated upstream in the scanner |
| Funnel emit lost on duplicate reconcile | FunnelRow carries `scan_id` UUID; duplicate emit returns False |

## License

MIT. See LICENSE file.