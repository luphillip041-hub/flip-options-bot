# Agent Setup Guide — flip-options-bot

Repository: `https://github.com/luphillip041-hub/flip-options-bot.git`

Concise skill entry point for agents setting up, configuring, operating, or extending flip-options-bot.

Quick flow for a new server: tell OpenClaw `install https://github.com/luphillip041-hub/flip-options-bot and init`.

## Core rules

- Run git from repo root.
- The daemon is **paper-only by default**. Live mode is double-gated (`FOB_PHASE=live` AND `LIVETRADE_ENABLED=true`). The executor must re-check `Settings.is_live()` at every submit, not trust the env at startup.
- **Idempotent journal writes.** Every event keyed by `client_order_id` (broker order ID). The journal uses `INSERT OR IGNORE` so a stuck reconcile loop cannot create duplicate close writes.
- **Position IDs are UUIDs.** Never derive position_id from symbol+date. Same (symbol, date) tuple can have multiple legitimate closes, distinguished by position_id.
- **No market orders.** Limit only. Cancel after fill window if not filled.
- **Funnel emits always.** Every scan emits a `FunnelRow` with a `scan_id` UUID, even when zero candidates pass. That's the diagnostic instrument.
- **Per-trade risk ≤ 2% of equity.** Daily loss cap 6% → kill switch.
- **Real-fill reconciliation.** The journal's canonical close ledger uses the broker's actual fill price from `list_filled_orders`, falling back to WebSocket trade-update stream, NEVER the conservative-loss heuristic.

## Prerequisites

```bash
python3.12 --version
uv --version 2>/dev/null || echo "NOT_INSTALLED"
```

## Install

```bash
git clone https://github.com/luphillip041-hub/flip-options-bot.git
cd flip-options-bot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example /root/.config/flip-options-bot/.env
chmod 600 /root/.config/flip-options-bot/.env
# Edit and fill in Alpaca paper creds:
#   APCA_API_KEY_ID_PAPER=PK...
#   APCA_API_SECRET_KEY_PAPER=...
```

## Test

```bash
source .venv/bin/activate
pytest tests/ -v
```

## Smoke-test (does NOT trade)

```bash
flip-options-bot --config-check   # validate env
flip-options-cli heartbeat        # print settings snapshot
flip-options-bot --once           # one scan cycle, exit
```

## Run as a service (paper only — verify before deploying)

```bash
sudo cp deploy/flip-options-bot.service /etc/systemd/system/
sudo cp deploy/flip-options-bot-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable flip-options-bot.service
sudo systemctl start flip-options-bot.service
```

**Do NOT enable or start the service until paper validation has run for at least 5-10 trading days.** This is the same gate as `paper-to-live-trading-bot-scaffold`.

## What this scaffold does NOT do (yet)

- **No real broker layer.** The broker/execution module is a placeholder. The journal + risk + funnel layers are complete and tested, but they don't connect to Alpaca yet.
- **No real scanner.** The long_call strategy module has the conviction function and DTE filter, but the actual market-data scanner that pulls chains + minute bars + computes conviction is a follow-up session.
- **No live trading.** The Settings has the live-mode double-gate wired in but the executor that would submit live orders doesn't exist.

## What this scaffold DOES (tested)

- **Config layer:** 6 tests
- **Risk engine:** 11 tests covering the structural fixes (tick_rollover, loss-caps, kill-switch ordering, idempotent close)
- **Journal:** 6 tests covering idempotent events, position_id uniqueness, multiple positions same symbol
- **Funnel recorder:** 5 tests covering UUIDs, malformed-line tolerance, scan_id dedup
- **Long-call strategy:** 5 tests covering DTE filter, expiry picker, conviction math
- **Total: 33 tests passing**

## Reference materials (already in this user's session)

- `paper-to-live-trading-bot-scaffold` — the workflow this scaffold implements
- `option-strategies-prior-art` — 38 option-strategy repos for inspiration
- `options-data-sources` — 20 options-data sources for the scanner layer
- `lean-algorithm-python-prior-art` — QuantConnect Lean patterns
- `go-trader-prior-art` — Go+Python hybrid prior art (architecture inspiration)

## Open work (next session)

1. **Broker layer** (`src/flip_options_bot/broker/alpaca.py`) — wrap alpaca-py with the same idempotent submit pattern, with a real-fill reconciliation loop polling `list_filled_orders`.
2. **Scanner** (`src/flip_options_bot/signal/scanner.py`) — pull option chains + minute bars, compute conviction, emit FunnelRow.
3. **Executor** (`src/flip_options_bot/execution/`) — translate a LongCallSignal into a broker order, place broker-resident TP/SL (the flip-alpaca-bot lesson).
4. **Paper observation window.** 10 trading days before live is eligible.