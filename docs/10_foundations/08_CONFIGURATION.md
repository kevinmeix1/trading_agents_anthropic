# Configuration

Step 8 moves important settings out of Python scripts and into a config file.

This matters for the hackathon because you may need to tune settings, but you do
not want to rewrite code every time you adjust risk or strategy thresholds.

## New File

```text
configs/default.toml
```

It contains:

- Competition timezone and checkpoint times.
- Starting equity.
- Risk limits.
- Active strategy selection.
- Strategy settings.
- Dry-run journal location.

## Hackathon Rules Reflected In Config

The config keeps the rule-aware assumptions visible:

- Timezone is `Europe/London`.
- Starting equity is `$1,000,000`.
- Checkpoint protection starts 90 minutes before configured checkpoints.
- Internal max gross leverage is `2.0x`, far below the event maximum of `1:30`.
- Internal daily loss stop is `2.5%`.
- Internal margin warning is `300%`, far above the official danger area.

These are conservative starter settings. They are not predictions and not trading
advice.

## Why TOML

TOML is readable:

```toml
[strategy]
active = "simple_momentum"

[strategy.simple_momentum]
symbol = "EURUSD"
threshold_bps = 8.0
target_notional_usd = 50000.0

[strategy.ma_crossover]
symbol = "EURUSD"
fast_window = 3
slow_window = 8
min_separation_bps = 2.0

[walk_forward]
ma_fast_windows = [2, 3]
ma_slow_windows = [5, 8]
ma_min_separation_bps = [1.0, 2.0]
```

Python 3.11 can read TOML using the standard library, so we do not need a new
dependency.

## New Commands

Show the config:

```bash
source .venv/bin/activate
python scripts/inspect/show_config.py
```

Run the configured strategy through risk and dry-run journaling:

```bash
python scripts/dry_run/configured_strategy_dry_run.py --scenario up
python scripts/dry_run/configured_strategy_dry_run.py --scenario down
python scripts/dry_run/configured_strategy_dry_run.py --scenario flat
```

Run all tests:

```bash
python -m unittest discover -s tests
```

## Reminder

The final hackathon schedule has a possible discrepancy between private notes and
the public event listing. Keep the official participant portal as the source of
truth, and update `configs/default.toml` once the exact final cutoff is confirmed.
