# Preflight Warnings

Step 19 makes preflight warnings real.

Before this step, preflight could report:

```text
OK
FAIL
```

but the `WARN` status existed mostly as unused structure.

Now preflight has three useful levels:

```text
OK    - safe enough for dry-run
WARN  - dry-run can continue, but pay attention
FAIL  - stop and fix this before continuing
```

## Why This Matters

The hackathon rewards performance, but account survival comes first.

A warning is for conditions that are still inside the hard guardrails but close
enough to deserve attention. This is the difference between:

```text
This quote is stale. Stop.
```

and:

```text
This quote is getting old. Be careful before acting on it.
```

## Warning Conditions

Market quality warns when:

- Spread is at least 50% of the configured spread limit.
- Quote age is at least 50% of the configured age limit.

Risk limits warn when:

- Gross leverage is above the conservative 2.0x target.
- Per-symbol notional is above 25% of equity.
- Daily loss stop is above 2.5%.
- Drawdown stop is above 6%.
- Margin floor is close to the 300% starter guard.

The harder fail guards still block the run.

## Run In VS Code Terminal

Default preflight:

```bash
python scripts/setup/preflight.py
```

Trigger a quote-age warning:

```bash
python scripts/setup/preflight.py --quote-as-of "2026-06-22T10:20:03+01:00"
```

Expected overall result:

```text
Overall: READY_WITH_WARNINGS
```

Trigger a quote-age failure:

```bash
python scripts/setup/preflight.py --quote-as-of "2026-06-22T10:20:10+01:00"
```

Expected overall result:

```text
Overall: ATTENTION_REQUIRED
```

## Important Detail

`READY_WITH_WARNINGS` does not exit with an error code. It is meant to be visible,
not blocking.

`ATTENTION_REQUIRED` still exits with an error code so automated checks can stop.
