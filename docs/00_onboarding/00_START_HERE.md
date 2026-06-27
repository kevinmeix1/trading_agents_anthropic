# Start Here

We are not choosing the final strategy yet.

The first phase is setup and orientation:

1. Open the project in VS Code.
2. Create a Python 3.11 virtual environment.
3. Confirm the terminal is using the project environment.
4. Read the rules summary.
5. Keep all trading routes in `dry_run`.

## Why This Order

The hackathon is not only about having a clever model. It is a live paper-trading
tournament with eliminations, margin rules, and London-time checkpoints. A simple
strategy with strong risk controls is safer than a complicated strategy with weak
execution controls.

## Our First Simple Strategy Later

When the workspace is ready, we will start with a small baseline:

- One or two liquid symbols.
- Low notional size.
- Momentum or mean-reversion signal.
- Strict risk firewall.
- Dry-run logs before any platform connection.

That gives us a working skeleton before we try anything more ambitious.

