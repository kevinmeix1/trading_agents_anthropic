# Hackathon Readiness Report

This command builds one go/no-go report from the pieces that matter for the
competition:

- official 15-instrument data coverage;
- forex/metals/crypto asset-class coverage;
- internal risk limits versus penalty zones;
- adaptive candidate promotion audit;
- current paper/live readiness verdict.

Run:

```bash
quanthack hackathon-readiness \
  --price-csv data/full_20gb_15m_prices.csv \
  --quote-csv data/full_20gb_15m_quotes.csv \
  --promotion-csv outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_promotion.csv \
  --summary-csv outputs/backtests/adaptive_strategy_selection_session_macd_cooldown1_summary.csv \
  --output outputs/reports/hackathon_readiness.md
```

Current result:

```text
Overall: FAIL
Data coverage: 10/15 official instruments
Missing asset classes: CRYPTO
Candidate: PAPER_ONLY
```

Interpretation:

```text
The project remains useful for FX/metals paper research, but it is not full
hackathon-live ready because the current local 15-minute dataset has no crypto
coverage for BARUSD, BTCUSD, ETHUSD, SOLUSD, or XRPUSD. The current adaptive
candidate also remains PAPER_ONLY because it has not cleared the stricter live
positive-fold gate.
```

Use `--strict` when you want the command to exit non-zero unless every check
passes.
