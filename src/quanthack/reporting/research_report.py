from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

from quanthack.backtesting.backtest import BacktestResult, EquityPoint
from quanthack.core.config import AppConfig
from quanthack.market.data_health import DataHealthSeverity, MarketDataHealthReport
from quanthack.trading.preflight import PreflightReport, PreflightStatus
from quanthack.backtesting.strategy_compare import StrategyComparison
from quanthack.backtesting.sweep import SweepResult
from quanthack.backtesting.walk_forward import WalkForwardResult


@dataclass(frozen=True)
class ResearchReport:
    title: str
    html: str


def build_research_report(
    *,
    config: AppConfig,
    preflight: PreflightReport,
    backtest: BacktestResult,
    comparison: StrategyComparison,
    sweep: SweepResult,
    strategy_name: str,
    data_health: MarketDataHealthReport | None = None,
    walk_forward: WalkForwardResult | None = None,
    title: str = "QuantHack Research Report",
    generated_at: datetime | None = None,
    sweep_limit: int = 6,
) -> ResearchReport:
    generated = generated_at or datetime.now(tz=ZoneInfo(config.competition.timezone))
    best_strategy = comparison.best.strategy_name if comparison.best is not None else "none"
    best_sweep = _best_sweep_label(sweep)
    best_walk_forward = (
        walk_forward.best.strategy_name
        if walk_forward is not None and walk_forward.best is not None
        else "not checked"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --ink: #151923;
      --muted: #5e6877;
      --line: #d7dde5;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #dff5f1;
      --warn: #9a3412;
      --warn-soft: #fff7ed;
      --bad: #991b1b;
      --bad-soft: #fef2f2;
      --blue: #1d4ed8;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 30px 20px 52px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 20px;
    }}
    h1 {{
      font-size: 30px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 18px;
      margin: 30px 0 10px;
      letter-spacing: 0;
    }}
    h3 {{
      font-size: 15px;
      margin: 18px 0 8px;
      letter-spacing: 0;
    }}
    .muted {{
      color: var(--muted);
      font-size: 14px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 650;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      padding: 9px 11px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      vertical-align: top;
    }}
    th {{
      background: #eef2f6;
      color: #313947;
      font-weight: 650;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status-ok {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .status-warn {{
      background: var(--warn-soft);
      color: var(--warn);
    }}
    .status-fail {{
      background: var(--bad-soft);
      color: var(--bad);
    }}
    .chart {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .chart svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .note {{
      background: var(--warn-soft);
      border: 1px solid #fed7aa;
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--warn);
      margin-top: 16px;
      font-size: 14px;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{escape(title)}</h1>
      <div class="muted">Generated {escape(generated.isoformat(timespec="seconds"))}</div>
    </div>
    <div class="muted">Offline research and dry-run only. No live or paper orders.</div>
  </header>

  <section>
    <h2>Decision Snapshot</h2>
    <div class="summary">
      {metric("Selected Strategy", strategy_name)}
      {metric("Best Strategy", best_strategy)}
      {metric("Best Momentum Sweep", best_sweep)}
      {metric("Best Walk-Forward", best_walk_forward)}
      {metric("Preflight", preflight.overall)}
      {metric("Data Health", data_health.overall.value if data_health is not None else "not checked")}
      {metric("Final Equity", money(backtest.metrics.final_equity))}
      {metric("Sharpe", f"{backtest.metrics.sharpe_ratio:.3f}")}
      {metric("Total Return", pct(backtest.metrics.total_return_pct))}
      {metric("Max Drawdown", pct(backtest.metrics.max_drawdown_pct))}
    </div>
  </section>

  {build_preflight_section(preflight)}
  {build_data_health_section(data_health)}
  {build_backtest_section(backtest)}
  {build_strategy_comparison_section(comparison)}
  {build_walk_forward_section(walk_forward)}
  {build_sweep_section(sweep, limit=sweep_limit)}
  {build_risk_section(config)}

  <div class="note">
    Hackathon safety reminder: this report is generated from local CSV data and
    simulated fills. It is useful for research and explanation, but it is not a
    live broker statement and does not place orders.
  </div>
</main>
</body>
</html>
"""
    return ResearchReport(title=title, html=html)


def build_preflight_section(preflight: PreflightReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(check.name)}</td>"
        f"<td>{status_badge(check.status)}</td>"
        f"<td>{escape(check.details)}</td>"
        "</tr>"
        for check in preflight.checks
    )
    return f"""  <section>
    <h2>Preflight</h2>
    <table>
      <thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>"""


def build_data_health_section(data_health: MarketDataHealthReport | None) -> str:
    if data_health is None:
        return """  <section>
    <h2>Market Data Health</h2>
    <div class="note">Market data validation was not run for this report.</div>
  </section>"""

    symbol_rows = "\n".join(
        "<tr>"
        f"<td>{escape(symbol.symbol)}</td>"
        f"<td>{symbol.price_count:,}</td>"
        f"<td>{symbol.quote_count:,}</td>"
        f"<td>{escape(_dt(symbol.price_start))}</td>"
        f"<td>{escape(_dt(symbol.price_end))}</td>"
        f"<td>{symbol.max_price_gap_seconds:.1f}s</td>"
        f"<td>{symbol.max_quote_gap_seconds:.1f}s</td>"
        f"<td>{symbol.max_spread_bps:.2f}</td>"
        "</tr>"
        for symbol in data_health.symbols
    )
    if not symbol_rows:
        symbol_rows = "<tr><td colspan=\"8\">No symbol data</td></tr>"

    issue_rows = "\n".join(
        "<tr>"
        f"<td>{status_badge(issue.severity)}</td>"
        f"<td>{escape(issue.symbol)}</td>"
        f"<td>{escape(issue.category)}</td>"
        f"<td>{escape(issue.details)}</td>"
        "</tr>"
        for issue in data_health.issues
    )
    if not issue_rows:
        issue_rows = "<tr><td colspan=\"4\">No issues</td></tr>"

    return f"""  <section>
    <h2>Market Data Health</h2>
    <div class="muted">Overall: {status_badge(data_health.overall)}</div>
    <h3>Coverage</h3>
    <table>
      <thead>
        <tr><th>Symbol</th><th>Prices</th><th>Quotes</th><th>Start</th>
        <th>End</th><th>Max Price Gap</th><th>Max Quote Gap</th><th>Max Spread bps</th></tr>
      </thead>
      <tbody>
        {symbol_rows}
      </tbody>
    </table>
    <h3>Issues</h3>
    <table>
      <thead><tr><th>Status</th><th>Symbol</th><th>Category</th><th>Details</th></tr></thead>
      <tbody>
        {issue_rows}
      </tbody>
    </table>
  </section>"""


def build_backtest_section(result: BacktestResult) -> str:
    ledger = result.pnl_ledger
    metrics_rows = [
        ("Observations", f"{result.metrics.observations:,}"),
        ("Fills", f"{len(result.fills):,}"),
        ("Final Equity", money(result.metrics.final_equity)),
        ("Total Return", pct(result.metrics.total_return_pct)),
        ("Sharpe Ratio", f"{result.metrics.sharpe_ratio:.3f}"),
        ("Max Drawdown", pct(result.metrics.max_drawdown_pct)),
        ("Win Rate", pct(result.metrics.win_rate)),
        ("Profit Factor", f"{result.metrics.profit_factor:.3f}"),
        ("Turnover", money(result.metrics.turnover_notional)),
        ("Realized P&L", money(ledger.realized_pnl_usd)),
        ("Open P&L", money(ledger.open_pnl_usd)),
        ("Total Attributed P&L", money(ledger.total_pnl_usd)),
    ]
    rows = "\n".join(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>" for label, value in metrics_rows)
    return f"""  <section>
    <h2>Backtest</h2>
    {build_equity_chart(result.equity_curve)}
    <h3>Metrics</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    {build_recent_fills_table(result)}
  </section>"""


def build_strategy_comparison_section(comparison: StrategyComparison) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{rank}</td>"
        f"<td>{escape(row.strategy_name)}</td>"
        f"<td>{escape(row.result.symbol)}</td>"
        f"<td>{money(row.result.metrics.final_equity)}</td>"
        f"<td>{pct(row.result.metrics.total_return_pct)}</td>"
        f"<td>{row.result.metrics.sharpe_ratio:.3f}</td>"
        f"<td>{pct(row.result.metrics.max_drawdown_pct)}</td>"
        f"<td>{len(row.result.fills):,}</td>"
        "</tr>"
        for rank, row in enumerate(comparison.rows, start=1)
    )
    if not rows:
        rows = "<tr><td colspan=\"8\">No strategy comparison rows</td></tr>"
    return f"""  <section>
    <h2>Strategy Comparison</h2>
    <table>
      <thead>
        <tr><th>Rank</th><th>Strategy</th><th>Symbol</th><th>Final Equity</th>
        <th>Return</th><th>Sharpe</th><th>Max DD</th><th>Fills</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>"""


def build_walk_forward_section(walk_forward: WalkForwardResult | None) -> str:
    if walk_forward is None:
        return """  <section>
    <h2>Walk-Forward Evaluation</h2>
    <div class="note">Walk-forward evaluation was not run for this report.</div>
  </section>"""

    rows = "\n".join(
        "<tr>"
        f"<td>{rank}</td>"
        f"<td>{escape(summary.strategy_name)}</td>"
        f"<td>{escape(str(summary.eligible))}</td>"
        f"<td>{len(summary.folds):,}</td>"
        f"<td>{summary.median_test_sharpe:.3f}</td>"
        f"<td>{pct(summary.lower_quartile_test_return)}</td>"
        f"<td>{pct(summary.worst_test_drawdown)}</td>"
        f"<td>{pct(summary.profitable_fold_fraction)}</td>"
        f"<td>{summary.total_test_fills:,}</td>"
        f"<td>{money(summary.total_test_turnover)}</td>"
        "</tr>"
        for rank, summary in enumerate(walk_forward.summaries, start=1)
    )
    if not rows:
        rows = "<tr><td colspan=\"10\">No walk-forward rows</td></tr>"

    return f"""  <section>
    <h2>Walk-Forward Evaluation</h2>
    <table>
      <thead>
        <tr><th>Rank</th><th>Strategy</th><th>Eligible</th><th>Folds</th>
        <th>Median Sharpe</th><th>LQ Return</th><th>Worst DD</th>
        <th>Profitable Folds</th><th>Fills</th><th>Turnover</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>"""


def build_sweep_section(sweep: SweepResult, *, limit: int) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{rank}</td>"
        f"<td>{candidate.lookback}</td>"
        f"<td>{candidate.threshold_bps:.1f}</td>"
        f"<td>{candidate.train.metrics.sharpe_ratio:.3f}</td>"
        f"<td>{candidate.test.metrics.sharpe_ratio:.3f}</td>"
        f"<td>{pct(candidate.test.metrics.total_return_pct)}</td>"
        f"<td>{pct(candidate.test.metrics.max_drawdown_pct)}</td>"
        f"<td>{len(candidate.test.fills):,}</td>"
        "</tr>"
        for rank, candidate in enumerate(sweep.candidates[:limit], start=1)
    )
    if not rows:
        rows = "<tr><td colspan=\"8\">No sweep candidates</td></tr>"
    return f"""  <section>
    <h2>Momentum Parameter Sweep</h2>
    <table>
      <thead>
        <tr><th>Rank</th><th>Lookback</th><th>Threshold bps</th>
        <th>Train Sharpe</th><th>Test Sharpe</th><th>Test Return</th>
        <th>Test DD</th><th>Test Fills</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>"""


def build_risk_section(config: AppConfig) -> str:
    limits = config.risk
    rows = [
        ("Max gross leverage", f"{limits.max_gross_leverage:.2f}x"),
        ("Max symbol notional", pct(limits.max_symbol_notional_pct)),
        ("Max daily loss", pct(limits.max_daily_loss_pct)),
        ("Max drawdown", pct(limits.max_drawdown_pct)),
        ("Checkpoint multiplier", f"{limits.checkpoint_risk_multiplier:.2f}x"),
        ("Minimum margin level", f"{limits.min_margin_level_pct:.0f}%"),
    ]
    body = "\n".join(f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>" for label, value in rows)
    return f"""  <section>
    <h2>Risk Settings</h2>
    <table>
      <thead><tr><th>Limit</th><th>Value</th></tr></thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </section>"""


def build_recent_fills_table(result: BacktestResult, limit: int = 8) -> str:
    recent = result.fills[-limit:]
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(fill.timestamp)}</td>"
        f"<td>{escape(fill.side.value)}</td>"
        f"<td>{fill.fill_price:.5f}</td>"
        f"<td>{fill.trade_units:,.2f}</td>"
        f"<td>{money(fill.adjusted_notional_usd)}</td>"
        "</tr>"
        for fill in recent
    )
    if not rows:
        rows = "<tr><td colspan=\"5\">No fills</td></tr>"
    return f"""    <h3>Recent Fills</h3>
    <table>
      <thead><tr><th>Time</th><th>Side</th><th>Fill</th><th>Units</th><th>Adjusted Notional</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>"""


def build_equity_chart(points: tuple[EquityPoint, ...]) -> str:
    if len(points) < 2:
        return "<div class=\"chart muted\">Not enough equity points to chart.</div>"

    width = 900
    height = 220
    pad = 24
    equities = [point.equity for point in points]
    low = min(equities)
    high = max(equities)
    span = high - low
    if span == 0:
        span = 1.0

    coordinates = []
    for index, equity in enumerate(equities):
        x = pad + (index / (len(equities) - 1)) * (width - 2 * pad)
        y = height - pad - ((equity - low) / span) * (height - 2 * pad)
        coordinates.append(f"{x:.1f},{y:.1f}")

    start = money(equities[0])
    end = money(equities[-1])
    return f"""    <div class="chart">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">
        <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#d7dde5" />
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#d7dde5" />
        <polyline points="{' '.join(coordinates)}" fill="none" stroke="#1d4ed8" stroke-width="3" />
        <text x="{pad}" y="18" font-size="12" fill="#5e6877">High {money(high)}</text>
        <text x="{pad}" y="{height - 6}" font-size="12" fill="#5e6877">Low {money(low)}</text>
        <text x="{width - pad - 150}" y="{height - 6}" font-size="12" fill="#5e6877">Start {start} / End {end}</text>
      </svg>
    </div>"""


def metric(label: str, value: str) -> str:
    return f"""<div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
      </div>"""


def status_badge(status: PreflightStatus | DataHealthSeverity) -> str:
    css_class = {
        "OK": "status-ok",
        "WARN": "status-warn",
        "FAIL": "status-fail",
    }[status.value]
    return f"<span class=\"status {css_class}\">{escape(status.value)}</span>"


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.3%}"


def _best_sweep_label(sweep: SweepResult) -> str:
    if sweep.best is None:
        return "none"
    return f"L{sweep.best.lookback} / {sweep.best.threshold_bps:.1f} bps"


def _dt(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.isoformat(timespec="seconds")
