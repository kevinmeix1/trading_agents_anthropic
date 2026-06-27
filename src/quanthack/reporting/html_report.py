from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

from quanthack.reporting.journal_report import JournalSummary, summarize_journal


@dataclass(frozen=True)
class HtmlReport:
    title: str
    html: str


def build_journal_html_report(
    *,
    records: list[dict],
    title: str = "Claude Agent Trader Dry-Run Journal",
    generated_at: datetime | None = None,
    recent_limit: int = 10,
) -> HtmlReport:
    generated = generated_at or datetime.now(tz=ZoneInfo("Europe/London"))
    summary = summarize_journal(records)
    recent_records = records[-recent_limit:] if recent_limit > 0 else []

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --ink: #151923;
      --muted: #5d6675;
      --line: #d8dde6;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #9a3412;
      --bad: #991b1b;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.4;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 22px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 6px;
    }}
    h2 {{
      font-size: 18px;
      margin: 28px 0 10px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
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
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    th {{
      background: #eef1f5;
      color: #303744;
      font-weight: 650;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .status-accepted {{
      color: var(--accent);
      font-weight: 650;
    }}
    .status-blocked {{
      color: var(--bad);
      font-weight: 650;
    }}
    .note {{
      background: #fff7ed;
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
    <div class="muted">Dry-run only. No live or paper orders are placed by this report.</div>
  </header>

  {build_metrics_html(summary)}
  {build_count_table("By Status", summary.by_status)}
  {build_count_table("By Mode", summary.by_mode)}
  {build_symbol_table(summary)}
  {build_recent_table(recent_records)}

  <div class="note">
    Hackathon rule reminder: this report shows local dry-run decisions only. It helps prove
    that strategy outputs pass through market quality and risk controls before any future
    execution adapter is allowed to act.
  </div>
</main>
</body>
</html>
"""
    return HtmlReport(title=title, html=html)


def build_metrics_html(summary: JournalSummary) -> str:
    metrics = [
        ("Records", f"{summary.total_records:,}"),
        ("Accepted", f"{summary.accepted:,}"),
        ("Blocked", f"{summary.blocked:,}"),
        ("Accepted Rate", f"{summary.accepted_rate:.1%}"),
        ("Requested Notional", money(summary.requested_notional_usd)),
        ("Adjusted Notional", money(summary.adjusted_notional_usd)),
        ("Trimmed By Risk", money(summary.trimmed_notional_usd)),
    ]
    cards = "\n".join(
        f"""    <div class="metric">
      <div class="label">{escape(label)}</div>
      <div class="value">{escape(value)}</div>
    </div>"""
        for label, value in metrics
    )
    return f"""  <section>
    <h2>Summary</h2>
    <div class="grid">
{cards}
    </div>
  </section>"""


def build_count_table(title: str, rows: dict[str, int]) -> str:
    body = "\n".join(
        f"<tr><td>{escape(key)}</td><td>{count:,}</td></tr>" for key, count in rows.items()
    )
    if not body:
        body = "<tr><td colspan=\"2\">No records</td></tr>"
    return f"""  <section>
    <h2>{escape(title)}</h2>
    <table>
      <thead><tr><th>Name</th><th>Count</th></tr></thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </section>"""


def build_symbol_table(summary: JournalSummary) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{escape(row.symbol)}</td>"
        f"<td>{row.count:,}</td>"
        f"<td>{row.accepted:,}</td>"
        f"<td>{row.blocked:,}</td>"
        f"<td>{money(row.requested_notional_usd)}</td>"
        f"<td>{money(row.adjusted_notional_usd)}</td>"
        f"<td>{money(row.trimmed_notional_usd)}</td>"
        "</tr>"
        for row in summary.by_symbol
    )
    if not body:
        body = "<tr><td colspan=\"7\">No records</td></tr>"
    return f"""  <section>
    <h2>By Symbol</h2>
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Records</th><th>Accepted</th><th>Blocked</th>
          <th>Requested</th><th>Adjusted</th><th>Trimmed</th>
        </tr>
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </section>"""


def build_recent_table(records: list[dict]) -> str:
    body = "\n".join(_recent_row(record) for record in records)
    if not body:
        body = "<tr><td colspan=\"7\">No records</td></tr>"
    return f"""  <section>
    <h2>Recent Records</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th><th>Status</th><th>Mode</th><th>Side</th>
          <th>Symbol</th><th>Requested</th><th>Adjusted</th>
        </tr>
      </thead>
      <tbody>
        {body}
      </tbody>
    </table>
  </section>"""


def _recent_row(record: dict) -> str:
    request = record.get("request", {})
    decision = record.get("decision", {})
    status = str(record.get("status", "UNKNOWN"))
    status_class = "status-accepted" if bool(decision.get("approved", False)) else "status-blocked"
    return (
        "<tr>"
        f"<td>{escape(str(record.get('created_at_utc', 'unknown')))}</td>"
        f"<td class=\"{status_class}\">{escape(status)}</td>"
        f"<td>{escape(str(record.get('mode', 'UNKNOWN')))}</td>"
        f"<td>{escape(str(request.get('side', 'UNKNOWN')))}</td>"
        f"<td>{escape(str(request.get('symbol', 'UNKNOWN')))}</td>"
        f"<td>{money(_as_float(request.get('target_notional_usd', 0.0)))}</td>"
        f"<td>{money(_as_float(decision.get('adjusted_notional_usd', 0.0)))}</td>"
        "</tr>"
    )


def money(value: float) -> str:
    return f"${value:,.0f}"


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

