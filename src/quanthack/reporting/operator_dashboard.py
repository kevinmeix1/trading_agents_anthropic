from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class OperatorDashboard:
    title: str
    html: str


@dataclass(frozen=True)
class DashboardSource:
    label: str
    path: Path
    rows: int
    status: str
    message: str
    modified_at: str


def build_operator_dashboard(
    *,
    profile_pack_json: str | Path,
    snapshot_csv: str | Path,
    allocation_csv: str | Path,
    monitor_csv: str | Path,
    ticket_csv: str | Path,
    title: str = "Claude Agent Trader Live Operator Dashboard",
    generated_at: datetime | None = None,
) -> OperatorDashboard:
    generated = generated_at or datetime.now(tz=ZoneInfo("Europe/London"))
    profile_pack, profile_source = _read_json_source("Profile pack", profile_pack_json)
    snapshot_rows, snapshot_source = _read_csv_source("Profile snapshot", snapshot_csv)
    allocation_rows, allocation_source = _read_csv_source("Live allocation", allocation_csv)
    monitor_rows, monitor_source = _read_csv_source("Live monitor", monitor_csv)
    ticket_rows, ticket_source = _read_csv_source("MT5 ticket sheet", ticket_csv)
    sources = (
        profile_source,
        snapshot_source,
        allocation_source,
        monitor_source,
        ticket_source,
    )
    profile_label = _profile_label(snapshot_rows, ticket_rows, profile_pack)
    latest_monitor = monitor_rows[-1] if monitor_rows else {}
    latest_allocation = allocation_rows[-1] if allocation_rows else {}
    actionable = [row for row in snapshot_rows if row.get("order_side") != "HOLD"]
    ready_tickets = [row for row in ticket_rows if row.get("status") == "READY"]
    ticket_status_counts = Counter(row.get("status", "UNKNOWN") for row in ticket_rows)
    operational_status, operational_reason = _operational_status(
        sources=sources,
        actionable_count=len(actionable),
        ticket_status_counts=ticket_status_counts,
        allocation_status=latest_allocation.get("estimated_risk_status", ""),
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
      --bg: #f4f6f8;
      --ink: #17202f;
      --muted: #5b6472;
      --panel: #ffffff;
      --line: #d7dee8;
      --ok: #0f766e;
      --ok-bg: #def7f1;
      --warn: #9a3412;
      --warn-bg: #fff7ed;
      --bad: #991b1b;
      --bad-bg: #fef2f2;
      --blue: #1d4ed8;
      --blue-bg: #eff6ff;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.42;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .metric, .panel {{
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
      font-weight: 700;
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
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #eef2f6;
      color: #303947;
      font-weight: 700;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ok {{ background: var(--ok-bg); color: var(--ok); }}
    .warn {{ background: var(--warn-bg); color: var(--warn); }}
    .bad {{ background: var(--bad-bg); color: var(--bad); }}
    .info {{ background: var(--blue-bg); color: var(--blue); }}
    .note {{
      background: var(--warn-bg);
      border: 1px solid #fed7aa;
      color: var(--warn);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 14px;
      margin-top: 16px;
    }}
    @media (max-width: 720px) {{
      header {{ display: block; }}
      table {{ display: block; overflow-x: auto; }}
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
    <div>
      {status_badge(operational_status)}
      <div class="muted">{escape(operational_reason)}</div>
    </div>
  </header>

  <section>
    <h2>Operating View</h2>
    <div class="grid">
      {metric("Profile", profile_label)}
      {metric("Snapshot Time", _latest_timestamp(snapshot_rows, ticket_rows))}
      {metric("Actionable Signals", str(len(actionable)))}
      {metric("Ready Tickets", str(len(ready_tickets)))}
      {metric("Ticket Blocks", str(_ticket_block_count(ticket_status_counts)))}
      {metric("Allocation Status", latest_allocation.get("estimated_risk_status", "missing"))}
      {metric("Equity", money(_float_value(latest_monitor, "equity")))}
      {metric("Live Leverage", ratio(_float_value(latest_monitor, "leverage")))}
      {metric("Risk Score", _risk_score_text(latest_monitor))}
      {metric("Accepted Trades", latest_monitor.get("accepted_trade_count", "missing"))}
    </div>
  </section>

  {build_next_actions_section(actionable, ticket_status_counts, sources)}
  {build_ticket_section(ticket_rows)}
  {build_snapshot_section(snapshot_rows)}
  {build_allocation_section(latest_allocation)}
  {build_monitor_section(latest_monitor)}
  {build_profile_pack_section(profile_pack)}
  {build_sources_section(sources)}

  <div class="note">
    Competition safety reminder: this dashboard is an operator review surface.
    It does not connect to MT5, place orders, or override risk gates. For crypto
    and metals, fill MT5 contract specifications before using ticket lot sizes.
  </div>
</main>
</body>
</html>
"""
    return OperatorDashboard(title=title, html=html)


def build_next_actions_section(
    actionable: list[dict[str, str]],
    ticket_status_counts: Counter[str],
    sources: tuple[DashboardSource, ...],
) -> str:
    actions: list[str] = []
    missing = [source.label for source in sources if source.status != "OK"]
    if missing:
        actions.append("Regenerate or provide missing artifacts: " + ", ".join(missing))
    if ticket_status_counts.get("NEEDS_CONTRACT_SPEC", 0):
        actions.append("Fill MT5 contract specs for crypto/metals before sizing tickets.")
    if ticket_status_counts.get("NEEDS_QUOTE_USD_RATE", 0):
        actions.append("Add quote-to-USD rates for non-USD quote crosses.")
    if ticket_status_counts.get("READY", 0):
        actions.append("Review READY tickets in MT5 manually before any order entry.")
    if not actionable and not ticket_status_counts:
        actions.append("No actionable signal in the latest snapshot; hold the book.")
    if not actions:
        actions.append("All visible artifacts are consistent; continue monitoring.")
    rows = "".join(f"<li>{escape(action)}</li>" for action in actions)
    return f"""  <section>
    <h2>Next Actions</h2>
    <div class="panel"><ul>{rows}</ul></div>
  </section>"""


def build_ticket_section(rows: list[dict[str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(row.get('symbol', ''))}</td>"
        f"<td>{escape(row.get('broker_symbol', ''))}</td>"
        f"<td>{escape(row.get('side', ''))}</td>"
        f"<td>{status_badge(row.get('status', 'UNKNOWN'))}</td>"
        f"<td>{money(_float_value(row, 'action_notional_usd'))}</td>"
        f"<td>{_lots(row)}</td>"
        f"<td>{escape(row.get('instruction', ''))}</td>"
        "</tr>"
        for row in rows
    )
    if not body:
        body = "<tr><td colspan=\"7\">No ticket rows</td></tr>"
    return f"""  <section>
    <h2>MT5 Ticket Sheet</h2>
    <table>
      <thead>
        <tr><th>Symbol</th><th>Broker</th><th>Side</th><th>Status</th>
        <th>Action Notional</th><th>Lots</th><th>Instruction</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </section>"""


def build_snapshot_section(rows: list[dict[str, str]]) -> str:
    actionable = [row for row in rows if row.get("order_side") != "HOLD"]
    display_rows = actionable if actionable else rows[:8]
    body = "".join(
        "<tr>"
        f"<td>{escape(row.get('symbol', ''))}</td>"
        f"<td>{escape(row.get('order_side', ''))}</td>"
        f"<td>{money(_float_value(row, 'change_notional_usd'))}</td>"
        f"<td>{money(_float_value(row, 'allocated_target_notional_usd'))}</td>"
        f"<td>{status_badge('OK' if row.get('risk_approved') == 'True' else 'FAIL')}</td>"
        f"<td>{escape(row.get('strategy_reason', ''))}</td>"
        "</tr>"
        for row in display_rows
    )
    if not body:
        body = "<tr><td colspan=\"6\">No snapshot rows</td></tr>"
    return f"""  <section>
    <h2>Profile Snapshot</h2>
    <table>
      <thead>
        <tr><th>Symbol</th><th>Side</th><th>Change</th><th>Target</th>
        <th>Risk</th><th>Reason</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </section>"""


def build_allocation_section(row: dict[str, str]) -> str:
    if not row:
        body = "<tr><td colspan=\"2\">No allocation report row</td></tr>"
    else:
        rows = (
            ("Timestamp", row.get("timestamp", "")),
            ("Requested Gross", money(_float_value(row, "requested_gross_notional_usd"))),
            ("Adjusted Gross", money(_float_value(row, "adjusted_gross_notional_usd"))),
            ("Net Directional", pct(_float_value(row, "net_directional_exposure"))),
            ("Largest Symbol", pct(_float_value(row, "largest_symbol_concentration"))),
            ("Active Symbols", row.get("active_symbols", "")),
            ("Status", row.get("estimated_risk_status", "")),
            ("Trim Reasons", row.get("trim_reasons", "")),
        )
        body = "".join(
            f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
            for label, value in rows
        )
    return f"""  <section>
    <h2>Latest Allocation</h2>
    <table><tbody>{body}</tbody></table>
  </section>"""


def build_monitor_section(row: dict[str, str]) -> str:
    if not row:
        body = "<tr><td colspan=\"2\">No monitor row</td></tr>"
    else:
        rows = (
            ("Timestamp", row.get("timestamp", "")),
            ("Equity", money(_float_value(row, "equity"))),
            ("Daily P&L", pct(_float_value(row, "daily_pnl_pct"))),
            ("Drawdown", pct(_float_value(row, "drawdown_pct"))),
            ("Margin Level", pct(_float_value(row, "margin_level_pct") / 100)),
            ("Gross Notional", money(_float_value(row, "gross_notional_usd"))),
            ("Net Notional", money(_float_value(row, "net_notional_usd"))),
            ("Accepted Trades", row.get("accepted_trade_count", "")),
        )
        body = "".join(
            f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
            for label, value in rows
        )
    return f"""  <section>
    <h2>Live Monitor</h2>
    <table><tbody>{body}</tbody></table>
  </section>"""


def build_profile_pack_section(profile_pack: dict) -> str:
    profiles = profile_pack.get("profiles", []) if isinstance(profile_pack, dict) else []
    body = "".join(
        "<tr>"
        f"<td>{escape(str(profile.get('slot', '')))}</td>"
        f"<td>{escape(str(profile.get('label', '')))}</td>"
        f"<td>{status_badge(str(profile.get('evidence_status', 'UNKNOWN')))}</td>"
        f"<td>{pct(float(profile.get('return_pct', 0.0)))}</td>"
        f"<td>{pct(float(profile.get('max_drawdown_pct', 0.0)))}</td>"
        f"<td>{escape(str(profile.get('fold_contribution', '')))}</td>"
        "</tr>"
        for profile in profiles
    )
    if not body:
        body = "<tr><td colspan=\"6\">No profile pack rows</td></tr>"
    recommended = (
        str(profile_pack.get("recommended_slot", "missing"))
        if profile_pack
        else "missing"
    )
    reason = str(profile_pack.get("recommendation_reason", "")) if profile_pack else ""
    return f"""  <section>
    <h2>Deployment Profiles</h2>
    <div class="muted">Recommendation: {escape(recommended)} {escape(reason)}</div>
    <table>
      <thead>
        <tr><th>Slot</th><th>Label</th><th>Evidence</th><th>Return</th>
        <th>Max DD</th><th>Fold Contribution</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </section>"""


def build_sources_section(sources: tuple[DashboardSource, ...]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{escape(source.label)}</td>"
        f"<td>{status_badge(source.status)}</td>"
        f"<td>{source.rows}</td>"
        f"<td>{escape(str(source.path))}</td>"
        f"<td>{escape(source.modified_at)}</td>"
        f"<td>{escape(source.message)}</td>"
        "</tr>"
        for source in sources
    )
    return f"""  <section>
    <h2>Sources</h2>
    <table>
      <thead>
        <tr><th>Source</th><th>Status</th><th>Rows</th><th>Path</th>
        <th>Modified</th><th>Message</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </section>"""


def metric(label: str, value: str) -> str:
    return f"""<div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
      </div>"""


def status_badge(status: str) -> str:
    normalized = status.strip().upper()
    css = "info"
    if normalized in {"OK", "READY", "PROMOTE", "TRUE"}:
        css = "ok"
    elif normalized in {
        "WARN",
        "PAPER_ONLY",
        "NEEDS_CONTRACT_SPEC",
        "NEEDS_QUOTE_USD_RATE",
        "BELOW_MIN_VOLUME",
        "HOLD",
    }:
        css = "warn"
    elif normalized in {"FAIL", "MISSING", "BLOCKED_BY_RISK", "PENALTY_RISK"}:
        css = "bad"
    return f"<span class=\"badge {css}\">{escape(status)}</span>"


def money(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"${value:,.0f}"


def pct(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.1%}"


def ratio(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.2f}x"


def _read_csv_source(label: str, path: str | Path) -> tuple[list[dict[str, str]], DashboardSource]:
    source_path = Path(path)
    if not source_path.exists():
        return [], _source(label, source_path, 0, "MISSING", "file not found")
    try:
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return [], _source(label, source_path, 0, "FAIL", str(exc))
    return rows, _source(label, source_path, len(rows), "OK", "loaded")


def _read_json_source(label: str, path: str | Path) -> tuple[dict, DashboardSource]:
    source_path = Path(path)
    if not source_path.exists():
        return {}, _source(label, source_path, 0, "MISSING", "file not found")
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, _source(label, source_path, 0, "FAIL", str(exc))
    row_count = len(data.get("profiles", [])) if isinstance(data, dict) else 1
    return data, _source(label, source_path, row_count, "OK", "loaded")


def _source(
    label: str,
    path: Path,
    rows: int,
    status: str,
    message: str,
) -> DashboardSource:
    modified = ""
    if path.exists():
        modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return DashboardSource(
        label=label,
        path=path,
        rows=rows,
        status=status,
        message=message,
        modified_at=modified,
    )


def _profile_label(
    snapshot_rows: list[dict[str, str]],
    ticket_rows: list[dict[str, str]],
    profile_pack: dict,
) -> str:
    rows = snapshot_rows or ticket_rows
    if rows:
        return f"{rows[0].get('profile_slot', '')}: {rows[0].get('profile_label', '')}"
    profiles = profile_pack.get("profiles", []) if isinstance(profile_pack, dict) else []
    if profiles:
        profile = profiles[0]
        return f"{profile.get('slot', '')}: {profile.get('label', '')}"
    return "missing"


def _latest_timestamp(
    snapshot_rows: list[dict[str, str]],
    ticket_rows: list[dict[str, str]],
) -> str:
    rows = ticket_rows or snapshot_rows
    return rows[0].get("timestamp", "missing") if rows else "missing"


def _operational_status(
    *,
    sources: tuple[DashboardSource, ...],
    actionable_count: int,
    ticket_status_counts: Counter[str],
    allocation_status: str,
) -> tuple[str, str]:
    if any(source.status != "OK" for source in sources):
        return "WARN", "one or more dashboard sources are missing"
    if ticket_status_counts.get("BLOCKED_BY_RISK", 0):
        return "FAIL", "risk-blocked ticket requires attention"
    if ticket_status_counts.get("READY", 0):
        return "READY", "manual ticket review required"
    if ticket_status_counts.get("NEEDS_CONTRACT_SPEC", 0):
        return "WARN", "fill MT5 contract specs before ticket sizing"
    if allocation_status == "WARN":
        return "WARN", "allocation guardrails show warning"
    if actionable_count == 0:
        return "OK", "latest profile evaluation is HOLD"
    return "OK", "artifacts loaded"


def _ticket_block_count(status_counts: Counter[str]) -> int:
    return sum(
        status_counts.get(status, 0)
        for status in ("NEEDS_CONTRACT_SPEC", "NEEDS_QUOTE_USD_RATE", "BLOCKED_BY_RISK")
    )


def _risk_score_text(row: dict[str, str]) -> str:
    if not row:
        return "missing"
    leverage = _float_value(row, "leverage") or 0.0
    concentration = _float_value(row, "single_symbol_concentration") or 0.0
    if leverage > 2.0 or concentration > 0.8:
        return "watch"
    return "normal"


def _float_value(row: dict[str, str], key: str) -> float | None:
    try:
        raw = row.get(key, "")
        if raw == "":
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _lots(row: dict[str, str]) -> str:
    lots = _float_value(row, "rounded_lots")
    if lots is None or lots <= 0:
        return ""
    return f"{lots:.4f}"
