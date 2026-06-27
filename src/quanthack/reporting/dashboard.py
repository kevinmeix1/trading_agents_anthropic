from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import webbrowser

from quanthack.reporting.journal_report import summarize_journal
from quanthack.trading.execution import portfolio_from_journal, read_journal


DEFAULT_BACKTEST_DIR = Path("outputs/backtests")
DEFAULT_LIVE_MONITOR = Path("outputs/live_competition_monitor.csv")
DEFAULT_LIVE_JOURNAL = Path("outputs/live_dry_run_journal.jsonl")
DEFAULT_DRY_JOURNAL = Path("outputs/dry_run_journal.jsonl")


@dataclass(frozen=True)
class DashboardOptions:
    backtest_dir: Path = DEFAULT_BACKTEST_DIR
    live_monitor_path: Path = DEFAULT_LIVE_MONITOR
    live_journal_path: Path = DEFAULT_LIVE_JOURNAL
    dry_journal_path: Path = DEFAULT_DRY_JOURNAL


def build_dashboard_payload(options: DashboardOptions | None = None) -> dict[str, Any]:
    settings = options or DashboardOptions()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "backtest_dir": _source_status(settings.backtest_dir),
            "live_monitor": _source_status(settings.live_monitor_path),
            "live_journal": _source_status(settings.live_journal_path),
            "dry_journal": _source_status(settings.dry_journal_path),
        },
        "backtests": {
            "comparisons": _comparison_files(settings.backtest_dir),
            "walk_forward": _walk_forward_files(settings.backtest_dir),
            "equity_curves": _equity_curve_files(settings.backtest_dir),
        },
        "live": _live_section(settings.live_monitor_path, settings.live_journal_path),
        "dry_run": _journal_section(settings.dry_journal_path),
    }


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    options: DashboardOptions | None = None,
    open_browser: bool = False,
) -> None:
    settings = options or DashboardOptions()
    handler = _make_handler(settings)
    server = _bind_server(host=host, port=port, handler=handler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print(f"Claude Agent Trader dashboard: {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Claude Agent Trader dashboard.")
    finally:
        server.server_close()


def _make_handler(options: DashboardOptions) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_text(DASHBOARD_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/dashboard":
                payload = build_dashboard_payload(options)
                self._send_json(payload)
                return
            if parsed.path == "/health":
                self._send_json({"ok": True})
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404, "Not found")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def _bind_server(
    *,
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    attempts = (port,) if port == 0 else tuple(range(port, port + 10))
    last_error: OSError | None = None
    for candidate in attempts:
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            last_error = exc
    raise OSError(f"could not bind dashboard server near port {port}") from last_error


def _comparison_files(backtest_dir: Path) -> list[dict[str, Any]]:
    files = sorted(backtest_dir.glob("*comparison.csv")) if backtest_dir.exists() else []
    comparisons: list[dict[str, Any]] = []
    for path in files:
        rows = _read_csv(path)
        normalized_rows = [_normalize_comparison_row(row) for row in rows]
        comparisons.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": _mtime(path),
                "kind": _comparison_kind(rows),
                "row_count": len(rows),
                "best": normalized_rows[0] if normalized_rows else None,
                "rows": normalized_rows,
            }
        )
    comparisons.sort(key=lambda item: item["mtime"] or "", reverse=True)
    return comparisons


def _walk_forward_files(backtest_dir: Path) -> list[dict[str, Any]]:
    files = sorted(backtest_dir.glob("*walk_forward*summary.csv")) if backtest_dir.exists() else []
    summaries: list[dict[str, Any]] = []
    for path in files:
        rows = _read_csv(path)
        summaries.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": _mtime(path),
                "row_count": len(rows),
                "rows": rows,
            }
        )
    summaries.sort(key=lambda item: item["mtime"] or "", reverse=True)
    return summaries


def _equity_curve_files(backtest_dir: Path) -> list[dict[str, Any]]:
    files = sorted(backtest_dir.glob("*equity*.csv")) if backtest_dir.exists() else []
    curves: list[dict[str, Any]] = []
    for path in files:
        rows = _read_csv(path)
        points = [
            {
                "timestamp": str(row.get("timestamp", "")),
                "equity": _as_float(row.get("equity")),
                "drawdown_pct": _as_float(row.get("drawdown_pct")),
                "gross_notional_usd": _as_float(row.get("gross_notional_usd")),
                "net_notional_usd": _as_float(row.get("net_notional_usd")),
            }
            for row in rows
            if row.get("timestamp") and row.get("equity") is not None
        ]
        curves.append(
            {
                "name": path.name,
                "path": str(path),
                "mtime": _mtime(path),
                "point_count": len(points),
                "start": points[0]["timestamp"] if points else None,
                "end": points[-1]["timestamp"] if points else None,
                "latest": points[-1] if points else None,
                "points": _thin_points(points, limit=500),
            }
        )
    curves.sort(key=lambda item: item["mtime"] or "", reverse=True)
    return curves


def _live_section(monitor_path: Path, journal_path: Path) -> dict[str, Any]:
    monitor_rows = _read_csv(monitor_path)
    latest = monitor_rows[-1] if monitor_rows else None
    return {
        "ready": bool(monitor_rows or journal_path.exists()),
        "monitor": {
            "path": str(monitor_path),
            "exists": monitor_path.exists(),
            "mtime": _mtime(monitor_path),
            "row_count": len(monitor_rows),
            "latest": latest,
            "points": _thin_points(
                [
                    {
                        "timestamp": str(row.get("timestamp", "")),
                        "equity": _as_float(row.get("equity")),
                        "drawdown_pct": _as_float(row.get("drawdown_pct")),
                        "gross_notional_usd": _as_float(row.get("gross_notional_usd")),
                        "net_notional_usd": _as_float(row.get("net_notional_usd")),
                    }
                    for row in monitor_rows
                ],
                limit=500,
            ),
        },
        "journal": _journal_section(journal_path),
    }


def _journal_section(path: Path) -> dict[str, Any]:
    records = read_journal(path)
    summary = summarize_journal(records)
    portfolio = portfolio_from_journal(records)
    return {
        "path": str(path),
        "exists": path.exists(),
        "mtime": _mtime(path),
        "records": len(records),
        "accepted": summary.accepted,
        "blocked": summary.blocked,
        "accepted_rate": summary.accepted_rate,
        "requested_notional_usd": summary.requested_notional_usd,
        "adjusted_notional_usd": summary.adjusted_notional_usd,
        "trimmed_notional_usd": summary.trimmed_notional_usd,
        "by_status": summary.by_status,
        "by_mode": summary.by_mode,
        "by_symbol": [
            {
                "symbol": row.symbol,
                "count": row.count,
                "accepted": row.accepted,
                "blocked": row.blocked,
                "requested_notional_usd": row.requested_notional_usd,
                "adjusted_notional_usd": row.adjusted_notional_usd,
                "trimmed_notional_usd": row.trimmed_notional_usd,
            }
            for row in summary.by_symbol
        ],
        "positions": [
            {"symbol": position.symbol, "notional_usd": position.notional_usd}
            for position in portfolio.positions
        ],
        "latest_records": records[-25:],
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def _source_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "mtime": _mtime(path),
    }


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )


def _comparison_kind(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    first = rows[0]
    if "symbols" in first:
        return "portfolio"
    if "symbol" in first:
        return "single_symbol"
    return "unknown"


def _normalize_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": _as_float(row.get("rank")),
        "strategy": str(row.get("strategy", "")),
        "symbol": str(row.get("symbol") or row.get("symbols") or ""),
        "proxy_score": _as_float(row.get("proxy_score")),
        "final_equity": _as_float(row.get("final_equity")),
        "return_pct": _as_float(row.get("total_return_pct", row.get("official_return_pct"))),
        "drawdown_pct": _as_float(
            row.get("max_drawdown_pct", row.get("official_max_drawdown_pct"))
        ),
        "sharpe": _as_float(row.get("sharpe_ratio", row.get("official_15m_sharpe"))),
        "fills": _as_float(row.get("fills", row.get("trade_count"))),
        "turnover_notional": _as_float(row.get("turnover_notional")),
        "risk_discipline_score": _as_float(row.get("risk_discipline_score")),
    }


def _thin_points(points: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    thinned = points[::step]
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])
    return thinned


def _coerce(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return ""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claude Agent Trader Dashboard</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9e0ea;
      --blue: #2563eb;
      --green: #147a50;
      --amber: #a15c07;
      --red: #b42318;
      --violet: #6d42c7;
      --shadow: 0 1px 3px rgba(16, 24, 40, 0.08);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 20;
    }

    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }

    .subline {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }

    .tabs {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #f8fafc;
      min-height: 36px;
    }

    .tab {
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      padding: 8px 12px;
      min-width: 96px;
      cursor: pointer;
      font: inherit;
      font-weight: 600;
    }

    .tab:last-child { border-right: 0; }
    .tab.active {
      background: var(--blue);
      color: #ffffff;
    }

    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 36px;
    }

    .view { display: none; }
    .view.active { display: block; }

    .grid {
      display: grid;
      gap: 14px;
    }

    .kpis {
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      margin-bottom: 14px;
    }

    .two {
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
    }

    .three {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }

    .card.pad { padding: 14px; }

    .card-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      min-height: 48px;
    }

    .card-title h2 {
      margin: 0;
      font-size: 14px;
      font-weight: 700;
    }

    .kpi {
      min-height: 104px;
      padding: 13px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .kpi .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }

    .kpi .value {
      font-size: 23px;
      font-weight: 760;
      margin-top: 8px;
      overflow-wrap: anywhere;
    }

    .kpi .note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }

    .status {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--line);
      white-space: nowrap;
    }

    .status.ok { color: var(--green); background: #ecfdf3; border-color: #abefc6; }
    .status.warn { color: var(--amber); background: #fffaeb; border-color: #fedf89; }
    .status.bad { color: var(--red); background: #fef3f2; border-color: #fecdca; }

    select, button.control {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      padding: 6px 10px;
      font: inherit;
      max-width: 100%;
    }

    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }

    .chart {
      width: 100%;
      height: 280px;
      display: block;
    }

    .chart.small { height: 220px; }

    .table-wrap {
      width: 100%;
      overflow-x: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
    }

    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }

    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) {
      text-align: left;
    }

    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #f8fafc;
    }

    .empty {
      padding: 22px 14px;
      color: var(--muted);
    }

    .sources {
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }

    .source-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 10px;
    }

    .source-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #ffffff;
      min-width: 0;
    }

    .path {
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 12px;
    }

    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .muted { color: var(--muted); }

    @media (max-width: 980px) {
      .topbar { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .two, .three { grid-template-columns: 1fr; }
      .source-grid { grid-template-columns: 1fr 1fr; }
    }

    @media (max-width: 620px) {
      main { padding: 14px 12px 28px; }
      .topbar { padding: 12px; }
      .tabs { width: 100%; }
      .tab { flex: 1; min-width: 0; padding: 8px 6px; }
      .kpis { grid-template-columns: 1fr; }
      .source-grid { grid-template-columns: 1fr; }
      .chart { height: 240px; }
      .kpi .value { font-size: 20px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div>
        <h1>Claude Agent Trader Trading Dashboard</h1>
        <div class="subline" id="freshness">Loading local results...</div>
      </div>
      <nav class="tabs" aria-label="Dashboard views">
        <button class="tab active" data-view="overview">Overview</button>
        <button class="tab" data-view="backtests">Backtests</button>
        <button class="tab" data-view="live">Live</button>
      </nav>
    </div>
  </header>

  <main>
    <section id="overview" class="view active">
      <div class="grid kpis" id="overviewKpis"></div>
      <div class="grid two">
        <section class="card">
          <div class="card-title">
            <h2>Strategy Ranking</h2>
            <div class="toolbar">
              <select id="overviewComparison"></select>
            </div>
          </div>
          <svg id="overviewChart" class="chart" role="img" aria-label="Strategy return chart"></svg>
        </section>
        <section class="card">
          <div class="card-title">
            <h2>Latest Equity Curve</h2>
            <div class="toolbar">
              <select id="overviewEquity"></select>
            </div>
          </div>
          <svg id="overviewEquityChart" class="chart" role="img" aria-label="Equity curve"></svg>
        </section>
      </div>
      <section class="card sources">
        <div class="card-title"><h2>Sources</h2></div>
        <div class="source-grid pad" id="sourceGrid"></div>
      </section>
    </section>

    <section id="backtests" class="view">
      <div class="grid kpis" id="backtestKpis"></div>
      <div class="grid two">
        <section class="card">
          <div class="card-title">
            <h2>Comparison</h2>
            <div class="toolbar">
              <select id="comparisonSelect"></select>
            </div>
          </div>
          <svg id="comparisonChart" class="chart" role="img" aria-label="Comparison chart"></svg>
          <div class="table-wrap"><table id="comparisonTable"></table></div>
        </section>
        <section class="card">
          <div class="card-title">
            <h2>Walk-Forward</h2>
            <div class="toolbar">
              <select id="walkForwardSelect"></select>
            </div>
          </div>
          <div class="table-wrap"><table id="walkForwardTable"></table></div>
        </section>
      </div>
    </section>

    <section id="live" class="view">
      <div class="grid kpis" id="liveKpis"></div>
      <div class="grid two">
        <section class="card">
          <div class="card-title">
            <h2>Live Equity And Exposure</h2>
            <span id="liveStatus" class="status warn">WAITING</span>
          </div>
          <svg id="liveChart" class="chart" role="img" aria-label="Live equity curve"></svg>
        </section>
        <section class="card">
          <div class="card-title"><h2>Current Positions</h2></div>
          <div class="table-wrap"><table id="positionTable"></table></div>
        </section>
      </div>
      <section class="card" style="margin-top:14px">
        <div class="card-title"><h2>Recent Live Journal</h2></div>
        <div class="table-wrap"><table id="journalTable"></table></div>
      </section>
    </section>
  </main>

  <script>
    const state = { data: null };

    const fmtMoney = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0
    });
    const fmtNum = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
    const fmtPct = new Intl.NumberFormat("en-US", {
      style: "percent",
      minimumFractionDigits: 3,
      maximumFractionDigits: 3
    });

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.view).classList.add("active");
        if (state.data) render();
      });
    });

    async function loadDashboard() {
      const response = await fetch("/api/dashboard", { cache: "no-store" });
      state.data = await response.json();
      render();
    }

    function render() {
      const data = state.data;
      document.getElementById("freshness").textContent =
        `Last refresh ${shortTime(data.generated_at)} from local project outputs`;
      populateSelectors(data);
      renderOverview(data);
      renderBacktests(data);
      renderLive(data);
      renderSources(data.sources);
    }

    function populateSelectors(data) {
      fillSelect("overviewComparison", data.backtests.comparisons, "name");
      fillSelect("comparisonSelect", data.backtests.comparisons, "name");
      fillSelect("overviewEquity", data.backtests.equity_curves, "name");
      fillSelect("walkForwardSelect", data.backtests.walk_forward, "name");
    }

    function fillSelect(id, items, key) {
      const select = document.getElementById(id);
      const current = select.value;
      select.innerHTML = "";
      if (!items.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No files found";
        select.appendChild(option);
        return;
      }
      items.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = item[key];
        select.appendChild(option);
      });
      if (current && Number(current) < items.length) select.value = current;
      select.onchange = render;
    }

    function renderOverview(data) {
      const comparisons = data.backtests.comparisons;
      const selectedComparison = selectedItem("overviewComparison", comparisons);
      const best = selectedComparison && selectedComparison.best;
      const live = data.live;
      const liveLatest = live.monitor.latest || {};
      const dry = data.dry_run;
      setKpis("overviewKpis", [
        ["Best Strategy", best ? best.strategy : "No backtests", best ? selectedComparison.name : ""],
        ["Best Return", best ? pct(best.return_pct) : "-", best ? `Sharpe ${num(best.sharpe)}` : ""],
        ["Live Status", live.ready ? "Ready" : "Waiting", live.monitor.exists ? live.monitor.path : "No live monitor yet"],
        ["Live Equity", live.monitor.latest ? money(liveLatest.equity) : "-", live.monitor.latest ? `Drawdown ${pct(liveLatest.drawdown_pct)}` : ""],
        ["Dry-Run Records", String(dry.records), `${dry.accepted} accepted, ${dry.blocked} blocked`]
      ]);
      renderBarChart("overviewChart", selectedComparison ? selectedComparison.rows : [], "return_pct", "Return");
      const equity = selectedItem("overviewEquity", data.backtests.equity_curves);
      renderLineChart("overviewEquityChart", equity ? equity.points : [], "equity", "Equity");
    }

    function renderBacktests(data) {
      const comparison = selectedItem("comparisonSelect", data.backtests.comparisons);
      const best = comparison && comparison.best;
      setKpis("backtestKpis", [
        ["Comparison Files", String(data.backtests.comparisons.length), "outputs/backtests"],
        ["Best Strategy", best ? best.strategy : "-", best ? comparison.name : ""],
        ["Best Final Equity", best ? money(best.final_equity) : "-", best ? `Return ${pct(best.return_pct)}` : ""],
        ["Equity Curves", String(data.backtests.equity_curves.length), "available curves"],
        ["Walk-Forward Files", String(data.backtests.walk_forward.length), "summary CSVs"]
      ]);
      renderBarChart("comparisonChart", comparison ? comparison.rows : [], "return_pct", "Return");
      renderComparisonTable(comparison ? comparison.rows : []);
      const wf = selectedItem("walkForwardSelect", data.backtests.walk_forward);
      renderGenericTable("walkForwardTable", wf ? wf.rows : [], [
        "rank", "strategy", "eligible", "folds", "median_test_sharpe",
        "lower_quartile_test_return", "worst_test_drawdown", "profitable_fold_fraction",
        "total_test_fills", "most_selected_strategy", "stable_fold_fraction",
        "average_risk_discipline_score"
      ]);
    }

    function renderLive(data) {
      const live = data.live;
      const latest = live.monitor.latest || {};
      const journal = live.journal;
      const status = document.getElementById("liveStatus");
      status.textContent = live.ready ? "READY" : "WAITING";
      status.className = `status ${live.ready ? "ok" : "warn"}`;
      setKpis("liveKpis", [
        ["Equity", live.monitor.latest ? money(latest.equity) : "-", live.monitor.exists ? live.monitor.path : "Live monitor not found"],
        ["Daily P&L", live.monitor.latest ? pct(latest.daily_pnl_pct) : "-", "latest monitor snapshot"],
        ["Drawdown", live.monitor.latest ? pct(latest.drawdown_pct) : "-", "latest monitor snapshot"],
        ["Margin Level", live.monitor.latest ? `${num(latest.margin_level_pct)}%` : "-", "internal safety floor applies"],
        ["Live Journal", `${journal.records} records`, `${journal.accepted} accepted, ${journal.blocked} blocked`]
      ]);
      renderLineChart("liveChart", live.monitor.points || [], "equity", "Equity");
      renderGenericTable("positionTable", journal.positions || [], ["symbol", "notional_usd"]);
      renderJournalTable(journal.latest_records || []);
    }

    function renderSources(sources) {
      const grid = document.getElementById("sourceGrid");
      grid.innerHTML = "";
      Object.entries(sources).forEach(([name, source]) => {
        const item = document.createElement("div");
        item.className = "source-item";
        const badge = source.exists ? '<span class="status ok">FOUND</span>' : '<span class="status warn">WAITING</span>';
        item.innerHTML = `<div style="display:flex;justify-content:space-between;gap:8px;align-items:center">
          <strong>${escapeHtml(name)}</strong>${badge}</div>
          <div class="path">${escapeHtml(source.path)}</div>
          <div class="muted">${source.mtime ? shortTime(source.mtime) : "No timestamp"}</div>`;
        grid.appendChild(item);
      });
    }

    function setKpis(id, items) {
      const root = document.getElementById(id);
      root.innerHTML = "";
      items.forEach(([label, value, note]) => {
        const card = document.createElement("div");
        card.className = "card kpi";
        const valClass = String(value).startsWith("-") ? "value negative" : "value";
        card.innerHTML = `<div><div class="label">${escapeHtml(label)}</div>
          <div class="${valClass}">${escapeHtml(value)}</div></div>
          <div class="note">${escapeHtml(note || "")}</div>`;
        root.appendChild(card);
      });
    }

    function renderComparisonTable(rows) {
      renderTable("comparisonTable", rows, [
        ["rank", "Rank"],
        ["strategy", "Strategy"],
        ["symbol", "Symbol(s)"],
        ["final_equity", "Final Equity", money],
        ["return_pct", "Return", pct],
        ["drawdown_pct", "Max DD", pct],
        ["sharpe", "Sharpe", num],
        ["fills", "Fills", int],
        ["proxy_score", "Proxy", num]
      ]);
    }

    function renderJournalTable(records) {
      const rows = records.slice().reverse().map((record) => ({
        created_at_utc: record.created_at_utc,
        status: record.status,
        mode: record.mode,
        symbol: record.request && record.request.symbol,
        side: record.request && record.request.side,
        requested: record.request && record.request.target_notional_usd,
        adjusted: record.decision && record.decision.adjusted_notional_usd,
        reason: record.decision && record.decision.reason
      }));
      renderTable("journalTable", rows, [
        ["created_at_utc", "Time"],
        ["status", "Status"],
        ["symbol", "Symbol"],
        ["side", "Side"],
        ["requested", "Requested", money],
        ["adjusted", "Adjusted", money],
        ["reason", "Reason"]
      ]);
    }

    function renderGenericTable(id, rows, keys) {
      const present = keys.filter((key) => rows.some((row) => row[key] !== undefined));
      renderTable(id, rows, present.map((key) => [key, titleize(key), formatterForKey(key)]));
    }

    function renderTable(id, rows, columns) {
      const table = document.getElementById(id);
      table.innerHTML = "";
      if (!rows.length) {
        const row = table.insertRow();
        const cell = row.insertCell();
        cell.className = "empty";
        cell.textContent = "No records yet";
        return;
      }
      const thead = table.createTHead();
      const headRow = thead.insertRow();
      columns.forEach(([, label]) => {
        const th = document.createElement("th");
        th.textContent = label;
        headRow.appendChild(th);
      });
      const tbody = table.createTBody();
      rows.forEach((row) => {
        const tr = tbody.insertRow();
        columns.forEach(([key, , formatter]) => {
          const td = tr.insertCell();
          const raw = row[key];
          td.textContent = formatter ? formatter(raw) : String(raw ?? "");
          if (typeof raw === "number" && raw < 0) td.className = "negative";
          if (typeof raw === "number" && raw > 0 && key.includes("return")) td.className = "positive";
        });
      });
    }

    function renderBarChart(id, rows, metric, label) {
      const svg = document.getElementById(id);
      clearSvg(svg);
      const width = svg.clientWidth || 600;
      const height = svg.clientHeight || 280;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      if (!rows.length) return emptyChart(svg, width, height, "No comparison data");
      const margin = { top: 22, right: 18, bottom: 64, left: 56 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const values = rows.map((row) => Number(row[metric]) || 0);
      const min = Math.min(0, ...values);
      const max = Math.max(0, ...values);
      const span = max - min || 1;
      axis(svg, margin, plotW, plotH, min, max, label);
      const barW = Math.max(12, plotW / rows.length - 10);
      rows.forEach((row, index) => {
        const value = Number(row[metric]) || 0;
        const x = margin.left + index * (plotW / rows.length) + 5;
        const y0 = margin.top + ((max - 0) / span) * plotH;
        const y = margin.top + ((max - value) / span) * plotH;
        const h = Math.max(1, Math.abs(y0 - y));
        const rect = svgEl("rect", {
          x, y: Math.min(y, y0), width: barW, height: h,
          rx: 3, fill: value >= 0 ? "var(--green)" : "var(--red)"
        });
        rect.appendChild(svgEl("title", {}, `${row.strategy}: ${pct(value)}`));
        svg.appendChild(rect);
        const labelText = String(row.strategy || "").slice(0, 14);
        svg.appendChild(svgEl("text", {
          x: x + barW / 2, y: height - 18, "text-anchor": "middle",
          fill: "#667085", "font-size": 11
        }, labelText));
      });
    }

    function renderLineChart(id, points, metric, label) {
      const svg = document.getElementById(id);
      clearSvg(svg);
      const width = svg.clientWidth || 600;
      const height = svg.clientHeight || 280;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      const valid = points.filter((point) => Number.isFinite(Number(point[metric])));
      if (!valid.length) return emptyChart(svg, width, height, "No equity data yet");
      const margin = { top: 22, right: 18, bottom: 42, left: 72 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const values = valid.map((point) => Number(point[metric]));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || Math.max(1, Math.abs(max));
      axis(svg, margin, plotW, plotH, min, max, label);
      const d = valid.map((point, index) => {
        const x = margin.left + (index / Math.max(1, valid.length - 1)) * plotW;
        const y = margin.top + ((max - Number(point[metric])) / span) * plotH;
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(" ");
      svg.appendChild(svgEl("path", {
        d, fill: "none", stroke: "var(--blue)", "stroke-width": 2.5,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
      const last = valid[valid.length - 1];
      svg.appendChild(svgEl("circle", {
        cx: margin.left + plotW, cy: margin.top + ((max - Number(last[metric])) / span) * plotH,
        r: 4, fill: "var(--violet)"
      }));
    }

    function axis(svg, margin, plotW, plotH, min, max, label) {
      const y0 = margin.top + plotH;
      svg.appendChild(svgEl("line", { x1: margin.left, y1: y0, x2: margin.left + plotW, y2: y0, stroke: "#d9e0ea" }));
      svg.appendChild(svgEl("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: y0, stroke: "#d9e0ea" }));
      [min, (min + max) / 2, max].forEach((value) => {
        const y = margin.top + ((max - value) / (max - min || 1)) * plotH;
        svg.appendChild(svgEl("line", { x1: margin.left, y1: y, x2: margin.left + plotW, y2: y, stroke: "#edf2f7" }));
        svg.appendChild(svgEl("text", { x: margin.left - 8, y: y + 4, "text-anchor": "end", fill: "#667085", "font-size": 11 }, compact(value)));
      });
      svg.appendChild(svgEl("text", { x: margin.left, y: 14, fill: "#667085", "font-size": 11 }, label));
    }

    function emptyChart(svg, width, height, message) {
      svg.appendChild(svgEl("text", {
        x: width / 2, y: height / 2, "text-anchor": "middle",
        fill: "#667085", "font-size": 13
      }, message));
    }

    function selectedItem(selectId, items) {
      if (!items.length) return null;
      const index = Number(document.getElementById(selectId).value || 0);
      return items[Math.max(0, Math.min(index, items.length - 1))];
    }

    function clearSvg(svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }

    function svgEl(name, attrs = {}, text = null) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      if (text !== null) el.textContent = text;
      return el;
    }

    function formatterForKey(key) {
      if (key.includes("equity") || key.includes("notional") || key.includes("turnover")) return money;
      if (key.includes("return") || key.includes("drawdown") || key.includes("fraction") || key.includes("pct")) return pct;
      if (key.includes("sharpe") || key.includes("score") || key.includes("leverage")) return num;
      return null;
    }

    function money(value) { return Number.isFinite(Number(value)) ? fmtMoney.format(Number(value)) : "-"; }
    function pct(value) { return Number.isFinite(Number(value)) ? fmtPct.format(Number(value)) : "-"; }
    function num(value) { return Number.isFinite(Number(value)) ? fmtNum.format(Number(value)) : "-"; }
    function int(value) { return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "-"; }
    function compact(value) {
      const number = Number(value);
      if (!Number.isFinite(number)) return "-";
      if (Math.abs(number) >= 1000) return fmtMoney.format(number).replace("$", "$");
      if (Math.abs(number) < 1) return pct(number);
      return num(number);
    }
    function titleize(key) {
      return key.replaceAll("_", " ").replace(/\b\w/g, (match) => match.toUpperCase());
    }
    function shortTime(value) {
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value || "");
      return date.toLocaleString();
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }

    loadDashboard();
    setInterval(loadDashboard, 5000);
    window.addEventListener("resize", () => { if (state.data) render(); });
  </script>
</body>
</html>"""
