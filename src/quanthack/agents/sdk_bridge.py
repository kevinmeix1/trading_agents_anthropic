from __future__ import annotations

import csv
import importlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quanthack.backtesting.experiment_leaderboard import build_experiment_leaderboard
from quanthack.core.config import load_config
from quanthack.market.data_health import validate_market_data
from quanthack.market.market_data import load_price_history, load_quote_history
from quanthack.reporting.hackathon_readiness import build_hackathon_readiness_report
from quanthack.agents.demo_director import build_judge_demo_runbook as build_local_judge_demo_runbook
from quanthack.agents.guardrails import build_agent_guardrail_suite
from quanthack.agents.judge_packet import (
    build_technology_prize_judge_packet as build_local_technology_prize_judge_packet,
)
from quanthack.agents.technology_prize import (
    build_agent_architecture,
    run_local_technology_prize_demo,
)
from quanthack.agents.topology import build_agent_topology_report
from quanthack.agents.trace_replay import build_agent_trace_replay


class AgentsSdkUnavailableError(RuntimeError):
    """Raised when the optional Agents SDK dependency is not installed."""


@dataclass(frozen=True)
class AgentsSdkApp:
    chief_agent: Any
    specialist_agents: dict[str, Any]
    tools: dict[str, Callable[..., Any]]


def create_agents_sdk_app(
    *,
    project_root: str | Path = ".",
    importer: Callable[[str], Any] = importlib.import_module,
) -> AgentsSdkApp:
    """Build Agents SDK objects without running an online model call.

    The default Claude Agent Trader workflow stays deterministic and offline. This bridge
    exists so the technology-prize demo can be armed later with real SDK
    `Runner` calls while tests can still verify the agent graph locally.
    """
    try:
        sdk = importer("agents")
    except Exception as exc:  # pragma: no cover - exact import errors vary by env
        raise AgentsSdkUnavailableError(
            "The optional Agents SDK package is not installed. "
            "Install it with `python -m pip install -e .[agent]` before running SDK demos."
        ) from exc

    missing = [
        name for name in ("Agent", "function_tool") if not hasattr(sdk, name)
    ]
    if missing:
        raise AgentsSdkUnavailableError(
            "The imported `agents` package is missing required features: "
            + ", ".join(missing)
        )

    function_tool = sdk.function_tool
    root = Path(project_root).resolve()

    @function_tool
    def summarize_research_artifacts() -> dict[str, Any]:
        """Summarize the current technology-prize evidence pack."""
        report = run_local_technology_prize_demo(project_root=root)
        return {
            "summary": list(report.summary_lines()),
            "artifacts": [artifact.__dict__ for artifact in report.artifacts],
            "trace": [event.__dict__ for event in report.trace],
        }

    @function_tool
    def summarize_csv(path: str, limit: int = 5) -> dict[str, Any]:
        """Read the first rows and columns of a project-local CSV artifact."""
        csv_path = _project_path(root, path)
        if not csv_path.exists():
            return {"path": str(csv_path), "present": False, "rows": []}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            bounded_limit = max(0, min(limit, 25))
            rows = [row for index, row in enumerate(reader) if index < bounded_limit]
            columns = reader.fieldnames or []
        return {
            "path": str(csv_path),
            "present": True,
            "columns": columns,
            "rows": rows,
        }

    @function_tool
    def validate_market_data_summary(
        price_csv: str = "data/backtest_prices.csv",
        quote_csv: str = "data/backtest_quotes.csv",
        symbols: str = "",
    ) -> dict[str, Any]:
        """Run the project market-data health checker on project-local CSV files."""
        price_path = _project_path(root, price_csv)
        quote_path = _project_path(root, quote_csv)
        if not price_path.exists() or not quote_path.exists():
            return {
                "status": "MISSING",
                "price_csv": str(price_path),
                "quote_csv": str(quote_path),
                "price_present": price_path.exists(),
                "quote_present": quote_path.exists(),
            }
        selected_symbols = _parse_symbols(symbols)
        report = validate_market_data(
            prices=load_price_history(price_path),
            quotes=load_quote_history(quote_path),
            symbols=selected_symbols or None,
        )
        return {
            "status": report.overall.value,
            "symbols": [
                {
                    "symbol": symbol.symbol,
                    "price_count": symbol.price_count,
                    "quote_count": symbol.quote_count,
                    "median_spread_bps": symbol.median_spread_bps,
                    "p95_spread_bps": symbol.p95_spread_bps,
                    "max_price_gap_seconds": symbol.max_price_gap_seconds,
                    "max_quote_gap_seconds": symbol.max_quote_gap_seconds,
                }
                for symbol in report.symbols[:12]
            ],
            "issue_count": len(report.issues),
            "issues": [
                {
                    "severity": issue.severity.value,
                    "symbol": issue.symbol,
                    "category": issue.category,
                    "details": issue.details,
                }
                for issue in report.issues[:12]
            ],
        }

    @function_tool
    def summarize_experiment_leaderboard(
        summary_csv_paths: str = "outputs/research/adaptive_current_top_recheck_summary.csv,outputs/research/adaptive_plus_squeeze_summary.csv,outputs/research/adaptive_plus_macd_squeeze_summary.csv",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Rank project-local walk-forward summary CSV files."""
        paths = tuple(
            str(_project_path(root, value.strip()))
            for value in summary_csv_paths.split(",")
            if value.strip()
        )
        rows = build_experiment_leaderboard(paths)
        return {
            "count": len(rows),
            "top": [
                {
                    "label": row.label,
                    "score": row.score,
                    "compounded_return_pct": row.compounded_return_pct,
                    "active_positive_fold_fraction": row.active_positive_fold_fraction,
                    "non_negative_fold_fraction": row.non_negative_fold_fraction,
                    "worst_drawdown_pct": row.worst_drawdown_pct,
                    "risk_discipline_score": row.average_risk_discipline_score,
                }
                for row in rows[: max(0, min(limit, 10))]
            ],
        }

    @function_tool
    def build_hackathon_readiness_snapshot(
        config_path: str = "configs/default.toml",
        price_csv: str = "",
        quote_csv: str = "",
        promotion_csv: str = "outputs/research/adaptive_current_top_recheck_promotion.csv",
        summary_csv: str = "outputs/research/adaptive_current_top_recheck_summary.csv",
    ) -> dict[str, Any]:
        """Build a project-local hackathon readiness snapshot."""
        config = load_config(_project_path(root, config_path))
        resolved_price = price_csv or config.backtest.price_csv
        resolved_quote = quote_csv or config.backtest.quote_csv
        report = build_hackathon_readiness_report(
            config=config,
            prices=load_price_history(_project_path(root, resolved_price)),
            quotes=load_quote_history(_project_path(root, resolved_quote)),
            promotion_csv=_project_path(root, promotion_csv),
            summary_csv=_project_path(root, summary_csv),
        )
        return {
            "overall_status": report.overall_status.value,
            "ready_for_live": report.ready_for_live,
            "common_symbols": list(report.coverage.common_symbols),
            "missing_common_symbols": list(report.coverage.missing_common_symbols),
            "covered_asset_classes": [
                asset_class.value for asset_class in report.coverage.covered_asset_classes
            ],
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "details": check.details,
                }
                for check in report.checks
            ],
            "promotion": None
            if report.promotion is None
            else {
                "status": report.promotion.status,
                "live_ready": report.promotion.live_ready,
                "reason": report.promotion.reason,
            },
        }

    @function_tool
    def summarize_mt5_ticket_sheet(
        ticket_csv: str = "outputs/research/mt5_ticket_sheet_asof_0815.csv",
    ) -> dict[str, Any]:
        """Summarize a project-local manual MT5 ticket sheet without placing orders."""
        ticket_path = _project_path(root, ticket_csv)
        rows = _read_csv_rows(ticket_path, limit=250)
        statuses = Counter(row.get("status", "UNKNOWN") for row in rows)
        symbols = sorted({row.get("symbol", "") for row in rows if row.get("symbol")})
        sides = Counter(row.get("side", "UNKNOWN") for row in rows)
        return {
            "path": str(ticket_path),
            "present": ticket_path.exists(),
            "rows": len(rows),
            "statuses": dict(statuses),
            "sides": dict(sides),
            "symbols": symbols,
            "has_order_authority": False,
            "safety_note": "manual ticket sheet only; no MT5 order placement is exposed",
        }

    @function_tool
    def summarize_operator_dashboard_sources(
        profile_pack_json: str = "outputs/research/deployment_profile_pack.json",
        snapshot_csv: str = "outputs/research/deployment_profile_conservative_signal_snapshot.csv",
        allocation_csv: str = "outputs/research/profile_live_allocation.csv",
        monitor_csv: str = "outputs/research/profile_live_monitor.csv",
        ticket_csv: str = "outputs/research/mt5_ticket_sheet_asof_0815.csv",
    ) -> dict[str, Any]:
        """Check whether operator dashboard source artifacts are present."""
        sources = {
            "profile_pack_json": profile_pack_json,
            "snapshot_csv": snapshot_csv,
            "allocation_csv": allocation_csv,
            "monitor_csv": monitor_csv,
            "ticket_csv": ticket_csv,
        }
        inspected = {}
        for name, relative_path in sources.items():
            path = _project_path(root, relative_path)
            inspected[name] = {
                "path": str(path),
                "present": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        return {
            "sources": inspected,
            "ready": all(value["present"] for value in inspected.values()),
            "has_order_authority": False,
        }

    @function_tool
    def build_technology_prize_judge_packet() -> dict[str, Any]:
        """Build a requirement-by-requirement judge packet summary."""
        packet = build_local_technology_prize_judge_packet(project_root=root)
        return {
            "status": packet.status,
            "summary": list(packet.summary_lines()),
            "requirements": [
                {
                    "name": requirement.name,
                    "prize_axis": requirement.prize_axis,
                    "status": requirement.status,
                    "judge_note": requirement.judge_note,
                    "evidence": list(requirement.evidence),
                }
                for requirement in packet.requirements
            ],
            "source_evidence_count": len(packet.source_evidence),
            "report_evidence_count": len(packet.report_evidence),
            "has_order_authority": packet.has_order_authority,
            "online_model_calls_default_to_off": packet.online_model_calls_default_to_off,
        }

    @function_tool
    def run_agent_guardrail_suite() -> dict[str, Any]:
        """Evaluate project-local AI/broker safety guardrails."""
        suite = build_agent_guardrail_suite(project_root=root)
        return {
            "status": suite.status,
            "summary": list(suite.summary_lines()),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "scope": check.scope,
                    "details": check.details,
                    "evidence": list(check.evidence),
                }
                for check in suite.checks
            ],
        }

    @function_tool
    def analyze_agent_topology() -> dict[str, Any]:
        """Validate AgentSDK graph topology and tool coverage."""
        report = build_agent_topology_report()
        return {
            "status": report.status,
            "summary": list(report.summary_lines()),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "details": check.details,
                }
                for check in report.checks
            ],
            "tool_coverage": [
                {
                    "tool": coverage.tool,
                    "status": coverage.status,
                    "read_only": coverage.read_only,
                    "used_by": list(coverage.used_by),
                }
                for coverage in report.tool_coverage
            ],
            "handoffs": [
                {
                    "from_agent": edge.from_agent,
                    "to_agent": edge.to_agent,
                    "status": edge.status,
                }
                for edge in report.handoffs
            ],
        }

    @function_tool
    def build_judge_demo_runbook() -> dict[str, Any]:
        """Build a timed judge-demo runbook from current evidence."""
        runbook = build_local_judge_demo_runbook(project_root=root)
        return {
            "status": runbook.status,
            "summary": list(runbook.summary_lines()),
            "steps": [
                {
                    "minute_mark": step.minute_mark,
                    "action": step.action,
                    "say": step.say,
                    "command_or_artifact": step.command_or_artifact,
                    "proof": step.proof,
                }
                for step in runbook.steps
            ],
            "risk_notes": [
                {
                    "risk": note.risk,
                    "mitigation": note.mitigation,
                    "proof": note.proof,
                }
                for note in runbook.risk_notes
            ],
        }

    @function_tool
    def replay_agent_trace() -> dict[str, Any]:
        """Export the local agent workflow as inspectable trace spans."""
        replay = build_agent_trace_replay(project_root=root)
        return {
            "status": replay.status,
            "trace_id": replay.trace_id,
            "summary": list(replay.summary_lines()),
            "spans": [
                {
                    "span_id": span.span_id,
                    "parent_id": span.parent_id,
                    "name": span.name,
                    "kind": span.kind,
                    "agent": span.agent,
                    "status": span.status,
                    "tool": span.tool,
                    "summary": span.summary,
                    "evidence": span.evidence,
                }
                for span in replay.spans
            ],
        }

    tools = {
        "summarize_research_artifacts": summarize_research_artifacts,
        "summarize_csv": summarize_csv,
        "validate_market_data_summary": validate_market_data_summary,
        "summarize_experiment_leaderboard": summarize_experiment_leaderboard,
        "build_hackathon_readiness_snapshot": build_hackathon_readiness_snapshot,
        "summarize_mt5_ticket_sheet": summarize_mt5_ticket_sheet,
        "summarize_operator_dashboard_sources": summarize_operator_dashboard_sources,
        "build_technology_prize_judge_packet": build_technology_prize_judge_packet,
        "run_agent_guardrail_suite": run_agent_guardrail_suite,
        "analyze_agent_topology": analyze_agent_topology,
        "build_judge_demo_runbook": build_judge_demo_runbook,
        "replay_agent_trace": replay_agent_trace,
    }

    specialist_agents: dict[str, Any] = {}
    for node in build_agent_architecture()[1:]:
        specialist_agents[node.name] = sdk.Agent(
            name=node.name,
            instructions=_agent_instructions(node.name, node.role, node.guardrails),
            tools=list(tools.values()),
        )

    chief = sdk.Agent(
        name="Chief Trading Agent",
        instructions=(
            "You are the Claude Agent Trader trading-system control-plane agent. "
            "Delegate research, risk, data, deployment, and reporting checks to specialists. "
            "Never place live trades. Summarize evidence, gaps, and next actions."
        ),
        tools=list(tools.values()),
        handoffs=list(specialist_agents.values()),
    )
    return AgentsSdkApp(
        chief_agent=chief,
        specialist_agents=specialist_agents,
        tools=tools,
    )


def _agent_instructions(
    name: str,
    role: str,
    guardrails: tuple[str, ...],
) -> str:
    return (
        f"You are {name}. {role} "
        "Use the provided read-only tools and cite artifact paths. "
        "Guardrails: " + "; ".join(guardrails)
    )


def _project_path(root: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {relative_path}") from exc
    return resolved


def _parse_symbols(symbols: str) -> tuple[str, ...]:
    return tuple(symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip())


def _read_csv_rows(path: Path, *, limit: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for index, row in enumerate(reader) if index < limit]
