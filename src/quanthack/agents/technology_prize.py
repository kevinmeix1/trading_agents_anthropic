from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE_FILES: tuple[tuple[str, str], ...] = (
    ("adaptive leaderboard", "outputs/research/adaptive_selector_variant_leaderboard.csv"),
    ("policy sweep", "outputs/research/adaptive_strategy_policy_sweep.csv"),
    ("oracle summary", "outputs/research/adaptive_current_top_oracle_summary.csv"),
    ("handoff diagnostic", "outputs/research/adaptive_current_top_handoff_diagnostic.csv"),
    ("deployment profile pack", "outputs/research/deployment_profile_pack.json"),
    ("live monitor", "outputs/research/profile_live_monitor.csv"),
    ("manual MT5 ticket sheet", "outputs/research/mt5_ticket_sheet_asof_0815.csv"),
)


@dataclass(frozen=True)
class AgentSdkStatus:
    package_name: str
    installed: bool
    detected_features: tuple[str, ...]
    import_error: str = ""

    @property
    def ready(self) -> bool:
        return self.installed and {"Agent", "Runner", "function_tool"}.issubset(
            self.detected_features
        )


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    purpose: str
    sdk_mapping: str
    local_source: str
    read_only: bool = True


@dataclass(frozen=True)
class AgentNode:
    name: str
    role: str
    tools: tuple[str, ...]
    handoffs: tuple[str, ...]
    guardrails: tuple[str, ...]


@dataclass(frozen=True)
class ModelProviderReadiness:
    name: str
    package_name: str
    api_key_env: str
    installed: bool
    api_key_present: bool
    role: str
    import_error: str = ""

    @property
    def ready(self) -> bool:
        return self.installed and self.api_key_present


@dataclass(frozen=True)
class EvidenceArtifact:
    name: str
    path: str
    present: bool
    kind: str
    row_count: int = 0
    column_count: int = 0
    sha256_12: str = ""
    summary: str = "missing"

    @property
    def status(self) -> str:
        return "OK" if self.present else "MISSING"


@dataclass(frozen=True)
class AgentTraceEvent:
    step: int
    agent: str
    action: str
    status: str
    detail: str
    artifact: str = ""


@dataclass(frozen=True)
class TechnologyPrizeReport:
    sdk_status: AgentSdkStatus
    providers: tuple[ModelProviderReadiness, ...]
    agents: tuple[AgentNode, ...]
    tools: tuple[AgentToolSpec, ...]
    artifacts: tuple[EvidenceArtifact, ...]
    trace: tuple[AgentTraceEvent, ...]
    next_actions: tuple[str, ...]

    def summary_lines(self) -> tuple[str, ...]:
        present = sum(1 for artifact in self.artifacts if artifact.present)
        missing = len(self.artifacts) - present
        ready_providers = sum(1 for provider in self.providers if provider.ready)
        return (
            "Technology Prize Agent Control Plane",
            f"  Agents SDK: {'ready' if self.sdk_status.ready else 'not ready'}",
            f"  Model providers: {ready_providers}/{len(self.providers)} API-ready",
            f"  Architecture agents: {len(self.agents)}",
            f"  Tool specs: {len(self.tools)}",
            f"  Evidence artifacts: {present} present, {missing} missing",
            f"  Trace events: {len(self.trace)}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sdk_status": _dataclass_dict(self.sdk_status),
            "providers": [_dataclass_dict(provider) for provider in self.providers],
            "agents": [_dataclass_dict(agent) for agent in self.agents],
            "tools": [_dataclass_dict(tool) for tool in self.tools],
            "artifacts": [_dataclass_dict(artifact) for artifact in self.artifacts],
            "trace": [_dataclass_dict(event) for event in self.trace],
            "next_actions": list(self.next_actions),
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Claude Agent Trader AgentSDK Technology Prize Control Plane",
                "",
                "This report turns the existing trading system into a demoable agentic workflow.",
                "It is safe to run locally: the default path reads research artifacts and never places orders.",
                "",
                "## Executive Snapshot",
                "",
                f"- Agents SDK package: `{self.sdk_status.package_name}`",
                f"- SDK ready: {'yes' if self.sdk_status.ready else 'no'}",
                "- Detected SDK features: "
                + (", ".join(self.sdk_status.detected_features) or "none"),
                "- API-ready model providers: "
                + (
                    ", ".join(provider.name for provider in self.providers if provider.ready)
                    or "none"
                ),
                f"- Evidence artifacts present: {sum(1 for item in self.artifacts if item.present)}/{len(self.artifacts)}",
                "",
                "## Architecture",
                "",
                "```mermaid",
                "flowchart LR",
                '  chief["Chief Trading Agent"]',
                '  data["Data Health Agent"]',
                '  alpha["Alpha Research Agent"]',
                '  risk["Risk Guardian Agent"]',
                '  regime["Regime Scientist Agent"]',
                '  audit["Experiment Auditor Agent"]',
                '  deploy["Deployment Operator Agent"]',
                '  critic["Anthropic Critic Agent"]',
                '  report["Technology Report Agent"]',
                "  chief --> data",
                "  chief --> alpha",
                "  chief --> risk",
                "  chief --> regime",
                "  chief --> audit",
                "  chief --> deploy",
                "  chief --> critic",
                "  data --> report",
                "  alpha --> report",
                "  risk --> report",
                "  regime --> alpha",
                "  regime --> report",
                "  audit --> critic",
                "  audit --> report",
                "  deploy --> report",
                "  critic --> report",
                "```",
                "",
                _agent_table(self.agents),
                "",
                "## Model Provider Readiness",
                "",
                _provider_table(self.providers),
                "",
                "## Tool Surface",
                "",
                _tool_table(self.tools),
                "",
                "## Evidence Artifacts",
                "",
                _artifact_table(self.artifacts),
                "",
                "## Agent Trace",
                "",
                _trace_table(self.trace),
                "",
                "## Next Actions",
                "",
                *(f"- {action}" for action in self.next_actions),
                "",
            ]
        )


def detect_agents_sdk(package_name: str = "agents") -> AgentSdkStatus:
    try:
        module = importlib.import_module(package_name)
    except Exception as exc:  # pragma: no cover - exact import errors vary by env
        return AgentSdkStatus(
            package_name=package_name,
            installed=False,
            detected_features=(),
            import_error=str(exc),
        )
    features = tuple(
        feature
        for feature in (
            "Agent",
            "Runner",
            "function_tool",
            "handoff",
            "GuardrailFunctionOutput",
            "RunConfig",
            "trace",
        )
        if hasattr(module, feature)
    )
    return AgentSdkStatus(
        package_name=package_name,
        installed=True,
        detected_features=features,
    )


def detect_model_providers(
    environ: Mapping[str, str] | None = None,
) -> tuple[ModelProviderReadiness, ...]:
    env = os.environ if environ is None else environ
    return (
        _detect_provider(
            name="OpenAI AgentSDK Orchestrator",
            package_name="openai",
            api_key_env="OPENAI_API_KEY",
            role="Primary AgentSDK model calls, handoffs, function tools, guardrails, and traces.",
            environ=env,
        ),
        _detect_provider(
            name="Anthropic Critic",
            package_name="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            role=(
                "Optional independent model critic for overfitting, risk, and presentation review. "
                "It has no order-routing authority."
            ),
            environ=env,
        ),
    )


def build_agent_architecture() -> tuple[AgentNode, ...]:
    return (
        AgentNode(
            name="Chief Trading Agent",
            role="Plans a research-to-deployment run and delegates to specialist agents.",
            tools=(
                "summarize_research_artifacts",
                "summarize_experiment_leaderboard",
                "build_hackathon_readiness_snapshot",
                "build_technology_prize_judge_packet",
                "run_agent_guardrail_suite",
                "analyze_agent_topology",
                "replay_agent_trace",
                "build_judge_demo_runbook",
            ),
            handoffs=(
                "Data Health Agent",
                "Alpha Research Agent",
                "Risk Guardian Agent",
                "Regime Scientist Agent",
                "Experiment Auditor Agent",
                "Deployment Operator Agent",
                "Anthropic Critic Agent",
            ),
            guardrails=(
                "Default mode is read-only.",
                "Live trading requires explicit MT5 execution arming.",
                "No strategy promotion without walk-forward and risk evidence.",
            ),
        ),
        AgentNode(
            name="Data Health Agent",
            role="Checks downloaded FX/metal/crypto coverage before strategy claims are trusted.",
            tools=("validate_market_data_summary", "summarize_csv"),
            handoffs=("Technology Report Agent",),
            guardrails=("Missing official symbols are warnings, not silently ignored.",),
        ),
        AgentNode(
            name="Alpha Research Agent",
            role="Ranks strategy evidence, oracle regret, selector policy sweeps, and fold diagnostics.",
            tools=(
                "summarize_experiment_leaderboard",
                "summarize_research_artifacts",
                "summarize_csv",
            ),
            handoffs=("Risk Guardian Agent", "Technology Report Agent"),
            guardrails=("Research-only proxy crypto evidence must remain labelled research-only.",),
        ),
        AgentNode(
            name="Risk Guardian Agent",
            role="Reviews allocator, drawdown, leverage, risk discipline, and promotion gates.",
            tools=(
                "build_hackathon_readiness_snapshot",
                "summarize_research_artifacts",
                "run_agent_guardrail_suite",
            ),
            handoffs=("Deployment Operator Agent", "Technology Report Agent"),
            guardrails=(
                "Keep internal leverage below development cap.",
                "Do not override official stop-out or margin safety constraints.",
            ),
        ),
        AgentNode(
            name="Regime Scientist Agent",
            role=(
                "Studies market regime, selector regret, compression/chop failures, and when "
                "strategy sleeves should stand down."
            ),
            tools=("summarize_research_artifacts", "summarize_csv"),
            handoffs=("Alpha Research Agent", "Risk Guardian Agent", "Technology Report Agent"),
            guardrails=(
                "Treat regime labels as evidence, not certainty.",
                "Do not promote a strategy without out-of-sample fold support.",
            ),
        ),
        AgentNode(
            name="Experiment Auditor Agent",
            role=(
                "Verifies provenance, command reproducibility, artifact hashes, and whether "
                "claims are supported by generated evidence."
            ),
            tools=(
                "summarize_research_artifacts",
                "summarize_operator_dashboard_sources",
                "build_technology_prize_judge_packet",
                "run_agent_guardrail_suite",
                "analyze_agent_topology",
                "replay_agent_trace",
                "build_judge_demo_runbook",
            ),
            handoffs=("Technology Report Agent", "Anthropic Critic Agent"),
            guardrails=(
                "Separate proven claims from planned extensions.",
                "Flag missing artifacts instead of filling gaps with narrative.",
            ),
        ),
        AgentNode(
            name="Deployment Operator Agent",
            role="Converts approved targets into live dry-run, monitor, and manual MT5 ticket artifacts.",
            tools=("summarize_mt5_ticket_sheet", "summarize_operator_dashboard_sources"),
            handoffs=("Technology Report Agent",),
            guardrails=(
                "MT5 adapter is read-only until explicit route-live switch is built and armed.",
                "Manual ticket sheets are advisory and must pass risk checks.",
            ),
        ),
        AgentNode(
            name="Anthropic Critic Agent",
            role=(
                "Uses optional Anthropic credits as an independent reviewer of strategy evidence, "
                "overfitting risk, and judge narrative."
            ),
            tools=("summarize_research_artifacts", "summarize_csv"),
            handoffs=("Risk Guardian Agent", "Technology Report Agent"),
            guardrails=(
                "Critique only; no direct trade approval authority.",
                "Never receives broker credentials or order-placement tools.",
            ),
        ),
        AgentNode(
            name="Technology Report Agent",
            role="Creates a judge-readable trace of agents, tools, evidence, and remaining gaps.",
            tools=(
                "summarize_research_artifacts",
                "summarize_operator_dashboard_sources",
                "build_technology_prize_judge_packet",
                "run_agent_guardrail_suite",
                "analyze_agent_topology",
            ),
            handoffs=(),
            guardrails=("Report must separate proven artifacts from planned work.",),
        ),
    )


def build_agent_tool_specs() -> tuple[AgentToolSpec, ...]:
    return (
        AgentToolSpec(
            name="summarize_research_artifacts",
            purpose="Read leaderboard, oracle, handoff, live monitor, and ticket artifacts.",
            sdk_mapping="@function_tool summarize_research_artifacts",
            local_source="quanthack.agents.technology_prize.run_local_technology_prize_demo",
        ),
        AgentToolSpec(
            name="summarize_csv",
            purpose="Read a bounded preview of a project-local CSV artifact.",
            sdk_mapping="@function_tool summarize_csv",
            local_source="quanthack.agents.sdk_bridge.create_agents_sdk_app",
        ),
        AgentToolSpec(
            name="validate_market_data_summary",
            purpose="Check price/quote coverage, gaps, duplicates, and spread quality.",
            sdk_mapping="@function_tool validate_market_data_summary",
            local_source="quanthack.market.data_health.validate_market_data",
        ),
        AgentToolSpec(
            name="summarize_experiment_leaderboard",
            purpose="Rank walk-forward experiment outputs with return, stability, drawdown, and risk score.",
            sdk_mapping="@function_tool summarize_experiment_leaderboard",
            local_source="quanthack.backtesting.experiment_leaderboard.build_experiment_leaderboard",
        ),
        AgentToolSpec(
            name="build_hackathon_readiness_snapshot",
            purpose="Build a competition go/no-go report from data coverage and promotion gates.",
            sdk_mapping="@function_tool build_hackathon_readiness_snapshot",
            local_source="quanthack.reporting.hackathon_readiness.build_hackathon_readiness_report",
        ),
        AgentToolSpec(
            name="summarize_mt5_ticket_sheet",
            purpose="Convert safe USD targets into manual MT5 ticket rows.",
            sdk_mapping="@function_tool summarize_mt5_ticket_sheet",
            local_source="quanthack.trading.mt5_ticket_sheet",
        ),
        AgentToolSpec(
            name="summarize_operator_dashboard_sources",
            purpose="Build a judge/operator HTML dashboard from profile, monitor, allocation, and ticket files.",
            sdk_mapping="@function_tool summarize_operator_dashboard_sources",
            local_source="quanthack.reporting.operator_dashboard.build_operator_dashboard",
        ),
        AgentToolSpec(
            name="build_technology_prize_judge_packet",
            purpose="Verify technology-prize requirements against hashed source, report, and workflow evidence.",
            sdk_mapping="@function_tool build_technology_prize_judge_packet",
            local_source="quanthack.agents.judge_packet.build_technology_prize_judge_packet",
        ),
        AgentToolSpec(
            name="run_agent_guardrail_suite",
            purpose="Evaluate AI/broker safety guardrails before judge claims or online model calls are trusted.",
            sdk_mapping="@function_tool run_agent_guardrail_suite",
            local_source="quanthack.agents.guardrails.build_agent_guardrail_suite",
        ),
        AgentToolSpec(
            name="analyze_agent_topology",
            purpose="Validate the AgentSDK graph for tool coverage, handoff integrity, guardrails, and authority boundaries.",
            sdk_mapping="@function_tool analyze_agent_topology",
            local_source="quanthack.agents.topology.build_agent_topology_report",
        ),
        AgentToolSpec(
            name="build_judge_demo_runbook",
            purpose="Generate a timed judge demo script grounded in topology, guardrails, workflow, and packet evidence.",
            sdk_mapping="@function_tool build_judge_demo_runbook",
            local_source="quanthack.agents.demo_director.build_judge_demo_runbook",
        ),
        AgentToolSpec(
            name="replay_agent_trace",
            purpose="Export the deterministic local workflow as AgentSDK-style trace spans without online model calls.",
            sdk_mapping="@function_tool replay_agent_trace",
            local_source="quanthack.agents.trace_replay.build_agent_trace_replay",
        ),
    )


def run_local_technology_prize_demo(
    *,
    project_root: str | Path = ".",
    evidence_files: tuple[tuple[str, str], ...] = DEFAULT_EVIDENCE_FILES,
    package_name: str = "agents",
    environ: Mapping[str, str] | None = None,
) -> TechnologyPrizeReport:
    root = Path(project_root)
    sdk_status = detect_agents_sdk(package_name=package_name)
    providers = detect_model_providers(environ=environ)
    agents = build_agent_architecture()
    tools = build_agent_tool_specs()
    artifacts = tuple(
        _inspect_artifact(name=name, path=root / relative_path)
        for name, relative_path in evidence_files
    )
    trace = _build_trace(artifacts=artifacts, sdk_status=sdk_status)
    return TechnologyPrizeReport(
        sdk_status=sdk_status,
        providers=providers,
        agents=agents,
        tools=tools,
        artifacts=artifacts,
        trace=trace,
        next_actions=_next_actions(
            artifacts=artifacts,
            sdk_status=sdk_status,
            providers=providers,
        ),
    )


def write_technology_prize_report(
    report: TechnologyPrizeReport,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _inspect_artifact(*, name: str, path: Path) -> EvidenceArtifact:
    if not path.exists():
        return EvidenceArtifact(name=name, path=str(path), present=False, kind=_kind(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    if path.suffix.lower() == ".csv":
        return _inspect_csv_artifact(name=name, path=path, sha256_12=digest)
    if path.suffix.lower() == ".json":
        return _inspect_json_artifact(name=name, path=path, sha256_12=digest)
    return EvidenceArtifact(
        name=name,
        path=str(path),
        present=True,
        kind=_kind(path),
        sha256_12=digest,
        summary=f"{path.stat().st_size} bytes",
    )


def _inspect_csv_artifact(*, name: str, path: Path, sha256_12: str) -> EvidenceArtifact:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    columns = reader.fieldnames or ()
    summary = _csv_summary(name=name, rows=rows)
    return EvidenceArtifact(
        name=name,
        path=str(path),
        present=True,
        kind="csv",
        row_count=len(rows),
        column_count=len(columns),
        sha256_12=sha256_12,
        summary=summary,
    )


def _inspect_json_artifact(*, name: str, path: Path, sha256_12: str) -> EvidenceArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary = f"invalid json: {exc}"
        count = 0
    else:
        if isinstance(payload, dict):
            keys = sorted(str(key) for key in payload.keys())
            summary = "keys: " + ", ".join(keys[:8])
            count = len(keys)
        elif isinstance(payload, list):
            summary = f"list items: {len(payload)}"
            count = len(payload)
        else:
            summary = type(payload).__name__
            count = 1
    return EvidenceArtifact(
        name=name,
        path=str(path),
        present=True,
        kind="json",
        row_count=count,
        column_count=0,
        sha256_12=sha256_12,
        summary=summary,
    )


def _csv_summary(*, name: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return "empty csv"
    first = rows[0]
    if "leaderboard" in name:
        label = first.get("label", "unknown")
        score = first.get("score", "n/a")
        compounded = first.get("compounded_return_pct", first.get("compounded_test_return_pct", "n/a"))
        return f"top={label}, score={score}, compounded_return={compounded}"
    if "policy sweep" in name:
        return (
            f"top_status={first.get('promotion_status', 'n/a')}, "
            f"selector_score={first.get('selector_score', 'n/a')}, "
            f"live_ready={first.get('live_ready', 'n/a')}"
        )
    if "oracle" in name:
        return (
            f"selected_was_oracle={first.get('selected_was_oracle_fraction', 'n/a')}, "
            f"total_regret={first.get('total_regret_pct', 'n/a')}"
        )
    if "handoff" in name:
        diagnoses = sorted({row.get("diagnosis", "") for row in rows if row.get("diagnosis")})
        return f"folds={len(rows)}, diagnoses={', '.join(diagnoses[:6])}"
    if "monitor" in name:
        return f"latest_row_fields={', '.join(first.keys())}"
    if "ticket" in name:
        symbols = sorted({row.get("symbol", "") for row in rows if row.get("symbol")})
        return f"tickets={len(rows)}, symbols={', '.join(symbols[:8])}"
    return f"rows={len(rows)}"


def _build_trace(
    *,
    artifacts: tuple[EvidenceArtifact, ...],
    sdk_status: AgentSdkStatus,
) -> tuple[AgentTraceEvent, ...]:
    missing = [artifact.name for artifact in artifacts if not artifact.present]
    leaderboard = _artifact_by_name(artifacts, "adaptive leaderboard")
    policy = _artifact_by_name(artifacts, "policy sweep")
    oracle = _artifact_by_name(artifacts, "oracle summary")
    handoff = _artifact_by_name(artifacts, "handoff diagnostic")
    deploy_files = [
        artifact
        for artifact in artifacts
        if artifact.name in {"deployment profile pack", "live monitor", "manual MT5 ticket sheet"}
    ]
    return (
        AgentTraceEvent(
            step=1,
            agent="Chief Trading Agent",
            action="plan",
            status="OK",
            detail="Built a read-only research-to-deployment agent run plan.",
        ),
        AgentTraceEvent(
            step=2,
            agent="Data Health Agent",
            action="artifact audit",
            status="WARN" if missing else "OK",
            detail=(
                "Missing artifacts: " + ", ".join(missing)
                if missing
                else "All expected evidence artifacts are present."
            ),
        ),
        AgentTraceEvent(
            step=3,
            agent="Alpha Research Agent",
            action="leaderboard review",
            status="OK" if leaderboard and leaderboard.present else "MISSING",
            detail=leaderboard.summary if leaderboard else "leaderboard missing",
            artifact=leaderboard.path if leaderboard else "",
        ),
        AgentTraceEvent(
            step=4,
            agent="Alpha Research Agent",
            action="selector policy review",
            status="OK" if policy and policy.present else "MISSING",
            detail=policy.summary if policy else "policy sweep missing",
            artifact=policy.path if policy else "",
        ),
        AgentTraceEvent(
            step=5,
            agent="Risk Guardian Agent",
            action="oracle and handoff risk review",
            status="OK" if oracle and oracle.present and handoff and handoff.present else "MISSING",
            detail=_join_existing_summaries(oracle, handoff),
            artifact=oracle.path if oracle else "",
        ),
        AgentTraceEvent(
            step=6,
            agent="Deployment Operator Agent",
            action="deployment artifact review",
            status="OK" if all(artifact.present for artifact in deploy_files) else "WARN",
            detail=_deployment_summary(deploy_files),
        ),
        AgentTraceEvent(
            step=7,
            agent="Technology Report Agent",
            action="sdk readiness",
            status="OK" if sdk_status.ready else "WARN",
            detail=(
                "Agents SDK primitives detected: " + ", ".join(sdk_status.detected_features)
                if sdk_status.ready
                else "Local deterministic demo is available; install optional Agents SDK extra for live SDK Runner demos."
            ),
        ),
    )


def _next_actions(
    *,
    artifacts: tuple[EvidenceArtifact, ...],
    sdk_status: AgentSdkStatus,
    providers: tuple[ModelProviderReadiness, ...],
) -> tuple[str, ...]:
    actions: list[str] = []
    if not sdk_status.ready:
        actions.append(
            "Install the optional Agents SDK dependency when API-key demos are wanted: `python -m pip install -e .[agent]`."
        )
    if any(not artifact.present for artifact in artifacts):
        actions.append(
            "Regenerate missing evidence with the existing research/backtest CLIs before a final judge demo."
        )
    if not _provider_ready(providers, "OpenAI AgentSDK Orchestrator"):
        actions.append(
            "Set `OPENAI_API_KEY` before running the guarded online AgentSDK demo."
        )
    if not _provider_ready(providers, "Anthropic Critic"):
        actions.append(
            "Set `ANTHROPIC_API_KEY` when Anthropic credits are available to enable the independent critic layer."
        )
    actions.extend(
        [
            "Wire `create_agents_sdk_app()` to a guarded API-key demo runner after strategy evidence is frozen.",
            "Record a final technology trace that links data, alpha selection, risk gates, MT5 ticket generation, and dashboard output.",
        ]
    )
    return tuple(actions)


def _detect_provider(
    *,
    name: str,
    package_name: str,
    api_key_env: str,
    role: str,
    environ: Mapping[str, str],
) -> ModelProviderReadiness:
    try:
        importlib.import_module(package_name)
    except Exception as exc:  # pragma: no cover - exact import errors vary by env
        return ModelProviderReadiness(
            name=name,
            package_name=package_name,
            api_key_env=api_key_env,
            installed=False,
            api_key_present=bool(environ.get(api_key_env)),
            role=role,
            import_error=str(exc),
        )
    return ModelProviderReadiness(
        name=name,
        package_name=package_name,
        api_key_env=api_key_env,
        installed=True,
        api_key_present=bool(environ.get(api_key_env)),
        role=role,
    )


def _provider_ready(
    providers: tuple[ModelProviderReadiness, ...],
    name: str,
) -> bool:
    return any(provider.name == name and provider.ready for provider in providers)


def _artifact_by_name(
    artifacts: tuple[EvidenceArtifact, ...],
    name: str,
) -> EvidenceArtifact | None:
    for artifact in artifacts:
        if artifact.name == name:
            return artifact
    return None


def _deployment_summary(artifacts: list[EvidenceArtifact]) -> str:
    if not artifacts:
        return "no deployment artifacts configured"
    return "; ".join(f"{artifact.name}: {artifact.status} ({artifact.summary})" for artifact in artifacts)


def _join_existing_summaries(*artifacts: EvidenceArtifact | None) -> str:
    summaries = [artifact.summary for artifact in artifacts if artifact is not None]
    return "; ".join(summaries) if summaries else "missing risk diagnostics"


def _kind(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "file"


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _agent_table(agents: tuple[AgentNode, ...]) -> str:
    rows = ["| Agent | Role | Tools | Handoffs | Guardrails |", "| --- | --- | --- | --- | --- |"]
    for agent in agents:
        rows.append(
            "| "
            + " | ".join(
                (
                    agent.name,
                    agent.role,
                    ", ".join(agent.tools) or "none",
                    ", ".join(agent.handoffs) or "none",
                    "<br>".join(agent.guardrails) or "none",
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _provider_table(providers: tuple[ModelProviderReadiness, ...]) -> str:
    rows = [
        "| Provider | Package | API Key | Status | Role |",
        "| --- | --- | --- | --- | --- |",
    ]
    for provider in providers:
        if provider.ready:
            status = "READY"
        elif provider.installed:
            status = "PACKAGE_ONLY"
        else:
            status = "MISSING_PACKAGE"
        rows.append(
            f"| {provider.name} | `{provider.package_name}` | `{provider.api_key_env}` "
            f"| {status} | {provider.role} |"
        )
    return "\n".join(rows)


def _tool_table(tools: tuple[AgentToolSpec, ...]) -> str:
    rows = ["| Tool | Purpose | SDK Mapping | Local Source |", "| --- | --- | --- | --- |"]
    for tool in tools:
        rows.append(
            f"| {tool.name} | {tool.purpose} | `{tool.sdk_mapping}` | `{tool.local_source}` |"
        )
    return "\n".join(rows)


def _artifact_table(artifacts: tuple[EvidenceArtifact, ...]) -> str:
    rows = [
        "| Artifact | Status | Rows | Columns | Hash | Summary |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for artifact in artifacts:
        rows.append(
            f"| {artifact.name} | {artifact.status} | {artifact.row_count} | "
            f"{artifact.column_count} | `{artifact.sha256_12 or '-'}` | {artifact.summary} |"
        )
    return "\n".join(rows)


def _trace_table(trace: tuple[AgentTraceEvent, ...]) -> str:
    rows = ["| Step | Agent | Action | Status | Detail |", "| ---: | --- | --- | --- | --- |"]
    for event in trace:
        rows.append(
            f"| {event.step} | {event.agent} | {event.action} | {event.status} | {event.detail} |"
        )
    return "\n".join(rows)
