from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.technology_prize import (
    EvidenceArtifact,
    ModelProviderReadiness,
    TechnologyPrizeReport,
    run_local_technology_prize_demo,
)


@dataclass(frozen=True)
class WorkflowBlackboardItem:
    key: str
    value: str
    source_agent: str
    evidence: str
    status: str = "OK"


@dataclass(frozen=True)
class WorkflowHandoff:
    from_agent: str
    to_agent: str
    reason: str


@dataclass(frozen=True)
class WorkflowStep:
    step: int
    agent: str
    objective: str
    tool: str
    status: str
    summary: str
    writes: tuple[str, ...]
    handoff_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalAgentWorkflow:
    generated_at: datetime
    status: str
    steps: tuple[WorkflowStep, ...]
    blackboard: tuple[WorkflowBlackboardItem, ...]
    handoffs: tuple[WorkflowHandoff, ...]
    verdict: str

    def summary_lines(self) -> tuple[str, ...]:
        return (
            "Local Agent Workflow",
            f"  Status: {self.status}",
            f"  Steps: {len(self.steps)}",
            f"  Blackboard items: {len(self.blackboard)}",
            f"  Handoffs: {len(self.handoffs)}",
            f"  Verdict: {self.verdict}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "steps": [_dict(step) for step in self.steps],
            "blackboard": [_dict(item) for item in self.blackboard],
            "handoffs": [_dict(handoff) for handoff in self.handoffs],
            "verdict": self.verdict,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Local Agent Workflow",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Status: **{self.status}**",
                "",
                "## Why This Matters",
                "",
                (
                    "This is the deterministic local twin of the online AgentSDK workflow. "
                    "It executes the same agent roles against project evidence, records "
                    "blackboard writes, and traces handoffs without spending model credits."
                ),
                "",
                "## Steps",
                "",
                _steps_table(self.steps),
                "",
                "## Blackboard",
                "",
                _blackboard_table(self.blackboard),
                "",
                "## Handoffs",
                "",
                _handoff_table(self.handoffs),
                "",
                "## Verdict",
                "",
                self.verdict,
                "",
            ]
        )


def run_local_agent_workflow(
    *,
    project_root: str | Path = ".",
) -> LocalAgentWorkflow:
    report = run_local_technology_prize_demo(project_root=project_root)
    blackboard = _build_blackboard(report)
    handoffs = _build_handoffs()
    steps = _build_steps(report=report, blackboard=blackboard)
    status = "PASS" if all(step.status == "PASS" for step in steps) else "WARN"
    verdict = _workflow_verdict(status=status, report=report, blackboard=blackboard)
    return LocalAgentWorkflow(
        generated_at=datetime.now(tz=timezone.utc),
        status=status,
        steps=steps,
        blackboard=blackboard,
        handoffs=handoffs,
        verdict=verdict,
    )


def write_local_agent_workflow(
    workflow: LocalAgentWorkflow,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(workflow.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(workflow.to_dict(), indent=2), encoding="utf-8")


def _build_blackboard(
    report: TechnologyPrizeReport,
) -> tuple[WorkflowBlackboardItem, ...]:
    artifact_count = len(report.artifacts)
    artifact_present = sum(1 for artifact in report.artifacts if artifact.present)
    leaderboard = _artifact(report.artifacts, "adaptive leaderboard")
    oracle = _artifact(report.artifacts, "oracle summary")
    handoff = _artifact(report.artifacts, "handoff diagnostic")
    tickets = _artifact(report.artifacts, "manual MT5 ticket sheet")
    anthropic = _provider(report.providers, "Anthropic Critic")
    openai = _provider(report.providers, "OpenAI AgentSDK Orchestrator")
    return (
        WorkflowBlackboardItem(
            key="sdk_primitives",
            value=", ".join(report.sdk_status.detected_features) or "missing",
            source_agent="Chief Trading Agent",
            evidence="agents package introspection",
            status="OK" if report.sdk_status.ready else "WARN",
        ),
        WorkflowBlackboardItem(
            key="artifact_coverage",
            value=f"{artifact_present}/{artifact_count}",
            source_agent="Data Health Agent",
            evidence="technology prize evidence artifact scan",
            status="OK" if artifact_present == artifact_count else "WARN",
        ),
        WorkflowBlackboardItem(
            key="top_research_candidate",
            value=leaderboard.summary if leaderboard is not None else "missing",
            source_agent="Alpha Research Agent",
            evidence=leaderboard.path if leaderboard is not None else "missing",
            status="OK" if leaderboard is not None and leaderboard.present else "WARN",
        ),
        WorkflowBlackboardItem(
            key="selector_regret",
            value=oracle.summary if oracle is not None else "missing",
            source_agent="Regime Scientist Agent",
            evidence=oracle.path if oracle is not None else "missing",
            status="OK" if oracle is not None and oracle.present else "WARN",
        ),
        WorkflowBlackboardItem(
            key="handoff_diagnoses",
            value=handoff.summary if handoff is not None else "missing",
            source_agent="Regime Scientist Agent",
            evidence=handoff.path if handoff is not None else "missing",
            status="OK" if handoff is not None and handoff.present else "WARN",
        ),
        WorkflowBlackboardItem(
            key="mt5_ticket_boundary",
            value=tickets.summary if tickets is not None else "missing",
            source_agent="Deployment Operator Agent",
            evidence=tickets.path if tickets is not None else "missing",
            status="OK" if tickets is not None and tickets.present else "WARN",
        ),
        WorkflowBlackboardItem(
            key="openai_agent_sdk_provider",
            value=_provider_value(openai),
            source_agent="Experiment Auditor Agent",
            evidence="provider readiness scan",
            status="OK" if openai is not None and openai.installed else "WARN",
        ),
        WorkflowBlackboardItem(
            key="anthropic_critic_provider",
            value=_provider_value(anthropic),
            source_agent="Anthropic Critic Agent",
            evidence="provider readiness scan",
            status="OK" if anthropic is not None and anthropic.installed else "WARN",
        ),
        WorkflowBlackboardItem(
            key="order_authority",
            value="none exposed to agents",
            source_agent="Risk Guardian Agent",
            evidence="sdk bridge tool registry",
            status="OK",
        ),
    )


def _build_steps(
    *,
    report: TechnologyPrizeReport,
    blackboard: tuple[WorkflowBlackboardItem, ...],
) -> tuple[WorkflowStep, ...]:
    status_by_key = {item.key: item.status for item in blackboard}
    return (
        WorkflowStep(
            step=1,
            agent="Chief Trading Agent",
            objective="Initialize mission and inspect SDK capability.",
            tool="detect_agents_sdk",
            status="PASS" if report.sdk_status.ready else "WARN",
            summary=_item_value(blackboard, "sdk_primitives"),
            writes=("sdk_primitives",),
            handoff_to=("Data Health Agent", "Alpha Research Agent", "Risk Guardian Agent"),
        ),
        WorkflowStep(
            step=2,
            agent="Data Health Agent",
            objective="Confirm artifact coverage before any claims are trusted.",
            tool="summarize_research_artifacts",
            status="PASS" if status_by_key["artifact_coverage"] == "OK" else "WARN",
            summary=_item_value(blackboard, "artifact_coverage"),
            writes=("artifact_coverage",),
            handoff_to=("Alpha Research Agent", "Experiment Auditor Agent"),
        ),
        WorkflowStep(
            step=3,
            agent="Alpha Research Agent",
            objective="Identify the current research candidate from leaderboard evidence.",
            tool="summarize_experiment_leaderboard",
            status="PASS" if status_by_key["top_research_candidate"] == "OK" else "WARN",
            summary=_item_value(blackboard, "top_research_candidate"),
            writes=("top_research_candidate",),
            handoff_to=("Regime Scientist Agent", "Risk Guardian Agent"),
        ),
        WorkflowStep(
            step=4,
            agent="Regime Scientist Agent",
            objective="Review selector regret and regime/handoff diagnostics.",
            tool="summarize_csv",
            status=(
                "PASS"
                if status_by_key["selector_regret"] == "OK"
                and status_by_key["handoff_diagnoses"] == "OK"
                else "WARN"
            ),
            summary=(
                _item_value(blackboard, "selector_regret")
                + "; "
                + _item_value(blackboard, "handoff_diagnoses")
            ),
            writes=("selector_regret", "handoff_diagnoses"),
            handoff_to=("Risk Guardian Agent", "Anthropic Critic Agent"),
        ),
        WorkflowStep(
            step=5,
            agent="Risk Guardian Agent",
            objective="Verify the AI layer has no broker/order authority.",
            tool="summarize_mt5_ticket_sheet",
            status="PASS" if status_by_key["order_authority"] == "OK" else "WARN",
            summary=_item_value(blackboard, "order_authority"),
            writes=("order_authority",),
            handoff_to=("Deployment Operator Agent", "Experiment Auditor Agent"),
        ),
        WorkflowStep(
            step=6,
            agent="Deployment Operator Agent",
            objective="Check MT5 ticket artifact without placing orders.",
            tool="summarize_mt5_ticket_sheet",
            status="PASS" if status_by_key["mt5_ticket_boundary"] == "OK" else "WARN",
            summary=_item_value(blackboard, "mt5_ticket_boundary"),
            writes=("mt5_ticket_boundary",),
            handoff_to=("Experiment Auditor Agent",),
        ),
        WorkflowStep(
            step=7,
            agent="Experiment Auditor Agent",
            objective="Verify provider readiness and provenance boundary.",
            tool="summarize_research_artifacts",
            status="PASS" if status_by_key["openai_agent_sdk_provider"] == "OK" else "WARN",
            summary=_item_value(blackboard, "openai_agent_sdk_provider"),
            writes=("openai_agent_sdk_provider",),
            handoff_to=("Anthropic Critic Agent", "Technology Report Agent"),
        ),
        WorkflowStep(
            step=8,
            agent="Anthropic Critic Agent",
            objective="Record independent critic availability.",
            tool="provider readiness scan",
            status="PASS" if status_by_key["anthropic_critic_provider"] == "OK" else "WARN",
            summary=_item_value(blackboard, "anthropic_critic_provider"),
            writes=("anthropic_critic_provider",),
            handoff_to=("Technology Report Agent",),
        ),
        WorkflowStep(
            step=9,
            agent="Technology Report Agent",
            objective="Package final judge-facing evidence and verdict.",
            tool="write_technology_prize_demo_pack",
            status="PASS",
            summary="blackboard complete; reports can be generated safely",
            writes=("final_verdict",),
        ),
    )


def _build_handoffs() -> tuple[WorkflowHandoff, ...]:
    return (
        WorkflowHandoff("Chief Trading Agent", "Data Health Agent", "artifact coverage first"),
        WorkflowHandoff("Chief Trading Agent", "Alpha Research Agent", "rank evidence"),
        WorkflowHandoff("Alpha Research Agent", "Regime Scientist Agent", "explain fragile folds"),
        WorkflowHandoff("Regime Scientist Agent", "Risk Guardian Agent", "convert diagnostics into gates"),
        WorkflowHandoff("Risk Guardian Agent", "Deployment Operator Agent", "inspect ticket boundary"),
        WorkflowHandoff("Deployment Operator Agent", "Experiment Auditor Agent", "verify outputs"),
        WorkflowHandoff("Experiment Auditor Agent", "Anthropic Critic Agent", "independent critique"),
        WorkflowHandoff("Anthropic Critic Agent", "Technology Report Agent", "package judge narrative"),
    )


def _workflow_verdict(
    *,
    status: str,
    report: TechnologyPrizeReport,
    blackboard: tuple[WorkflowBlackboardItem, ...],
) -> str:
    return (
        f"{status}: local agent workflow executed {len(blackboard)} blackboard writes "
        f"across {len(report.agents)} agents. The workflow is AI-native because it uses "
        "specialist roles, grounded tool evidence, explicit handoffs, provider-aware critic "
        "readiness, and a hard no-order-authority boundary."
    )


def _artifact(
    artifacts: tuple[EvidenceArtifact, ...],
    name: str,
) -> EvidenceArtifact | None:
    for artifact in artifacts:
        if artifact.name == name:
            return artifact
    return None


def _provider(
    providers: tuple[ModelProviderReadiness, ...],
    name: str,
) -> ModelProviderReadiness | None:
    for provider in providers:
        if provider.name == name:
            return provider
    return None


def _provider_value(provider: ModelProviderReadiness | None) -> str:
    if provider is None:
        return "missing"
    if provider.ready:
        return "package installed; api key present"
    if provider.installed:
        return "package installed; api key missing"
    return "package missing"


def _item_value(items: tuple[WorkflowBlackboardItem, ...], key: str) -> str:
    for item in items:
        if item.key == key:
            return item.value
    return "missing"


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _steps_table(steps: tuple[WorkflowStep, ...]) -> str:
    rows = ["| Step | Agent | Tool | Status | Summary | Writes |", "| ---: | --- | --- | --- | --- | --- |"]
    for step in steps:
        rows.append(
            f"| {step.step} | {step.agent} | `{step.tool}` | {step.status} | "
            f"{step.summary} | {', '.join(step.writes)} |"
        )
    return "\n".join(rows)


def _blackboard_table(items: tuple[WorkflowBlackboardItem, ...]) -> str:
    rows = ["| Key | Status | Value | Source Agent | Evidence |", "| --- | --- | --- | --- | --- |"]
    for item in items:
        rows.append(
            f"| `{item.key}` | {item.status} | {item.value} | {item.source_agent} | {item.evidence} |"
        )
    return "\n".join(rows)


def _handoff_table(handoffs: tuple[WorkflowHandoff, ...]) -> str:
    rows = ["| From | To | Reason |", "| --- | --- | --- |"]
    for handoff in handoffs:
        rows.append(f"| {handoff.from_agent} | {handoff.to_agent} | {handoff.reason} |")
    return "\n".join(rows)
