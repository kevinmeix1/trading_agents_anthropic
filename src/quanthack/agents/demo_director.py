from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.guardrails import AgentGuardrailSuite, build_agent_guardrail_suite
from quanthack.agents.judge_packet import TechnologyPrizeJudgePacket, build_technology_prize_judge_packet
from quanthack.agents.topology import AgentTopologyReport, build_agent_topology_report
from quanthack.agents.workflow import LocalAgentWorkflow, run_local_agent_workflow


@dataclass(frozen=True)
class DemoRunbookStep:
    minute_mark: str
    action: str
    say: str
    command_or_artifact: str
    proof: str


@dataclass(frozen=True)
class DemoRiskNote:
    risk: str
    mitigation: str
    proof: str


@dataclass(frozen=True)
class JudgeDemoRunbook:
    generated_at: datetime
    status: str
    steps: tuple[DemoRunbookStep, ...]
    risk_notes: tuple[DemoRiskNote, ...]
    judge_packet: TechnologyPrizeJudgePacket
    guardrails: AgentGuardrailSuite
    topology: AgentTopologyReport
    workflow: LocalAgentWorkflow

    def summary_lines(self) -> tuple[str, ...]:
        core_requirements = tuple(
            item
            for item in self.judge_packet.requirements
            if item.name != "Judge demo readiness"
        )
        return (
            "Technology Prize Judge Demo Runbook",
            f"  Overall: {self.status}",
            f"  Steps: {len(self.steps)}",
            f"  Risk notes: {len(self.risk_notes)}",
            f"  Core judge requirements: {sum(1 for item in core_requirements if item.status == 'PASS')}/{len(core_requirements)} pass",
            f"  Guardrails: {sum(1 for item in self.guardrails.checks if item.status == 'PASS')}/{len(self.guardrails.checks)} pass",
            f"  Topology: {self.topology.status}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "steps": [_dict(step) for step in self.steps],
            "risk_notes": [_dict(note) for note in self.risk_notes],
            "judge_packet_status": self.judge_packet.status,
            "guardrail_status": self.guardrails.status,
            "topology_status": self.topology.status,
            "workflow_status": self.workflow.status,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Technology Prize Judge Demo Runbook",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## 3-Minute Demo Flow",
                "",
                _steps_table(self.steps),
                "",
                "## Risk Handling",
                "",
                _risk_table(self.risk_notes),
                "",
                "## Closing Line",
                "",
                (
                    "QuanHack is not only a trading bot; it is an AI-native research and deployment "
                    "control plane where agents inspect evidence, validate their topology, run safety "
                    "guardrails, and keep MT5 execution authority outside the model boundary."
                ),
                "",
            ]
        )


def build_judge_demo_runbook(
    *,
    project_root: str | Path = ".",
) -> JudgeDemoRunbook:
    root = Path(project_root)
    judge_packet = build_technology_prize_judge_packet(project_root=root)
    guardrails = build_agent_guardrail_suite(project_root=root)
    topology = build_agent_topology_report()
    workflow = run_local_agent_workflow(project_root=root)
    steps = _build_steps(
        judge_packet=judge_packet,
        guardrails=guardrails,
        topology=topology,
        workflow=workflow,
    )
    risk_notes = _build_risk_notes()
    status = _overall_status(
        judge_packet=judge_packet,
        guardrails=guardrails,
        topology=topology,
        workflow=workflow,
    )
    return JudgeDemoRunbook(
        generated_at=datetime.now(tz=timezone.utc),
        status=status,
        steps=steps,
        risk_notes=risk_notes,
        judge_packet=judge_packet,
        guardrails=guardrails,
        topology=topology,
        workflow=workflow,
    )


def write_judge_demo_runbook(
    runbook: JudgeDemoRunbook,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(runbook.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(runbook.to_dict(), indent=2), encoding="utf-8")


def _build_steps(
    *,
    judge_packet: TechnologyPrizeJudgePacket,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    workflow: LocalAgentWorkflow,
) -> tuple[DemoRunbookStep, ...]:
    return (
        DemoRunbookStep(
            minute_mark="0:00",
            action="Open the dashboard",
            say=(
                "This is the technical-merit view: AgentSDK orchestration, Anthropic critic readiness, "
                "guardrails, topology checks, and provenance in one page."
            ),
            command_or_artifact="outputs/reports/technology_prize_dashboard.html",
            proof="Dashboard generated from technology_prize_demo_pack with PASS status.",
        ),
        DemoRunbookStep(
            minute_mark="0:30",
            action="Show the AgentSDK topology",
            say=(
                f"The graph has {len(topology.agents)} agents, {len(topology.tools)} tools, "
                f"and {len(topology.handoffs)} validated handoffs; every tool is read-only."
            ),
            command_or_artifact="quanthack tech-prize-topology",
            proof=f"Topology report: {topology.status}",
        ),
        DemoRunbookStep(
            minute_mark="1:00",
            action="Show executable guardrails",
            say=(
                "The safety model is executable: no broker authority, path confinement, credit-spend gates, "
                "and Anthropic review-only boundaries are checked before claims are trusted."
            ),
            command_or_artifact="quanthack tech-prize-guardrails",
            proof=f"Guardrails: {sum(1 for item in guardrails.checks if item.status == 'PASS')}/{len(guardrails.checks)} pass",
        ),
        DemoRunbookStep(
            minute_mark="1:30",
            action="Show local blackboard workflow",
            say=(
                f"The offline workflow records {len(workflow.steps)} steps and "
                f"{len(workflow.blackboard)} blackboard writes without spending model credits."
            ),
            command_or_artifact="quanthack tech-prize-workflow",
            proof=f"Workflow: {workflow.status}",
        ),
        DemoRunbookStep(
            minute_mark="2:00",
            action="Show the judge packet",
            say=(
                "This turns the prize criteria into a requirement matrix with source/report hashes, "
                "so the claims are inspectable instead of just asserted."
            ),
            command_or_artifact="quanthack tech-prize-judge-packet",
            proof=(
                f"Judge packet: {sum(1 for item in judge_packet.requirements if item.status == 'PASS')}/"
                f"{len(judge_packet.requirements)} requirements pass"
            ),
        ),
        DemoRunbookStep(
            minute_mark="2:30",
            action="Optional online model proof",
            say=(
                "Online OpenAI AgentSDK and Anthropic critic runs are available only when explicitly armed; "
                "the safe demo path defaults to no credit spend."
            ),
            command_or_artifact="quanthack tech-prize-demo --run-sdk --run-anthropic-critic",
            proof="Online calls require --allow-online-sdk or --allow-online-anthropic.",
        ),
    )


def _build_risk_notes() -> tuple[DemoRiskNote, ...]:
    return (
        DemoRiskNote(
            risk="Judge asks whether the AI can place trades.",
            mitigation="Show the guardrail and topology reports: all AgentSDK tools are read-only and no MT5 order tool is registered.",
            proof="outputs/reports/technology_prize_guardrails.md",
        ),
        DemoRiskNote(
            risk="Judge asks why Anthropic credits matter.",
            mitigation="Explain Anthropic is an independent critic for overfitting, risk, and narrative review with no broker authority.",
            proof="outputs/reports/technology_prize_anthropic_critic.md",
        ),
        DemoRiskNote(
            risk="Judge asks whether the graph is real or just a diagram.",
            mitigation="Run the topology analyzer; it validates tool usage, handoffs, guardrails, and authority boundaries.",
            proof="outputs/reports/technology_prize_topology.md",
        ),
    )


def _overall_status(
    *,
    judge_packet: TechnologyPrizeJudgePacket,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    workflow: LocalAgentWorkflow,
) -> str:
    core_requirement_statuses = tuple(
        requirement.status
        for requirement in judge_packet.requirements
        if requirement.name != "Judge demo readiness"
    )
    statuses = (*core_requirement_statuses, guardrails.status, topology.status, workflow.status)
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "WARN" for status in statuses):
        return "WARN"
    return "PASS"


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _steps_table(steps: tuple[DemoRunbookStep, ...]) -> str:
    rows = [
        "| Time | Action | Say | Command / Artifact | Proof |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in steps:
        rows.append(
            f"| {step.minute_mark} | {step.action} | {step.say} | "
            f"`{step.command_or_artifact}` | {step.proof} |"
        )
    return "\n".join(rows)


def _risk_table(notes: tuple[DemoRiskNote, ...]) -> str:
    rows = ["| Risk | Mitigation | Proof |", "| --- | --- | --- |"]
    for note in notes:
        rows.append(f"| {note.risk} | {note.mitigation} | `{note.proof}` |")
    return "\n".join(rows)
