from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.guardrails import AgentGuardrailSuite, build_agent_guardrail_suite
from quanthack.agents.technology_prize import TechnologyPrizeReport, run_local_technology_prize_demo
from quanthack.agents.topology import AgentTopologyReport, build_agent_topology_report
from quanthack.agents.workflow import LocalAgentWorkflow, run_local_agent_workflow


EXPECTED_SDK_BRIDGE_TOOLS: tuple[str, ...] = (
    "analyze_agent_topology",
    "build_hackathon_readiness_snapshot",
    "build_judge_demo_runbook",
    "build_technology_prize_judge_packet",
    "replay_agent_trace",
    "run_agent_guardrail_suite",
    "summarize_csv",
    "summarize_experiment_leaderboard",
    "summarize_mt5_ticket_sheet",
    "summarize_operator_dashboard_sources",
    "summarize_research_artifacts",
    "validate_market_data_summary",
)


DEFAULT_JUDGE_SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("agent architecture", "src/quanthack/agents/technology_prize.py"),
    ("agents sdk bridge", "src/quanthack/agents/sdk_bridge.py"),
    ("agent guardrails", "src/quanthack/agents/guardrails.py"),
    ("agent topology", "src/quanthack/agents/topology.py"),
    ("agent trace replay", "src/quanthack/agents/trace_replay.py"),
    ("demo director", "src/quanthack/agents/demo_director.py"),
    ("demo rehearsal", "src/quanthack/agents/demo_rehearsal.py"),
    ("demo rehearsal cli", "src/quanthack/cli/tech_prize_rehearse.py"),
    ("local workflow", "src/quanthack/agents/workflow.py"),
    ("judge packet", "src/quanthack/agents/judge_packet.py"),
    ("guarded sdk runner", "src/quanthack/agents/sdk_runner.py"),
    ("anthropic critic", "src/quanthack/agents/anthropic_critic.py"),
    ("demo pack", "src/quanthack/agents/demo_pack.py"),
    ("technology rubric", "src/quanthack/agents/rubric.py"),
    ("technology rubric cli", "src/quanthack/cli/tech_prize_rubric.py"),
    ("technology red team", "src/quanthack/agents/red_team.py"),
    ("technology red team cli", "src/quanthack/cli/tech_prize_red_team.py"),
    ("technology dashboard", "src/quanthack/reporting/technology_prize_dashboard.py"),
    ("judge packet cli", "src/quanthack/cli/tech_prize_judge_packet.py"),
    ("technology docs", "docs/50_technology_prize/126_AGENTSDK_CONTROL_PLANE.md"),
)


DEFAULT_JUDGE_REPORT_FILES: tuple[tuple[str, str], ...] = (
    ("architecture report", "outputs/reports/technology_prize_agent_report.md"),
    ("architecture json", "outputs/reports/technology_prize_agent_report.json"),
    ("sdk runner report", "outputs/reports/technology_prize_sdk_runner.md"),
    ("anthropic critic report", "outputs/reports/technology_prize_anthropic_critic.md"),
    ("local workflow report", "outputs/reports/technology_prize_workflow.md"),
    ("local workflow json", "outputs/reports/technology_prize_workflow.json"),
    ("agent guardrails report", "outputs/reports/technology_prize_guardrails.md"),
    ("agent guardrails json", "outputs/reports/technology_prize_guardrails.json"),
    ("agent topology report", "outputs/reports/technology_prize_topology.md"),
    ("agent topology json", "outputs/reports/technology_prize_topology.json"),
    ("agent trace replay", "outputs/reports/technology_prize_trace_replay.md"),
    ("agent trace replay json", "outputs/reports/technology_prize_trace_replay.json"),
    ("judge demo runbook", "outputs/reports/technology_prize_demo_runbook.md"),
    ("judge demo runbook json", "outputs/reports/technology_prize_demo_runbook.json"),
    ("demo pack report", "outputs/reports/technology_prize_demo_pack.md"),
    ("demo pack json", "outputs/reports/technology_prize_demo_pack.json"),
    ("technology dashboard", "outputs/reports/technology_prize_dashboard.html"),
)


@dataclass(frozen=True)
class JudgeRequirement:
    name: str
    prize_axis: str
    status: str
    evidence: tuple[str, ...]
    judge_note: str


@dataclass(frozen=True)
class JudgeEvidenceLink:
    name: str
    path: str
    present: bool
    kind: str
    bytes_size: int = 0
    sha256_12: str = ""

    @property
    def status(self) -> str:
        return "OK" if self.present else "MISSING"


@dataclass(frozen=True)
class TechnologyPrizeJudgePacket:
    generated_at: datetime
    status: str
    requirements: tuple[JudgeRequirement, ...]
    source_evidence: tuple[JudgeEvidenceLink, ...]
    report_evidence: tuple[JudgeEvidenceLink, ...]
    architecture: TechnologyPrizeReport
    workflow: LocalAgentWorkflow
    guardrails: AgentGuardrailSuite
    topology: AgentTopologyReport
    has_order_authority: bool
    online_model_calls_default_to_off: bool

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for requirement in self.requirements if requirement.status == "PASS")
        warn_count = sum(1 for requirement in self.requirements if requirement.status == "WARN")
        fail_count = sum(1 for requirement in self.requirements if requirement.status == "FAIL")
        return (
            "Technology Prize Judge Packet",
            f"  Overall: {self.status}",
            f"  Requirements: {pass_count} pass, {warn_count} warn, {fail_count} fail",
            f"  Agents: {len(self.architecture.agents)}",
            f"  SDK tools: {len(self.architecture.tools)}",
            f"  Workflow steps: {len(self.workflow.steps)}",
            f"  Blackboard writes: {len(self.workflow.blackboard)}",
            f"  Handoffs: {len(self.workflow.handoffs)}",
            f"  Guardrails: {sum(1 for check in self.guardrails.checks if check.status == 'PASS')}/{len(self.guardrails.checks)} pass",
            f"  Topology: {self.topology.status}",
            f"  Order authority exposed: {'yes' if self.has_order_authority else 'no'}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "requirements": [_dict(requirement) for requirement in self.requirements],
            "source_evidence": [_dict(item) for item in self.source_evidence],
            "report_evidence": [_dict(item) for item in self.report_evidence],
            "architecture": self.architecture.to_dict(),
            "workflow": self.workflow.to_dict(),
            "guardrails": self.guardrails.to_dict(),
            "topology": self.topology.to_dict(),
            "has_order_authority": self.has_order_authority,
            "online_model_calls_default_to_off": self.online_model_calls_default_to_off,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Claude Agent Trader Technology Prize Judge Packet",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## One-Screen Pitch",
                "",
                (
                    "Claude Agent Trader is presented as an AgentSDK-centered trading research control plane: "
                    "specialist agents call read-only evidence tools, write shared blackboard state, "
                    "handoff to risk/deployment/critic roles, and package a judge-verifiable audit trail."
                ),
                "",
                "## Requirement Matrix",
                "",
                _requirements_table(self.requirements),
                "",
                "## AgentSDK Evidence",
                "",
                f"- Agents: {len(self.architecture.agents)}",
                f"- Tool specs: {len(self.architecture.tools)}",
                f"- Expected SDK bridge tools: {', '.join(EXPECTED_SDK_BRIDGE_TOOLS)}",
                f"- Workflow steps: {len(self.workflow.steps)}",
                f"- Blackboard writes: {len(self.workflow.blackboard)}",
                f"- Handoffs: {len(self.workflow.handoffs)}",
                f"- Guardrail suite: {self.guardrails.status} ({sum(1 for check in self.guardrails.checks if check.status == 'PASS')}/{len(self.guardrails.checks)} pass)",
                f"- Topology report: {self.topology.status} ({sum(1 for check in self.topology.checks if check.status == 'PASS')}/{len(self.topology.checks)} pass)",
                f"- Online model calls default off: {'yes' if self.online_model_calls_default_to_off else 'no'}",
                f"- Agent order authority exposed: {'yes' if self.has_order_authority else 'no'}",
                "",
                "## Source Evidence",
                "",
                _evidence_table(self.source_evidence),
                "",
                "## Generated Report Evidence",
                "",
                _evidence_table(self.report_evidence),
                "",
                "## Judge Demo Script",
                "",
                "1. Run `quanthack tech-prize-pack` to regenerate the safe offline demo pack.",
                "2. Run `quanthack tech-prize-guardrails` to show executable AI/broker safety gates.",
                "3. Run `quanthack tech-prize-workflow` to show the deterministic blackboard trace.",
                "4. Run `quanthack tech-prize-judge-packet` to verify the requirement matrix.",
                "5. Optionally arm `--allow-online-sdk` and `--allow-online-anthropic` only when API keys and credit-spend approval are ready.",
                "",
            ]
        )


def build_technology_prize_judge_packet(
    *,
    project_root: str | Path = ".",
    source_files: tuple[tuple[str, str], ...] = DEFAULT_JUDGE_SOURCE_FILES,
    report_files: tuple[tuple[str, str], ...] = DEFAULT_JUDGE_REPORT_FILES,
) -> TechnologyPrizeJudgePacket:
    root = Path(project_root)
    architecture = run_local_technology_prize_demo(project_root=root)
    workflow = run_local_agent_workflow(project_root=root)
    guardrails = build_agent_guardrail_suite(project_root=root)
    topology = build_agent_topology_report()
    source_evidence = tuple(
        _inspect_evidence(name=name, path=root / relative_path) for name, relative_path in source_files
    )
    report_evidence = tuple(
        _inspect_evidence(name=name, path=root / relative_path) for name, relative_path in report_files
    )
    has_order_authority = _has_order_authority(architecture=architecture, workflow=workflow)
    online_model_calls_default_to_off = _online_model_calls_default_to_off(source_evidence)
    requirements = _build_requirements(
        architecture=architecture,
        workflow=workflow,
        guardrails=guardrails,
        topology=topology,
        source_evidence=source_evidence,
        report_evidence=report_evidence,
        has_order_authority=has_order_authority,
        online_model_calls_default_to_off=online_model_calls_default_to_off,
    )
    return TechnologyPrizeJudgePacket(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(requirements),
        requirements=requirements,
        source_evidence=source_evidence,
        report_evidence=report_evidence,
        architecture=architecture,
        workflow=workflow,
        guardrails=guardrails,
        topology=topology,
        has_order_authority=has_order_authority,
        online_model_calls_default_to_off=online_model_calls_default_to_off,
    )


def write_technology_prize_judge_packet(
    packet: TechnologyPrizeJudgePacket,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(packet.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(packet.to_dict(), indent=2), encoding="utf-8")


def _build_requirements(
    *,
    architecture: TechnologyPrizeReport,
    workflow: LocalAgentWorkflow,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    source_evidence: tuple[JudgeEvidenceLink, ...],
    report_evidence: tuple[JudgeEvidenceLink, ...],
    has_order_authority: bool,
    online_model_calls_default_to_off: bool,
) -> tuple[JudgeRequirement, ...]:
    tool_names = tuple(sorted(tool.name for tool in architecture.tools))
    all_tools_read_only = all(tool.read_only for tool in architecture.tools)
    all_evidence_present = all(artifact.present for artifact in architecture.artifacts)
    source_count = sum(1 for item in source_evidence if item.present)
    report_count = sum(1 for item in report_evidence if item.present)
    anthropic_agent = any(agent.name == "Anthropic Critic Agent" for agent in architecture.agents)
    anthropic_provider = any(provider.name == "Anthropic Critic" for provider in architecture.providers)
    sdk_graph_ok = len(architecture.agents) >= 9 and len(workflow.handoffs) >= 8
    workflow_ok = (
        workflow.status == "PASS"
        and len(workflow.steps) >= 9
        and len(workflow.blackboard) >= 9
    )
    guardrails_ok = guardrails.status == "PASS" and len(guardrails.checks) >= 8
    topology_ok = topology.status == "PASS" and len(topology.checks) >= 8
    trace_source_present = _evidence_present(source_evidence, "agent trace replay")
    trace_report_present = _evidence_present(report_evidence, "agent trace replay")
    runbook_source_present = _evidence_present(source_evidence, "demo director")
    runbook_report_present = _evidence_present(report_evidence, "judge demo runbook")
    return (
        JudgeRequirement(
            name="AgentSDK-centered control plane",
            prize_axis="Best use of AgentSDK",
            status="PASS" if sdk_graph_ok else "FAIL",
            evidence=(
                "src/quanthack/agents/technology_prize.py",
                "outputs/reports/technology_prize_agent_report.json",
                "outputs/reports/technology_prize_workflow.json",
            ),
            judge_note=(
                f"{len(architecture.agents)} specialist agents, {len(workflow.handoffs)} handoffs, "
                "and a Chief Trading Agent coordinator."
            ),
        ),
        JudgeRequirement(
            name="Grounded function-tool bridge",
            prize_axis="Best use of AgentSDK",
            status="PASS" if tool_names == EXPECTED_SDK_BRIDGE_TOOLS and all_tools_read_only else "FAIL",
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "src/quanthack/agents/technology_prize.py",
            ),
            judge_note=(
                f"{len(tool_names)} read-only tools expose artifact summaries, data health, "
                "leaderboards, MT5 tickets, dashboards, and this judge packet."
            ),
        ),
        JudgeRequirement(
            name="Anthropic critic integration",
            prize_axis="Additional Anthropic Credits",
            status="PASS" if anthropic_agent and anthropic_provider else "FAIL",
            evidence=(
                "src/quanthack/agents/anthropic_critic.py",
                "outputs/reports/technology_prize_anthropic_critic.md",
            ),
            judge_note=(
                "Anthropic is wired as an independent critic role for overfitting, risk, "
                "and presentation review without broker authority."
            ),
        ),
        JudgeRequirement(
            name="AI-native blackboard workflow",
            prize_axis="Why this is AI-native",
            status="PASS" if workflow_ok else "WARN",
            evidence=(
                "src/quanthack/agents/workflow.py",
                "outputs/reports/technology_prize_workflow.md",
            ),
            judge_note=(
                f"{len(workflow.steps)} executable steps and {len(workflow.blackboard)} blackboard "
                "writes make the agent reasoning trace inspectable."
            ),
        ),
        JudgeRequirement(
            name="Executable AI guardrails",
            prize_axis="Best use of AgentSDK",
            status="PASS" if guardrails_ok else "FAIL",
            evidence=(
                "src/quanthack/agents/guardrails.py",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            judge_note=(
                f"{len(guardrails.checks)} guardrail checks cover read-only tools, path confinement, "
                "credit spend arming, Anthropic review-only authority, and MT5 no-order boundaries."
            ),
        ),
        JudgeRequirement(
            name="Validated AgentSDK topology",
            prize_axis="Best use of AgentSDK",
            status="PASS" if topology_ok else "FAIL",
            evidence=(
                "src/quanthack/agents/topology.py",
                "outputs/reports/technology_prize_topology.md",
            ),
            judge_note=(
                f"{len(topology.checks)} topology checks validate tool usage, handoff targets, "
                "guardrail coverage, read-only authority, and the Anthropic critic boundary."
            ),
        ),
        JudgeRequirement(
            name="Offline AgentSDK trace replay",
            prize_axis="Best use of AgentSDK",
            status="PASS" if trace_source_present and trace_report_present else "WARN",
            evidence=(
                "src/quanthack/agents/trace_replay.py",
                "outputs/reports/technology_prize_trace_replay.md",
            ),
            judge_note=(
                "The local workflow is exported as trace-like spans for agent steps, tool calls, "
                "blackboard writes, and handoffs without online model calls."
            ),
        ),
        JudgeRequirement(
            name="Judge demo readiness",
            prize_axis="Technical merit",
            status="PASS" if runbook_source_present and runbook_report_present else "WARN",
            evidence=(
                "src/quanthack/agents/demo_director.py",
                "outputs/reports/technology_prize_demo_runbook.md",
            ),
            judge_note=(
                "A timed demo runbook maps commands, talk track, proof artifacts, and risk answers "
                "for a reliable live presentation."
            ),
        ),
        JudgeRequirement(
            name="Evidence-grounded trading research",
            prize_axis="Technical merit",
            status="PASS" if all_evidence_present else "WARN",
            evidence=tuple(artifact.path for artifact in architecture.artifacts),
            judge_note=(
                f"{sum(1 for artifact in architecture.artifacts if artifact.present)}/"
                f"{len(architecture.artifacts)} research and deployment artifacts are present."
            ),
        ),
        JudgeRequirement(
            name="Broker-safe autonomy boundary",
            prize_axis="Responsible AI architecture",
            status="PASS" if not has_order_authority and all_tools_read_only else "FAIL",
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "outputs/reports/technology_prize_sdk_runner.md",
            ),
            judge_note="Agents can inspect MT5 ticket sheets but no registered tool can place orders.",
        ),
        JudgeRequirement(
            name="Guarded model-credit spend",
            prize_axis="Operational discipline",
            status="PASS" if online_model_calls_default_to_off else "FAIL",
            evidence=(
                "src/quanthack/agents/sdk_runner.py",
                "src/quanthack/agents/anthropic_critic.py",
            ),
            judge_note="OpenAI AgentSDK and Anthropic calls default to skipped unless explicitly armed.",
        ),
        JudgeRequirement(
            name="Judge reproducibility packet",
            prize_axis="Technical merit",
            status="PASS" if source_count == len(source_evidence) and report_count >= 6 else "WARN",
            evidence=tuple(item.path for item in (*source_evidence, *report_evidence) if item.present),
            judge_note=(
                f"Hashed {source_count}/{len(source_evidence)} source files and "
                f"{report_count}/{len(report_evidence)} generated reports."
            ),
        ),
    )


def _has_order_authority(
    *,
    architecture: TechnologyPrizeReport,
    workflow: LocalAgentWorkflow,
) -> bool:
    if any(not tool.read_only for tool in architecture.tools):
        return True
    for item in workflow.blackboard:
        if item.key == "order_authority":
            return "none" not in item.value.lower()
    return True


def _online_model_calls_default_to_off(
    source_evidence: tuple[JudgeEvidenceLink, ...],
) -> bool:
    required_sources = {"guarded sdk runner", "anthropic critic"}
    present = {item.name for item in source_evidence if item.present}
    return required_sources.issubset(present)


def _overall_status(requirements: tuple[JudgeRequirement, ...]) -> str:
    if any(requirement.status == "FAIL" for requirement in requirements):
        return "FAIL"
    if any(requirement.status == "WARN" for requirement in requirements):
        return "WARN"
    return "PASS"


def _inspect_evidence(*, name: str, path: Path) -> JudgeEvidenceLink:
    if not path.exists():
        return JudgeEvidenceLink(
            name=name,
            path=str(path),
            present=False,
            kind=_kind(path),
        )
    payload = path.read_bytes()
    return JudgeEvidenceLink(
        name=name,
        path=str(path),
        present=True,
        kind=_kind(path),
        bytes_size=len(payload),
        sha256_12=hashlib.sha256(payload).hexdigest()[:12],
    )


def _kind(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "file"


def _evidence_present(items: tuple[JudgeEvidenceLink, ...], name: str) -> bool:
    return any(item.name == name and item.present for item in items)


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _requirements_table(requirements: tuple[JudgeRequirement, ...]) -> str:
    rows = [
        "| Requirement | Prize Axis | Status | Evidence | Judge Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for requirement in requirements:
        rows.append(
            f"| {requirement.name} | {requirement.prize_axis} | {requirement.status} | "
            f"{'<br>'.join(requirement.evidence)} | {requirement.judge_note} |"
        )
    return "\n".join(rows)


def _evidence_table(items: tuple[JudgeEvidenceLink, ...]) -> str:
    rows = [
        "| Evidence | Status | Kind | Bytes | SHA-256 Prefix | Path |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in items:
        rows.append(
            f"| {item.name} | {item.status} | {item.kind} | {item.bytes_size} | "
            f"`{item.sha256_12 or '-'}` | `{item.path}` |"
        )
    return "\n".join(rows)
