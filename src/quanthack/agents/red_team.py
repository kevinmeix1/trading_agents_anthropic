from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.demo_director import JudgeDemoRunbook, build_judge_demo_runbook
from quanthack.agents.guardrails import AgentGuardrailSuite, build_agent_guardrail_suite
from quanthack.agents.judge_packet import (
    JudgeRequirement,
    TechnologyPrizeJudgePacket,
    build_technology_prize_judge_packet,
)
from quanthack.agents.rubric import TechnologyPrizeRubric, build_technology_prize_rubric
from quanthack.agents.topology import AgentTopologyReport, build_agent_topology_report
from quanthack.agents.trace_replay import AgentTraceReplay, build_agent_trace_replay


@dataclass(frozen=True)
class RedTeamCheck:
    name: str
    status: str
    details: str


@dataclass(frozen=True)
class RedTeamChallenge:
    name: str
    prize_axis: str
    skeptical_question: str
    status: str
    demo_answer: str
    evidence: tuple[str, ...]
    checks: tuple[RedTeamCheck, ...]


@dataclass(frozen=True)
class TechnologyPrizeRedTeamReport:
    generated_at: datetime
    status: str
    challenges: tuple[RedTeamChallenge, ...]
    judge_packet_status: str
    guardrail_status: str
    topology_status: str
    trace_status: str
    runbook_status: str
    rubric_status: str

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for challenge in self.challenges if challenge.status == "PASS")
        warn_count = sum(1 for challenge in self.challenges if challenge.status == "WARN")
        fail_count = sum(1 for challenge in self.challenges if challenge.status == "FAIL")
        return (
            "Technology Prize Red-Team Report",
            f"  Overall: {self.status}",
            f"  Challenges: {pass_count} pass, {warn_count} warn, {fail_count} fail",
            f"  Judge packet: {self.judge_packet_status}",
            f"  Guardrails: {self.guardrail_status}",
            f"  Topology: {self.topology_status}",
            f"  Trace replay: {self.trace_status}",
            f"  Runbook: {self.runbook_status}",
            f"  Rubric: {self.rubric_status}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "challenges": [_challenge_dict(challenge) for challenge in self.challenges],
            "judge_packet_status": self.judge_packet_status,
            "guardrail_status": self.guardrail_status,
            "topology_status": self.topology_status,
            "trace_status": self.trace_status,
            "runbook_status": self.runbook_status,
            "rubric_status": self.rubric_status,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Technology Prize Red-Team Report",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## Purpose",
                "",
                (
                    "This report rehearses skeptical judge questions and verifies that each answer "
                    "is backed by current source, artifact, guardrail, topology, trace, and rubric evidence."
                ),
                "",
                "## Challenge Matrix",
                "",
                _challenge_table(self.challenges),
                "",
                "## Demo Answers",
                "",
                *(_challenge_sections(self.challenges)),
                "",
            ]
        )


def build_technology_prize_red_team_report(
    *,
    project_root: str | Path = ".",
) -> TechnologyPrizeRedTeamReport:
    root = Path(project_root)
    judge_packet = build_technology_prize_judge_packet(project_root=root)
    guardrails = build_agent_guardrail_suite(project_root=root)
    topology = build_agent_topology_report()
    trace = build_agent_trace_replay(project_root=root)
    runbook = build_judge_demo_runbook(project_root=root)
    rubric = build_technology_prize_rubric(project_root=root)
    challenges = _build_challenges(
        judge_packet=judge_packet,
        guardrails=guardrails,
        topology=topology,
        trace=trace,
        runbook=runbook,
        rubric=rubric,
    )
    return TechnologyPrizeRedTeamReport(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(challenges),
        challenges=challenges,
        judge_packet_status=judge_packet.status,
        guardrail_status=guardrails.status,
        topology_status=topology.status,
        trace_status=trace.status,
        runbook_status=runbook.status,
        rubric_status=rubric.status,
    )


def write_technology_prize_red_team_report(
    report: TechnologyPrizeRedTeamReport,
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


def _build_challenges(
    *,
    judge_packet: TechnologyPrizeJudgePacket,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    trace: AgentTraceReplay,
    runbook: JudgeDemoRunbook,
    rubric: TechnologyPrizeRubric,
) -> tuple[RedTeamChallenge, ...]:
    source_count = sum(1 for item in judge_packet.source_evidence if item.present)
    report_count = sum(1 for item in judge_packet.report_evidence if item.present)
    return (
        _challenge(
            name="No Hidden Trading Authority",
            prize_axis="Responsible AI architecture",
            skeptical_question="Can the AI agent secretly place MT5 trades or bypass the risk engine?",
            demo_answer=(
                "No. The AgentSDK surface is read-only, the workflow records no order authority, "
                "and the MT5 path is limited to advisory ticket sheets unless a human operates MT5 separately."
            ),
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "src/quanthack/agents/guardrails.py",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            checks=(
                _check(
                    "All AgentSDK tool specs are read-only",
                    all(tool.read_only for tool in judge_packet.architecture.tools),
                    f"{len(judge_packet.architecture.tools)} tool specs inspected.",
                ),
                _check(
                    "Judge packet reports no order authority",
                    not judge_packet.has_order_authority,
                    f"has_order_authority={judge_packet.has_order_authority}",
                ),
                _check(
                    "No broker order authority guardrail passes",
                    _guardrail_pass(guardrails, "No broker order authority"),
                    "Guardrail suite explicitly checks the MT5/deployment boundary.",
                ),
            ),
        ),
        _challenge(
            name="Real AgentSDK Architecture",
            prize_axis="Best use of AgentSDK",
            skeptical_question="Is AgentSDK actually central here, or is this just a normal Python project with an agent label?",
            demo_answer=(
                "AgentSDK is the control-plane design center: the project defines specialist agents, "
                "handoffs, read-only function tools, topology validation, and an optional real SDK bridge."
            ),
            evidence=(
                "src/quanthack/agents/technology_prize.py",
                "src/quanthack/agents/sdk_bridge.py",
                "outputs/reports/technology_prize_topology.md",
            ),
            checks=(
                _check(
                    "Specialist agent graph is broad enough",
                    len(judge_packet.architecture.agents) >= 9,
                    f"{len(judge_packet.architecture.agents)} agents defined.",
                ),
                _check(
                    "Read-only function tool registry is broad enough",
                    len(judge_packet.architecture.tools) >= 12,
                    f"{len(judge_packet.architecture.tools)} tool specs defined.",
                ),
                _check(
                    "AgentSDK topology validates",
                    topology.status == "PASS",
                    f"topology_status={topology.status}",
                ),
                _check(
                    "AgentSDK prize requirements pass",
                    _requirements_pass(
                        judge_packet.requirements,
                        (
                            "AgentSDK-centered control plane",
                            "Grounded function-tool bridge",
                            "Validated AgentSDK topology",
                            "Offline AgentSDK trace replay",
                        ),
                    ),
                    "Judge packet validates AgentSDK control, tools, topology, and trace replay.",
                ),
            ),
        ),
        _challenge(
            name="Anthropic Credits Have A Real Role",
            prize_axis="Additional Anthropic Credits",
            skeptical_question="Where would Anthropic credits actually improve the system?",
            demo_answer=(
                "Anthropic is wired as an independent critic for overfitting, risk, and demo review. "
                "It is deliberately critique-only and cannot approve trades."
            ),
            evidence=(
                "src/quanthack/agents/anthropic_critic.py",
                "outputs/reports/technology_prize_anthropic_critic.md",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            checks=(
                _check(
                    "Anthropic critic requirement passes",
                    _requirements_pass(judge_packet.requirements, ("Anthropic critic integration",)),
                    "Judge packet recognizes the Anthropic critic integration.",
                ),
                _check(
                    "Anthropic online calls require arming",
                    _guardrail_pass(guardrails, "Online Anthropic calls require arming"),
                    "Critic defaults to skipped/blocked unless explicitly armed.",
                ),
                _check(
                    "Anthropic critic is review-only",
                    _guardrail_pass(guardrails, "Anthropic remains critique-only"),
                    "Critic has no direct trade approval or broker authority.",
                ),
            ),
        ),
        _challenge(
            name="AI-Native, Not Script-Native",
            prize_axis="Why this is AI-native and innovative",
            skeptical_question="What makes this AI-native instead of many scripts wrapped in a dashboard?",
            demo_answer=(
                "The project models reasoning as agents, handoffs, blackboard writes, tool calls, "
                "critic loops, trace spans, and executable guardrails."
            ),
            evidence=(
                "src/quanthack/agents/workflow.py",
                "src/quanthack/agents/trace_replay.py",
                "outputs/reports/technology_prize_workflow.md",
                "outputs/reports/technology_prize_trace_replay.md",
            ),
            checks=(
                _check(
                    "Local agent workflow passes",
                    judge_packet.workflow.status == "PASS",
                    f"workflow_status={judge_packet.workflow.status}",
                ),
                _check(
                    "Blackboard state is inspectable",
                    len(judge_packet.workflow.blackboard) >= 9,
                    f"{len(judge_packet.workflow.blackboard)} blackboard writes.",
                ),
                _check(
                    "Trace replay is span-like and passing",
                    trace.status == "PASS" and len(trace.spans) >= 30,
                    f"trace_status={trace.status}; spans={len(trace.spans)}.",
                ),
                _check(
                    "Rubric AI-native criterion passes",
                    _rubric_pass(rubric, "AI-Native Innovation"),
                    "Rubric scores the AI-native architecture as passing.",
                ),
            ),
        ),
        _challenge(
            name="No Surprise Model Spend",
            prize_axis="Operational discipline",
            skeptical_question="Could a demo command accidentally spend OpenAI or Anthropic credits?",
            demo_answer=(
                "No default technology-prize command makes online model calls. Both OpenAI AgentSDK "
                "and Anthropic paths require explicit arming flags and API keys."
            ),
            evidence=(
                "src/quanthack/agents/sdk_runner.py",
                "src/quanthack/agents/anthropic_critic.py",
                "outputs/reports/technology_prize_submission.md",
            ),
            checks=(
                _check(
                    "OpenAI online calls require arming",
                    _guardrail_pass(guardrails, "Online OpenAI calls require arming"),
                    "AgentSDK Runner calls default to skipped/blocked.",
                ),
                _check(
                    "Anthropic online calls require arming",
                    _guardrail_pass(guardrails, "Online Anthropic calls require arming"),
                    "Anthropic critic calls default to skipped/blocked.",
                ),
                _check(
                    "Judge packet confirms offline defaults",
                    judge_packet.online_model_calls_default_to_off,
                    f"online_model_calls_default_to_off={judge_packet.online_model_calls_default_to_off}",
                ),
            ),
        ),
        _challenge(
            name="Evidence And Provenance Survive Scrutiny",
            prize_axis="Technical merit",
            skeptical_question="Can the judges verify claims from files instead of trusting the presentation?",
            demo_answer=(
                "Yes. The judge packet, demo pack, rubric, and submission bundle hash source and "
                "generated reports, and the red-team checks reuse those current artifacts."
            ),
            evidence=(
                "outputs/reports/technology_prize_judge_packet.md",
                "outputs/reports/technology_prize_demo_pack.md",
                "outputs/reports/technology_prize_rubric.md",
            ),
            checks=(
                _check(
                    "Judge packet passes",
                    judge_packet.status == "PASS",
                    f"judge_packet_status={judge_packet.status}",
                ),
                _check(
                    "All tracked source evidence is present",
                    source_count == len(judge_packet.source_evidence),
                    f"{source_count}/{len(judge_packet.source_evidence)} source files present.",
                ),
                _check(
                    "All tracked report evidence is present",
                    report_count == len(judge_packet.report_evidence),
                    f"{report_count}/{len(judge_packet.report_evidence)} reports present.",
                ),
                _check(
                    "Rubric reaches full score",
                    rubric.status == "PASS" and rubric.total_score == rubric.max_score,
                    f"rubric={rubric.total_score}/{rubric.max_score}; status={rubric.status}.",
                ),
            ),
        ),
        _challenge(
            name="Demo Can Be Reproduced Under Time Pressure",
            prize_axis="Judge demo readiness",
            skeptical_question="If the live demo gets rushed, is there a reliable proof path?",
            demo_answer=(
                "Yes. The runbook gives a timed sequence, the dashboard is the first artifact to open, "
                "and every proof command is safe offline by default."
            ),
            evidence=(
                "outputs/reports/technology_prize_dashboard.html",
                "outputs/reports/technology_prize_demo_runbook.md",
                "outputs/reports/technology_prize_submission.md",
            ),
            checks=(
                _check(
                    "Runbook passes",
                    runbook.status == "PASS",
                    f"runbook_status={runbook.status}",
                ),
                _check(
                    "Runbook has enough timed steps",
                    len(runbook.steps) >= 6,
                    f"{len(runbook.steps)} runbook steps.",
                ),
                _check(
                    "Runbook includes risk answers",
                    len(runbook.risk_notes) >= 3,
                    f"{len(runbook.risk_notes)} risk notes.",
                ),
                _check(
                    "Demo readiness rubric passes",
                    _rubric_pass(rubric, "Demo Readiness"),
                    "Rubric validates the timed demo path.",
                ),
            ),
        ),
        _challenge(
            name="Technology Prize Story Is Separate From PnL",
            prize_axis="Technology prize positioning",
            skeptical_question="If trading P&L is weak, why is this still competitive for the separate technology prize?",
            demo_answer=(
                "The technology submission is judged on architecture: AgentSDK control plane, "
                "multi-provider critique, reproducibility, guardrails, and operator-safe deployment boundaries."
            ),
            evidence=(
                "outputs/reports/technology_prize_rubric.md",
                "outputs/reports/technology_prize_judge_packet.md",
                "docs/50_technology_prize/126_AGENTSDK_CONTROL_PLANE.md",
            ),
            checks=(
                _check(
                    "Technical merit requirement passes",
                    _requirements_pass(judge_packet.requirements, ("Judge reproducibility packet",)),
                    "Judge packet proves the reproducibility story.",
                ),
                _check(
                    "Responsible safety criterion passes",
                    _rubric_pass(rubric, "Responsible Trading Safety"),
                    "Rubric separates safe architecture from return performance.",
                ),
                _check(
                    "Best-use AgentSDK criterion passes",
                    _rubric_pass(rubric, "Best Use Of AgentSDK"),
                    "Rubric validates the central technology-prize axis.",
                ),
            ),
        ),
    )


def _challenge(
    *,
    name: str,
    prize_axis: str,
    skeptical_question: str,
    demo_answer: str,
    evidence: tuple[str, ...],
    checks: tuple[RedTeamCheck, ...],
) -> RedTeamChallenge:
    return RedTeamChallenge(
        name=name,
        prize_axis=prize_axis,
        skeptical_question=skeptical_question,
        status=_checks_status(checks),
        demo_answer=demo_answer,
        evidence=evidence,
        checks=checks,
    )


def _check(name: str, passed: bool, details: str) -> RedTeamCheck:
    return RedTeamCheck(
        name=name,
        status="PASS" if passed else "FAIL",
        details=details,
    )


def _checks_status(checks: tuple[RedTeamCheck, ...]) -> str:
    if all(check.status == "PASS" for check in checks):
        return "PASS"
    if any(check.status == "PASS" for check in checks):
        return "WARN"
    return "FAIL"


def _overall_status(challenges: tuple[RedTeamChallenge, ...]) -> str:
    if any(challenge.status == "FAIL" for challenge in challenges):
        return "FAIL"
    if any(challenge.status == "WARN" for challenge in challenges):
        return "WARN"
    return "PASS"


def _requirements_pass(
    requirements: tuple[JudgeRequirement, ...],
    names: tuple[str, ...],
) -> bool:
    required = set(names)
    passed = {
        requirement.name
        for requirement in requirements
        if requirement.name in required and requirement.status == "PASS"
    }
    return passed == required


def _guardrail_pass(suite: AgentGuardrailSuite, name: str) -> bool:
    return any(check.name == name and check.status == "PASS" for check in suite.checks)


def _rubric_pass(rubric: TechnologyPrizeRubric, name: str) -> bool:
    return any(criterion.name == name and criterion.status == "PASS" for criterion in rubric.criteria)


def _challenge_dict(challenge: RedTeamChallenge) -> dict[str, Any]:
    return {
        "name": challenge.name,
        "prize_axis": challenge.prize_axis,
        "skeptical_question": challenge.skeptical_question,
        "status": challenge.status,
        "demo_answer": challenge.demo_answer,
        "evidence": list(challenge.evidence),
        "checks": [_dict(check) for check in challenge.checks],
    }


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _challenge_table(challenges: tuple[RedTeamChallenge, ...]) -> str:
    rows = [
        "| Challenge | Prize Axis | Status | Skeptical Question | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for challenge in challenges:
        rows.append(
            f"| {challenge.name} | {challenge.prize_axis} | {challenge.status} | "
            f"{challenge.skeptical_question} | {'<br>'.join(challenge.evidence)} |"
        )
    return "\n".join(rows)


def _challenge_sections(challenges: tuple[RedTeamChallenge, ...]) -> tuple[str, ...]:
    lines: list[str] = []
    for challenge in challenges:
        lines.extend(
            [
                f"### {challenge.name}",
                "",
                f"Question: {challenge.skeptical_question}",
                "",
                f"Answer: {challenge.demo_answer}",
                "",
                f"Status: **{challenge.status}**",
                "",
                _checks_table(challenge.checks),
                "",
            ]
        )
    return tuple(lines)


def _checks_table(checks: tuple[RedTeamCheck, ...]) -> str:
    rows = ["| Check | Status | Details |", "| --- | --- | --- |"]
    for check in checks:
        rows.append(f"| {check.name} | {check.status} | {check.details} |")
    return "\n".join(rows)
