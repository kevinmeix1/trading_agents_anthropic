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
from quanthack.agents.topology import AgentTopologyReport, build_agent_topology_report
from quanthack.agents.trace_replay import AgentTraceReplay, build_agent_trace_replay


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    prize_axis: str
    status: str
    score: int
    max_score: int
    evidence: tuple[str, ...]
    judge_note: str
    next_improvement: str


@dataclass(frozen=True)
class TechnologyPrizeRubric:
    generated_at: datetime
    status: str
    total_score: int
    max_score: int
    criteria: tuple[RubricCriterion, ...]
    judge_packet_status: str
    guardrail_status: str
    topology_status: str
    trace_status: str
    runbook_status: str

    @property
    def score_pct(self) -> float:
        if self.max_score == 0:
            return 0.0
        return self.total_score / self.max_score * 100.0

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for criterion in self.criteria if criterion.status == "PASS")
        warn_count = sum(1 for criterion in self.criteria if criterion.status == "WARN")
        fail_count = sum(1 for criterion in self.criteria if criterion.status == "FAIL")
        return (
            "Technology Prize Rubric",
            f"  Overall: {self.status}",
            f"  Score: {self.total_score}/{self.max_score} ({self.score_pct:.1f}%)",
            f"  Criteria: {pass_count} pass, {warn_count} warn, {fail_count} fail",
            f"  Judge packet: {self.judge_packet_status}",
            f"  Guardrails: {self.guardrail_status}",
            f"  Topology: {self.topology_status}",
            f"  Trace replay: {self.trace_status}",
            f"  Runbook: {self.runbook_status}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "score_pct": self.score_pct,
            "criteria": [_dict(criterion) for criterion in self.criteria],
            "judge_packet_status": self.judge_packet_status,
            "guardrail_status": self.guardrail_status,
            "topology_status": self.topology_status,
            "trace_status": self.trace_status,
            "runbook_status": self.runbook_status,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Claude Agent Trader Technology Prize Rubric",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                f"Score: **{self.total_score}/{self.max_score} ({self.score_pct:.1f}%)**",
                "",
                "## Judge-Facing Verdict",
                "",
                _verdict(self),
                "",
                "## Scored Criteria",
                "",
                _criteria_table(self.criteria),
                "",
                "## Component Status",
                "",
                f"- Judge packet: `{self.judge_packet_status}`",
                f"- Guardrails: `{self.guardrail_status}`",
                f"- AgentSDK topology: `{self.topology_status}`",
                f"- Offline trace replay: `{self.trace_status}`",
                f"- Demo runbook: `{self.runbook_status}`",
                "",
                "## How To Use In The Demo",
                "",
                "1. Open `outputs/reports/technology_prize_dashboard.html`.",
                "2. Run `quanthack tech-prize-rubric` to show the scored prize framing.",
                "3. Run `quanthack tech-prize-topology`, `quanthack tech-prize-trace`, and `quanthack tech-prize-guardrails` for executable proof.",
                "4. Keep optional online OpenAI/Anthropic calls disabled unless API keys and credit-spend approval are ready.",
                "",
            ]
        )


def build_technology_prize_rubric(
    *,
    project_root: str | Path = ".",
) -> TechnologyPrizeRubric:
    root = Path(project_root)
    judge_packet = build_technology_prize_judge_packet(project_root=root)
    guardrails = build_agent_guardrail_suite(project_root=root)
    topology = build_agent_topology_report()
    trace = build_agent_trace_replay(project_root=root)
    runbook = build_judge_demo_runbook(project_root=root)
    criteria = _build_criteria(
        judge_packet=judge_packet,
        guardrails=guardrails,
        topology=topology,
        trace=trace,
        runbook=runbook,
    )
    return TechnologyPrizeRubric(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(criteria),
        total_score=sum(criterion.score for criterion in criteria),
        max_score=sum(criterion.max_score for criterion in criteria),
        criteria=criteria,
        judge_packet_status=judge_packet.status,
        guardrail_status=guardrails.status,
        topology_status=topology.status,
        trace_status=trace.status,
        runbook_status=runbook.status,
    )


def write_technology_prize_rubric(
    rubric: TechnologyPrizeRubric,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rubric.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(rubric.to_dict(), indent=2), encoding="utf-8")


def _build_criteria(
    *,
    judge_packet: TechnologyPrizeJudgePacket,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    trace: AgentTraceReplay,
    runbook: JudgeDemoRunbook,
) -> tuple[RubricCriterion, ...]:
    sdk_requirements = (
        "AgentSDK-centered control plane",
        "Grounded function-tool bridge",
        "Validated AgentSDK topology",
        "Offline AgentSDK trace replay",
    )
    anthropic_requirements = ("Anthropic critic integration",)
    ai_native_requirements = ("AI-native blackboard workflow", "Executable AI guardrails")
    reproducibility_requirements = ("Judge reproducibility packet", "Evidence-grounded trading research")
    safety_requirements = ("Broker-safe autonomy boundary", "Guarded model-credit spend")
    demo_requirements = ("Judge demo readiness",)
    source_count = sum(1 for item in judge_packet.source_evidence if item.present)
    report_count = sum(1 for item in judge_packet.report_evidence if item.present)
    return (
        RubricCriterion(
            name="Best Use Of AgentSDK",
            prize_axis="Best use of AgentSDK",
            max_score=25,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, sdk_requirements),
                    topology.status == "PASS",
                    trace.status == "PASS",
                    len(judge_packet.architecture.agents) >= 9,
                    len(judge_packet.architecture.tools) >= 12,
                ),
                max_score=25,
            ),
            evidence=(
                "src/quanthack/agents/technology_prize.py",
                "src/quanthack/agents/sdk_bridge.py",
                "outputs/reports/technology_prize_topology.md",
                "outputs/reports/technology_prize_trace_replay.md",
            ),
            judge_note=(
                f"{len(judge_packet.architecture.agents)} agents, "
                f"{len(judge_packet.architecture.tools)} read-only tool specs, "
                f"{len(judge_packet.workflow.handoffs)} handoffs, and {len(trace.spans)} trace spans."
            ),
            next_improvement="Run the optional online AgentSDK judge brief when API key and credit approval are ready.",
        ),
        RubricCriterion(
            name="Additional Anthropic Credits",
            prize_axis="Additional Anthropic Credits",
            max_score=15,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, anthropic_requirements),
                    _guardrail_pass(guardrails, "Online Anthropic calls require arming"),
                    _guardrail_pass(guardrails, "Anthropic remains critique-only"),
                ),
                max_score=15,
            ),
            evidence=(
                "src/quanthack/agents/anthropic_critic.py",
                "outputs/reports/technology_prize_anthropic_critic.md",
                "src/quanthack/agents/technology_prize.py",
            ),
            judge_note=(
                "Anthropic is wired as an independent critique path for overfitting, risk, "
                "and demo review, with no trade approval authority."
            ),
            next_improvement="Use Anthropic live only for final critique once credits and keys are intentionally armed.",
        ),
        RubricCriterion(
            name="AI-Native Innovation",
            prize_axis="Why this is AI-native and innovative",
            max_score=20,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, ai_native_requirements),
                    judge_packet.workflow.status == "PASS",
                    len(judge_packet.workflow.blackboard) >= 9,
                    len(judge_packet.workflow.handoffs) >= 8,
                    guardrails.status == "PASS",
                ),
                max_score=20,
            ),
            evidence=(
                "src/quanthack/agents/workflow.py",
                "src/quanthack/agents/guardrails.py",
                "outputs/reports/technology_prize_workflow.md",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            judge_note=(
                "The system is organized around agent roles, grounded tools, blackboard state, "
                "handoffs, critic loops, and executable safety checks."
            ),
            next_improvement="Add a live trace export from an armed AgentSDK run after the offline proof is accepted.",
        ),
        RubricCriterion(
            name="Technical Merit And Reproducibility",
            prize_axis="Technical merit",
            max_score=20,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, reproducibility_requirements),
                    judge_packet.status == "PASS",
                    source_count == len(judge_packet.source_evidence),
                    report_count == len(judge_packet.report_evidence),
                ),
                max_score=20,
            ),
            evidence=(
                "outputs/reports/technology_prize_judge_packet.md",
                "outputs/reports/technology_prize_demo_pack.md",
                "outputs/reports/technology_prize_submission.md",
            ),
            judge_note=(
                f"Source evidence: {source_count}/{len(judge_packet.source_evidence)} present; "
                f"report evidence: {report_count}/{len(judge_packet.report_evidence)} present."
            ),
            next_improvement="Refresh the bundle immediately before judging so every hash reflects the current worktree.",
        ),
        RubricCriterion(
            name="Responsible Trading Safety",
            prize_axis="Responsible AI architecture",
            max_score=10,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, safety_requirements),
                    guardrails.status == "PASS",
                    not judge_packet.has_order_authority,
                    all(tool.read_only for tool in judge_packet.architecture.tools),
                ),
                max_score=10,
            ),
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "src/quanthack/agents/guardrails.py",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            judge_note="Agents can inspect research, risk, dashboard, and ticket artifacts but cannot place MT5 orders.",
            next_improvement="Keep live MT5 execution as a separate human-armed path after strategy validation.",
        ),
        RubricCriterion(
            name="Demo Readiness",
            prize_axis="Judge demo readiness",
            max_score=10,
            **_criterion_score(
                checks=(
                    _requirements_pass(judge_packet.requirements, demo_requirements),
                    runbook.status == "PASS",
                    len(runbook.steps) >= 6,
                    len(runbook.risk_notes) >= 3,
                ),
                max_score=10,
            ),
            evidence=(
                "outputs/reports/technology_prize_dashboard.html",
                "outputs/reports/technology_prize_demo_runbook.md",
                "outputs/reports/technology_prize_rubric.md",
            ),
            judge_note="The demo has a timed runbook, proof commands, risk answers, and one-page dashboard path.",
            next_improvement="Rehearse the three-minute flow and keep optional online demos disabled unless needed.",
        ),
    )


def _criterion_score(*, checks: tuple[bool, ...], max_score: int) -> dict[str, int | str]:
    passed = sum(1 for check in checks if check)
    if passed == len(checks):
        return {"status": "PASS", "score": max_score}
    if passed == 0:
        return {"status": "FAIL", "score": 0}
    return {"status": "WARN", "score": round(max_score * passed / len(checks))}


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


def _overall_status(criteria: tuple[RubricCriterion, ...]) -> str:
    if any(criterion.status == "FAIL" for criterion in criteria):
        return "FAIL"
    if any(criterion.status == "WARN" for criterion in criteria):
        return "WARN"
    return "PASS"


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _criteria_table(criteria: tuple[RubricCriterion, ...]) -> str:
    rows = [
        "| Criterion | Prize Axis | Status | Score | Evidence | Judge Note | Next Improvement |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for criterion in criteria:
        rows.append(
            f"| {criterion.name} | {criterion.prize_axis} | {criterion.status} | "
            f"{criterion.score}/{criterion.max_score} | {'<br>'.join(criterion.evidence)} | "
            f"{criterion.judge_note} | {criterion.next_improvement} |"
        )
    return "\n".join(rows)


def _verdict(rubric: TechnologyPrizeRubric) -> str:
    if rubric.status == "PASS":
        return (
            "The current build is judge-ready for the technology prize: it has a broad "
            "AgentSDK-centered control plane, an Anthropic critic path, executable guardrails, "
            "offline traceability, and reproducible evidence artifacts."
        )
    if rubric.status == "WARN":
        return (
            "The current build is close but should refresh the flagged evidence before judging. "
            "The score identifies exactly which prize axis still needs proof."
        )
    return (
        "The current build is not ready for technology-prize judging. Resolve the failing "
        "criteria before presenting the system."
    )
