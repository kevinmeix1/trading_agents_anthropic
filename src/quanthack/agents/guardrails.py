from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.technology_prize import TechnologyPrizeReport, run_local_technology_prize_demo
from quanthack.agents.workflow import LocalAgentWorkflow, run_local_agent_workflow


@dataclass(frozen=True)
class AgentGuardrailCheck:
    name: str
    status: str
    scope: str
    evidence: tuple[str, ...]
    details: str


@dataclass(frozen=True)
class AgentGuardrailSuite:
    generated_at: datetime
    status: str
    checks: tuple[AgentGuardrailCheck, ...]

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for check in self.checks if check.status == "PASS")
        warn_count = sum(1 for check in self.checks if check.status == "WARN")
        fail_count = sum(1 for check in self.checks if check.status == "FAIL")
        return (
            "Agent Guardrail Suite",
            f"  Overall: {self.status}",
            f"  Checks: {pass_count} pass, {warn_count} warn, {fail_count} fail",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "checks": [_dict(check) for check in self.checks],
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Agent Guardrail Suite",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## Guardrails",
                "",
                _checks_table(self.checks),
                "",
            ]
        )


def build_agent_guardrail_suite(
    *,
    project_root: str | Path = ".",
) -> AgentGuardrailSuite:
    root = Path(project_root)
    architecture = run_local_technology_prize_demo(project_root=root)
    workflow = run_local_agent_workflow(project_root=root)
    sdk_bridge_source = _read_text(root / "src/quanthack/agents/sdk_bridge.py")
    sdk_runner_source = _read_text(root / "src/quanthack/agents/sdk_runner.py")
    anthropic_source = _read_text(root / "src/quanthack/agents/anthropic_critic.py")
    checks = _build_checks(
        architecture=architecture,
        workflow=workflow,
        sdk_bridge_source=sdk_bridge_source,
        sdk_runner_source=sdk_runner_source,
        anthropic_source=anthropic_source,
    )
    return AgentGuardrailSuite(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(checks),
        checks=checks,
    )


def write_agent_guardrail_suite(
    suite: AgentGuardrailSuite,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(suite.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(suite.to_dict(), indent=2), encoding="utf-8")


def _build_checks(
    *,
    architecture: TechnologyPrizeReport,
    workflow: LocalAgentWorkflow,
    sdk_bridge_source: str,
    sdk_runner_source: str,
    anthropic_source: str,
) -> tuple[AgentGuardrailCheck, ...]:
    all_tools_read_only = all(tool.read_only for tool in architecture.tools)
    order_authority = _blackboard_value(workflow, "order_authority")
    artifact_present = sum(1 for artifact in architecture.artifacts if artifact.present)
    return (
        AgentGuardrailCheck(
            name="Read-only AgentSDK tools",
            status="PASS" if all_tools_read_only else "FAIL",
            scope="AgentSDK tool surface",
            evidence=("src/quanthack/agents/technology_prize.py",),
            details=f"{len(architecture.tools)} tool specs inspected; every tool must be read-only.",
        ),
        AgentGuardrailCheck(
            name="No broker order authority",
            status="PASS" if "none" in order_authority.lower() else "FAIL",
            scope="MT5/deployment boundary",
            evidence=("src/quanthack/agents/workflow.py", "src/quanthack/agents/sdk_bridge.py"),
            details=f"Workflow blackboard order authority: {order_authority}",
        ),
        AgentGuardrailCheck(
            name="Project-local path confinement",
            status=(
                "PASS"
                if "relative_to(root)" in sdk_bridge_source
                and "path is outside project root" in sdk_bridge_source
                else "FAIL"
            ),
            scope="AgentSDK function tools",
            evidence=("src/quanthack/agents/sdk_bridge.py",),
            details="SDK CSV/artifact tools must reject paths outside the project root.",
        ),
        AgentGuardrailCheck(
            name="Online OpenAI calls require arming",
            status=(
                "PASS"
                if "allow_online_sdk" in sdk_runner_source
                and "OPENAI_API_KEY" in sdk_runner_source
                else "FAIL"
            ),
            scope="Credit spend gate",
            evidence=("src/quanthack/agents/sdk_runner.py",),
            details="AgentSDK Runner calls default to skipped/blocked unless explicitly allowed.",
        ),
        AgentGuardrailCheck(
            name="Online Anthropic calls require arming",
            status=(
                "PASS"
                if "allow_online_anthropic" in anthropic_source
                and "ANTHROPIC_API_KEY" in anthropic_source
                else "FAIL"
            ),
            scope="Additional Anthropic Credits",
            evidence=("src/quanthack/agents/anthropic_critic.py",),
            details="Anthropic critic calls default to skipped/blocked unless explicitly allowed.",
        ),
        AgentGuardrailCheck(
            name="Anthropic remains critique-only",
            status="PASS" if _anthropic_is_critique_only(architecture) else "FAIL",
            scope="Model authority separation",
            evidence=("src/quanthack/agents/technology_prize.py",),
            details="Anthropic Critic Agent must not approve trades or receive broker credentials.",
        ),
        AgentGuardrailCheck(
            name="Evidence before claims",
            status="PASS" if artifact_present == len(architecture.artifacts) else "WARN",
            scope="Judge narrative grounding",
            evidence=tuple(artifact.path for artifact in architecture.artifacts),
            details=f"{artifact_present}/{len(architecture.artifacts)} required research/deployment artifacts are present.",
        ),
        AgentGuardrailCheck(
            name="Deterministic workflow trace",
            status=(
                "PASS"
                if workflow.status == "PASS"
                and len(workflow.steps) >= 9
                and len(workflow.blackboard) >= 9
                else "WARN"
            ),
            scope="Agentic auditability",
            evidence=("outputs/reports/technology_prize_workflow.json",),
            details=(
                f"{len(workflow.steps)} steps, {len(workflow.blackboard)} blackboard writes, "
                f"{len(workflow.handoffs)} handoffs."
            ),
        ),
    )


def _blackboard_value(workflow: LocalAgentWorkflow, key: str) -> str:
    for item in workflow.blackboard:
        if item.key == key:
            return item.value
    return "missing"


def _anthropic_is_critique_only(architecture: TechnologyPrizeReport) -> bool:
    for agent in architecture.agents:
        if agent.name != "Anthropic Critic Agent":
            continue
        text = " ".join((*agent.guardrails, agent.role)).lower()
        return "no direct trade approval" in text and "broker" in text
    return False


def _overall_status(checks: tuple[AgentGuardrailCheck, ...]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _checks_table(checks: tuple[AgentGuardrailCheck, ...]) -> str:
    rows = [
        "| Guardrail | Status | Scope | Evidence | Details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in checks:
        rows.append(
            f"| {check.name} | {check.status} | {check.scope} | "
            f"{'<br>'.join(check.evidence)} | {check.details} |"
        )
    return "\n".join(rows)
