from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.technology_prize import AgentNode, AgentToolSpec
from quanthack.agents.technology_prize import build_agent_architecture, build_agent_tool_specs


@dataclass(frozen=True)
class AgentToolCoverage:
    tool: str
    used_by: tuple[str, ...]
    read_only: bool

    @property
    def status(self) -> str:
        if not self.used_by:
            return "ORPHAN"
        if not self.read_only:
            return "UNSAFE"
        return "OK"


@dataclass(frozen=True)
class AgentHandoffEdge:
    from_agent: str
    to_agent: str
    status: str


@dataclass(frozen=True)
class AgentTopologyCheck:
    name: str
    status: str
    details: str


@dataclass(frozen=True)
class AgentTopologyReport:
    generated_at: datetime
    status: str
    agents: tuple[AgentNode, ...]
    tools: tuple[AgentToolSpec, ...]
    tool_coverage: tuple[AgentToolCoverage, ...]
    handoffs: tuple[AgentHandoffEdge, ...]
    checks: tuple[AgentTopologyCheck, ...]

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for check in self.checks if check.status == "PASS")
        warn_count = sum(1 for check in self.checks if check.status == "WARN")
        fail_count = sum(1 for check in self.checks if check.status == "FAIL")
        return (
            "AgentSDK Topology Report",
            f"  Overall: {self.status}",
            f"  Agents: {len(self.agents)}",
            f"  Tools: {len(self.tools)}",
            f"  Handoffs: {len(self.handoffs)}",
            f"  Checks: {pass_count} pass, {warn_count} warn, {fail_count} fail",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "agents": [_dict(agent) for agent in self.agents],
            "tools": [_dict(tool) for tool in self.tools],
            "tool_coverage": [_dict(coverage) | {"status": coverage.status} for coverage in self.tool_coverage],
            "handoffs": [_dict(edge) for edge in self.handoffs],
            "checks": [_dict(check) for check in self.checks],
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack AgentSDK Topology Report",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## Checks",
                "",
                _checks_table(self.checks),
                "",
                "## Tool Coverage",
                "",
                _tool_coverage_table(self.tool_coverage),
                "",
                "## Handoffs",
                "",
                _handoffs_table(self.handoffs),
                "",
            ]
        )


def build_agent_topology_report() -> AgentTopologyReport:
    agents = build_agent_architecture()
    tools = build_agent_tool_specs()
    tool_coverage = _build_tool_coverage(agents=agents, tools=tools)
    handoffs = _build_handoffs(agents=agents)
    checks = _build_checks(agents=agents, tools=tools, tool_coverage=tool_coverage, handoffs=handoffs)
    return AgentTopologyReport(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(checks),
        agents=agents,
        tools=tools,
        tool_coverage=tool_coverage,
        handoffs=handoffs,
        checks=checks,
    )


def write_agent_topology_report(
    report: AgentTopologyReport,
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


def _build_tool_coverage(
    *,
    agents: tuple[AgentNode, ...],
    tools: tuple[AgentToolSpec, ...],
) -> tuple[AgentToolCoverage, ...]:
    usage = {
        tool.name: tuple(agent.name for agent in agents if tool.name in agent.tools)
        for tool in tools
    }
    return tuple(
        AgentToolCoverage(
            tool=tool.name,
            used_by=usage[tool.name],
            read_only=tool.read_only,
        )
        for tool in tools
    )


def _build_handoffs(*, agents: tuple[AgentNode, ...]) -> tuple[AgentHandoffEdge, ...]:
    agent_names = {agent.name for agent in agents}
    return tuple(
        AgentHandoffEdge(
            from_agent=agent.name,
            to_agent=target,
            status="OK" if target in agent_names else "MISSING_TARGET",
        )
        for agent in agents
        for target in agent.handoffs
    )


def _build_checks(
    *,
    agents: tuple[AgentNode, ...],
    tools: tuple[AgentToolSpec, ...],
    tool_coverage: tuple[AgentToolCoverage, ...],
    handoffs: tuple[AgentHandoffEdge, ...],
) -> tuple[AgentTopologyCheck, ...]:
    tool_names = {tool.name for tool in tools}
    agent_tool_names = {tool for agent in agents for tool in agent.tools}
    unknown_agent_tools = tuple(sorted(agent_tool_names - tool_names))
    orphan_tools = tuple(sorted(coverage.tool for coverage in tool_coverage if not coverage.used_by))
    writable_tools = tuple(sorted(tool.name for tool in tools if not tool.read_only))
    missing_handoffs = tuple(edge for edge in handoffs if edge.status != "OK")
    agents_without_guardrails = tuple(sorted(agent.name for agent in agents if not agent.guardrails))
    agents_without_tools = tuple(sorted(agent.name for agent in agents if not agent.tools))
    critic = next((agent for agent in agents if agent.name == "Anthropic Critic Agent"), None)
    critic_text = "" if critic is None else " ".join((*critic.guardrails, critic.role)).lower()
    return (
        AgentTopologyCheck(
            name="All declared tools are used",
            status="PASS" if not orphan_tools else "FAIL",
            details=", ".join(orphan_tools) if orphan_tools else f"{len(tool_coverage)} tools have at least one agent.",
        ),
        AgentTopologyCheck(
            name="Agents reference known tools",
            status="PASS" if not unknown_agent_tools else "FAIL",
            details=", ".join(unknown_agent_tools) if unknown_agent_tools else "Every agent tool appears in the tool registry.",
        ),
        AgentTopologyCheck(
            name="All tools are read-only",
            status="PASS" if not writable_tools else "FAIL",
            details=", ".join(writable_tools) if writable_tools else f"{len(tools)} tools are read-only.",
        ),
        AgentTopologyCheck(
            name="Handoffs target real agents",
            status="PASS" if not missing_handoffs else "FAIL",
            details=(
                ", ".join(f"{edge.from_agent}->{edge.to_agent}" for edge in missing_handoffs)
                if missing_handoffs
                else f"{len(handoffs)} handoffs validated."
            ),
        ),
        AgentTopologyCheck(
            name="Every agent has guardrails",
            status="PASS" if not agents_without_guardrails else "FAIL",
            details=(
                ", ".join(agents_without_guardrails)
                if agents_without_guardrails
                else f"{len(agents)} agents include guardrails."
            ),
        ),
        AgentTopologyCheck(
            name="Every agent has tools",
            status="PASS" if not agents_without_tools else "WARN",
            details=(
                ", ".join(agents_without_tools)
                if agents_without_tools
                else f"{len(agents)} agents can call at least one tool."
            ),
        ),
        AgentTopologyCheck(
            name="Anthropic critic is review-only",
            status=(
                "PASS"
                if critic is not None
                and "no direct trade approval" in critic_text
                and "broker" in critic_text
                else "FAIL"
            ),
            details="Anthropic Critic Agent has no trade approval or broker authority.",
        ),
        AgentTopologyCheck(
            name="Chief agent delegates broadly",
            status="PASS" if len(agents[0].handoffs) >= 6 else "WARN",
            details=f"Chief Trading Agent has {len(agents[0].handoffs)} specialist handoffs.",
        ),
    )


def _overall_status(checks: tuple[AgentTopologyCheck, ...]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "WARN" for check in checks):
        return "WARN"
    return "PASS"


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _checks_table(checks: tuple[AgentTopologyCheck, ...]) -> str:
    rows = ["| Check | Status | Details |", "| --- | --- | --- |"]
    for check in checks:
        rows.append(f"| {check.name} | {check.status} | {check.details} |")
    return "\n".join(rows)


def _tool_coverage_table(coverage: tuple[AgentToolCoverage, ...]) -> str:
    rows = ["| Tool | Status | Read Only | Used By |", "| --- | --- | --- | --- |"]
    for item in coverage:
        rows.append(
            f"| {item.tool} | {item.status} | {'yes' if item.read_only else 'no'} | "
            f"{'<br>'.join(item.used_by) or '-'} |"
        )
    return "\n".join(rows)


def _handoffs_table(handoffs: tuple[AgentHandoffEdge, ...]) -> str:
    rows = ["| From | To | Status |", "| --- | --- | --- |"]
    for edge in handoffs:
        rows.append(f"| {edge.from_agent} | {edge.to_agent} | {edge.status} |")
    return "\n".join(rows)
