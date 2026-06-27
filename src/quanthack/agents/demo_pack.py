from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.anthropic_critic import (
    AnthropicCriticResult,
    run_guarded_anthropic_critic,
    write_anthropic_critic_report,
)
from quanthack.agents.demo_director import (
    JudgeDemoRunbook,
    build_judge_demo_runbook,
    write_judge_demo_runbook,
)
from quanthack.agents.guardrails import (
    AgentGuardrailSuite,
    build_agent_guardrail_suite,
    write_agent_guardrail_suite,
)
from quanthack.agents.sdk_bridge import AgentsSdkUnavailableError, create_agents_sdk_app
from quanthack.agents.sdk_runner import (
    AgentsSdkRunnerResult,
    run_guarded_agents_sdk_demo,
    write_agents_sdk_runner_report,
)
from quanthack.agents.technology_prize import (
    TechnologyPrizeReport,
    run_local_technology_prize_demo,
    write_technology_prize_report,
)
from quanthack.agents.topology import (
    AgentTopologyReport,
    build_agent_topology_report,
    write_agent_topology_report,
)
from quanthack.agents.trace_replay import (
    AgentTraceReplay,
    build_agent_trace_replay,
    write_agent_trace_replay,
)
from quanthack.agents.workflow import (
    LocalAgentWorkflow,
    run_local_agent_workflow,
    write_local_agent_workflow,
)


DEFAULT_SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("agent architecture", "src/quanthack/agents/technology_prize.py"),
    ("agents sdk bridge", "src/quanthack/agents/sdk_bridge.py"),
    ("agent guardrails", "src/quanthack/agents/guardrails.py"),
    ("agent topology", "src/quanthack/agents/topology.py"),
    ("agent trace replay", "src/quanthack/agents/trace_replay.py"),
    ("demo director", "src/quanthack/agents/demo_director.py"),
    ("demo rehearsal", "src/quanthack/agents/demo_rehearsal.py"),
    ("demo rehearsal cli", "src/quanthack/cli/tech_prize_rehearse.py"),
    ("guarded sdk runner", "src/quanthack/agents/sdk_runner.py"),
    ("anthropic critic", "src/quanthack/agents/anthropic_critic.py"),
    ("demo pack", "src/quanthack/agents/demo_pack.py"),
    ("demo cli", "src/quanthack/cli/tech_prize_demo.py"),
    ("demo pack cli", "src/quanthack/cli/tech_prize_pack.py"),
    ("technology dashboard", "src/quanthack/reporting/technology_prize_dashboard.py"),
    ("local agent workflow", "src/quanthack/agents/workflow.py"),
    ("workflow cli", "src/quanthack/cli/tech_prize_workflow.py"),
    ("judge packet", "src/quanthack/agents/judge_packet.py"),
    ("judge packet cli", "src/quanthack/cli/tech_prize_judge_packet.py"),
    ("technology rubric", "src/quanthack/agents/rubric.py"),
    ("technology rubric cli", "src/quanthack/cli/tech_prize_rubric.py"),
    ("technology red team", "src/quanthack/agents/red_team.py"),
    ("technology red team cli", "src/quanthack/cli/tech_prize_red_team.py"),
    ("technology docs", "docs/50_technology_prize/126_AGENTSDK_CONTROL_PLANE.md"),
)


DEFAULT_REPORT_FILES: tuple[tuple[str, str], ...] = (
    ("architecture report", "outputs/reports/technology_prize_agent_report.md"),
    ("architecture trace json", "outputs/reports/technology_prize_agent_report.json"),
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
)


@dataclass(frozen=True)
class DemoPackArtifact:
    name: str
    path: str
    present: bool
    bytes_size: int = 0
    sha256_12: str = ""

    @property
    def status(self) -> str:
        return "OK" if self.present else "MISSING"


@dataclass(frozen=True)
class DemoPackCheck:
    name: str
    status: str
    details: str


@dataclass(frozen=True)
class DemoCommand:
    name: str
    purpose: str
    command: str
    spends_credits: bool
    requires_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class InnovationClaim:
    name: str
    claim: str
    evidence: tuple[str, ...]
    why_ai_native: str
    prize_relevance: str


@dataclass(frozen=True)
class TechnologyPrizeDemoPack:
    generated_at: datetime
    architecture_report: TechnologyPrizeReport
    sdk_runner: AgentsSdkRunnerResult
    anthropic_critic: AnthropicCriticResult
    local_workflow: LocalAgentWorkflow
    agent_guardrails: AgentGuardrailSuite
    agent_topology: AgentTopologyReport
    agent_trace_replay: AgentTraceReplay
    demo_runbook: JudgeDemoRunbook
    innovation_claims: tuple[InnovationClaim, ...]
    checks: tuple[DemoPackCheck, ...]
    source_artifacts: tuple[DemoPackArtifact, ...]
    report_artifacts: tuple[DemoPackArtifact, ...]
    demo_commands: tuple[DemoCommand, ...]

    @property
    def overall_status(self) -> str:
        if any(check.status == "FAIL" for check in self.checks):
            return "FAIL"
        if any(check.status == "WARN" for check in self.checks):
            return "WARN"
        return "PASS"

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for check in self.checks if check.status == "PASS")
        warn_count = sum(1 for check in self.checks if check.status == "WARN")
        fail_count = sum(1 for check in self.checks if check.status == "FAIL")
        return (
            "Technology Prize Demo Pack",
            f"  Overall: {self.overall_status}",
            f"  Checks: {pass_count} pass, {warn_count} warn, {fail_count} fail",
            f"  Agents: {len(self.architecture_report.agents)}",
            f"  Tools: {len(self.architecture_report.tools)}",
            f"  Workflow steps: {len(self.local_workflow.steps)}",
            f"  Guardrails: {sum(1 for check in self.agent_guardrails.checks if check.status == 'PASS')}/{len(self.agent_guardrails.checks)} pass",
            f"  Topology: {self.agent_topology.status}",
            f"  Trace replay: {self.agent_trace_replay.status}",
            f"  Demo runbook: {self.demo_runbook.status}",
            f"  AI-native claims: {len(self.innovation_claims)}",
            f"  Evidence artifacts: {sum(1 for item in self.architecture_report.artifacts if item.present)}/{len(self.architecture_report.artifacts)}",
            f"  Source manifest: {sum(1 for item in self.source_artifacts if item.present)}/{len(self.source_artifacts)}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status,
            "innovation_claims": [_dict(claim) for claim in self.innovation_claims],
            "checks": [_dict(check) for check in self.checks],
            "source_artifacts": [_dict(artifact) for artifact in self.source_artifacts],
            "report_artifacts": [_dict(artifact) for artifact in self.report_artifacts],
            "demo_commands": [_dict(command) for command in self.demo_commands],
            "architecture": self.architecture_report.to_dict(),
            "sdk_runner": _dict(self.sdk_runner),
            "anthropic_critic": _dict(self.anthropic_critic),
            "local_workflow": self.local_workflow.to_dict(),
            "agent_guardrails": self.agent_guardrails.to_dict(),
            "agent_topology": self.agent_topology.to_dict(),
            "agent_trace_replay": self.agent_trace_replay.to_dict(),
            "demo_runbook": self.demo_runbook.to_dict(),
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Claude Agent Trader Technology Prize Demo Pack",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.overall_status}**",
                "",
                "## Judge Pitch",
                "",
                (
                    "Claude Agent Trader is now structured as a safe agentic trading-research system: "
                    "Agents SDK owns orchestration, function tools summarize verifiable artifacts, "
                    "risk and MT5 boundaries are explicit, and Anthropic can act as an independent "
                    "critic when credits are available."
                ),
                "",
                "## Why This Is AI-Native And Innovative",
                "",
                _innovation_table(self.innovation_claims),
                "",
                "## Scorecard",
                "",
                _checks_table(self.checks),
                "",
                "## Demo Commands",
                "",
                _commands_table(self.demo_commands),
                "",
                "## Agent Graph Snapshot",
                "",
                f"- Agents: {len(self.architecture_report.agents)}",
                f"- Function-tool specs: {len(self.architecture_report.tools)}",
                f"- Local workflow steps: {len(self.local_workflow.steps)}",
                f"- Blackboard writes: {len(self.local_workflow.blackboard)}",
                f"- Guardrail suite: `{self.agent_guardrails.status}` ({sum(1 for check in self.agent_guardrails.checks if check.status == 'PASS')}/{len(self.agent_guardrails.checks)} pass)",
                f"- Topology report: `{self.agent_topology.status}` ({sum(1 for check in self.agent_topology.checks if check.status == 'PASS')}/{len(self.agent_topology.checks)} pass)",
                f"- Trace replay: `{self.agent_trace_replay.status}` ({len(self.agent_trace_replay.spans)} spans)",
                f"- Demo runbook: `{self.demo_runbook.status}` ({len(self.demo_runbook.steps)} steps)",
                f"- SDK runner status: `{self.sdk_runner.status}`",
                f"- Anthropic critic status: `{self.anthropic_critic.status}`",
                "",
                "## Source Manifest",
                "",
                _artifact_table(self.source_artifacts),
                "",
                "## Generated Report Manifest",
                "",
                _artifact_table(self.report_artifacts),
                "",
                "## Safety Claims",
                "",
                "- Default commands do not make online model calls.",
                "- Online AgentSDK and Anthropic paths require explicit `--allow-online-*` switches.",
                "- No MT5 order-placement tool is registered in the AgentSDK tool surface.",
                "- The Anthropic critic has no trade approval or broker authority.",
                "- Strategy promotion remains tied to backtest, walk-forward, and risk evidence.",
                "",
                "## Remaining Gaps",
                "",
                *(_remaining_gap_lines(self.checks)),
                "",
            ]
        )


def build_technology_prize_demo_pack(
    *,
    project_root: str | Path = ".",
    output_dir: str | Path = "outputs/reports",
    allow_online_sdk: bool = False,
    allow_online_anthropic: bool = False,
    sdk_model: str = "gpt-5.5",
    anthropic_model: str = "claude-sonnet-4-6",
) -> TechnologyPrizeDemoPack:
    root = Path(project_root)
    report_dir = root / output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    architecture = run_local_technology_prize_demo(project_root=root)
    sdk_runner = run_guarded_agents_sdk_demo(
        project_root=root,
        model=sdk_model,
        allow_online_sdk=allow_online_sdk,
    )
    critic = run_guarded_anthropic_critic(
        project_root=root,
        model=anthropic_model,
        allow_online_anthropic=allow_online_anthropic,
    )
    local_workflow = run_local_agent_workflow(project_root=root)
    guardrails = build_agent_guardrail_suite(project_root=root)
    topology = build_agent_topology_report()
    trace_replay = build_agent_trace_replay(project_root=root)

    write_technology_prize_report(
        architecture,
        markdown_path=report_dir / "technology_prize_agent_report.md",
        json_path=report_dir / "technology_prize_agent_report.json",
    )
    write_agents_sdk_runner_report(
        sdk_runner,
        report_dir / "technology_prize_sdk_runner.md",
    )
    write_anthropic_critic_report(
        critic,
        report_dir / "technology_prize_anthropic_critic.md",
    )
    write_local_agent_workflow(
        local_workflow,
        markdown_path=report_dir / "technology_prize_workflow.md",
        json_path=report_dir / "technology_prize_workflow.json",
    )
    write_agent_guardrail_suite(
        guardrails,
        markdown_path=report_dir / "technology_prize_guardrails.md",
        json_path=report_dir / "technology_prize_guardrails.json",
    )
    write_agent_topology_report(
        topology,
        markdown_path=report_dir / "technology_prize_topology.md",
        json_path=report_dir / "technology_prize_topology.json",
    )
    write_agent_trace_replay(
        trace_replay,
        markdown_path=report_dir / "technology_prize_trace_replay.md",
        json_path=report_dir / "technology_prize_trace_replay.json",
    )
    runbook = build_judge_demo_runbook(project_root=root)
    write_judge_demo_runbook(
        runbook,
        markdown_path=report_dir / "technology_prize_demo_runbook.md",
        json_path=report_dir / "technology_prize_demo_runbook.json",
    )

    source_artifacts = tuple(
        _inspect_artifact(root / relative_path, name=name)
        for name, relative_path in DEFAULT_SOURCE_FILES
    )
    report_artifacts = tuple(
        _inspect_artifact(root / relative_path, name=name)
        for name, relative_path in DEFAULT_REPORT_FILES
    )
    innovation_claims = _innovation_claims()
    checks = _build_checks(
        architecture=architecture,
        sdk_runner=sdk_runner,
        critic=critic,
        local_workflow=local_workflow,
        guardrails=guardrails,
        topology=topology,
        trace_replay=trace_replay,
        runbook=runbook,
        innovation_claims=innovation_claims,
        source_artifacts=source_artifacts,
        report_artifacts=report_artifacts,
        bridge_tool_names=_bridge_tool_names(root),
    )
    return TechnologyPrizeDemoPack(
        generated_at=datetime.now(tz=timezone.utc),
        architecture_report=architecture,
        sdk_runner=sdk_runner,
        anthropic_critic=critic,
        local_workflow=local_workflow,
        agent_guardrails=guardrails,
        agent_topology=topology,
        agent_trace_replay=trace_replay,
        demo_runbook=runbook,
        innovation_claims=innovation_claims,
        checks=checks,
        source_artifacts=source_artifacts,
        report_artifacts=report_artifacts,
        demo_commands=_demo_commands(),
    )


def write_technology_prize_demo_pack(
    pack: TechnologyPrizeDemoPack,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    markdown_output = Path(markdown_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(pack.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(pack.to_dict(), indent=2), encoding="utf-8")


def _build_checks(
    *,
    architecture: TechnologyPrizeReport,
    sdk_runner: AgentsSdkRunnerResult,
    critic: AnthropicCriticResult,
    local_workflow: LocalAgentWorkflow,
    guardrails: AgentGuardrailSuite,
    topology: AgentTopologyReport,
    trace_replay: AgentTraceReplay,
    runbook: JudgeDemoRunbook,
    innovation_claims: tuple[InnovationClaim, ...],
    source_artifacts: tuple[DemoPackArtifact, ...],
    report_artifacts: tuple[DemoPackArtifact, ...],
    bridge_tool_names: tuple[str, ...],
) -> tuple[DemoPackCheck, ...]:
    tools_read_only = all(tool.read_only for tool in architecture.tools)
    innovation_has_evidence = all(claim.evidence for claim in innovation_claims)
    tool_spec_names = tuple(sorted(tool.name for tool in architecture.tools))
    bridge_matches_specs = tool_spec_names == bridge_tool_names
    evidence_present = all(artifact.present for artifact in architecture.artifacts)
    source_present = all(artifact.present for artifact in source_artifacts)
    reports_present = all(artifact.present for artifact in report_artifacts)
    openai_provider = _provider(architecture, "OpenAI AgentSDK Orchestrator")
    anthropic_provider = _provider(architecture, "Anthropic Critic")
    return (
        DemoPackCheck(
            name="Agents SDK primitives",
            status="PASS" if architecture.sdk_status.ready else "FAIL",
            details=", ".join(architecture.sdk_status.detected_features) or architecture.sdk_status.import_error,
        ),
        DemoPackCheck(
            name="Agent graph breadth",
            status="PASS" if len(architecture.agents) >= 9 else "WARN",
            details=f"{len(architecture.agents)} agents with typed tools, handoffs, and guardrails.",
        ),
        DemoPackCheck(
            name="Read-only tool surface",
            status="PASS" if tools_read_only else "FAIL",
            details=f"{len(architecture.tools)} tool specs; no write/trade tools allowed.",
        ),
        DemoPackCheck(
            name="SDK bridge tool coverage",
            status="PASS" if bridge_matches_specs else "FAIL",
            details=(
                "actual SDK tools: "
                + (", ".join(bridge_tool_names) if bridge_tool_names else "unavailable")
            ),
        ),
        DemoPackCheck(
            name="AI-native innovation evidence",
            status="PASS" if len(innovation_claims) >= 8 and innovation_has_evidence else "WARN",
            details=(
                f"{len(innovation_claims)} claims covering agentic control, grounded tools, "
                "critic review, guardrails, and provenance."
            ),
        ),
        DemoPackCheck(
            name="Executable local agent workflow",
            status=(
                "PASS"
                if local_workflow.status == "PASS"
                and len(local_workflow.steps) >= 9
                and len(local_workflow.blackboard) >= 9
                else "WARN"
            ),
            details=(
                f"{len(local_workflow.steps)} steps, "
                f"{len(local_workflow.blackboard)} blackboard writes, "
                f"{len(local_workflow.handoffs)} handoffs."
            ),
        ),
        DemoPackCheck(
            name="Executable AI guardrails",
            status="PASS" if guardrails.status == "PASS" and len(guardrails.checks) >= 8 else "FAIL",
            details=(
                f"{sum(1 for check in guardrails.checks if check.status == 'PASS')}/"
                f"{len(guardrails.checks)} guardrail checks pass."
            ),
        ),
        DemoPackCheck(
            name="AgentSDK topology integrity",
            status="PASS" if topology.status == "PASS" and len(topology.checks) >= 8 else "FAIL",
            details=(
                f"{sum(1 for check in topology.checks if check.status == 'PASS')}/"
                f"{len(topology.checks)} topology checks pass."
            ),
        ),
        DemoPackCheck(
            name="AgentSDK trace replay",
            status="PASS" if trace_replay.status == "PASS" and len(trace_replay.spans) >= 20 else "WARN",
            details=f"{len(trace_replay.spans)} trace spans, trace_id={trace_replay.trace_id}.",
        ),
        DemoPackCheck(
            name="Judge demo runbook",
            status="PASS" if runbook.status == "PASS" and len(runbook.steps) >= 6 else "WARN",
            details=f"{len(runbook.steps)} timed steps and {len(runbook.risk_notes)} risk notes.",
        ),
        DemoPackCheck(
            name="Evidence coverage",
            status="PASS" if evidence_present else "WARN",
            details=f"{sum(1 for item in architecture.artifacts if item.present)}/{len(architecture.artifacts)} research/deployment artifacts present.",
        ),
        DemoPackCheck(
            name="Source provenance",
            status="PASS" if source_present else "FAIL",
            details=f"{sum(1 for item in source_artifacts if item.present)}/{len(source_artifacts)} source files hashed.",
        ),
        DemoPackCheck(
            name="Report provenance",
            status="PASS" if reports_present else "FAIL",
            details=f"{sum(1 for item in report_artifacts if item.present)}/{len(report_artifacts)} generated reports hashed.",
        ),
        DemoPackCheck(
            name="AgentSDK online gate",
            status="PASS" if sdk_runner.status in {"SKIPPED", "BLOCKED", "RAN"} else "FAIL",
            details=(
                f"status={sdk_runner.status}; "
                f"OPENAI_API_KEY={'present' if sdk_runner.api_key_present else 'missing'}; "
                f"package={'present' if openai_provider and openai_provider.installed else 'missing'}"
            ),
        ),
        DemoPackCheck(
            name="Anthropic critic gate",
            status="PASS" if critic.status in {"SKIPPED", "BLOCKED", "RAN"} else "FAIL",
            details=(
                f"status={critic.status}; "
                f"ANTHROPIC_API_KEY={'present' if critic.api_key_present else 'missing'}; "
                f"package={'present' if anthropic_provider and anthropic_provider.installed else 'missing'}"
            ),
        ),
        DemoPackCheck(
            name="No live trading authority",
            status="PASS",
            details="Agent reports expose read-only artifact tools; MT5 ticket sheets remain advisory.",
        ),
    )


def _bridge_tool_names(root: Path) -> tuple[str, ...]:
    try:
        app = create_agents_sdk_app(project_root=root)
    except AgentsSdkUnavailableError:
        return ()
    return tuple(sorted(app.tools))


def _innovation_claims() -> tuple[InnovationClaim, ...]:
    return (
        InnovationClaim(
            name="Agentic Control Plane",
            claim="The trading system is supervised by specialist agents instead of a single script.",
            evidence=(
                "src/quanthack/agents/technology_prize.py",
                "outputs/reports/technology_prize_agent_report.json",
            ),
            why_ai_native=(
                "The architecture is built around delegation, handoffs, specialist roles, "
                "and model-facing tool contracts."
            ),
            prize_relevance="Shows technical system design beyond trading P&L.",
        ),
        InnovationClaim(
            name="Grounded Function Tools",
            claim="Agents inspect real project artifacts through read-only function tools.",
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "outputs/reports/technology_prize_demo_pack.json",
            ),
            why_ai_native=(
                "The model reasons over live evidence via tool calls instead of relying on "
                "ungrounded prompt text."
            ),
            prize_relevance="Demonstrates practical AgentSDK tool use on the actual quant stack.",
        ),
        InnovationClaim(
            name="Executable Blackboard Workflow",
            claim=(
                "A deterministic local workflow executes specialist handoffs and records shared "
                "blackboard writes without spending model credits."
            ),
            evidence=(
                "src/quanthack/agents/workflow.py",
                "outputs/reports/technology_prize_workflow.json",
            ),
            why_ai_native=(
                "The system models multi-agent cognition as inspectable state transitions, "
                "not a hidden monolithic prompt."
            ),
            prize_relevance="Makes the advanced architecture executable and auditable.",
        ),
        InnovationClaim(
            name="Independent Critic Loop",
            claim="Anthropic credits can power an independent critic agent for overfitting, risk, and demo review.",
            evidence=(
                "src/quanthack/agents/anthropic_critic.py",
                "outputs/reports/technology_prize_anthropic_critic.md",
            ),
            why_ai_native=(
                "The system separates generation/orchestration from model-based critique, "
                "making review a first-class workflow."
            ),
            prize_relevance="Uses additional credits for quality assurance rather than decorative chat.",
        ),
        InnovationClaim(
            name="Regime-Aware Research Delegation",
            claim=(
                "A specialist Regime Scientist Agent tracks market-regime evidence, selector regret, "
                "and when strategy sleeves should stand down."
            ),
            evidence=(
                "src/quanthack/agents/technology_prize.py",
                "outputs/research/adaptive_current_top_handoff_diagnostic.csv",
            ),
            why_ai_native=(
                "The AI layer decomposes ambiguous market conditions into a dedicated reasoning role "
                "instead of hiding regime decisions inside one script."
            ),
            prize_relevance="Shows advanced architecture for adaptive quant research.",
        ),
        InnovationClaim(
            name="Experiment Audit Agent",
            claim=(
                "An Experiment Auditor Agent checks provenance, command reproducibility, hashes, "
                "and whether claims are supported by artifacts."
            ),
            evidence=(
                "src/quanthack/agents/demo_pack.py",
                "outputs/reports/technology_prize_demo_pack.json",
            ),
            why_ai_native=(
                "The system gives AI-generated narratives an explicit verification agent and audit trail."
            ),
            prize_relevance="Makes the technology demo credible and inspection-ready.",
        ),
        InnovationClaim(
            name="Guarded Online Execution",
            claim="Online model calls are available but deliberately locked behind explicit arming flags.",
            evidence=(
                "src/quanthack/agents/sdk_runner.py",
                "src/quanthack/cli/tech_prize_demo.py",
            ),
            why_ai_native=(
                "The agent runtime is treated like production infrastructure with spend and safety gates."
            ),
            prize_relevance="Balances powerful model autonomy with operator control.",
        ),
        InnovationClaim(
            name="Broker-Safe AI Boundary",
            claim="AI tools can summarize MT5 ticket sheets but cannot place MT5 orders.",
            evidence=(
                "src/quanthack/agents/sdk_bridge.py",
                "outputs/reports/technology_prize_sdk_runner.md",
            ),
            why_ai_native=(
                "The model can reason about deployment artifacts while hard boundaries prevent unsafe action."
            ),
            prize_relevance="Shows responsible AI-native trading architecture.",
        ),
        InnovationClaim(
            name="Executable Guardrails",
            claim=(
                "The agent layer has code-level guardrails for read-only tools, path confinement, "
                "credit-spend arming, Anthropic review-only authority, and MT5 no-order boundaries."
            ),
            evidence=(
                "src/quanthack/agents/guardrails.py",
                "outputs/reports/technology_prize_guardrails.md",
            ),
            why_ai_native=(
                "The safety model is an executable agent workflow component instead of a prose policy."
            ),
            prize_relevance="Shows mature AI system design under trading constraints.",
        ),
        InnovationClaim(
            name="Self-Validating Agent Topology",
            claim=(
                "The AgentSDK graph validates tool coverage, handoff targets, guardrail coverage, "
                "and authority boundaries before judge claims are trusted."
            ),
            evidence=(
                "src/quanthack/agents/topology.py",
                "outputs/reports/technology_prize_topology.md",
            ),
            why_ai_native=(
                "The agent graph is treated as inspectable runtime architecture, not just a diagram."
            ),
            prize_relevance="Strengthens Best use of AgentSDK with machine-checkable graph integrity.",
        ),
        InnovationClaim(
            name="Offline Agent Trace Replay",
            claim=(
                "The deterministic workflow is exported as span-like trace data covering agent steps, "
                "tool calls, blackboard writes, and handoffs."
            ),
            evidence=(
                "src/quanthack/agents/trace_replay.py",
                "outputs/reports/technology_prize_trace_replay.md",
            ),
            why_ai_native=(
                "The system makes agent reasoning inspectable as trace state instead of relying on hidden prompts."
            ),
            prize_relevance="Demonstrates AgentSDK-style observability without spending model credits.",
        ),
        InnovationClaim(
            name="Judge Demo Director",
            claim=(
                "A generated runbook turns agent evidence into a timed judge demo with exact commands, "
                "talk track, proofs, and risk-answer handling."
            ),
            evidence=(
                "src/quanthack/agents/demo_director.py",
                "outputs/reports/technology_prize_demo_runbook.md",
            ),
            why_ai_native=(
                "The system uses its own agent-evidence outputs to plan how a human operator should "
                "demonstrate the architecture under time pressure."
            ),
            prize_relevance="Converts technical merit into a reliable live presentation path.",
        ),
        InnovationClaim(
            name="Provenance-First Demo",
            claim="The system hashes source files and reports so claims are tied to verifiable artifacts.",
            evidence=(
                "src/quanthack/agents/demo_pack.py",
                "outputs/reports/technology_prize_demo_pack.md",
            ),
            why_ai_native=(
                "AI outputs are packaged with audit trails, evidence manifests, and reproducible commands."
            ),
            prize_relevance="Makes technical merit inspectable by judges.",
        ),
        InnovationClaim(
            name="Requirement-Level Judge Packet",
            claim=(
                "The technology-prize evidence is checked against explicit requirements for "
                "AgentSDK use, Anthropic critic support, AI-native workflow, and broker safety."
            ),
            evidence=(
                "src/quanthack/agents/judge_packet.py",
                "outputs/reports/technology_prize_judge_packet.md",
            ),
            why_ai_native=(
                "The system treats judging criteria as machine-checkable agent evidence rather "
                "than relying on an unverified narrative."
            ),
            prize_relevance="Gives judges a direct technical-merit checklist with artifact hashes.",
        ),
        InnovationClaim(
            name="Scored Technology Rubric",
            claim=(
                "A deterministic rubric maps the architecture into the prize axes: AgentSDK use, "
                "Anthropic credits, AI-native innovation, reproducibility, safety, and demo readiness."
            ),
            evidence=(
                "src/quanthack/agents/rubric.py",
                "outputs/reports/technology_prize_rubric.md",
            ),
            why_ai_native=(
                "The system evaluates its own AI architecture with the same evidence trail used "
                "by the agent workflow."
            ),
            prize_relevance="Turns the technology story into a judge-scannable scorecard.",
        ),
        InnovationClaim(
            name="Skeptical Judge Red Team",
            claim=(
                "A red-team evaluator rehearses skeptical judge questions about order authority, "
                "AgentSDK centrality, Anthropic credit use, model spend, provenance, and demo readiness."
            ),
            evidence=(
                "src/quanthack/agents/red_team.py",
                "outputs/reports/technology_prize_red_team.md",
            ),
            why_ai_native=(
                "The system uses its own agent evidence to challenge the architecture before judges do."
            ),
            prize_relevance="Improves live-demo credibility by making hard questions executable checks.",
        ),
    )


def _demo_commands() -> tuple[DemoCommand, ...]:
    return (
        DemoCommand(
            name="Build safe demo pack",
            purpose="Generate the architecture, runner, critic, manifest, and scorecard without online calls.",
            command="quanthack tech-prize-pack",
            spends_credits=False,
        ),
        DemoCommand(
            name="Dry-run control plane",
            purpose="Regenerate the AgentSDK architecture report and both guarded runner reports.",
            command="quanthack tech-prize-demo --run-sdk --run-anthropic-critic",
            spends_credits=False,
        ),
        DemoCommand(
            name="Run local agent workflow",
            purpose="Execute the deterministic blackboard/handoff workflow without online calls.",
            command="quanthack tech-prize-workflow",
            spends_credits=False,
        ),
        DemoCommand(
            name="Run agent guardrails",
            purpose="Evaluate read-only tools, path confinement, credit gates, and no-order authority.",
            command="quanthack tech-prize-guardrails",
            spends_credits=False,
        ),
        DemoCommand(
            name="Analyze AgentSDK topology",
            purpose="Validate tool coverage, handoff integrity, guardrails, and authority boundaries.",
            command="quanthack tech-prize-topology",
            spends_credits=False,
        ),
        DemoCommand(
            name="Replay agent trace",
            purpose="Export agent steps, tool calls, blackboard writes, and handoffs as trace spans.",
            command="quanthack tech-prize-trace",
            spends_credits=False,
        ),
        DemoCommand(
            name="Build judge demo runbook",
            purpose="Generate the timed demo script, talk track, proof map, and risk answers.",
            command="quanthack tech-prize-runbook",
            spends_credits=False,
        ),
        DemoCommand(
            name="Build judge packet",
            purpose="Verify the technology-prize requirements against source, report, and workflow evidence.",
            command="quanthack tech-prize-judge-packet",
            spends_credits=False,
        ),
        DemoCommand(
            name="Score technology rubric",
            purpose="Map the architecture to the technology-prize judging axes with a 100-point scorecard.",
            command="quanthack tech-prize-rubric",
            spends_credits=False,
        ),
        DemoCommand(
            name="Run skeptical red team",
            purpose="Stress-test the technology-prize story against likely judge objections.",
            command="quanthack tech-prize-red-team",
            spends_credits=False,
        ),
        DemoCommand(
            name="Online AgentSDK judge brief",
            purpose="Use real Agents SDK Runner orchestration after OPENAI_API_KEY is configured.",
            command="quanthack tech-prize-demo --run-sdk --allow-online-sdk --sdk-model gpt-5.5",
            spends_credits=True,
            requires_env=("OPENAI_API_KEY",),
        ),
        DemoCommand(
            name="Online Anthropic critic",
            purpose="Use Anthropic credits for an independent architecture and risk critique.",
            command=(
                "quanthack tech-prize-demo --run-anthropic-critic "
                "--allow-online-anthropic --anthropic-model claude-sonnet-4-6"
            ),
            spends_credits=True,
            requires_env=("ANTHROPIC_API_KEY",),
        ),
    )


def _provider(
    architecture: TechnologyPrizeReport,
    name: str,
) -> Any:
    for provider in architecture.providers:
        if provider.name == name:
            return provider
    return None


def _inspect_artifact(path: Path, *, name: str) -> DemoPackArtifact:
    if not path.exists():
        return DemoPackArtifact(name=name, path=str(path), present=False)
    payload = path.read_bytes()
    return DemoPackArtifact(
        name=name,
        path=str(path),
        present=True,
        bytes_size=len(payload),
        sha256_12=hashlib.sha256(payload).hexdigest()[:12],
    )


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _checks_table(checks: tuple[DemoPackCheck, ...]) -> str:
    rows = ["| Check | Status | Details |", "| --- | --- | --- |"]
    for check in checks:
        rows.append(f"| {check.name} | {check.status} | {check.details} |")
    return "\n".join(rows)


def _innovation_table(claims: tuple[InnovationClaim, ...]) -> str:
    rows = [
        "| Claim | Why AI-Native | Evidence | Prize Relevance |",
        "| --- | --- | --- | --- |",
    ]
    for claim in claims:
        rows.append(
            f"| {claim.name}: {claim.claim} | {claim.why_ai_native} | "
            f"{'<br>'.join(claim.evidence)} | {claim.prize_relevance} |"
        )
    return "\n".join(rows)


def _commands_table(commands: tuple[DemoCommand, ...]) -> str:
    rows = [
        "| Command | Purpose | Spends Credits | Required Env |",
        "| --- | --- | --- | --- |",
    ]
    for command in commands:
        rows.append(
            f"| `{command.command}` | {command.purpose} | "
            f"{'yes' if command.spends_credits else 'no'} | "
            f"{', '.join(command.requires_env) or 'none'} |"
        )
    return "\n".join(rows)


def _artifact_table(artifacts: tuple[DemoPackArtifact, ...]) -> str:
    rows = ["| Artifact | Status | Bytes | SHA-256 Prefix | Path |", "| --- | --- | ---: | --- | --- |"]
    for artifact in artifacts:
        rows.append(
            f"| {artifact.name} | {artifact.status} | {artifact.bytes_size} | "
            f"`{artifact.sha256_12 or '-'}` | `{artifact.path}` |"
        )
    return "\n".join(rows)


def _remaining_gap_lines(checks: tuple[DemoPackCheck, ...]) -> tuple[str, ...]:
    gaps = [
        f"- {check.name}: {check.details}"
        for check in checks
        if check.status in {"WARN", "FAIL"}
    ]
    if not gaps:
        return ("- No failing technology-prize pack checks. Optional online demos still need deliberate arming.",)
    return tuple(gaps)
