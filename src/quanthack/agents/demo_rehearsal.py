from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from quanthack.agents.demo_director import build_judge_demo_runbook, write_judge_demo_runbook
from quanthack.agents.demo_pack import build_technology_prize_demo_pack, write_technology_prize_demo_pack
from quanthack.agents.guardrails import build_agent_guardrail_suite, write_agent_guardrail_suite
from quanthack.agents.judge_packet import build_technology_prize_judge_packet, write_technology_prize_judge_packet
from quanthack.agents.red_team import build_technology_prize_red_team_report, write_technology_prize_red_team_report
from quanthack.agents.rubric import build_technology_prize_rubric, write_technology_prize_rubric
from quanthack.agents.topology import build_agent_topology_report, write_agent_topology_report
from quanthack.agents.trace_replay import build_agent_trace_replay, write_agent_trace_replay
from quanthack.reporting.technology_prize_dashboard import build_technology_prize_dashboard


@dataclass(frozen=True)
class DemoRehearsalStep:
    label: str
    command: str
    purpose: str
    status: str
    duration_ms: int
    artifacts: tuple[str, ...]
    evidence: str
    spends_credits: bool = False


@dataclass(frozen=True)
class TechnologyPrizeDemoRehearsal:
    generated_at: datetime
    status: str
    total_duration_ms: int
    steps: tuple[DemoRehearsalStep, ...]
    offline_only: bool

    def summary_lines(self) -> tuple[str, ...]:
        pass_count = sum(1 for step in self.steps if step.status == "PASS")
        warn_count = sum(1 for step in self.steps if step.status == "WARN")
        fail_count = sum(1 for step in self.steps if step.status == "FAIL")
        return (
            "Technology Prize Demo Rehearsal",
            f"  Overall: {self.status}",
            f"  Steps: {pass_count} pass, {warn_count} warn, {fail_count} fail",
            f"  Duration: {self.total_duration_ms} ms",
            f"  Offline only: {'yes' if self.offline_only else 'no'}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
            "offline_only": self.offline_only,
            "steps": [_dict(step) for step in self.steps],
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Technology Prize Demo Rehearsal",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                f"Total duration: **{self.total_duration_ms} ms**",
                f"Offline only: **{'yes' if self.offline_only else 'no'}**",
                "",
                "## Why This Matters",
                "",
                (
                    "This rehearses the safe judge-demo path without online model calls. It verifies "
                    "that each command-equivalent step can regenerate its expected artifacts before "
                    "a live presentation."
                ),
                "",
                "## Rehearsed Steps",
                "",
                _steps_table(self.steps),
                "",
                "## Recommended Live Flow",
                "",
                "1. Run `quanthack tech-prize-rehearse` to verify the safe demo flow.",
                "2. Run `quanthack tech-prize-submit` before judging.",
                "3. Open `outputs/reports/technology_prize_dashboard.html`.",
                "4. Run `quanthack tech-prize-rubric` and `quanthack tech-prize-red-team` if judges ask for proof.",
                "5. Run `quanthack tech-prize-topology`, `quanthack tech-prize-trace`, and `quanthack tech-prize-guardrails` for executable detail.",
                "6. Keep online OpenAI/Anthropic demos disabled unless explicitly armed.",
                "",
            ]
        )


def build_technology_prize_demo_rehearsal(
    *,
    project_root: str | Path = ".",
    output_dir: str | Path = "outputs/reports",
) -> TechnologyPrizeDemoRehearsal:
    root = Path(project_root)
    report_dir = root / output_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    steps = (
        _run_step(
            label="Build demo pack and dashboard",
            command="quanthack tech-prize-pack && quanthack tech-prize-dashboard",
            purpose="Regenerate the main scorecard and browser-first dashboard.",
            artifacts=(
                "outputs/reports/technology_prize_demo_pack.md",
                "outputs/reports/technology_prize_demo_pack.json",
                "outputs/reports/technology_prize_dashboard.html",
            ),
            action=lambda: _build_pack_and_dashboard(root=root, report_dir=report_dir, output_dir=output_dir),
        ),
        _run_step(
            label="Build runbook",
            command="quanthack tech-prize-runbook",
            purpose="Regenerate the timed judge talk track.",
            artifacts=(
                "outputs/reports/technology_prize_demo_runbook.md",
                "outputs/reports/technology_prize_demo_runbook.json",
            ),
            action=lambda: _build_runbook(root=root, report_dir=report_dir),
        ),
        _run_step(
            label="Analyze topology",
            command="quanthack tech-prize-topology",
            purpose="Validate the AgentSDK graph, handoffs, and tool authority.",
            artifacts=(
                "outputs/reports/technology_prize_topology.md",
                "outputs/reports/technology_prize_topology.json",
            ),
            action=lambda: _build_topology(report_dir=report_dir),
        ),
        _run_step(
            label="Replay trace",
            command="quanthack tech-prize-trace",
            purpose="Export agent steps, tool calls, blackboard writes, and handoffs as spans.",
            artifacts=(
                "outputs/reports/technology_prize_trace_replay.md",
                "outputs/reports/technology_prize_trace_replay.json",
            ),
            action=lambda: _build_trace(root=root, report_dir=report_dir),
        ),
        _run_step(
            label="Run guardrails",
            command="quanthack tech-prize-guardrails",
            purpose="Verify read-only tools, path confinement, credit gates, and MT5 no-order authority.",
            artifacts=(
                "outputs/reports/technology_prize_guardrails.md",
                "outputs/reports/technology_prize_guardrails.json",
            ),
            action=lambda: _build_guardrails(root=root, report_dir=report_dir),
        ),
        _run_step(
            label="Build judge packet",
            command="quanthack tech-prize-judge-packet",
            purpose="Verify the prize requirements against source and report evidence.",
            artifacts=(
                "outputs/reports/technology_prize_judge_packet.md",
                "outputs/reports/technology_prize_judge_packet.json",
            ),
            action=lambda: _build_judge_packet(root=root, report_dir=report_dir),
        ),
        _run_step(
            label="Score rubric",
            command="quanthack tech-prize-rubric",
            purpose="Score the project against the technology-prize judging axes.",
            artifacts=(
                "outputs/reports/technology_prize_rubric.md",
                "outputs/reports/technology_prize_rubric.json",
            ),
            action=lambda: _build_rubric(root=root, report_dir=report_dir),
        ),
        _run_step(
            label="Run red team",
            command="quanthack tech-prize-red-team",
            purpose="Answer skeptical judge questions with executable evidence checks.",
            artifacts=(
                "outputs/reports/technology_prize_red_team.md",
                "outputs/reports/technology_prize_red_team.json",
            ),
            action=lambda: _build_red_team(root=root, report_dir=report_dir),
        ),
    )
    total_duration_ms = round((perf_counter() - started) * 1000)
    return TechnologyPrizeDemoRehearsal(
        generated_at=datetime.now(tz=timezone.utc),
        status=_overall_status(steps),
        total_duration_ms=total_duration_ms,
        steps=steps,
        offline_only=not any(step.spends_credits for step in steps),
    )


def write_technology_prize_demo_rehearsal(
    rehearsal: TechnologyPrizeDemoRehearsal,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rehearsal.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(rehearsal.to_dict(), indent=2), encoding="utf-8")


def _run_step(
    *,
    label: str,
    command: str,
    purpose: str,
    artifacts: tuple[str, ...],
    action: Callable[[], tuple[str, str]],
) -> DemoRehearsalStep:
    started = perf_counter()
    try:
        status, evidence = action()
    except Exception as exc:  # pragma: no cover - defensive report path.
        status = "FAIL"
        evidence = f"{type(exc).__name__}: {exc}"
    return DemoRehearsalStep(
        label=label,
        command=command,
        purpose=purpose,
        status=status,
        duration_ms=round((perf_counter() - started) * 1000),
        artifacts=artifacts,
        evidence=evidence,
    )


def _build_pack_and_dashboard(*, root: Path, report_dir: Path, output_dir: str | Path) -> tuple[str, str]:
    pack = build_technology_prize_demo_pack(project_root=root, output_dir=output_dir)
    write_technology_prize_demo_pack(
        pack,
        markdown_path=report_dir / "technology_prize_demo_pack.md",
        json_path=report_dir / "technology_prize_demo_pack.json",
    )
    dashboard = build_technology_prize_dashboard(pack)
    (report_dir / "technology_prize_dashboard.html").write_text(dashboard.html, encoding="utf-8")
    return pack.overall_status, f"demo_pack={pack.overall_status}; claims={len(pack.innovation_claims)}"


def _build_runbook(*, root: Path, report_dir: Path) -> tuple[str, str]:
    runbook = build_judge_demo_runbook(project_root=root)
    write_judge_demo_runbook(
        runbook,
        markdown_path=report_dir / "technology_prize_demo_runbook.md",
        json_path=report_dir / "technology_prize_demo_runbook.json",
    )
    return runbook.status, f"steps={len(runbook.steps)}; risk_notes={len(runbook.risk_notes)}"


def _build_topology(*, report_dir: Path) -> tuple[str, str]:
    topology = build_agent_topology_report()
    write_agent_topology_report(
        topology,
        markdown_path=report_dir / "technology_prize_topology.md",
        json_path=report_dir / "technology_prize_topology.json",
    )
    return topology.status, f"agents={len(topology.agents)}; tools={len(topology.tools)}; handoffs={len(topology.handoffs)}"


def _build_trace(*, root: Path, report_dir: Path) -> tuple[str, str]:
    trace = build_agent_trace_replay(project_root=root)
    write_agent_trace_replay(
        trace,
        markdown_path=report_dir / "technology_prize_trace_replay.md",
        json_path=report_dir / "technology_prize_trace_replay.json",
    )
    return trace.status, f"trace_id={trace.trace_id}; spans={len(trace.spans)}"


def _build_guardrails(*, root: Path, report_dir: Path) -> tuple[str, str]:
    guardrails = build_agent_guardrail_suite(project_root=root)
    write_agent_guardrail_suite(
        guardrails,
        markdown_path=report_dir / "technology_prize_guardrails.md",
        json_path=report_dir / "technology_prize_guardrails.json",
    )
    return guardrails.status, f"checks={sum(1 for item in guardrails.checks if item.status == 'PASS')}/{len(guardrails.checks)}"


def _build_judge_packet(*, root: Path, report_dir: Path) -> tuple[str, str]:
    packet = build_technology_prize_judge_packet(project_root=root)
    write_technology_prize_judge_packet(
        packet,
        markdown_path=report_dir / "technology_prize_judge_packet.md",
        json_path=report_dir / "technology_prize_judge_packet.json",
    )
    return packet.status, f"requirements={sum(1 for item in packet.requirements if item.status == 'PASS')}/{len(packet.requirements)}"


def _build_rubric(*, root: Path, report_dir: Path) -> tuple[str, str]:
    rubric = build_technology_prize_rubric(project_root=root)
    write_technology_prize_rubric(
        rubric,
        markdown_path=report_dir / "technology_prize_rubric.md",
        json_path=report_dir / "technology_prize_rubric.json",
    )
    return rubric.status, f"score={rubric.total_score}/{rubric.max_score}"


def _build_red_team(*, root: Path, report_dir: Path) -> tuple[str, str]:
    red_team = build_technology_prize_red_team_report(project_root=root)
    write_technology_prize_red_team_report(
        red_team,
        markdown_path=report_dir / "technology_prize_red_team.md",
        json_path=report_dir / "technology_prize_red_team.json",
    )
    return red_team.status, f"challenges={sum(1 for item in red_team.challenges if item.status == 'PASS')}/{len(red_team.challenges)}"


def _overall_status(steps: tuple[DemoRehearsalStep, ...]) -> str:
    if any(step.status == "FAIL" for step in steps):
        return "FAIL"
    if any(step.status == "WARN" for step in steps):
        return "WARN"
    return "PASS"


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.__dict__.items()
    }


def _steps_table(steps: tuple[DemoRehearsalStep, ...]) -> str:
    rows = [
        "| Step | Command | Status | Duration | Artifacts | Evidence |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for step in steps:
        rows.append(
            f"| {step.label} | `{step.command}` | {step.status} | {step.duration_ms} ms | "
            f"{'<br>'.join(step.artifacts)} | {step.evidence} |"
        )
    return "\n".join(rows)
