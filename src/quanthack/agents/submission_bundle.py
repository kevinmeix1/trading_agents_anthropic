from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.demo_director import build_judge_demo_runbook, write_judge_demo_runbook
from quanthack.agents.demo_pack import build_technology_prize_demo_pack, write_technology_prize_demo_pack
from quanthack.agents.demo_rehearsal import (
    build_technology_prize_demo_rehearsal,
    write_technology_prize_demo_rehearsal,
)
from quanthack.agents.judge_packet import build_technology_prize_judge_packet, write_technology_prize_judge_packet
from quanthack.agents.red_team import (
    build_technology_prize_red_team_report,
    write_technology_prize_red_team_report,
)
from quanthack.agents.rubric import build_technology_prize_rubric, write_technology_prize_rubric
from quanthack.reporting.technology_prize_dashboard import build_technology_prize_dashboard


SUBMISSION_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("Open first: dashboard", "outputs/reports/technology_prize_dashboard.html"),
    ("Judge demo runbook", "outputs/reports/technology_prize_demo_runbook.md"),
    ("Judge demo rehearsal", "outputs/reports/technology_prize_demo_rehearsal.md"),
    ("Judge demo rehearsal JSON", "outputs/reports/technology_prize_demo_rehearsal.json"),
    ("Judge packet", "outputs/reports/technology_prize_judge_packet.md"),
    ("Judge packet JSON", "outputs/reports/technology_prize_judge_packet.json"),
    ("Technology prize rubric", "outputs/reports/technology_prize_rubric.md"),
    ("Technology prize rubric JSON", "outputs/reports/technology_prize_rubric.json"),
    ("Technology prize red team", "outputs/reports/technology_prize_red_team.md"),
    ("Technology prize red team JSON", "outputs/reports/technology_prize_red_team.json"),
    ("Demo pack", "outputs/reports/technology_prize_demo_pack.md"),
    ("Demo pack JSON", "outputs/reports/technology_prize_demo_pack.json"),
    ("AgentSDK topology", "outputs/reports/technology_prize_topology.md"),
    ("AgentSDK trace replay", "outputs/reports/technology_prize_trace_replay.md"),
    ("Agent guardrails", "outputs/reports/technology_prize_guardrails.md"),
    ("Local workflow", "outputs/reports/technology_prize_workflow.md"),
    ("Agent architecture report", "outputs/reports/technology_prize_agent_report.md"),
    ("AgentSDK runner report", "outputs/reports/technology_prize_sdk_runner.md"),
    ("Anthropic critic report", "outputs/reports/technology_prize_anthropic_critic.md"),
)


@dataclass(frozen=True)
class SubmissionArtifact:
    name: str
    path: str
    present: bool
    bytes_size: int = 0
    sha256_12: str = ""

    @property
    def status(self) -> str:
        return "OK" if self.present else "MISSING"


@dataclass(frozen=True)
class SubmissionCommand:
    label: str
    command: str
    purpose: str
    spends_credits: bool = False


@dataclass(frozen=True)
class TechnologyPrizeSubmissionBundle:
    generated_at: datetime
    status: str
    demo_pack_status: str
    judge_packet_status: str
    rubric_status: str
    red_team_status: str
    rehearsal_status: str
    runbook_status: str
    artifact_manifest: tuple[SubmissionArtifact, ...]
    commands: tuple[SubmissionCommand, ...]

    def summary_lines(self) -> tuple[str, ...]:
        present = sum(1 for item in self.artifact_manifest if item.present)
        return (
            "Technology Prize Submission Bundle",
            f"  Overall: {self.status}",
            f"  Demo pack: {self.demo_pack_status}",
            f"  Judge packet: {self.judge_packet_status}",
            f"  Rubric: {self.rubric_status}",
            f"  Red team: {self.red_team_status}",
            f"  Rehearsal: {self.rehearsal_status}",
            f"  Runbook: {self.runbook_status}",
            f"  Artifacts: {present}/{len(self.artifact_manifest)} present",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "demo_pack_status": self.demo_pack_status,
            "judge_packet_status": self.judge_packet_status,
            "rubric_status": self.rubric_status,
            "red_team_status": self.red_team_status,
            "rehearsal_status": self.rehearsal_status,
            "runbook_status": self.runbook_status,
            "artifact_manifest": [_dict(item) | {"status": item.status} for item in self.artifact_manifest],
            "commands": [_dict(command) for command in self.commands],
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# QuanHack Technology Prize Submission Bundle",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Overall status: **{self.status}**",
                "",
                "## Open First",
                "",
                "1. `outputs/reports/technology_prize_dashboard.html`",
                "2. `outputs/reports/technology_prize_demo_runbook.md`",
                "3. `outputs/reports/technology_prize_demo_rehearsal.md`",
                "4. `outputs/reports/technology_prize_judge_packet.md`",
                "5. `outputs/reports/technology_prize_rubric.md`",
                "6. `outputs/reports/technology_prize_red_team.md`",
                "",
                "## Demo Commands",
                "",
                _commands_table(self.commands),
                "",
                "## Artifact Manifest",
                "",
                _artifact_table(self.artifact_manifest),
                "",
                "## Safety Reminder",
                "",
                (
                    "Default commands do not spend OpenAI or Anthropic credits. Online model calls "
                    "require explicit `--allow-online-sdk` or `--allow-online-anthropic` flags."
                ),
                "",
            ]
        )


def build_technology_prize_submission_bundle(
    *,
    project_root: str | Path = ".",
    output_dir: str | Path = "outputs/reports",
) -> TechnologyPrizeSubmissionBundle:
    root = Path(project_root)
    report_dir = root / output_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    pack = build_technology_prize_demo_pack(project_root=root, output_dir=output_dir)
    write_technology_prize_demo_pack(
        pack,
        markdown_path=report_dir / "technology_prize_demo_pack.md",
        json_path=report_dir / "technology_prize_demo_pack.json",
    )

    dashboard = build_technology_prize_dashboard(pack)
    (report_dir / "technology_prize_dashboard.html").write_text(dashboard.html, encoding="utf-8")

    runbook = build_judge_demo_runbook(project_root=root)
    write_judge_demo_runbook(
        runbook,
        markdown_path=report_dir / "technology_prize_demo_runbook.md",
        json_path=report_dir / "technology_prize_demo_runbook.json",
    )

    judge_packet = build_technology_prize_judge_packet(project_root=root)
    write_technology_prize_judge_packet(
        judge_packet,
        markdown_path=report_dir / "technology_prize_judge_packet.md",
        json_path=report_dir / "technology_prize_judge_packet.json",
    )

    rubric = build_technology_prize_rubric(project_root=root)
    write_technology_prize_rubric(
        rubric,
        markdown_path=report_dir / "technology_prize_rubric.md",
        json_path=report_dir / "technology_prize_rubric.json",
    )

    red_team = build_technology_prize_red_team_report(project_root=root)
    write_technology_prize_red_team_report(
        red_team,
        markdown_path=report_dir / "technology_prize_red_team.md",
        json_path=report_dir / "technology_prize_red_team.json",
    )

    rehearsal = build_technology_prize_demo_rehearsal(project_root=root, output_dir=output_dir)
    write_technology_prize_demo_rehearsal(
        rehearsal,
        markdown_path=report_dir / "technology_prize_demo_rehearsal.md",
        json_path=report_dir / "technology_prize_demo_rehearsal.json",
    )

    artifacts = tuple(
        _inspect_artifact(name=name, path=root / relative_path)
        for name, relative_path in SUBMISSION_ARTIFACTS
    )
    status = _overall_status(
        demo_pack_status=pack.overall_status,
        judge_packet_status=judge_packet.status,
        rubric_status=rubric.status,
        red_team_status=red_team.status,
        rehearsal_status=rehearsal.status,
        runbook_status=runbook.status,
        artifacts=artifacts,
    )
    return TechnologyPrizeSubmissionBundle(
        generated_at=datetime.now(tz=timezone.utc),
        status=status,
        demo_pack_status=pack.overall_status,
        judge_packet_status=judge_packet.status,
        rubric_status=rubric.status,
        red_team_status=red_team.status,
        rehearsal_status=rehearsal.status,
        runbook_status=runbook.status,
        artifact_manifest=artifacts,
        commands=_commands(),
    )


def write_technology_prize_submission_bundle(
    bundle: TechnologyPrizeSubmissionBundle,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundle.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")


def _overall_status(
    *,
    demo_pack_status: str,
    judge_packet_status: str,
    rubric_status: str,
    red_team_status: str,
    rehearsal_status: str,
    runbook_status: str,
    artifacts: tuple[SubmissionArtifact, ...],
) -> str:
    statuses = (
        demo_pack_status,
        judge_packet_status,
        rubric_status,
        red_team_status,
        rehearsal_status,
        runbook_status,
    )
    if any(status == "FAIL" for status in statuses) or any(not item.present for item in artifacts):
        return "FAIL"
    if any(status == "WARN" for status in statuses):
        return "WARN"
    return "PASS"


def _commands() -> tuple[SubmissionCommand, ...]:
    return (
        SubmissionCommand(
            label="Final bundle",
            command="quanthack tech-prize-submit",
            purpose="Regenerate every judge-facing artifact in safe offline mode.",
        ),
        SubmissionCommand(
            label="Open dashboard",
            command="open outputs/reports/technology_prize_dashboard.html",
            purpose="Show the browser-readable technology-prize overview.",
        ),
        SubmissionCommand(
            label="Run rubric",
            command="quanthack tech-prize-rubric",
            purpose="Score the architecture against the technology-prize judging axes.",
        ),
        SubmissionCommand(
            label="Run red team",
            command="quanthack tech-prize-red-team",
            purpose="Answer skeptical judge questions with executable evidence checks.",
        ),
        SubmissionCommand(
            label="Rehearse demo",
            command="quanthack tech-prize-rehearse",
            purpose="Dry-run the safe offline judge-demo command flow and artifacts.",
        ),
        SubmissionCommand(
            label="Run topology",
            command="quanthack tech-prize-topology",
            purpose="Prove the AgentSDK graph is coherent and read-only.",
        ),
        SubmissionCommand(
            label="Replay trace",
            command="quanthack tech-prize-trace",
            purpose="Show agent steps, tool calls, blackboard writes, and handoffs as trace spans.",
        ),
        SubmissionCommand(
            label="Run guardrails",
            command="quanthack tech-prize-guardrails",
            purpose="Prove broker, path, model-spend, and critic-authority boundaries.",
        ),
        SubmissionCommand(
            label="Optional online AgentSDK",
            command="quanthack tech-prize-demo --run-sdk --allow-online-sdk --sdk-model gpt-5.5",
            purpose="Run real AgentSDK orchestration only after API key and credit-spend approval.",
            spends_credits=True,
        ),
        SubmissionCommand(
            label="Optional Anthropic critic",
            command=(
                "quanthack tech-prize-demo --run-anthropic-critic "
                "--allow-online-anthropic --anthropic-model claude-sonnet-4-6"
            ),
            purpose="Run independent Anthropic critique only after API key and credit-spend approval.",
            spends_credits=True,
        ),
    )


def _inspect_artifact(*, name: str, path: Path) -> SubmissionArtifact:
    if not path.exists():
        return SubmissionArtifact(name=name, path=str(path), present=False)
    payload = path.read_bytes()
    return SubmissionArtifact(
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


def _commands_table(commands: tuple[SubmissionCommand, ...]) -> str:
    rows = [
        "| Label | Command | Purpose | Spends Credits |",
        "| --- | --- | --- | --- |",
    ]
    for command in commands:
        rows.append(
            f"| {command.label} | `{command.command}` | {command.purpose} | "
            f"{'yes' if command.spends_credits else 'no'} |"
        )
    return "\n".join(rows)


def _artifact_table(artifacts: tuple[SubmissionArtifact, ...]) -> str:
    rows = [
        "| Artifact | Status | Bytes | SHA-256 Prefix | Path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for artifact in artifacts:
        rows.append(
            f"| {artifact.name} | {artifact.status} | {artifact.bytes_size} | "
            f"`{artifact.sha256_12 or '-'}` | `{artifact.path}` |"
        )
    return "\n".join(rows)
