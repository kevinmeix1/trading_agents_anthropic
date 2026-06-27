from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quanthack.agents.technology_prize import TechnologyPrizeReport, run_local_technology_prize_demo


@dataclass(frozen=True)
class AnthropicCriticResult:
    status: str
    model: str
    max_tokens: int
    online_allowed: bool
    api_key_present: bool
    prompt: str
    final_output: str = ""
    error: str = ""

    @property
    def ran(self) -> bool:
        return self.status == "RAN"

    def summary_lines(self) -> tuple[str, ...]:
        return (
            "Anthropic Guarded Critic",
            f"  Status: {self.status}",
            f"  Model: {self.model}",
            f"  Online allowed: {'yes' if self.online_allowed else 'no'}",
            f"  API key present: {'yes' if self.api_key_present else 'no'}",
        )

    def to_markdown(self) -> str:
        lines = [
            "# QuanHack Guarded Anthropic Critic",
            "",
            "This is the optional Anthropic-credits path for independent review.",
            "It is disabled unless the operator explicitly arms it.",
            "",
            "## Status",
            "",
            f"- Status: **{self.status}**",
            f"- Model: `{self.model}`",
            f"- Online allowed: {'yes' if self.online_allowed else 'no'}",
            f"- API key present: {'yes' if self.api_key_present else 'no'}",
            f"- Max tokens: {self.max_tokens}",
            "",
            "## Authority Boundary",
            "",
            "- Critique only.",
            "- No MT5 credentials.",
            "- No broker or order-placement tools.",
            "- No direct strategy promotion authority.",
            "",
            "## Prompt",
            "",
            "```text",
            self.prompt,
            "```",
            "",
        ]
        if self.final_output:
            lines.extend(["## Critic Output", "", str(self.final_output), ""])
        if self.error:
            lines.extend(["## Critic Error", "", f"```text\n{self.error}\n```", ""])
        return "\n".join(lines)


def run_guarded_anthropic_critic(
    *,
    project_root: str | Path = ".",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 900,
    allow_online_anthropic: bool = False,
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], Any] = importlib.import_module,
) -> AnthropicCriticResult:
    env = os.environ if environ is None else environ
    local_report = run_local_technology_prize_demo(project_root=project_root, environ=env)
    prompt = build_anthropic_critic_prompt(local_report)
    api_key = env.get("ANTHROPIC_API_KEY", "")

    if not allow_online_anthropic:
        return AnthropicCriticResult(
            status="SKIPPED",
            model=model,
            max_tokens=max_tokens,
            online_allowed=False,
            api_key_present=bool(api_key),
            prompt=prompt,
            error=(
                "Online Anthropic critic is disabled. "
                "Pass --run-anthropic-critic --allow-online-anthropic to arm it."
            ),
        )
    if not api_key:
        return AnthropicCriticResult(
            status="BLOCKED",
            model=model,
            max_tokens=max_tokens,
            online_allowed=True,
            api_key_present=False,
            prompt=prompt,
            error="ANTHROPIC_API_KEY is not set, so the online critic was not started.",
        )

    try:
        anthropic = importer("anthropic")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                "You are an independent quant-technology judge. "
                "Critique architecture, risk evidence, overfitting risk, and demo clarity. "
                "You have no trade authority."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # pragma: no cover - live API errors vary by account/network
        return AnthropicCriticResult(
            status="FAILED",
            model=model,
            max_tokens=max_tokens,
            online_allowed=True,
            api_key_present=True,
            prompt=prompt,
            error=str(exc),
        )

    return AnthropicCriticResult(
        status="RAN",
        model=model,
        max_tokens=max_tokens,
        online_allowed=True,
        api_key_present=True,
        prompt=prompt,
        final_output=_extract_message_text(message),
    )


def build_anthropic_critic_prompt(report: TechnologyPrizeReport) -> str:
    artifact_lines = [
        f"- {artifact.name}: {artifact.status}, rows={artifact.row_count}, summary={artifact.summary}"
        for artifact in report.artifacts
    ]
    trace_lines = [
        f"- {event.agent} / {event.action}: {event.status} - {event.detail}"
        for event in report.trace
    ]
    return "\n".join(
        [
            "Review QuanHack for the separate technology prize.",
            "Focus on technical merit, not trading P&L.",
            "",
            "Please return:",
            "1. Strongest technical claims",
            "2. Weakest or least proven claims",
            "3. AgentSDK use quality",
            "4. Risk and live-trading safety review",
            "5. Specific next changes before judging",
            "",
            "Do not suggest live trading or credential sharing.",
            "",
            "Local summary:",
            *report.summary_lines(),
            "",
            "Artifacts:",
            *artifact_lines,
            "",
            "Trace:",
            *trace_lines,
        ]
    )


def write_anthropic_critic_report(
    result: AnthropicCriticResult,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.to_markdown(), encoding="utf-8")


def _extract_message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        elif isinstance(block, dict) and block.get("text") is not None:
            parts.append(str(block["text"]))
    return "\n".join(parts) if parts else str(message)
