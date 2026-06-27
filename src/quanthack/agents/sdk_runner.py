from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from quanthack.agents.sdk_bridge import (
    AgentsSdkUnavailableError,
    create_agents_sdk_app,
)
from quanthack.agents.technology_prize import TechnologyPrizeReport, run_local_technology_prize_demo


@dataclass(frozen=True)
class AgentsSdkRunnerResult:
    status: str
    model: str
    max_turns: int
    online_allowed: bool
    api_key_present: bool
    prompt: str
    final_output: str = ""
    error: str = ""
    tool_names: tuple[str, ...] = ()
    specialist_agents: tuple[str, ...] = ()

    @property
    def ran(self) -> bool:
        return self.status == "RAN"

    def summary_lines(self) -> tuple[str, ...]:
        return (
            "Agents SDK Guarded Runner",
            f"  Status: {self.status}",
            f"  Model: {self.model}",
            f"  Online allowed: {'yes' if self.online_allowed else 'no'}",
            f"  API key present: {'yes' if self.api_key_present else 'no'}",
            f"  Tools: {', '.join(self.tool_names) or 'none'}",
            f"  Specialists: {len(self.specialist_agents)}",
        )

    def to_markdown(self) -> str:
        lines = [
            "# Claude Agent Trader Guarded Agents SDK Runner",
            "",
            "This is the optional online demo path for the technology prize.",
            "It is disabled unless the operator explicitly arms it.",
            "",
            "## Status",
            "",
            f"- Status: **{self.status}**",
            f"- Model: `{self.model}`",
            f"- Online allowed: {'yes' if self.online_allowed else 'no'}",
            f"- API key present: {'yes' if self.api_key_present else 'no'}",
            f"- Max turns: {self.max_turns}",
            f"- Tools: {', '.join(self.tool_names) or 'none'}",
            f"- Specialist agents: {', '.join(self.specialist_agents) or 'none'}",
            "",
            "## Safety Boundary",
            "",
            "- Only read-only artifact summary tools are exposed.",
            "- No MT5 order-placement function is registered.",
            "- Manual ticket sheets remain advisory.",
            "- Strategy promotion still depends on existing backtest and risk gates.",
            "",
            "## Prompt",
            "",
            "```text",
            self.prompt,
            "```",
            "",
        ]
        if self.final_output:
            lines.extend(["## Runner Output", "", str(self.final_output), ""])
        if self.error:
            lines.extend(["## Runner Error", "", f"```text\n{self.error}\n```", ""])
        return "\n".join(lines)


def run_guarded_agents_sdk_demo(
    *,
    project_root: str | Path = ".",
    model: str = "gpt-5.5",
    max_turns: int = 6,
    allow_online_sdk: bool = False,
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], Any] = importlib.import_module,
) -> AgentsSdkRunnerResult:
    env = os.environ if environ is None else environ
    local_report = run_local_technology_prize_demo(project_root=project_root, environ=env)
    prompt = build_judge_demo_prompt(local_report)
    api_key_present = bool(env.get("OPENAI_API_KEY"))

    if not allow_online_sdk:
        return AgentsSdkRunnerResult(
            status="SKIPPED",
            model=model,
            max_turns=max_turns,
            online_allowed=False,
            api_key_present=api_key_present,
            prompt=prompt,
            error="Online Agents SDK runner is disabled. Pass --run-sdk --allow-online-sdk to arm it.",
        )
    if not api_key_present:
        return AgentsSdkRunnerResult(
            status="BLOCKED",
            model=model,
            max_turns=max_turns,
            online_allowed=True,
            api_key_present=False,
            prompt=prompt,
            error="OPENAI_API_KEY is not set, so the online AgentSDK Runner was not started.",
        )

    try:
        sdk = importer("agents")
        app = create_agents_sdk_app(project_root=project_root, importer=importer)
    except AgentsSdkUnavailableError as exc:
        return AgentsSdkRunnerResult(
            status="BLOCKED",
            model=model,
            max_turns=max_turns,
            online_allowed=True,
            api_key_present=True,
            prompt=prompt,
            error=str(exc),
        )

    if not hasattr(sdk, "Runner"):
        return AgentsSdkRunnerResult(
            status="BLOCKED",
            model=model,
            max_turns=max_turns,
            online_allowed=True,
            api_key_present=True,
            prompt=prompt,
            error="The imported agents package does not expose Runner.",
            tool_names=tuple(sorted(app.tools)),
            specialist_agents=tuple(sorted(app.specialist_agents)),
        )

    try:
        run_config = _build_run_config(sdk=sdk, model=model)
        result = sdk.Runner.run_sync(
            app.chief_agent,
            prompt,
            max_turns=max_turns,
            run_config=run_config,
        )
    except Exception as exc:  # pragma: no cover - live SDK errors vary by account/network
        return AgentsSdkRunnerResult(
            status="FAILED",
            model=model,
            max_turns=max_turns,
            online_allowed=True,
            api_key_present=True,
            prompt=prompt,
            error=str(exc),
            tool_names=tuple(sorted(app.tools)),
            specialist_agents=tuple(sorted(app.specialist_agents)),
        )

    return AgentsSdkRunnerResult(
        status="RAN",
        model=model,
        max_turns=max_turns,
        online_allowed=True,
        api_key_present=True,
        prompt=prompt,
        final_output=str(getattr(result, "final_output", result)),
        tool_names=tuple(sorted(app.tools)),
        specialist_agents=tuple(sorted(app.specialist_agents)),
    )


def build_judge_demo_prompt(report: TechnologyPrizeReport) -> str:
    provider_lines = [
        f"- {provider.name}: package={'yes' if provider.installed else 'no'}, "
        f"api_key={'yes' if provider.api_key_present else 'no'}"
        for provider in report.providers
    ]
    artifact_lines = [
        f"- {artifact.name}: {artifact.status}, {artifact.summary}"
        for artifact in report.artifacts
    ]
    return "\n".join(
        [
            "You are presenting Claude Agent Trader for the separate technology prize.",
            "Use the read-only tools if you need more detail, then produce a concise judge-facing brief.",
            "",
            "Required sections:",
            "1. AgentSDK architecture",
            "2. Evidence inspected",
            "3. Risk and MT5 safety boundary",
            "4. Anthropic critic extension",
            "5. Remaining build gaps",
            "",
            "Hard constraints:",
            "- Do not place, suggest placing, or simulate placing live orders.",
            "- Do not ask for broker credentials.",
            "- Distinguish proven artifacts from planned extensions.",
            "",
            "Current local summary:",
            *report.summary_lines(),
            "",
            "Provider readiness:",
            *provider_lines,
            "",
            "Evidence artifacts:",
            *artifact_lines,
        ]
    )


def write_agents_sdk_runner_report(
    result: AgentsSdkRunnerResult,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.to_markdown(), encoding="utf-8")


def _build_run_config(*, sdk: Any, model: str) -> Any:
    if not hasattr(sdk, "RunConfig"):
        return None
    return sdk.RunConfig(
        model=model,
        workflow_name="Claude Agent Trader Technology Prize Demo",
        trace_metadata={
            "project": "quanthack",
            "mode": "read_only_technology_prize",
        },
    )
