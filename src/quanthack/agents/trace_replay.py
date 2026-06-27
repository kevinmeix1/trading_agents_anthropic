from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quanthack.agents.workflow import LocalAgentWorkflow, run_local_agent_workflow


@dataclass(frozen=True)
class AgentTraceSpan:
    span_id: str
    parent_id: str
    name: str
    kind: str
    agent: str
    status: str
    summary: str
    tool: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class AgentTraceReplay:
    generated_at: datetime
    trace_id: str
    status: str
    spans: tuple[AgentTraceSpan, ...]
    workflow: LocalAgentWorkflow

    def summary_lines(self) -> tuple[str, ...]:
        return (
            "AgentSDK Trace Replay",
            f"  Overall: {self.status}",
            f"  Trace ID: {self.trace_id}",
            f"  Spans: {len(self.spans)}",
            f"  Workflow steps: {len(self.workflow.steps)}",
            f"  Blackboard writes: {len(self.workflow.blackboard)}",
            f"  Handoffs: {len(self.workflow.handoffs)}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "trace_id": self.trace_id,
            "status": self.status,
            "spans": [_dict(span) for span in self.spans],
            "workflow_status": self.workflow.status,
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Claude Agent Trader AgentSDK Trace Replay",
                "",
                f"Generated: `{self.generated_at.isoformat()}`",
                f"Trace ID: `{self.trace_id}`",
                f"Overall status: **{self.status}**",
                "",
                "## Why This Matters",
                "",
                (
                    "This is an offline, credit-free trace replay of the local AgentSDK-style "
                    "workflow. It makes agent steps, tool calls, blackboard writes, and handoffs "
                    "inspectable as spans."
                ),
                "",
                "## Spans",
                "",
                _spans_table(self.spans),
                "",
            ]
        )


def build_agent_trace_replay(
    *,
    project_root: str | Path = ".",
) -> AgentTraceReplay:
    workflow = run_local_agent_workflow(project_root=project_root)
    spans = _build_spans(workflow)
    status = "PASS" if workflow.status == "PASS" and all(span.status != "FAIL" for span in spans) else "WARN"
    return AgentTraceReplay(
        generated_at=datetime.now(tz=timezone.utc),
        trace_id=_trace_id(workflow),
        status=status,
        spans=spans,
        workflow=workflow,
    )


def write_agent_trace_replay(
    replay: AgentTraceReplay,
    *,
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(replay.to_markdown(), encoding="utf-8")
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(replay.to_dict(), indent=2), encoding="utf-8")


def _build_spans(workflow: LocalAgentWorkflow) -> tuple[AgentTraceSpan, ...]:
    spans: list[AgentTraceSpan] = [
        AgentTraceSpan(
            span_id="span-000",
            parent_id="",
            name="technology_prize_workflow",
            kind="trace_root",
            agent="Chief Trading Agent",
            status=workflow.status,
            summary=workflow.verdict,
        )
    ]
    for step in workflow.steps:
        step_span_id = f"span-step-{step.step:02d}"
        spans.append(
            AgentTraceSpan(
                span_id=step_span_id,
                parent_id="span-000",
                name=step.objective,
                kind="agent_step",
                agent=step.agent,
                status=step.status,
                summary=step.summary,
                tool=step.tool,
            )
        )
        spans.append(
            AgentTraceSpan(
                span_id=f"span-tool-{step.step:02d}",
                parent_id=step_span_id,
                name=f"tool:{step.tool}",
                kind="tool_call",
                agent=step.agent,
                status=step.status,
                summary=f"writes={', '.join(step.writes) or '-'}",
                tool=step.tool,
            )
        )
    for index, item in enumerate(workflow.blackboard, start=1):
        spans.append(
            AgentTraceSpan(
                span_id=f"span-blackboard-{index:02d}",
                parent_id="span-000",
                name=f"blackboard:{item.key}",
                kind="blackboard_write",
                agent=item.source_agent,
                status=item.status,
                summary=item.value,
                evidence=item.evidence,
            )
        )
    for index, handoff in enumerate(workflow.handoffs, start=1):
        spans.append(
            AgentTraceSpan(
                span_id=f"span-handoff-{index:02d}",
                parent_id="span-000",
                name=f"handoff:{handoff.from_agent}->{handoff.to_agent}",
                kind="handoff",
                agent=handoff.from_agent,
                status="PASS",
                summary=handoff.reason,
                evidence=handoff.to_agent,
            )
        )
    return tuple(spans)


def _trace_id(workflow: LocalAgentWorkflow) -> str:
    payload = "|".join(f"{step.step}:{step.agent}:{step.tool}:{step.status}" for step in workflow.steps)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dict(value: Any) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.__dict__.items()
    }


def _spans_table(spans: tuple[AgentTraceSpan, ...]) -> str:
    rows = [
        "| Span | Parent | Kind | Agent | Status | Tool | Summary | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for span in spans:
        rows.append(
            f"| `{span.span_id}` | `{span.parent_id or '-'}` | {span.kind} | {span.agent} | "
            f"{span.status} | `{span.tool or '-'}` | {span.summary} | `{span.evidence or '-'}` |"
        )
    return "\n".join(rows)
