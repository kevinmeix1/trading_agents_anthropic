from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape

from quanthack.agents.demo_pack import TechnologyPrizeDemoPack


@dataclass(frozen=True)
class TechnologyPrizeDashboard:
    title: str
    html: str


def build_technology_prize_dashboard(
    pack: TechnologyPrizeDemoPack,
    *,
    title: str = "QuanHack Technology Prize Dashboard",
    generated_at: datetime | None = None,
) -> TechnologyPrizeDashboard:
    generated = generated_at or datetime.now(tz=timezone.utc)
    pass_count = sum(1 for check in pack.checks if check.status == "PASS")
    warn_count = sum(1 for check in pack.checks if check.status == "WARN")
    fail_count = sum(1 for check in pack.checks if check.status == "FAIL")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --ink: #17202f;
      --muted: #5d6675;
      --panel: #ffffff;
      --line: #d7dee8;
      --ok: #0f766e;
      --ok-bg: #def7f1;
      --warn: #9a3412;
      --warn-bg: #fff7ed;
      --bad: #991b1b;
      --bad-bg: #fef2f2;
      --blue: #1d4ed8;
      --blue-bg: #eff6ff;
      --code: #0f172a;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.42;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 30px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 16px 0 8px;
      font-size: 15px;
      letter-spacing: 0;
    }}
    .muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      background: #eef2f6;
      color: #303947;
      font-weight: 700;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    code {{
      color: var(--code);
      background: #eef2f6;
      border-radius: 5px;
      padding: 2px 5px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ok {{ background: var(--ok-bg); color: var(--ok); }}
    .warn {{ background: var(--warn-bg); color: var(--warn); }}
    .bad {{ background: var(--bad-bg); color: var(--bad); }}
    .info {{ background: var(--blue-bg); color: var(--blue); }}
    .graph {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }}
    .agent-node {{
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px;
      min-height: 98px;
    }}
    .agent-node strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 14px;
    }}
    .agent-node span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .callout {{
      background: var(--blue-bg);
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--blue);
      margin-top: 16px;
      font-size: 14px;
    }}
    @media (max-width: 720px) {{
      header {{ display: block; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{escape(title)}</h1>
      <div class="muted">Generated {escape(generated.isoformat(timespec="seconds"))}</div>
    </div>
    <div>
      {status_badge(pack.overall_status)}
      <div class="muted">Technology-prize demo pack</div>
    </div>
  </header>

  <section>
    <h2>Executive View</h2>
    <div class="grid">
      {metric("Overall", pack.overall_status)}
      {metric("Checks", f"{pass_count} pass / {warn_count} warn / {fail_count} fail")}
      {metric("Agents", str(len(pack.architecture_report.agents)))}
      {metric("Tool Specs", str(len(pack.architecture_report.tools)))}
      {metric("Workflow Steps", str(len(pack.local_workflow.steps)))}
      {metric("Guardrails", f"{sum(1 for check in pack.agent_guardrails.checks if check.status == 'PASS')}/{len(pack.agent_guardrails.checks)} pass")}
      {metric("Topology", pack.agent_topology.status)}
      {metric("Trace Spans", str(len(pack.agent_trace_replay.spans)))}
      {metric("Runbook", f"{len(pack.demo_runbook.steps)} steps")}
      {metric("AI-Native Claims", str(len(pack.innovation_claims)))}
      {metric("Evidence", _coverage_text(pack.architecture_report.artifacts))}
      {metric("Source Manifest", _coverage_text(pack.source_artifacts))}
      {metric("SDK Runner", pack.sdk_runner.status)}
      {metric("Anthropic Critic", pack.anthropic_critic.status)}
    </div>
  </section>

  <section>
    <h2>Judge Pitch</h2>
    <div class="panel">
      QuanHack is a safe agentic trading-research system: Agents SDK owns the
      orchestration layer, read-only tools summarize verifiable artifacts,
      risk and MT5 boundaries are explicit, and Anthropic can provide an
      independent critic pass when credits are armed.
    </div>
  </section>

  <section>
    <h2>Agent Graph</h2>
    <div class="graph">
      {agent_nodes(pack)}
    </div>
  </section>

  <section>
    <h2>Why This Is AI-Native</h2>
    {innovation_table(pack)}
  </section>

  <section>
    <h2>Executable Workflow</h2>
    {workflow_table(pack)}
  </section>

  <section>
    <h2>Guardrails</h2>
    {guardrails_table(pack)}
  </section>

  <section>
    <h2>Topology</h2>
    {topology_table(pack)}
  </section>

  <section>
    <h2>Trace Replay</h2>
    {trace_table(pack)}
  </section>

  <section>
    <h2>Scorecard</h2>
    {checks_table(pack)}
  </section>

  <section>
    <h2>Judge Packet</h2>
    <div class="panel">
      <p>
        The requirement-level judge packet verifies AgentSDK use, Anthropic critic
        support, AI-native workflow, evidence provenance, guarded credit spend,
        and the broker-safe no-order-authority boundary.
      </p>
      <p>
        Run <code>quanthack tech-prize-judge-packet</code>, then open
        <code>outputs/reports/technology_prize_judge_packet.md</code> or
        <code>outputs/reports/technology_prize_judge_packet.json</code>.
      </p>
    </div>
  </section>

  <section>
    <h2>Demo Runbook</h2>
    {runbook_table(pack)}
  </section>

  <section>
    <h2>Demo Commands</h2>
    {commands_table(pack)}
  </section>

  <section>
    <h2>Provider Readiness</h2>
    {provider_table(pack)}
  </section>

  <section>
    <h2>Evidence Artifacts</h2>
    {evidence_table(pack)}
  </section>

  <section>
    <h2>Provenance</h2>
    <h3>Source Files</h3>
    {artifact_table(pack.source_artifacts)}
    <h3>Generated Reports</h3>
    {artifact_table(pack.report_artifacts)}
  </section>

  <section>
    <h2>Safety Boundary</h2>
    <div class="panel">
      <ul>
        <li>Default dashboard and pack commands do not make online model calls.</li>
        <li>Online AgentSDK and Anthropic paths require explicit <code>--allow-online-*</code> switches.</li>
        <li>No MT5 order-placement tool is registered in the AgentSDK tool surface.</li>
        <li>The Anthropic critic has no trade approval or broker authority.</li>
        <li>Strategy promotion remains tied to backtest, walk-forward, and risk evidence.</li>
      </ul>
    </div>
  </section>

  <div class="callout">
    Judge demo path: open this dashboard, then show
    <code>outputs/reports/technology_prize_judge_packet.md</code> for the
    requirement matrix and <code>outputs/reports/technology_prize_demo_pack.md</code>
    for the hashed manifest and exact command recipe.
  </div>
</main>
</body>
</html>
"""
    return TechnologyPrizeDashboard(title=title, html=html)


def status_badge(status: str) -> str:
    css = {"PASS": "ok", "OK": "ok", "WARN": "warn", "FAIL": "bad"}.get(status, "info")
    return f'<span class="badge {css}">{escape(status)}</span>'


def metric(label: str, value: str) -> str:
    return (
        '<div class="metric">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        "</div>"
    )


def agent_nodes(pack: TechnologyPrizeDemoPack) -> str:
    return "\n".join(
        '<div class="agent-node">'
        f"<strong>{escape(agent.name)}</strong>"
        f"<span>{escape(agent.role)}</span>"
        f"<span>Tools: {escape(str(len(agent.tools)))}</span>"
        f"<span>Handoffs: {escape(str(len(agent.handoffs)))}</span>"
        "</div>"
        for agent in pack.architecture_report.agents
    )


def checks_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = ["<table><thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead><tbody>"]
    for check in pack.checks:
        rows.append(
            "<tr>"
            f"<td>{escape(check.name)}</td>"
            f"<td>{status_badge(check.status)}</td>"
            f"<td>{escape(check.details)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def innovation_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Claim</th><th>Why AI-Native</th><th>Evidence</th><th>Prize Relevance</th></tr></thead><tbody>"
    ]
    for claim in pack.innovation_claims:
        rows.append(
            "<tr>"
            f"<td><strong>{escape(claim.name)}</strong><br>{escape(claim.claim)}</td>"
            f"<td>{escape(claim.why_ai_native)}</td>"
            f"<td>{'<br>'.join(f'<code>{escape(item)}</code>' for item in claim.evidence)}</td>"
            f"<td>{escape(claim.prize_relevance)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def workflow_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Step</th><th>Agent</th><th>Tool</th><th>Status</th><th>Summary</th><th>Writes</th></tr></thead><tbody>"
    ]
    for step in pack.local_workflow.steps:
        rows.append(
            "<tr>"
            f"<td>{step.step}</td>"
            f"<td>{escape(step.agent)}</td>"
            f"<td><code>{escape(step.tool)}</code></td>"
            f"<td>{status_badge(step.status)}</td>"
            f"<td>{escape(step.summary)}</td>"
            f"<td>{escape(', '.join(step.writes))}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        '<div class="callout">'
        f"Blackboard writes: {len(pack.local_workflow.blackboard)}. "
        f"Handoffs: {len(pack.local_workflow.handoffs)}. "
        f"Verdict: {escape(pack.local_workflow.verdict)}"
        "</div>"
    )
    return "\n".join(rows)


def guardrails_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Guardrail</th><th>Status</th><th>Scope</th><th>Details</th></tr></thead><tbody>"
    ]
    for check in pack.agent_guardrails.checks:
        rows.append(
            "<tr>"
            f"<td>{escape(check.name)}</td>"
            f"<td>{status_badge(check.status)}</td>"
            f"<td>{escape(check.scope)}</td>"
            f"<td>{escape(check.details)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def topology_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead><tbody>"
    ]
    for check in pack.agent_topology.checks:
        rows.append(
            "<tr>"
            f"<td>{escape(check.name)}</td>"
            f"<td>{status_badge(check.status)}</td>"
            f"<td>{escape(check.details)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        '<div class="callout">'
        f"Tool coverage rows: {len(pack.agent_topology.tool_coverage)}. "
        f"Handoff edges: {len(pack.agent_topology.handoffs)}."
        "</div>"
    )
    return "\n".join(rows)


def trace_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Span</th><th>Kind</th><th>Agent</th><th>Status</th><th>Tool</th><th>Summary</th></tr></thead><tbody>"
    ]
    for span in pack.agent_trace_replay.spans[:18]:
        rows.append(
            "<tr>"
            f"<td><code>{escape(span.span_id)}</code></td>"
            f"<td>{escape(span.kind)}</td>"
            f"<td>{escape(span.agent)}</td>"
            f"<td>{status_badge(span.status)}</td>"
            f"<td><code>{escape(span.tool or '-')}</code></td>"
            f"<td>{escape(span.summary)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        '<div class="callout">'
        f"Trace ID: {escape(pack.agent_trace_replay.trace_id)}. "
        f"Total spans: {len(pack.agent_trace_replay.spans)}."
        "</div>"
    )
    return "\n".join(rows)


def runbook_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Time</th><th>Action</th><th>Command / Artifact</th><th>Proof</th></tr></thead><tbody>"
    ]
    for step in pack.demo_runbook.steps:
        rows.append(
            "<tr>"
            f"<td>{escape(step.minute_mark)}</td>"
            f"<td>{escape(step.action)}</td>"
            f"<td><code>{escape(step.command_or_artifact)}</code></td>"
            f"<td>{escape(step.proof)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def commands_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Command</th><th>Purpose</th><th>Credits</th><th>Required Env</th></tr></thead><tbody>"
    ]
    for command in pack.demo_commands:
        rows.append(
            "<tr>"
            f"<td><code>{escape(command.command)}</code></td>"
            f"<td>{escape(command.purpose)}</td>"
            f"<td>{'yes' if command.spends_credits else 'no'}</td>"
            f"<td>{escape(', '.join(command.requires_env) or 'none')}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def provider_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Provider</th><th>Package</th><th>API Key</th><th>Status</th><th>Role</th></tr></thead><tbody>"
    ]
    for provider in pack.architecture_report.providers:
        if provider.ready:
            status = "READY"
        elif provider.installed:
            status = "PACKAGE_ONLY"
        else:
            status = "MISSING_PACKAGE"
        rows.append(
            "<tr>"
            f"<td>{escape(provider.name)}</td>"
            f"<td><code>{escape(provider.package_name)}</code></td>"
            f"<td><code>{escape(provider.api_key_env)}</code></td>"
            f"<td>{status_badge('PASS' if provider.ready else 'WARN')}</td>"
            f"<td>{escape(provider.role)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def evidence_table(pack: TechnologyPrizeDemoPack) -> str:
    rows = [
        "<table><thead><tr><th>Artifact</th><th>Status</th><th>Rows</th><th>Columns</th><th>Hash</th><th>Summary</th></tr></thead><tbody>"
    ]
    for artifact in pack.architecture_report.artifacts:
        rows.append(
            "<tr>"
            f"<td>{escape(artifact.name)}</td>"
            f"<td>{status_badge('OK' if artifact.present else 'FAIL')}</td>"
            f"<td>{artifact.row_count}</td>"
            f"<td>{artifact.column_count}</td>"
            f"<td><code>{escape(artifact.sha256_12 or '-')}</code></td>"
            f"<td>{escape(artifact.summary)}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def artifact_table(artifacts: tuple[object, ...]) -> str:
    rows = [
        "<table><thead><tr><th>Artifact</th><th>Status</th><th>Bytes</th><th>SHA-256 Prefix</th><th>Path</th></tr></thead><tbody>"
    ]
    for artifact in artifacts:
        rows.append(
            "<tr>"
            f"<td>{escape(str(getattr(artifact, 'name')))}</td>"
            f"<td>{status_badge(str(getattr(artifact, 'status')))}</td>"
            f"<td>{getattr(artifact, 'bytes_size')}</td>"
            f"<td><code>{escape(str(getattr(artifact, 'sha256_12') or '-'))}</code></td>"
            f"<td><code>{escape(str(getattr(artifact, 'path')))}</code></td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _coverage_text(items: tuple[object, ...]) -> str:
    present = sum(1 for item in items if bool(getattr(item, "present")))
    return f"{present}/{len(items)}"
