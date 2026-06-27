from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
PDF_PATH = OUTPUT_DIR / "claude_agent_trader_technology_walkthrough.pdf"


NAVY = colors.HexColor("#111827")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#64748B")
BLUE = colors.HexColor("#1D4ED8")
CYAN = colors.HexColor("#475569")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#92400E")
RED = colors.HexColor("#DC2626")
LAVENDER = colors.HexColor("#F3F4F6")
SKY = colors.HexColor("#EFF6FF")
MINT = colors.HexColor("#ECFDF5")
CREAM = colors.HexColor("#F9FAFB")
GRAY_BG = colors.HexColor("#F8FAFC")
LINE = colors.HexColor("#D1D5DB")


@dataclass(frozen=True)
class DocData:
    submission: dict[str, Any]
    rubric: dict[str, Any]
    red_team: dict[str, Any]
    rehearsal: dict[str, Any]
    topology: dict[str, Any]
    guardrails: dict[str, Any]
    trace: dict[str, Any]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.55 * inch,
        title="Claude Agent Trader Technology Walkthrough",
        author="Claude Agent Trader",
        subject="AgentSDK technology-prize architecture walkthrough",
    )
    styles = build_styles()
    story: list[Any] = []
    build_story(story, styles, data)
    doc.build(story, onFirstPage=draw_cover_footer, onLaterPages=draw_page_frame)
    print(PDF_PATH)


def load_data() -> DocData:
    return DocData(
        submission=read_json("technology_prize_submission.json"),
        rubric=read_json("technology_prize_rubric.json"),
        red_team=read_json("technology_prize_red_team.json"),
        rehearsal=read_json("technology_prize_demo_rehearsal.json"),
        topology=read_json("technology_prize_topology.json"),
        guardrails=read_json("technology_prize_guardrails.json"),
        trace=read_json("technology_prize_trace_replay.json"),
    )


def read_json(name: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=11.5,
            leading=16,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=10.4,
            textColor=INK,
        ),
        "small_muted": ParagraphStyle(
            "SmallMuted",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.6,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.25,
            leading=9.2,
            textColor=INK,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
        ),
    }


def build_story(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    add_cover(story, styles, data)
    add_executive_summary(story, styles, data)
    add_system_architecture(story, styles)
    add_agent_control_plane(story, styles, data)
    add_tool_surface(story, styles)
    add_workflow_and_trace(story, styles, data)
    add_safety_and_governance(story, styles, data)
    add_anthropic_section(story, styles)
    add_evaluators(story, styles, data)
    add_demo_flow(story, styles, data)
    add_artifact_map(story, styles, data)
    add_appendix(story, styles, data)


def add_cover(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    score = data.rubric.get("total_score", 0)
    max_score = data.rubric.get("max_score", 0)
    generated = data.submission.get("generated_at", datetime.now(tz=timezone.utc).isoformat())
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Claude Agent Trader Technology Walkthrough", styles["title"]))
    story.append(
        Paragraph(
            "A detailed PDF guide to the AgentSDK-centered, AI-native architecture built for the separate technology prize. "
            "The focus is technical merit: agent orchestration, read-only tools, Anthropic critique, guardrails, traceability, and demo reproducibility.",
            styles["subtitle"],
        )
    )
    story.append(
        BadgeRow(
            [
                ("Submission", data.submission.get("status", "UNKNOWN"), GREEN),
                ("Rubric", f"{score}/{max_score}", BLUE),
                ("Red team", data.red_team.get("status", "UNKNOWN"), AMBER),
                ("Rehearsal", data.rehearsal.get("status", "UNKNOWN"), CYAN),
            ]
        )
    )
    story.append(Spacer(1, 0.22 * inch))
    story.append(ArchitectureDiagram(width=7.1 * inch, height=3.25 * inch))
    story.append(Paragraph("Figure 1. Technology stack overview: trading research remains deterministic; the AgentSDK layer governs evidence, critique, safety, and presentation.", styles["caption"]))
    story.append(
        InfoBox(
            [
                "Built for the separate technology prize, not only trading P&L.",
                "Default demo path is offline and credit-safe.",
                "Agent tools are read-only; no MT5 order placement is exposed to models.",
                f"Final bundle generated: {generated}",
            ],
            fill=GRAY_BG,
        )
    )
    story.append(PageBreak())


def add_executive_summary(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("1. Executive Snapshot", styles["h1"]))
    story.append(
        Paragraph(
            "Claude Agent Trader has been wrapped as a technology-prize submission around a safe agentic control plane. "
            "The trading engine still supplies research artifacts, backtests, risk outputs, dashboards, and MT5 ticket sheets. "
            "The new technology layer makes those artifacts inspectable by specialist agents and proves the boundary between AI reasoning and trading authority.",
            styles["body"],
        )
    )
    story.append(
        status_table(
            [
                ("Submission bundle", data.submission.get("status", "UNKNOWN"), "Final artifact manifest and command index."),
                ("Rubric", data.rubric.get("status", "UNKNOWN"), f"Score {data.rubric.get('total_score', 0)}/{data.rubric.get('max_score', 0)} across technology-prize axes."),
                ("Red-team report", data.red_team.get("status", "UNKNOWN"), f"{len(data.red_team.get('challenges', []))} skeptical judge challenges."),
                ("Demo rehearsal", data.rehearsal.get("status", "UNKNOWN"), f"{len(data.rehearsal.get('steps', []))} offline demo steps."),
                ("Guardrails", data.guardrails.get("status", "UNKNOWN"), f"{len(data.guardrails.get('checks', []))} executable safety checks."),
                ("Topology", data.topology.get("status", "UNKNOWN"), f"{len(data.topology.get('agents', []))} agents and {len(data.topology.get('tools', []))} tools."),
                ("Trace replay", data.trace.get("status", "UNKNOWN"), f"{len(data.trace.get('spans', []))} span-like events."),
            ],
            styles,
            widths=[1.35 * inch, 0.75 * inch, 4.65 * inch],
        )
    )
    story.append(Paragraph("What the judges should understand in one minute", styles["h2"]))
    story.append(bullets(styles, [
        "The system is no longer just a collection of scripts. It has a multi-agent control plane with explicit roles, handoffs, tools, and guardrails.",
        "OpenAI AgentSDK is the orchestration center: agents inspect evidence through typed, read-only function tools.",
        "Anthropic credits have a defined role: an independent critic reviews risk, overfitting, and presentation quality without trading authority.",
        "The safety model is executable. The project can prove that AI tools cannot place MT5 orders and cannot silently spend model credits.",
        "The demo is reproducible: `tech-prize-rehearse` and `tech-prize-submit` regenerate the proof pack offline.",
    ]))
    story.append(PageBreak())


def add_system_architecture(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph("2. System Architecture", styles["h1"]))
    story.append(
        Paragraph(
            "The project is organized as a layered system. Lower layers do deterministic trading work: ingest data, run strategies, route signals through risk, backtest portfolios, and prepare MT5 ticket artifacts. "
            "The technology-prize layer sits above that engine. It does not replace the trading logic; it supervises and explains it.",
            styles["body"],
        )
    )
    story.append(LayerDiagram(width=7.1 * inch, height=3.4 * inch))
    story.append(Paragraph("Figure 2. Layered architecture: deterministic quant engine below, agentic evidence and governance layer above.", styles["caption"]))
    story.append(Paragraph("Why this matters", styles["h2"]))
    story.append(
        Paragraph(
            "A judge can inspect how the system is built without trusting a live model response. Every layer produces artifacts, and every technology-prize claim points back to files, hashes, reports, or executable checks.",
            styles["body"],
        )
    )
    story.append(PageBreak())


def add_agent_control_plane(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("3. AgentSDK Control Plane", styles["h1"]))
    agents = data.topology.get("agents", [])
    tools = data.topology.get("tools", [])
    handoffs = data.topology.get("handoffs", [])
    story.append(
        Paragraph(
            f"The control plane defines {len(agents)} specialist agents, {len(tools)} read-only tool specs, and {len(handoffs)} validated handoffs. "
            "The Chief Trading Agent coordinates specialist roles instead of letting one prompt do everything.",
            styles["body"],
        )
    )
    story.append(AgentGraphDiagram(width=7.1 * inch, height=3.7 * inch))
    story.append(Paragraph("Figure 3. Agent graph: the Chief Trading Agent delegates to specialist roles and all roads lead to judge-facing reporting.", styles["caption"]))
    rows = [
        ("Chief Trading Agent", "Coordinates the evidence review and delegates to specialists."),
        ("Data Health Agent", "Checks downloaded data, quote coverage, and market-data quality."),
        ("Alpha Research Agent", "Reads research leaderboards, strategy comparisons, and walk-forward evidence."),
        ("Risk Guardian Agent", "Inspects risk limits, guardrail status, MT5 boundary, and deployment safety."),
        ("Regime Scientist Agent", "Explains regime-aware selector and handoff evidence."),
        ("Experiment Auditor Agent", "Checks hashes, provenance, commands, and reproducibility."),
        ("Deployment Operator Agent", "Summarizes dashboards, MT5 ticket sheets, and live dry-run readiness."),
        ("Anthropic Critic Agent", "Independent critique of overfitting, risk, and narrative quality."),
        ("Technology Report Agent", "Packages proof for judges."),
    ]
    story.append(simple_table([("Agent", "Technology role"), *rows], styles, [1.8 * inch, 4.9 * inch]))
    story.append(PageBreak())


def add_tool_surface(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph("4. Read-Only Function Tool Surface", styles["h1"]))
    story.append(
        Paragraph(
            "The AgentSDK bridge exposes evidence tools, not trading controls. This is the main safety and technical-design point: models can inspect project artifacts, but they cannot route orders.",
            styles["body"],
        )
    )
    tool_rows = [
        ("summarize_research_artifacts", "Summarize research/deployment files that support claims."),
        ("summarize_csv", "Inspect bounded rows and schema from project-local CSV files."),
        ("validate_market_data_summary", "Check price/quote data health."),
        ("summarize_experiment_leaderboard", "Read walk-forward and candidate leaderboards."),
        ("build_hackathon_readiness_snapshot", "Summarize competition readiness evidence."),
        ("summarize_mt5_ticket_sheet", "Inspect manual MT5 ticket artifacts without order authority."),
        ("summarize_operator_dashboard_sources", "Check live operator/dashboard source readiness."),
        ("build_technology_prize_judge_packet", "Generate requirement-level evidence matrix."),
        ("run_agent_guardrail_suite", "Run AI/broker safety checks."),
        ("analyze_agent_topology", "Validate graph, handoffs, tools, and guardrails."),
        ("build_judge_demo_runbook", "Build timed demo script and risk answers."),
        ("replay_agent_trace", "Export local workflow as span-like trace data."),
    ]
    story.append(simple_table([("Tool", "Purpose"), *tool_rows], styles, [2.35 * inch, 4.35 * inch], font_size=6.7))
    story.append(
        InfoBox(
            [
                "All tools are read-only.",
                "Paths are confined to the project root.",
                "No MT5 order-placement tool is registered.",
                "Online model calls are disabled unless explicitly armed.",
            ],
            fill=GRAY_BG,
        )
    )
    story.append(PageBreak())


def add_workflow_and_trace(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("5. AI-Native Workflow And Traceability", styles["h1"]))
    story.append(
        Paragraph(
            "The project treats agent reasoning as inspectable state. The local workflow writes blackboard entries, records handoffs, and exports an offline trace replay. "
            "That makes the architecture demonstrable even without spending API credits.",
            styles["body"],
        )
    )
    story.append(TraceDiagram(width=7.1 * inch, height=2.9 * inch))
    story.append(Paragraph("Figure 4. The workflow produces agent steps, tool calls, blackboard writes, and handoffs, then exports trace-like spans.", styles["caption"]))
    trace_spans = data.trace.get("spans", [])
    kinds: dict[str, int] = {}
    for span in trace_spans:
        kinds[span.get("kind", "unknown")] = kinds.get(span.get("kind", "unknown"), 0) + 1
    rows = [("Trace kind", "Count")] + [(kind, str(count)) for kind, count in sorted(kinds.items())]
    story.append(simple_table(rows, styles, [2.2 * inch, 1.2 * inch]))
    story.append(
        Paragraph(
            "The most important detail is not the count itself. It is that a judge can see how specialist agents move from evidence to claims through explicit state transitions.",
            styles["body"],
        )
    )
    story.append(PageBreak())


def add_safety_and_governance(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("6. Safety, MT5 Boundary, And Credit Controls", styles["h1"]))
    story.append(
        Paragraph(
            "The technology prize judges may ask whether an AI agent can accidentally trade or spend credits. The answer is no by default. "
            "The project makes both boundaries executable: no broker authority is exposed to agents, and online OpenAI/Anthropic calls require explicit flags.",
            styles["body"],
        )
    )
    story.append(SafetyBoundaryDiagram(width=7.1 * inch, height=3.05 * inch))
    story.append(Paragraph("Figure 5. Safety boundary: models inspect artifacts; humans or separately armed systems control live MT5 execution.", styles["caption"]))
    guardrail_rows = [("Guardrail", "Status", "Details")]
    for check in data.guardrails.get("checks", []):
        guardrail_rows.append((check.get("name", ""), check.get("status", ""), check.get("details", "")))
    story.append(simple_table(guardrail_rows, styles, [2.05 * inch, 0.65 * inch, 4.0 * inch], font_size=6.7))
    story.append(PageBreak())


def add_anthropic_section(story: list[Any], styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph("7. Anthropic Critic Layer", styles["h1"]))
    story.append(
        Paragraph(
            "The Additional Anthropic Credits are not decorative. They have a defined technical role: independent critique. "
            "The Anthropic Critic Agent reviews overfitting risk, risk controls, and demo narrative while remaining outside broker authority.",
            styles["body"],
        )
    )
    story.append(CriticLoopDiagram(width=7.1 * inch, height=2.7 * inch))
    story.append(Paragraph("Figure 6. Anthropic is used as a reviewer, not a trader. Its output improves quality assurance and judge readiness.", styles["caption"]))
    story.append(Paragraph("Design rules", styles["h2"]))
    story.append(bullets(styles, [
        "Anthropic online calls default to skipped unless `--allow-online-anthropic` is passed.",
        "The critic cannot approve trades and has no MT5 credentials.",
        "The critic reviews evidence produced by the deterministic system and AgentSDK control plane.",
        "This keeps multi-provider AI useful while preserving operational safety.",
    ]))
    story.append(PageBreak())


def add_evaluators(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("8. Evidence Evaluators", styles["h1"]))
    story.append(
        Paragraph(
            "The final system includes evaluators that test the technology story from several angles. This is what makes the submission more than an architecture diagram: the claims are machine-checkable.",
            styles["body"],
        )
    )
    rows = [
        ("Evaluator", "Status", "What it proves"),
        ("Topology", data.topology.get("status", "UNKNOWN"), "Tools are used, handoffs target real agents, and every tool is read-only."),
        ("Guardrails", data.guardrails.get("status", "UNKNOWN"), "No broker authority, path confinement, model-spend gates, and critic boundaries."),
        ("Trace replay", data.trace.get("status", "UNKNOWN"), "Agent steps, tool calls, blackboard writes, and handoffs are inspectable."),
        ("Rubric", data.rubric.get("status", "UNKNOWN"), f"{data.rubric.get('total_score', 0)}/{data.rubric.get('max_score', 0)} score across technology-prize axes."),
        ("Red team", data.red_team.get("status", "UNKNOWN"), "Skeptical judge objections are answered with evidence checks."),
        ("Rehearsal", data.rehearsal.get("status", "UNKNOWN"), "Safe demo flow regenerates expected artifacts offline."),
    ]
    story.append(simple_table(rows, styles, [1.35 * inch, 0.75 * inch, 4.6 * inch]))
    story.append(Paragraph("Red-team challenges", styles["h2"]))
    red_rows = [("Challenge", "Status", "Skeptical question")]
    for challenge in data.red_team.get("challenges", []):
        red_rows.append((challenge.get("name", ""), challenge.get("status", ""), challenge.get("skeptical_question", "")))
    story.append(simple_table(red_rows, styles, [1.9 * inch, 0.62 * inch, 4.15 * inch], font_size=6.6))
    story.append(PageBreak())


def add_demo_flow(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("9. Judge Demo Flow", styles["h1"]))
    story.append(
        Paragraph(
            "The demo flow is intentionally offline by default. The goal is to avoid surprises during judging while still proving advanced technical architecture.",
            styles["body"],
        )
    )
    story.append(DemoFlowDiagram(width=7.1 * inch, height=2.8 * inch))
    story.append(Paragraph("Figure 7. Recommended live flow: rehearse, submit, open dashboard, then run targeted proof commands on request.", styles["caption"]))
    rows = [("Step", "Command", "Purpose")]
    for index, step in enumerate(data.rehearsal.get("steps", []), start=1):
        rows.append((str(index), step.get("command", ""), step.get("purpose", "")))
    story.append(simple_table(rows, styles, [0.35 * inch, 2.4 * inch, 3.95 * inch], font_size=6.6))
    story.append(PageBreak())


def add_artifact_map(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("10. Final Artifact Map", styles["h1"]))
    story.append(
        Paragraph(
            "The final bundle is the handoff package. It tells the judge what to open first, which commands are safe, and which reports prove each claim.",
            styles["body"],
        )
    )
    artifacts = data.submission.get("artifact_manifest", [])
    rows = [("Artifact", "Status", "Path")]
    for item in artifacts:
        rows.append((item.get("name", ""), item.get("status", ""), item.get("path", "")))
    story.append(simple_table(rows, styles, [1.85 * inch, 0.55 * inch, 4.3 * inch], font_size=6.1))
    story.append(PageBreak())


def add_appendix(story: list[Any], styles: dict[str, ParagraphStyle], data: DocData) -> None:
    story.append(Paragraph("11. Appendix: Commands And Talking Points", styles["h1"]))
    story.append(Paragraph("Core commands", styles["h2"]))
    story.append(simple_table(
        [
            ("Command", "Use"),
            ("quanthack tech-prize-rehearse", "Verifies the safe offline demo flow."),
            ("quanthack tech-prize-submit", "Regenerates the final submission bundle and manifest."),
            ("open outputs/reports/technology_prize_dashboard.html", "Opens the first judge-facing visual artifact."),
            ("quanthack tech-prize-rubric", "Shows the scored prize framing."),
            ("quanthack tech-prize-red-team", "Answers skeptical judge questions with evidence."),
            ("quanthack tech-prize-topology", "Validates AgentSDK graph integrity."),
            ("quanthack tech-prize-trace", "Shows trace-like agent spans."),
            ("quanthack tech-prize-guardrails", "Shows executable AI/broker safety checks."),
        ],
        styles,
        [2.45 * inch, 4.25 * inch],
    ))
    story.append(Paragraph("Closing pitch", styles["h2"]))
    story.append(
        InfoBox(
            [
                "Claude Agent Trader is not just a trading bot. It is an AI-native research and deployment control plane.",
                "AgentSDK coordinates specialist agents over real project evidence.",
                "Anthropic credits support independent critique rather than uncontrolled action.",
                "Guardrails prove that models cannot place trades or spend credits without explicit arming.",
                "The final demo is reproducible offline and backed by hashed artifacts.",
            ],
            fill=GRAY_BG,
        )
    )


def bullets(styles: dict[str, ParagraphStyle], items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["body"]), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=6,
    )


def status_table(rows: list[tuple[str, str, str]], styles: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    data = [
        [Paragraph("Component", styles["table_header"]), Paragraph("Status", styles["table_header"]), Paragraph("Meaning", styles["table_header"])]
    ]
    for name, status, meaning in rows:
        data.append([
            Paragraph(name, styles["table_cell"]),
            Paragraph(status, status_style(status, styles)),
            Paragraph(meaning, styles["table_cell"]),
        ])
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(default_table_style())
    return table


def simple_table(rows: list[tuple[Any, ...]], styles: dict[str, ParagraphStyle], widths: list[float], font_size: float = 7.25) -> Table:
    local_cell = ParagraphStyle(
        "TableCellLocal",
        parent=styles["table_cell"],
        fontSize=font_size,
        leading=font_size + 2,
    )
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        row_style = styles["table_header"] if row_index == 0 else local_cell
        data.append([Paragraph(escape(str(cell)), row_style) for cell in row])
    table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(default_table_style())
    return table


def status_style(status: str, styles: dict[str, ParagraphStyle]) -> ParagraphStyle:
    color = GREEN if status == "PASS" or status == "OK" else AMBER if status == "WARN" else RED
    return ParagraphStyle(
        f"Status{status}",
        parent=styles["table_cell"],
        fontName="Helvetica-Bold",
        textColor=color,
        alignment=TA_CENTER,
    )


def default_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRAY_BG]),
        ]
    )


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


class BadgeRow(Flowable):
    def __init__(self, badges: list[tuple[str, str, colors.Color]]) -> None:
        super().__init__()
        self.badges = badges
        self.width = 7.1 * inch
        self.height = 0.45 * inch

    def draw(self) -> None:
        x = 0
        gap = 8
        w = (self.width - gap * (len(self.badges) - 1)) / len(self.badges)
        for label, value, color in self.badges:
            self.canv.setFillColor(colors.white)
            self.canv.setStrokeColor(color)
            self.canv.setLineWidth(1)
            self.canv.roundRect(x, 0, w, self.height, 4, fill=1, stroke=1)
            self.canv.setFillColor(MUTED)
            self.canv.setFont("Helvetica-Bold", 8)
            self.canv.drawString(x + 10, 20, label.upper())
            self.canv.setFillColor(color)
            self.canv.setFont("Helvetica-Bold", 12)
            self.canv.drawString(x + 10, 6, value)
            x += w + gap


class InfoBox(Flowable):
    def __init__(self, lines: list[str], fill: colors.Color = SKY) -> None:
        super().__init__()
        self.lines = lines
        self.fill = fill
        self.width = 7.1 * inch
        self.height = max(0.62 * inch, 0.2 * inch + 0.18 * inch * len(lines))

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(self.fill)
        c.setStrokeColor(LINE)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 8.5)
        y = self.height - 16
        for line in self.lines:
            c.drawString(12, y, "- " + line)
            y -= 13


class BaseDiagram(Flowable):
    def __init__(self, width: float, height: float) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def box(self, x: float, y: float, w: float, h: float, title: str, body: str, fill: colors.Color, stroke: colors.Color = LINE) -> None:
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8)
        draw_centered(c, title, x + w / 2, y + h - 15, w - 8)
        c.setFillColor(INK)
        c.setFont("Helvetica", 6.5)
        draw_wrapped(c, body, x + 7, y + h - 28, w - 14, 8)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, color: colors.Color = BLUE) -> None:
        c = self.canv
        c.setStrokeColor(color)
        c.setLineWidth(1.1)
        c.line(x1, y1, x2, y2)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) >= abs(dy):
            direction = 1 if dx >= 0 else -1
            c.line(x2, y2, x2 - 5 * direction, y2 + 3)
            c.line(x2, y2, x2 - 5 * direction, y2 - 3)
        else:
            direction = 1 if dy >= 0 else -1
            c.line(x2, y2, x2 - 3, y2 - 5 * direction)
            c.line(x2, y2, x2 + 3, y2 - 5 * direction)


class ArchitectureDiagram(BaseDiagram):
    def draw(self) -> None:
        self.box(0.1 * inch, 2.15 * inch, 1.5 * inch, 0.65 * inch, "Data + Backtests", "20GB historical data, portfolio tests, leaderboards", SKY)
        self.box(0.1 * inch, 1.1 * inch, 1.5 * inch, 0.65 * inch, "Trading Engine", "strategies, router, allocator, risk, MT5 tickets", CREAM)
        self.box(2.0 * inch, 1.62 * inch, 1.7 * inch, 0.8 * inch, "Read-Only Tools", "summaries, validation, readiness, ticket sheets", LAVENDER)
        self.box(4.05 * inch, 1.62 * inch, 1.55 * inch, 0.8 * inch, "AgentSDK Graph", "9 agents, 12 tools, 20 handoffs", MINT)
        self.box(5.95 * inch, 2.15 * inch, 1.0 * inch, 0.65 * inch, "Reports", "dashboard, packets, PDF", SKY)
        self.box(5.95 * inch, 1.1 * inch, 1.0 * inch, 0.65 * inch, "Safety", "guardrails, no orders", MINT)
        self.arrow(1.6 * inch, 2.47 * inch, 2.0 * inch, 2.05 * inch)
        self.arrow(1.6 * inch, 1.42 * inch, 2.0 * inch, 1.96 * inch)
        self.arrow(3.7 * inch, 2.02 * inch, 4.05 * inch, 2.02 * inch)
        self.arrow(5.6 * inch, 2.02 * inch, 5.95 * inch, 2.47 * inch)
        self.arrow(5.6 * inch, 1.88 * inch, 5.95 * inch, 1.42 * inch)


class LayerDiagram(BaseDiagram):
    def draw(self) -> None:
        layers = [
            ("Judge-Facing Package", "dashboard, PDF, runbook, rubric, red-team, submission bundle", LAVENDER),
            ("AgentSDK Control Plane", "specialist agents, read-only tools, handoffs, traces, blackboard", MINT),
            ("Governance + Safety", "guardrails, credit gates, path confinement, no MT5 order authority", SKY),
            ("Quant Engine", "strategies, router, allocator, risk engine, backtests, metrics", CREAM),
            ("Data + Broker Boundary", "historical data, live dry-run, MT5 ticket sheets, manual execution", GRAY_BG),
        ]
        y = self.height - 0.55 * inch
        for title, body, fill in layers:
            self.box(0.35 * inch, y, 6.4 * inch, 0.44 * inch, title, body, fill)
            y -= 0.57 * inch
        for i in range(4):
            yy = self.height - 0.64 * inch - i * 0.57 * inch
            self.arrow(3.55 * inch, yy, 3.55 * inch, yy - 0.12 * inch, MUTED)


class AgentGraphDiagram(BaseDiagram):
    def draw(self) -> None:
        c = self.canv
        chief = (2.75 * inch, 3.0 * inch, 1.55 * inch, 0.42 * inch)
        self.box(*chief, "Chief Trading Agent", "coordinates", LAVENDER)
        nodes = [
            (0.1 * inch, 2.22 * inch, "Data Health", SKY),
            (1.65 * inch, 2.22 * inch, "Alpha Research", CREAM),
            (3.2 * inch, 2.22 * inch, "Risk Guardian", MINT),
            (4.75 * inch, 2.22 * inch, "Regime Scientist", SKY),
            (0.85 * inch, 1.28 * inch, "Experiment Auditor", LAVENDER),
            (2.55 * inch, 1.28 * inch, "Deployment Operator", MINT),
            (4.25 * inch, 1.28 * inch, "Anthropic Critic", CREAM),
        ]
        report = (2.55 * inch, 0.32 * inch, 1.75 * inch, 0.48 * inch)
        for x, y, title, fill in nodes:
            self.box(x, y, 1.28 * inch, 0.44 * inch, title, "", fill)
            self.arrow(chief[0] + chief[2] / 2, chief[1], x + 0.64 * inch, y + 0.44 * inch, MUTED)
            self.arrow(x + 0.64 * inch, y, report[0] + report[2] / 2, report[1] + report[3], BLUE)
        self.box(*report, "Technology Report Agent", "judge package", LAVENDER)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(3.55 * inch, 0.08 * inch, "Specialists inspect evidence, then converge into reproducible judge artifacts.")


class TraceDiagram(BaseDiagram):
    def draw(self) -> None:
        items = [
            ("Agent step", "objective + role", SKY),
            ("Tool call", "read-only evidence", LAVENDER),
            ("Blackboard write", "shared state", MINT),
            ("Handoff", "next specialist", CREAM),
            ("Trace replay", "36 spans", SKY),
        ]
        x = 0.15 * inch
        y = 1.42 * inch
        w = 1.22 * inch
        for idx, (title, body, fill) in enumerate(items):
            self.box(x, y, w, 0.65 * inch, title, body, fill)
            if idx < len(items) - 1:
                self.arrow(x + w, y + 0.33 * inch, x + w + 0.22 * inch, y + 0.33 * inch)
            x += w + 0.27 * inch
        self.box(1.25 * inch, 0.35 * inch, 4.7 * inch, 0.48 * inch, "Why it matters", "The reasoning path becomes inspectable without online model calls.", MINT)


class SafetyBoundaryDiagram(BaseDiagram):
    def draw(self) -> None:
        self.box(0.25 * inch, 1.8 * inch, 1.65 * inch, 0.65 * inch, "Model Agents", "read evidence, produce summaries", LAVENDER)
        self.box(2.25 * inch, 1.8 * inch, 1.65 * inch, 0.65 * inch, "Read-Only Tools", "project-local, no writes, no orders", MINT)
        self.box(4.25 * inch, 1.8 * inch, 1.55 * inch, 0.65 * inch, "Artifacts", "tickets, reports, dashboards", SKY)
        self.box(4.25 * inch, 0.65 * inch, 1.55 * inch, 0.65 * inch, "Human MT5 Use", "manual or explicitly armed path", CREAM)
        self.box(6.05 * inch, 0.65 * inch, 0.85 * inch, 0.65 * inch, "Broker", "MT5", GRAY_BG)
        self.arrow(1.9 * inch, 2.12 * inch, 2.25 * inch, 2.12 * inch)
        self.arrow(3.9 * inch, 2.12 * inch, 4.25 * inch, 2.12 * inch)
        self.arrow(5.8 * inch, 0.98 * inch, 6.05 * inch, 0.98 * inch, AMBER)
        self.canv.setStrokeColor(RED)
        self.canv.setLineWidth(1.2)
        self.canv.line(2.05 * inch, 0.35 * inch, 2.05 * inch, 2.8 * inch)
        self.canv.setFillColor(RED)
        self.canv.setFont("Helvetica-Bold", 7)
        self.canv.drawCentredString(2.05 * inch, 0.18 * inch, "hard safety boundary")


class CriticLoopDiagram(BaseDiagram):
    def draw(self) -> None:
        self.box(0.3 * inch, 1.42 * inch, 1.55 * inch, 0.62 * inch, "Evidence Pack", "backtests, risk, reports", SKY)
        self.box(2.25 * inch, 1.42 * inch, 1.65 * inch, 0.62 * inch, "Anthropic Critic", "independent review", CREAM)
        self.box(4.35 * inch, 1.42 * inch, 1.65 * inch, 0.62 * inch, "Actionable Feedback", "risk, overfit, clarity", MINT)
        self.box(2.25 * inch, 0.42 * inch, 1.65 * inch, 0.52 * inch, "Guardrail", "no trade approval", LAVENDER)
        self.arrow(1.85 * inch, 1.73 * inch, 2.25 * inch, 1.73 * inch)
        self.arrow(3.9 * inch, 1.73 * inch, 4.35 * inch, 1.73 * inch)
        self.arrow(3.07 * inch, 1.42 * inch, 3.07 * inch, 0.94 * inch, AMBER)


class DemoFlowDiagram(BaseDiagram):
    def draw(self) -> None:
        items = [
            ("Rehearse", "PASS"),
            ("Submit", "19/19 artifacts"),
            ("Dashboard", "open first"),
            ("Rubric", "100/100"),
            ("Red team", "8/8"),
            ("Proof cmds", "topology/trace/guardrails"),
        ]
        x = 0.08 * inch
        y = 1.35 * inch
        w = 1.05 * inch
        for idx, (title, body) in enumerate(items):
            self.box(x, y, w, 0.58 * inch, title, body, [MINT, SKY, LAVENDER, CREAM, MINT, SKY][idx])
            if idx < len(items) - 1:
                self.arrow(x + w, y + 0.29 * inch, x + w + 0.12 * inch, y + 0.29 * inch, BLUE)
            x += w + 0.17 * inch
        self.box(1.2 * inch, 0.35 * inch, 4.9 * inch, 0.42 * inch, "Default path", "offline, no OpenAI/Anthropic spend, no MT5 order authority", MINT)


def draw_centered(c: Any, text: str, x: float, y: float, max_width: float) -> None:
    if stringWidth(text, "Helvetica-Bold", 8) <= max_width:
        c.drawCentredString(x, y, text)
        return
    words = text.split()
    line = ""
    lines: list[str] = []
    for word in words:
        candidate = (line + " " + word).strip()
        if stringWidth(candidate, "Helvetica-Bold", 8) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    yy = y + 4
    for line in lines[:2]:
        c.drawCentredString(x, yy, line)
        yy -= 9


def draw_wrapped(c: Any, text: str, x: float, y: float, max_width: float, leading: float) -> None:
    words = text.split()
    line = ""
    lines: list[str] = []
    for word in words:
        candidate = (line + " " + word).strip()
        if stringWidth(candidate, "Helvetica", 6.5) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    yy = y
    for line in lines[:3]:
        c.drawString(x, yy, line)
        yy -= leading


def draw_page_frame(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.42 * inch, letter[0] - doc.rightMargin, 0.42 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 0.25 * inch, "Claude Agent Trader Technology Walkthrough")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.25 * inch, f"Page {doc.page}")
    canvas.restoreState()


def draw_cover_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, letter[1] - 0.18 * inch, letter[0], 0.18 * inch, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 0.25 * inch, "Generated from current Claude Agent Trader technology-prize artifacts.")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.25 * inch, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    main()
