from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from quanthack.agents.sdk_bridge import (
    AgentsSdkUnavailableError,
    create_agents_sdk_app,
)
from quanthack.agents.anthropic_critic import run_guarded_anthropic_critic
from quanthack.agents.demo_pack import (
    DEFAULT_SOURCE_FILES,
    build_technology_prize_demo_pack,
    write_technology_prize_demo_pack,
)
from quanthack.agents.demo_rehearsal import (
    build_technology_prize_demo_rehearsal,
    write_technology_prize_demo_rehearsal,
)
from quanthack.agents.demo_director import build_judge_demo_runbook, write_judge_demo_runbook
from quanthack.agents.guardrails import build_agent_guardrail_suite, write_agent_guardrail_suite
from quanthack.agents.judge_packet import (
    build_technology_prize_judge_packet,
    write_technology_prize_judge_packet,
)
from quanthack.agents.rubric import (
    build_technology_prize_rubric,
    write_technology_prize_rubric,
)
from quanthack.agents.red_team import (
    build_technology_prize_red_team_report,
    write_technology_prize_red_team_report,
)
from quanthack.agents.sdk_runner import run_guarded_agents_sdk_demo
from quanthack.agents.submission_bundle import (
    build_technology_prize_submission_bundle,
    write_technology_prize_submission_bundle,
)
from quanthack.agents.technology_prize import (
    DEFAULT_EVIDENCE_FILES,
    run_local_technology_prize_demo,
    write_technology_prize_report,
)
from quanthack.agents.topology import build_agent_topology_report, write_agent_topology_report
from quanthack.agents.trace_replay import build_agent_trace_replay, write_agent_trace_replay
from quanthack.agents.workflow import run_local_agent_workflow, write_local_agent_workflow
from quanthack.reporting.technology_prize_dashboard import build_technology_prize_dashboard


class TechnologyPrizeAgentsTest(TestCase):
    def test_local_demo_summarizes_present_and_missing_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            leaderboard = root / "leaderboard.csv"
            leaderboard.write_text(
                "rank,label,score,compounded_return_pct\n"
                "1,current_top,1.5,0.052\n",
                encoding="utf-8",
            )
            report = run_local_technology_prize_demo(
                project_root=root,
                package_name="definitely_missing_agents_package",
                evidence_files=(
                    ("adaptive leaderboard", "leaderboard.csv"),
                    ("policy sweep", "missing.csv"),
                ),
            )

        self.assertEqual(len(report.agents), 9)
        self.assertFalse(report.sdk_status.ready)
        self.assertEqual(len(report.providers), 2)
        self.assertEqual(report.artifacts[0].status, "OK")
        self.assertEqual(report.artifacts[1].status, "MISSING")
        self.assertTrue(any(event.status == "WARN" for event in report.trace))
        self.assertIn("current_top", report.artifacts[0].summary)

    def test_report_writes_markdown_and_json_trace(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "leaderboard.csv").write_text(
                "rank,label,score,compounded_return_pct\n"
                "1,current_top,1.5,0.052\n",
                encoding="utf-8",
            )
            report = run_local_technology_prize_demo(
                project_root=root,
                package_name="definitely_missing_agents_package",
                evidence_files=(("adaptive leaderboard", "leaderboard.csv"),),
            )
            markdown_path = root / "report.md"
            json_path = root / "report.json"
            write_technology_prize_report(
                report,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertIn("AgentSDK Technology Prize", markdown)
        self.assertIn("```mermaid", markdown)
        self.assertIn("Anthropic Critic Agent", markdown)
        self.assertIn('"sdk_status"', json_text)
        self.assertIn('"providers"', json_text)

    def test_sdk_bridge_can_build_fake_agents_sdk_graph(self) -> None:
        class FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.name = kwargs["name"]

        class FakeSdk:
            Agent = FakeAgent

            @staticmethod
            def function_tool(func: object) -> object:
                return func

        def importer(name: str) -> object:
            self.assertEqual(name, "agents")
            return FakeSdk

        app = create_agents_sdk_app(importer=importer)

        self.assertEqual(app.chief_agent.name, "Chief Trading Agent")
        self.assertIn("Data Health Agent", app.specialist_agents)
        self.assertIn("Regime Scientist Agent", app.specialist_agents)
        self.assertIn("Experiment Auditor Agent", app.specialist_agents)
        self.assertIn("Anthropic Critic Agent", app.specialist_agents)
        self.assertIn("summarize_research_artifacts", app.tools)
        self.assertIn("validate_market_data_summary", app.tools)
        self.assertIn("summarize_mt5_ticket_sheet", app.tools)
        self.assertIn("summarize_operator_dashboard_sources", app.tools)
        self.assertIn("build_technology_prize_judge_packet", app.tools)
        self.assertIn("run_agent_guardrail_suite", app.tools)
        self.assertIn("analyze_agent_topology", app.tools)
        self.assertIn("build_judge_demo_runbook", app.tools)
        self.assertIn("replay_agent_trace", app.tools)
        self.assertGreater(len(app.chief_agent.kwargs["handoffs"]), 0)

    def test_sdk_bridge_tools_are_read_only_project_summaries(self) -> None:
        class FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.name = kwargs["name"]

        class FakeSdk:
            Agent = FakeAgent

            @staticmethod
            def function_tool(func: object) -> object:
                return func

        def importer(_: str) -> object:
            return FakeSdk

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            (root / "tickets.csv").write_text(
                "symbol,side,status\nEURUSD,BUY,READY\nBTCUSD,SELL,NEEDS_CONTRACT_SPEC\n",
                encoding="utf-8",
            )
            (root / "summary.csv").write_text(
                "positive_fold_fraction,active_fold_fraction,active_positive_fold_fraction,"
                "non_negative_fold_fraction,median_active_test_return_pct,"
                "worst_test_drawdown_pct,average_risk_discipline_score,"
                "total_evaluation_fills,compounded_test_return_pct,strategies,symbols,folds\n"
                "0.5,0.6,0.7,0.8,0.01,0.02,100,10,0.05,demo,EURUSD,2\n",
                encoding="utf-8",
            )
            (root / "prices.csv").write_text(
                "timestamp,symbol,close\n"
                "2026-06-01T00:00:00+00:00,EURUSD,1.1\n"
                "2026-06-01T00:05:00+00:00,EURUSD,1.2\n",
                encoding="utf-8",
            )
            (root / "quotes.csv").write_text(
                "timestamp,symbol,bid,ask\n"
                "2026-06-01T00:00:00+00:00,EURUSD,1.099,1.101\n"
                "2026-06-01T00:05:00+00:00,EURUSD,1.199,1.201\n",
                encoding="utf-8",
            )

            app = create_agents_sdk_app(project_root=root, importer=importer)
            csv_summary = app.tools["summarize_csv"]("sample.csv", limit=1)
            ticket_summary = app.tools["summarize_mt5_ticket_sheet"]("tickets.csv")
            leaderboard = app.tools["summarize_experiment_leaderboard"]("summary.csv")
            health = app.tools["validate_market_data_summary"]("prices.csv", "quotes.csv")
            dashboard_sources = app.tools["summarize_operator_dashboard_sources"](
                ticket_csv="tickets.csv"
            )
            judge_packet = app.tools["build_technology_prize_judge_packet"]()
            guardrails = app.tools["run_agent_guardrail_suite"]()
            topology = app.tools["analyze_agent_topology"]()
            runbook = app.tools["build_judge_demo_runbook"]()
            trace = app.tools["replay_agent_trace"]()

            with self.assertRaises(ValueError):
                app.tools["summarize_csv"]("../outside.csv")

        self.assertEqual(csv_summary["rows"], [{"a": "1", "b": "2"}])
        self.assertEqual(ticket_summary["statuses"]["READY"], 1)
        self.assertFalse(ticket_summary["has_order_authority"])
        self.assertEqual(leaderboard["count"], 1)
        self.assertEqual(health["status"], "OK")
        self.assertFalse(dashboard_sources["ready"])
        self.assertFalse(dashboard_sources["has_order_authority"])
        self.assertIn(judge_packet["status"], {"PASS", "WARN", "FAIL"})
        self.assertFalse(judge_packet["has_order_authority"])
        self.assertIn("online_model_calls_default_to_off", judge_packet)
        self.assertIn(guardrails["status"], {"PASS", "WARN", "FAIL"})
        self.assertGreaterEqual(len(guardrails["checks"]), 8)
        self.assertEqual(topology["status"], "PASS")
        self.assertGreaterEqual(len(topology["tool_coverage"]), 12)
        self.assertIn(runbook["status"], {"PASS", "WARN", "FAIL"})
        self.assertGreaterEqual(len(runbook["steps"]), 6)
        self.assertIn(trace["status"], {"PASS", "WARN"})
        self.assertGreaterEqual(len(trace["spans"]), 30)

    def test_sdk_bridge_reports_missing_sdk_cleanly(self) -> None:
        def importer(_: str) -> object:
            raise ModuleNotFoundError("missing")

        with self.assertRaises(AgentsSdkUnavailableError):
            create_agents_sdk_app(importer=importer)

    def test_guarded_runner_skips_without_explicit_online_arm(self) -> None:
        result = run_guarded_agents_sdk_demo(
            allow_online_sdk=False,
            environ={},
        )

        self.assertEqual(result.status, "SKIPPED")
        self.assertFalse(result.ran)
        self.assertIn("disabled", result.error)

    def test_guarded_runner_blocks_without_openai_api_key(self) -> None:
        result = run_guarded_agents_sdk_demo(
            allow_online_sdk=True,
            environ={},
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("OPENAI_API_KEY", result.error)

    def test_guarded_runner_can_use_fake_sdk_runner_when_armed(self) -> None:
        class FakeAgent:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.name = kwargs["name"]

        class FakeResult:
            final_output = "judge brief"

        class FakeRunConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class FakeRunner:
            @staticmethod
            def run_sync(
                starting_agent: object,
                prompt: str,
                *,
                max_turns: int,
                run_config: object,
            ) -> FakeResult:
                self.assertEqual(getattr(starting_agent, "name"), "Chief Trading Agent")
                self.assertIn("technology prize", prompt)
                self.assertEqual(max_turns, 3)
                self.assertIsInstance(run_config, FakeRunConfig)
                return FakeResult()

        class FakeSdk:
            Agent = FakeAgent
            Runner = FakeRunner
            RunConfig = FakeRunConfig

            @staticmethod
            def function_tool(func: object) -> object:
                return func

        def importer(name: str) -> object:
            self.assertEqual(name, "agents")
            return FakeSdk

        result = run_guarded_agents_sdk_demo(
            model="fake-model",
            max_turns=3,
            allow_online_sdk=True,
            environ={"OPENAI_API_KEY": "test"},
            importer=importer,
        )

        self.assertTrue(result.ran)
        self.assertEqual(result.final_output, "judge brief")
        self.assertIn("summarize_csv", result.tool_names)
        self.assertIn("Anthropic Critic Agent", result.specialist_agents)

    def test_anthropic_critic_skips_without_explicit_online_arm(self) -> None:
        result = run_guarded_anthropic_critic(
            allow_online_anthropic=False,
            environ={"ANTHROPIC_API_KEY": "test"},
        )

        self.assertEqual(result.status, "SKIPPED")
        self.assertFalse(result.ran)
        self.assertIn("disabled", result.error)

    def test_anthropic_critic_blocks_without_api_key(self) -> None:
        result = run_guarded_anthropic_critic(
            allow_online_anthropic=True,
            environ={},
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertIn("ANTHROPIC_API_KEY", result.error)

    def test_anthropic_critic_can_use_fake_client_when_armed(self) -> None:
        case = self

        class FakeBlock:
            text = "critic brief"

        class FakeMessage:
            content = [FakeBlock()]

        class FakeMessages:
            def create(self, **kwargs: object) -> FakeMessage:
                self_model = kwargs["model"]
                case.assertEqual(self_model, "fake-claude")
                case.assertIn("technology prize", str(kwargs["messages"]).lower())
                case.assertEqual(kwargs["max_tokens"], 123)
                return FakeMessage()

        class FakeClient:
            def __init__(self, *, api_key: str) -> None:
                case.assertEqual(api_key, "test")
                self.messages = FakeMessages()

        class FakeAnthropic:
            Anthropic = FakeClient

        def importer(name: str) -> object:
            case.assertEqual(name, "anthropic")
            return FakeAnthropic

        result = run_guarded_anthropic_critic(
            model="fake-claude",
            max_tokens=123,
            allow_online_anthropic=True,
            environ={"ANTHROPIC_API_KEY": "test"},
            importer=importer,
        )

        self.assertTrue(result.ran)
        self.assertEqual(result.final_output, "critic brief")

    def test_demo_pack_writes_judge_markdown_and_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)

            pack = build_technology_prize_demo_pack(project_root=root)
            markdown_path = root / "outputs" / "reports" / "pack.md"
            json_path = root / "outputs" / "reports" / "pack.json"
            write_technology_prize_demo_pack(
                pack,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(pack.overall_status, "PASS")
        self.assertIn("Technology Prize Demo Pack", markdown)
        self.assertIn("Scorecard", markdown)
        self.assertIn("quanthack tech-prize-pack", markdown)
        self.assertIn("Why This Is AI-Native", markdown)
        self.assertIn("Local workflow steps", markdown)
        self.assertIn('"innovation_claims"', json_text)
        self.assertIn('"local_workflow"', json_text)
        self.assertIn('"agent_guardrails"', json_text)
        self.assertIn('"agent_topology"', json_text)
        self.assertIn('"agent_trace_replay"', json_text)
        self.assertIn('"demo_runbook"', json_text)
        self.assertIn('"overall_status": "PASS"', json_text)
        self.assertGreaterEqual(len(pack.innovation_claims), 8)
        self.assertGreaterEqual(len(pack.local_workflow.steps), 9)
        self.assertEqual(pack.agent_guardrails.status, "PASS")
        self.assertEqual(pack.agent_topology.status, "PASS")
        self.assertEqual(pack.agent_trace_replay.status, "PASS")
        self.assertEqual(pack.demo_runbook.status, "PASS")
        self.assertEqual(pack.sdk_runner.status, "SKIPPED")
        self.assertEqual(pack.anthropic_critic.status, "SKIPPED")
        self.assertIn("Requirement-Level Judge Packet", markdown)
        self.assertIn("Executable Guardrails", markdown)
        self.assertIn("Self-Validating Agent Topology", markdown)
        self.assertIn("Offline Agent Trace Replay", markdown)
        self.assertIn("Judge Demo Director", markdown)
        self.assertIn("quanthack tech-prize-judge-packet", markdown)
        self.assertIn("Scored Technology Rubric", markdown)
        self.assertIn("Skeptical Judge Red Team", markdown)
        self.assertIn("quanthack tech-prize-red-team", markdown)

    def test_local_agent_workflow_records_steps_and_blackboard(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_evidence_files(root)
            workflow = run_local_agent_workflow(project_root=root)
            markdown_path = root / "workflow.md"
            json_path = root / "workflow.json"
            write_local_agent_workflow(
                workflow,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(workflow.status, "PASS")
        self.assertGreaterEqual(len(workflow.steps), 9)
        self.assertGreaterEqual(len(workflow.blackboard), 9)
        self.assertIn("Local Agent Workflow", markdown)
        self.assertIn('"blackboard"', json_text)

    def test_agent_guardrail_suite_records_safety_checks(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            suite = build_agent_guardrail_suite(project_root=root)
            markdown_path = root / "guardrails.md"
            json_path = root / "guardrails.json"
            write_agent_guardrail_suite(
                suite,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(suite.status, "PASS")
        self.assertGreaterEqual(len(suite.checks), 8)
        self.assertIn("Read-only AgentSDK tools", markdown)
        self.assertIn("No broker order authority", markdown)
        self.assertIn('"checks"', json_text)

    def test_agent_topology_report_validates_graph_integrity(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = build_agent_topology_report()
            markdown_path = root / "topology.md"
            json_path = root / "topology.json"
            write_agent_topology_report(
                report,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(report.status, "PASS")
        self.assertGreaterEqual(len(report.tool_coverage), 12)
        self.assertGreaterEqual(len(report.handoffs), 8)
        self.assertIn("All declared tools are used", markdown)
        self.assertIn("Handoffs", markdown)
        self.assertIn('"tool_coverage"', json_text)

    def test_agent_trace_replay_exports_span_model(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_evidence_files(root)
            replay = build_agent_trace_replay(project_root=root)
            markdown_path = root / "trace.md"
            json_path = root / "trace.json"
            write_agent_trace_replay(
                replay,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(replay.status, "PASS")
        self.assertGreaterEqual(len(replay.spans), 30)
        self.assertIn("AgentSDK Trace Replay", markdown)
        self.assertIn("tool_call", markdown)
        self.assertIn('"spans"', json_text)

    def test_judge_demo_runbook_builds_timed_talk_track(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            _write_demo_pack_report_files(root)

            runbook = build_judge_demo_runbook(project_root=root)
            markdown_path = root / "runbook.md"
            json_path = root / "runbook.json"
            write_judge_demo_runbook(
                runbook,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(runbook.status, "PASS")
        self.assertGreaterEqual(len(runbook.steps), 6)
        self.assertGreaterEqual(len(runbook.risk_notes), 3)
        self.assertIn("3-Minute Demo Flow", markdown)
        self.assertIn("quanthack tech-prize-topology", markdown)
        self.assertIn('"steps"', json_text)

    def test_submission_bundle_builds_final_artifact_index(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            bundle = build_technology_prize_submission_bundle(project_root=root)
            markdown_path = root / "outputs" / "reports" / "submission.md"
            json_path = root / "outputs" / "reports" / "submission.json"
            write_technology_prize_submission_bundle(
                bundle,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(bundle.status, "PASS")
        self.assertEqual(bundle.demo_pack_status, "PASS")
        self.assertEqual(bundle.judge_packet_status, "PASS")
        self.assertEqual(bundle.rubric_status, "PASS")
        self.assertEqual(bundle.red_team_status, "PASS")
        self.assertEqual(bundle.rehearsal_status, "PASS")
        self.assertTrue(all(artifact.present for artifact in bundle.artifact_manifest))
        self.assertIn("Technology Prize Submission Bundle", markdown)
        self.assertIn("technology_prize_dashboard.html", markdown)
        self.assertIn("technology_prize_demo_rehearsal.md", markdown)
        self.assertIn("technology_prize_rubric.md", markdown)
        self.assertIn("technology_prize_red_team.md", markdown)
        self.assertIn('"artifact_manifest"', json_text)
        self.assertIn('"rubric_status": "PASS"', json_text)
        self.assertIn('"red_team_status": "PASS"', json_text)
        self.assertIn('"rehearsal_status": "PASS"', json_text)

    def test_judge_packet_verifies_prize_requirements(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            _write_demo_pack_report_files(root)

            packet = build_technology_prize_judge_packet(project_root=root)
            markdown_path = root / "judge.md"
            json_path = root / "judge.json"
            write_technology_prize_judge_packet(
                packet,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(packet.status, "PASS")
        self.assertFalse(packet.has_order_authority)
        self.assertTrue(packet.online_model_calls_default_to_off)
        self.assertEqual(packet.guardrails.status, "PASS")
        self.assertEqual(packet.topology.status, "PASS")
        self.assertEqual(len(packet.requirements), 12)
        self.assertIn("AgentSDK-centered control plane", markdown)
        self.assertIn("Executable AI guardrails", markdown)
        self.assertIn("Validated AgentSDK topology", markdown)
        self.assertIn("Offline AgentSDK trace replay", markdown)
        self.assertIn("Judge demo readiness", markdown)
        self.assertIn("Additional Anthropic Credits", markdown)
        self.assertIn("Why this is AI-native", markdown)
        self.assertIn('"requirements"', json_text)

    def test_technology_prize_rubric_scores_judging_axes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            build_technology_prize_submission_bundle(project_root=root)

            rubric = build_technology_prize_rubric(project_root=root)
            markdown_path = root / "rubric.md"
            json_path = root / "rubric.json"
            write_technology_prize_rubric(
                rubric,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(rubric.status, "PASS")
        self.assertEqual(rubric.total_score, 100)
        self.assertEqual(rubric.max_score, 100)
        self.assertEqual(len(rubric.criteria), 6)
        self.assertIn("Best Use Of AgentSDK", markdown)
        self.assertIn("Additional Anthropic Credits", markdown)
        self.assertIn("AI-Native Innovation", markdown)
        self.assertIn("Responsible Trading Safety", markdown)
        self.assertIn('"total_score": 100', json_text)
        self.assertIn('"status": "PASS"', json_text)

    def test_technology_prize_red_team_answers_skeptical_judge_questions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)
            build_technology_prize_submission_bundle(project_root=root)

            report = build_technology_prize_red_team_report(project_root=root)
            markdown_path = root / "red_team.md"
            json_path = root / "red_team.json"
            write_technology_prize_red_team_report(
                report,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(report.status, "PASS")
        self.assertGreaterEqual(len(report.challenges), 8)
        self.assertTrue(all(challenge.status == "PASS" for challenge in report.challenges))
        self.assertIn("Technology Prize Red-Team Report", markdown)
        self.assertIn("Can the AI agent secretly place MT5 trades", markdown)
        self.assertIn("Real AgentSDK Architecture", markdown)
        self.assertIn("Anthropic Credits Have A Real Role", markdown)
        self.assertIn("AI-Native, Not Script-Native", markdown)
        self.assertIn('"challenges"', json_text)
        self.assertIn('"status": "PASS"', json_text)

    def test_technology_prize_demo_rehearsal_regenerates_demo_flow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)

            rehearsal = build_technology_prize_demo_rehearsal(project_root=root)
            markdown_path = root / "rehearsal.md"
            json_path = root / "rehearsal.json"
            write_technology_prize_demo_rehearsal(
                rehearsal,
                markdown_path=markdown_path,
                json_path=json_path,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertEqual(rehearsal.status, "PASS")
        self.assertTrue(rehearsal.offline_only)
        self.assertGreaterEqual(len(rehearsal.steps), 8)
        self.assertTrue(all(step.status == "PASS" for step in rehearsal.steps))
        self.assertIn("Technology Prize Demo Rehearsal", markdown)
        self.assertIn("quanthack tech-prize-rehearse", markdown)
        self.assertIn("quanthack tech-prize-red-team", markdown)
        self.assertIn('"offline_only": true', json_text)
        self.assertIn('"status": "PASS"', json_text)

    def test_technology_prize_dashboard_renders_pack(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_demo_pack_source_files(root)
            _write_demo_pack_evidence_files(root)

            pack = build_technology_prize_demo_pack(project_root=root)
            dashboard = build_technology_prize_dashboard(pack)

        self.assertIn("<!doctype html>", dashboard.html)
        self.assertIn("QuanHack Technology Prize Dashboard", dashboard.html)
        self.assertIn("Agent Graph", dashboard.html)
        self.assertIn("Why This Is AI-Native", dashboard.html)
        self.assertIn("Executable Workflow", dashboard.html)
        self.assertIn("Guardrails", dashboard.html)
        self.assertIn("Topology", dashboard.html)
        self.assertIn("Trace Replay", dashboard.html)
        self.assertIn("Demo Runbook", dashboard.html)
        self.assertIn("Scorecard", dashboard.html)
        self.assertIn("Judge Packet", dashboard.html)
        self.assertIn("quanthack tech-prize-pack", dashboard.html)
        self.assertIn("quanthack tech-prize-guardrails", dashboard.html)
        self.assertIn("quanthack tech-prize-topology", dashboard.html)
        self.assertIn("quanthack tech-prize-trace", dashboard.html)
        self.assertIn("quanthack tech-prize-runbook", dashboard.html)
        self.assertIn("quanthack tech-prize-judge-packet", dashboard.html)
        self.assertIn("No MT5 order-placement tool", dashboard.html)


def _write_demo_pack_source_files(root: Path) -> None:
    for _, relative_path in DEFAULT_SOURCE_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.endswith("sdk_bridge.py"):
            text = "resolved.relative_to(root)\npath is outside project root\n"
        elif relative_path.endswith("sdk_runner.py"):
            text = "allow_online_sdk\nOPENAI_API_KEY\n"
        elif relative_path.endswith("anthropic_critic.py"):
            text = "allow_online_anthropic\nANTHROPIC_API_KEY\n"
        else:
            text = "# source\n"
        path.write_text(text, encoding="utf-8")


def _write_demo_pack_evidence_files(root: Path) -> None:
    for name, relative_path in DEFAULT_EVIDENCE_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text('{"profiles": [], "data_source": "unit"}\n', encoding="utf-8")
            continue
        if name == "adaptive leaderboard":
            path.write_text(
                "rank,label,score,compounded_return_pct\n1,current_top,1.0,0.05\n",
                encoding="utf-8",
            )
        elif name == "policy sweep":
            path.write_text(
                "promotion_status,selector_score,live_ready\nPAPER_ONLY,80,no\n",
                encoding="utf-8",
            )
        elif name == "oracle summary":
            path.write_text(
                "selected_was_oracle_fraction,total_regret_pct\n0.5,0.01\n",
                encoding="utf-8",
            )
        elif name == "handoff diagnostic":
            path.write_text("fold,diagnosis\n1,NO_REGRET\n", encoding="utf-8")
        elif name == "manual MT5 ticket sheet":
            path.write_text("symbol,side,volume\nEURUSD,BUY,0.1\n", encoding="utf-8")
        else:
            path.write_text("timestamp,equity\n2026-06-01T00:00:00+00:00,1000000\n", encoding="utf-8")


def _write_demo_pack_report_files(root: Path) -> None:
    report_paths = (
        "outputs/reports/technology_prize_agent_report.md",
        "outputs/reports/technology_prize_agent_report.json",
        "outputs/reports/technology_prize_sdk_runner.md",
        "outputs/reports/technology_prize_anthropic_critic.md",
        "outputs/reports/technology_prize_workflow.md",
        "outputs/reports/technology_prize_workflow.json",
        "outputs/reports/technology_prize_guardrails.md",
        "outputs/reports/technology_prize_guardrails.json",
        "outputs/reports/technology_prize_topology.md",
        "outputs/reports/technology_prize_topology.json",
        "outputs/reports/technology_prize_trace_replay.md",
        "outputs/reports/technology_prize_trace_replay.json",
        "outputs/reports/technology_prize_demo_runbook.md",
        "outputs/reports/technology_prize_demo_runbook.json",
        "outputs/reports/technology_prize_demo_pack.md",
        "outputs/reports/technology_prize_demo_pack.json",
        "outputs/reports/technology_prize_dashboard.html",
    )
    for relative_path in report_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("report\n", encoding="utf-8")
