from __future__ import annotations

import argparse
from collections.abc import Sequence

from quanthack.agents.anthropic_critic import (
    run_guarded_anthropic_critic,
    write_anthropic_critic_report,
)
from quanthack.agents.sdk_runner import (
    run_guarded_agents_sdk_demo,
    write_agents_sdk_runner_report,
)
from quanthack.agents.technology_prize import (
    run_local_technology_prize_demo,
    write_technology_prize_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a technology-prize AgentSDK control-plane report."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root used to resolve output artifacts.",
    )
    parser.add_argument(
        "--output",
        default="outputs/reports/technology_prize_agent_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/technology_prize_agent_report.json",
        help="JSON trace output path. Use an empty string to disable.",
    )
    parser.add_argument(
        "--sdk-package",
        default="agents",
        help="Python package name to probe for Agents SDK compatibility.",
    )
    parser.add_argument(
        "--run-sdk",
        action="store_true",
        help="Also build the guarded Agents SDK Runner report.",
    )
    parser.add_argument(
        "--allow-online-sdk",
        action="store_true",
        help="Allow the SDK Runner to make online model calls when OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--sdk-model",
        default="gpt-5.5",
        help="Model used only when --run-sdk and --allow-online-sdk are both set.",
    )
    parser.add_argument(
        "--sdk-max-turns",
        type=int,
        default=6,
        help="Max SDK runner turns for the optional online demo.",
    )
    parser.add_argument(
        "--sdk-runner-output",
        default="outputs/reports/technology_prize_sdk_runner.md",
        help="Markdown output for the guarded SDK runner report.",
    )
    parser.add_argument(
        "--run-anthropic-critic",
        action="store_true",
        help="Also build the guarded Anthropic critic report.",
    )
    parser.add_argument(
        "--allow-online-anthropic",
        action="store_true",
        help="Allow the Anthropic critic to make online model calls when ANTHROPIC_API_KEY is set.",
    )
    parser.add_argument(
        "--anthropic-model",
        default="claude-sonnet-4-6",
        help="Model used only when --run-anthropic-critic and --allow-online-anthropic are both set.",
    )
    parser.add_argument(
        "--anthropic-max-tokens",
        type=int,
        default=900,
        help="Max tokens for the optional Anthropic critic.",
    )
    parser.add_argument(
        "--anthropic-output",
        default="outputs/reports/technology_prize_anthropic_critic.md",
        help="Markdown output for the guarded Anthropic critic report.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    report = run_local_technology_prize_demo(
        project_root=args.project_root,
        package_name=args.sdk_package,
    )
    json_output = args.json_output or None
    write_technology_prize_report(
        report,
        markdown_path=args.output,
        json_path=json_output,
    )
    for line in report.summary_lines():
        print(line)
    print(f"  Markdown: {args.output}")
    if json_output is not None:
        print(f"  JSON: {json_output}")

    if args.run_sdk:
        runner_result = run_guarded_agents_sdk_demo(
            project_root=args.project_root,
            model=args.sdk_model,
            max_turns=args.sdk_max_turns,
            allow_online_sdk=args.allow_online_sdk,
        )
        write_agents_sdk_runner_report(
            runner_result,
            args.sdk_runner_output,
        )
        for line in runner_result.summary_lines():
            print(line)
        print(f"  SDK Runner Markdown: {args.sdk_runner_output}")

    if args.run_anthropic_critic:
        critic_result = run_guarded_anthropic_critic(
            project_root=args.project_root,
            model=args.anthropic_model,
            max_tokens=args.anthropic_max_tokens,
            allow_online_anthropic=args.allow_online_anthropic,
        )
        write_anthropic_critic_report(
            critic_result,
            args.anthropic_output,
        )
        for line in critic_result.summary_lines():
            print(line)
        print(f"  Anthropic Critic Markdown: {args.anthropic_output}")


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))
