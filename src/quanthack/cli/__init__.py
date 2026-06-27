"""Command-line entry points for QuanHack workflows."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


COMMANDS: dict[str, tuple[str, str]] = {
    "check-environment": (
        "quanthack.cli.check_environment",
        "Check Python and local environment readiness.",
    ),
    "project-status": ("quanthack.cli.project_status", "Show current project status."),
    "show-config": ("quanthack.cli.show_config", "Print configured hackathon settings."),
    "show-mode": (
        "quanthack.cli.show_competition_mode",
        "Show the current competition clock mode.",
    ),
    "show-prices": ("quanthack.cli.show_prices", "Inspect offline price CSV data."),
    "show-quotes": ("quanthack.cli.show_quotes", "Inspect offline quote CSV data."),
    "show-instruments": (
        "quanthack.cli.show_instruments",
        "Show official Syphonix instrument metadata.",
    ),
    "show-journal": ("quanthack.cli.show_journal", "Show recent dry-run journal records."),
    "show-positions": (
        "quanthack.cli.show_positions",
        "Reconstruct dry-run positions from the journal.",
    ),
    "risk-demo": ("quanthack.cli.risk_demo", "Run a simple risk engine decision demo."),
    "dry-run-trade": ("quanthack.cli.dry_run_trade", "Journal a manual dry-run trade."),
    "manual-ticket": (
        "quanthack.cli.manual_ticket",
        "Build a safe manual MT5 order ticket from a USD target.",
    ),
    "mt5-ticket-sheet": (
        "quanthack.cli.mt5_ticket_sheet",
        "Convert a profile signal snapshot CSV into manual MT5 tickets.",
    ),
    "strategy-dry-run": (
        "quanthack.cli.strategy_dry_run",
        "Run the simple strategy through risk and journal.",
    ),
    "configured-strategy-dry-run": (
        "quanthack.cli.configured_strategy_dry_run",
        "Run the configured strategy through risk and journal.",
    ),
    "data-strategy-dry-run": (
        "quanthack.cli.data_strategy_dry_run",
        "Run configured strategy from offline price CSV data.",
    ),
    "quality-data-strategy-dry-run": (
        "quanthack.cli.quality_data_strategy_dry_run",
        "Run quality, strategy, risk, and dry-run journaling.",
    ),
    "live-dry-run": (
        "quanthack.cli.live_dry_run",
        "Run read-only CSV/MT5 live dry-run monitoring.",
    ),
    "mt5-probe": (
        "quanthack.cli.mt5_probe",
        "Probe the read-only MT5 connection without placing orders.",
    ),
    "mt5-capture": (
        "quanthack.cli.mt5_capture",
        "Capture read-only MT5 quotes/account snapshots into CSV files.",
    ),
    "preflight": ("quanthack.cli.preflight", "Run local readiness checks."),
    "journal-summary": (
        "quanthack.cli.journal_summary",
        "Summarize the dry-run decision journal.",
    ),
    "html-report": (
        "quanthack.cli.build_html_report",
        "Build a standalone HTML dry-run report.",
    ),
    "dashboard": (
        "quanthack.cli.dashboard",
        "Run the local backtest and live-results dashboard.",
    ),
    "operator-dashboard": (
        "quanthack.cli.operator_dashboard",
        "Build a static live operator dashboard from profile artifacts.",
    ),
    "tech-prize-demo": (
        "quanthack.cli.tech_prize_demo",
        "Build an AgentSDK-style technology prize control-plane report.",
    ),
    "tech-prize-pack": (
        "quanthack.cli.tech_prize_pack",
        "Build a judge-ready technology prize demo pack.",
    ),
    "tech-prize-dashboard": (
        "quanthack.cli.tech_prize_dashboard",
        "Build a static HTML technology prize dashboard.",
    ),
    "tech-prize-workflow": (
        "quanthack.cli.tech_prize_workflow",
        "Run the local technology prize agent workflow trace.",
    ),
    "tech-prize-trace": (
        "quanthack.cli.tech_prize_trace",
        "Build an offline AgentSDK-style trace replay.",
    ),
    "tech-prize-guardrails": (
        "quanthack.cli.tech_prize_guardrails",
        "Run the technology prize AI/broker guardrail suite.",
    ),
    "tech-prize-topology": (
        "quanthack.cli.tech_prize_topology",
        "Analyze the technology prize AgentSDK topology.",
    ),
    "tech-prize-runbook": (
        "quanthack.cli.tech_prize_runbook",
        "Build a timed technology prize judge demo runbook.",
    ),
    "tech-prize-rehearse": (
        "quanthack.cli.tech_prize_rehearse",
        "Rehearse the safe offline technology-prize judge demo flow.",
    ),
    "tech-prize-rubric": (
        "quanthack.cli.tech_prize_rubric",
        "Score the system against the technology prize judging rubric.",
    ),
    "tech-prize-red-team": (
        "quanthack.cli.tech_prize_red_team",
        "Run skeptical judge red-team checks for the technology prize.",
    ),
    "tech-prize-submit": (
        "quanthack.cli.tech_prize_submit",
        "Build the final technology prize submission bundle.",
    ),
    "tech-prize-judge-packet": (
        "quanthack.cli.tech_prize_judge_packet",
        "Build a requirement-by-requirement technology prize judge packet.",
    ),
    "backtest": ("quanthack.cli.backtest", "Run one offline strategy backtest."),
    "portfolio-backtest": (
        "quanthack.cli.portfolio_backtest",
        "Run a shared-risk portfolio backtest.",
    ),
    "compare": ("quanthack.cli.compare_strategies", "Compare strategies by backtest."),
    "walk-forward": (
        "quanthack.cli.walk_forward",
        "Run chronological walk-forward evaluation.",
    ),
    "sweep": ("quanthack.cli.parameter_sweep", "Sweep momentum parameters."),
    "research-report": (
        "quanthack.cli.research_report",
        "Build the demo-ready research HTML report.",
    ),
    "hackathon-readiness": (
        "quanthack.cli.hackathon_readiness",
        "Build a competition go/no-go readiness report.",
    ),
    "strategy-demo": ("quanthack.cli.strategy_demo", "Inspect a strategy decision."),
    "ml-alpha-report": (
        "quanthack.cli.ml_alpha_report",
        "Evaluate the ML alpha signal on historical bars.",
    ),
    "time-series-report": (
        "quanthack.cli.time_series_report",
        "Classify trend/chop regimes with a Kalman-style filter.",
    ),
    "router-report": (
        "quanthack.cli.router_report",
        "Summarize router signal attribution from a backtest.",
    ),
    "portfolio-attribution-report": (
        "quanthack.cli.portfolio_attribution_report",
        "Summarize portfolio P&L by symbol, signal, hour, and side.",
    ),
    "experiment-leaderboard": (
        "quanthack.cli.experiment_leaderboard",
        "Rank walk-forward experiment summary CSVs.",
    ),
    "candidate-scorecard": (
        "quanthack.cli.candidate_scorecard",
        "Rank portfolio backtest output bundles by competition-style metrics.",
    ),
    "research-candidate-gate": (
        "quanthack.cli.research_candidate_gate",
        "Gate research comparison rows by data quality and promotion evidence.",
    ),
    "fold-complement": (
        "quanthack.cli.fold_complement",
        "Measure whether candidate folds complement a baseline strategy.",
    ),
    "fold-trade-attribution": (
        "quanthack.cli.fold_trade_attribution",
        "Attribute realized P&L by fold, symbol, signal, hour, and side.",
    ),
    "fold-regime-diagnostics": (
        "quanthack.cli.fold_regime_diagnostics",
        "Explain fixed-warmup folds with ex-ante regime diagnostics.",
    ),
    "fold-symbol-evidence": (
        "quanthack.cli.fold_symbol_evidence",
        "Simulate past-only symbol evidence gates from fold attribution.",
    ),
    "fold-symbol-evidence-sweep": (
        "quanthack.cli.fold_symbol_evidence_sweep",
        "Sweep past-only symbol evidence gate policies.",
    ),
    "signal-diagnostics": (
        "quanthack.cli.signal_diagnostics",
        "Screen signal sleeves with fast forward-return diagnostics.",
    ),
    "router-optimize": (
        "quanthack.cli.router_optimize",
        "Optimize alpha-router weights with allocator-aware backtests.",
    ),
    "relative-strength-optimize": (
        "quanthack.cli.relative_strength_optimize",
        "Optimize relative-strength parameters with portfolio backtests.",
    ),
    "volatility-squeeze-optimize": (
        "quanthack.cli.volatility_squeeze_optimize",
        "Optimize volatility-squeeze parameters with portfolio backtests.",
    ),
    "dual-squeeze-optimize": (
        "quanthack.cli.dual_squeeze_optimize",
        "Optimize dual-squeeze parameters with portfolio backtests.",
    ),
    "trend-pullback-optimize": (
        "quanthack.cli.trend_pullback_optimize",
        "Optimize trend-pullback parameters with portfolio backtests.",
    ),
    "fixing-reversal-optimize": (
        "quanthack.cli.fixing_reversal_optimize",
        "Optimize fixing-reversal parameters with portfolio backtests.",
    ),
    "liquidity-sweep-reversal-optimize": (
        "quanthack.cli.liquidity_sweep_reversal_optimize",
        "Optimize liquidity-sweep reversal parameters with portfolio backtests.",
    ),
    "kalman-trend-optimize": (
        "quanthack.cli.kalman_trend_optimize",
        "Optimize Kalman trend parameters with portfolio backtests.",
    ),
    "macd-momentum-optimize": (
        "quanthack.cli.macd_momentum_optimize",
        "Optimize MACD momentum parameters with portfolio backtests.",
    ),
    "multi-horizon-momentum-optimize": (
        "quanthack.cli.multi_horizon_momentum_optimize",
        "Optimize multi-horizon momentum parameters with portfolio backtests.",
    ),
    "autocorrelation-regime-optimize": (
        "quanthack.cli.autocorrelation_regime_optimize",
        "Optimize autocorrelation-regime parameters with portfolio backtests.",
    ),
    "session-momentum-optimize": (
        "quanthack.cli.session_momentum_optimize",
        "Optimize session-filtered momentum parameters with portfolio backtests.",
    ),
    "champion-ensemble-optimize": (
        "quanthack.cli.champion_ensemble_optimize",
        "Optimize champion-ensemble weights with portfolio backtests.",
    ),
    "cross-rate-optimize": (
        "quanthack.cli.cross_rate_optimize",
        "Optimize FX cross-rate reversion parameters with fast diagnostics.",
    ),
    "crypto-allocation-compare": (
        "quanthack.cli.crypto_allocation_compare",
        "Compare symbol-level crypto allocations across strategy sleeves.",
    ),
    "crypto-overlay-compare": (
        "quanthack.cli.crypto_overlay_compare",
        "Compare full-portfolio baselines against crypto overlay maps.",
    ),
    "crypto-overlay-sizing-compare": (
        "quanthack.cli.crypto_overlay_sizing_compare",
        "Compare crypto overlay size multipliers inside the full portfolio.",
    ),
    "crypto-overlay-fold-diagnostic": (
        "quanthack.cli.crypto_overlay_fold_diagnostic",
        "Export fold-level fills and attribution for one crypto overlay candidate.",
    ),
    "crypto-overlay-component-ablation": (
        "quanthack.cli.crypto_overlay_component_ablation",
        "Ablate components of the sized crypto overlay candidate.",
    ),
    "crypto-promotion-pipeline": (
        "quanthack.cli.crypto_promotion_pipeline",
        "Run crypto overlay evidence gates and a go/no-go promotion summary.",
    ),
    "crypto-fold-stability-optimize": (
        "quanthack.cli.crypto_fold_stability_optimize",
        "Optimize crypto overlay variants against fold concentration.",
    ),
    "asset-class-stability-optimize": (
        "quanthack.cli.asset_class_stability_optimize",
        "Optimize FX and metal exposure around crypto overlay profiles.",
    ),
    "deployment-profile-pack": (
        "quanthack.cli.deployment_profile_pack",
        "Build aggressive, conservative, and survival deployment profile evidence.",
    ),
    "deployment-profile-backtest": (
        "quanthack.cli.deployment_profile_backtest",
        "Backtest an exact slot from deployment_profile_pack.json.",
    ),
    "deployment-profile-snapshot": (
        "quanthack.cli.deployment_profile_snapshot",
        "Preview current targets from an exact deployment profile slot.",
    ),
    "deployment-profile-selector": (
        "quanthack.cli.deployment_profile_selector",
        "Compare fixed deployment profiles with an adaptive past-evidence selector.",
    ),
    "deployment-profile-selector-sweep": (
        "quanthack.cli.deployment_profile_selector_sweep",
        "Sweep adaptive deployment-profile selector policy settings.",
    ),
    "deployment-profile-recommendation": (
        "quanthack.cli.deployment_profile_recommendation",
        "Recommend the next deployment profile slot from selector-sweep evidence.",
    ),
    "deployment-profile-action-scan": (
        "quanthack.cli.deployment_profile_action_scan",
        "Scan when a deployment profile produces risk-approved actions.",
    ),
    "deployment-profile-session-attribution": (
        "quanthack.cli.deployment_profile_session_attribution",
        "Attribute deployment-profile P&L by UTC hour, symbol, signal, and side.",
    ),
    "deployment-profile-attribution-refine": (
        "quanthack.cli.deployment_profile_attribution_refine",
        "Test profile variants that scale weak attribution symbols.",
    ),
    "deployment-profile-session-gate-refine": (
        "quanthack.cli.deployment_profile_session_gate_refine",
        "Test profile variants that remove weak asset-class UTC hours.",
    ),
    "deployment-profile-symbol-gate-refine": (
        "quanthack.cli.deployment_profile_symbol_gate_refine",
        "Test profile variants that remove weak symbol-specific UTC hours.",
    ),
    "deployment-profile-challenger": (
        "quanthack.cli.deployment_profile_challenger",
        "Compare deployment profile challengers against a baseline.",
    ),
    "deployment-profile-robustness": (
        "quanthack.cli.deployment_profile_robustness",
        "Stress an exact deployment profile for cost and symbol dependence.",
    ),
    "deployment-profile-dependency-refine": (
        "quanthack.cli.deployment_profile_dependency_refine",
        "Test profile variants that reduce fragile single-symbol dependence.",
    ),
    "deployment-profile-dependency-replace": (
        "quanthack.cli.deployment_profile_dependency_replace",
        "Try replacing fragile dependency exposure with diversified refill symbols.",
    ),
    "deployment-profile-symbol-evidence-refine": (
        "quanthack.cli.deployment_profile_symbol_evidence_refine",
        "Sweep targeted symbol-evidence gates for fragile profile symbols.",
    ),
    "deployment-profile-symbol-universe-refine": (
        "quanthack.cli.deployment_profile_symbol_universe_refine",
        "Build a deployment profile slot from symbol-eligibility research.",
    ),
    "crypto-sleeve-compare": (
        "quanthack.cli.crypto_sleeve_compare",
        "Compare crypto alpha sleeves with full-sample and walk-forward evidence.",
    ),
    "validate-data": (
        "quanthack.cli.validate_market_data",
        "Validate price/quote CSV coverage and alignment.",
    ),
    "import-backtest-data": (
        "quanthack.cli.import_backtest_data",
        "Convert downloaded Parquet backtest data into QuanHack CSVs.",
    ),
    "archive-data-coverage": (
        "quanthack.cli.archive_data_coverage",
        "Inspect downloaded pricer zip symbol coverage without importing parquet.",
    ),
    "fetch-crypto-proxy-data": (
        "quanthack.cli.fetch_crypto_proxy_data",
        "Fetch research-only Binance crypto proxy data into QuanHack CSVs.",
    ),
    "merge-market-data": (
        "quanthack.cli.merge_market_data",
        "Merge and crop QuanHack price/quote CSV files.",
    ),
    "generate-sample-data": (
        "quanthack.cli.generate_sample_data",
        "Generate deterministic competition sample data.",
    ),
    "portfolio-compare": (
        "quanthack.cli.portfolio_compare",
        "Compare strategies on a shared-risk portfolio.",
    ),
    "strategy-attribution": (
        "quanthack.cli.strategy_attribution",
        "Write per-symbol P&L attribution for one or more strategies.",
    ),
    "asset-class-attribution": (
        "quanthack.cli.asset_class_attribution",
        "Group portfolio P&L attribution by asset class.",
    ),
    "symbol-eligibility-optimize": (
        "quanthack.cli.symbol_eligibility_optimize",
        "Optimize strategy symbol eligibility from attribution.",
    ),
    "strategy-map-optimize": (
        "quanthack.cli.strategy_map_optimize",
        "Optimize per-symbol strategy maps with shared-risk backtests.",
    ),
    "sizing-frontier": (
        "quanthack.cli.sizing_frontier",
        "Sweep strategy-map sizing caps with portfolio backtests.",
    ),
    "adaptive-strategy-select": (
        "quanthack.cli.adaptive_strategy_select",
        "Walk-forward select the best recent portfolio strategy.",
    ),
    "adaptive-strategy-policy-sweep": (
        "quanthack.cli.adaptive_strategy_policy_sweep",
        "Sweep adaptive strategy-selector policy settings.",
    ),
    "adaptive-strategy-oracle": (
        "quanthack.cli.adaptive_strategy_oracle",
        "Diagnose adaptive selector regret versus fold-level oracle.",
    ),
    "adaptive-handoff-diagnostic": (
        "quanthack.cli.adaptive_handoff_diagnostic",
        "Classify adaptive selector oracle misses with ex-ante regime features.",
    ),
    "portfolio-universe-scan": (
        "quanthack.cli.portfolio_universe_scan",
        "Rank diversified symbol baskets with portfolio backtests.",
    ),
    "portfolio-robustness": (
        "quanthack.cli.portfolio_robustness",
        "Run leave-one-symbol-out robustness checks for a portfolio strategy.",
    ),
    "portfolio-walk-forward": (
        "quanthack.cli.portfolio_walk_forward",
        "Validate portfolio basket selection on unseen windows.",
    ),
    "portfolio-fixed-warmup-walk-forward": (
        "quanthack.cli.portfolio_fixed_warmup_walk_forward",
        "Score a fixed portfolio on walk-forward windows after warmup history.",
    ),
    "portfolio-router-walk-forward": (
        "quanthack.cli.portfolio_router_walk_forward",
        "Tune alpha-router weights in portfolio walk-forward.",
    ),
}


def _print_help() -> None:
    print("QuanHack CLI")
    print("Usage: quanthack <command> [options]")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command = args[0]
    if command not in COMMANDS:
        available = ", ".join(COMMANDS)
        raise SystemExit(f"Unknown command '{command}'. Available commands: {available}")

    module_name, _ = COMMANDS[command]
    module = importlib.import_module(module_name)
    module.main(args[1:])
