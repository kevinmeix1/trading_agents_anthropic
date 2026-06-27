from __future__ import annotations

from quanthack.backtesting.competition_score import CompetitionMetrics, RiskDisciplineReport


def print_competition_view(
    *,
    metrics: CompetitionMetrics,
    risk_discipline: RiskDisciplineReport,
) -> None:
    cap_text = (
        f"yes, max Sharpe rank {metrics.sharpe_rank_cap:.0f}"
        if metrics.sharpe_rank_is_capped
        else "no"
    )
    trade_text = "yes" if metrics.sharpe_prize_trade_count_met else "no"
    review_text = "yes" if risk_discipline.compliance_review_required else "no"

    print("Competition scoring view")
    print(f"  Official return: {metrics.return_pct:.3%}")
    print(f"  Official max drawdown: {metrics.max_drawdown_pct:.3%}")
    print(f"  Official 15m Sharpe: {metrics.sharpe_15m:.3f}")
    print(f"  15m return observations: {metrics.return_observations}")
    print(f"  Sharpe rank cap active: {cap_text}")
    print(f"  Trades: {metrics.trade_count} (30-trade Sharpe prize minimum: {trade_text})")
    print(f"  Risk discipline score: {risk_discipline.score}/100")
    print(f"  Compliance review required: {review_text}")
    if risk_discipline.breaches:
        print("  Risk discipline breaches:")
        for breach in risk_discipline.breaches[:5]:
            print(
                f"    {breach.severity.value} | {breach.rule_id} | "
                f"{breach.details} | points={breach.penalty_points}"
            )

