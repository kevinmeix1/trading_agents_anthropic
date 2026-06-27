from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PNL_EPSILON = 1e-12


@dataclass(frozen=True)
class FoldSymbolEvidencePolicy:
    lookback_folds: int = 2
    min_prior_active_folds: int = 1
    min_prior_realized_events: int = 1
    min_prior_pnl_usd: float = 0.0
    min_prior_win_rate: float = 0.0
    allow_without_history: bool = True

    def __post_init__(self) -> None:
        if self.lookback_folds < 1:
            raise ValueError("lookback_folds must be at least 1")
        if self.min_prior_active_folds < 0:
            raise ValueError("min_prior_active_folds cannot be negative")
        if self.min_prior_realized_events < 0:
            raise ValueError("min_prior_realized_events cannot be negative")
        if not 0 <= self.min_prior_win_rate <= 1:
            raise ValueError("min_prior_win_rate must be between 0 and 1")


@dataclass(frozen=True)
class FoldSymbolContribution:
    fold: int
    fold_return_pct: float
    symbol: str
    realized_pnl_usd: float
    fills: int
    realized_events: int
    wins: int
    losses: int

    @property
    def active(self) -> bool:
        return self.fills > 0 or self.realized_events > 0 or abs(self.realized_pnl_usd) > PNL_EPSILON


@dataclass(frozen=True)
class FoldSymbolEvidenceRow:
    fold: int
    fold_return_pct: float
    symbol: str
    current_realized_pnl_usd: float
    current_fills: int
    current_realized_events: int
    prior_folds_seen: int
    prior_active_folds: int
    prior_realized_events: int
    prior_wins: int
    prior_losses: int
    prior_win_rate: float
    prior_realized_pnl_usd: float
    allowed: bool
    decision_reason: str

    @property
    def gated_realized_pnl_usd(self) -> float:
        return self.current_realized_pnl_usd if self.allowed else 0.0

    @property
    def avoided_loss_usd(self) -> float:
        if self.allowed or self.current_realized_pnl_usd >= -PNL_EPSILON:
            return 0.0
        return abs(self.current_realized_pnl_usd)

    @property
    def missed_gain_usd(self) -> float:
        if self.allowed or self.current_realized_pnl_usd <= PNL_EPSILON:
            return 0.0
        return self.current_realized_pnl_usd

    @property
    def kept_loss_usd(self) -> float:
        if not self.allowed or self.current_realized_pnl_usd >= -PNL_EPSILON:
            return 0.0
        return abs(self.current_realized_pnl_usd)

    @property
    def kept_gain_usd(self) -> float:
        if not self.allowed or self.current_realized_pnl_usd <= PNL_EPSILON:
            return 0.0
        return self.current_realized_pnl_usd


@dataclass(frozen=True)
class FoldSymbolEvidenceFoldRow:
    fold: int
    fold_return_pct: float
    symbols: int
    allowed_symbols: int
    blocked_symbols: int
    ungated_realized_pnl_usd: float
    gated_realized_pnl_usd: float
    simulated_delta_usd: float
    avoided_loss_usd: float
    missed_gain_usd: float
    kept_gain_usd: float
    kept_loss_usd: float


@dataclass(frozen=True)
class FoldSymbolEvidenceReport:
    policy: FoldSymbolEvidencePolicy
    rows: tuple[FoldSymbolEvidenceRow, ...]
    fold_rows: tuple[FoldSymbolEvidenceFoldRow, ...]

    @property
    def ungated_realized_pnl_usd(self) -> float:
        return sum(row.current_realized_pnl_usd for row in self.rows)

    @property
    def gated_realized_pnl_usd(self) -> float:
        return sum(row.gated_realized_pnl_usd for row in self.rows)

    @property
    def simulated_delta_usd(self) -> float:
        return self.gated_realized_pnl_usd - self.ungated_realized_pnl_usd

    @property
    def avoided_loss_usd(self) -> float:
        return sum(row.avoided_loss_usd for row in self.rows)

    @property
    def missed_gain_usd(self) -> float:
        return sum(row.missed_gain_usd for row in self.rows)

    @property
    def allowed_fraction(self) -> float:
        if not self.rows:
            return 0.0
        return sum(1 for row in self.rows if row.allowed) / len(self.rows)


@dataclass(frozen=True)
class FoldSymbolEvidenceSweepCandidate:
    rank: int
    policy: FoldSymbolEvidencePolicy
    ungated_realized_pnl_usd: float
    gated_realized_pnl_usd: float
    simulated_delta_usd: float
    avoided_loss_usd: float
    missed_gain_usd: float
    allowed_fraction: float

    @property
    def rank_key(self) -> tuple[float, float, float, float]:
        return (
            self.simulated_delta_usd,
            self.gated_realized_pnl_usd,
            self.avoided_loss_usd,
            -self.missed_gain_usd,
        )


@dataclass(frozen=True)
class FoldSymbolEvidenceSweepReport:
    candidates: tuple[FoldSymbolEvidenceSweepCandidate, ...]

    @property
    def best(self) -> FoldSymbolEvidenceSweepCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[0]


def build_fold_symbol_evidence_report(
    *,
    attribution_csv: str | Path,
    folds_csv: str | Path,
    symbols: tuple[str, ...] | list[str] | None = None,
    policy: FoldSymbolEvidencePolicy | None = None,
) -> FoldSymbolEvidenceReport:
    gate_policy = policy or FoldSymbolEvidencePolicy()
    contributions = _read_symbol_contributions(attribution_csv)
    fold_return_by_fold = _read_fold_returns(folds_csv)
    selected_symbols = tuple(symbols or sorted({row.symbol for row in contributions}))
    if not selected_symbols:
        raise ValueError("fold symbol evidence requires at least one symbol")

    contribution_by_key = {
        (row.fold, row.symbol): row
        for row in _complete_contributions(
            contributions=contributions,
            fold_return_by_fold=fold_return_by_fold,
            symbols=selected_symbols,
        )
    }
    sorted_folds = tuple(sorted(fold_return_by_fold))
    rows: list[FoldSymbolEvidenceRow] = []
    observed_contribution_by_key: dict[tuple[int, str], FoldSymbolContribution] = {}
    for fold_index, fold in enumerate(sorted_folds):
        prior_folds = sorted_folds[
            max(0, fold_index - gate_policy.lookback_folds) : fold_index
        ]
        for symbol in selected_symbols:
            current = contribution_by_key[(fold, symbol)]
            prior = tuple(
                observed_contribution_by_key.get(
                    (prior_fold, symbol),
                    _zero_contribution(
                        fold=prior_fold,
                        fold_return_pct=fold_return_by_fold[prior_fold],
                        symbol=symbol,
                    ),
                )
                for prior_fold in prior_folds
            )
            allowed, reason = _decision_for_symbol(prior, gate_policy)
            prior_events = sum(row.realized_events for row in prior)
            prior_wins = sum(row.wins for row in prior)
            prior_losses = sum(row.losses for row in prior)
            prior_win_rate = prior_wins / prior_events if prior_events > 0 else 0.0
            rows.append(
                FoldSymbolEvidenceRow(
                    fold=fold,
                    fold_return_pct=current.fold_return_pct,
                    symbol=symbol,
                    current_realized_pnl_usd=current.realized_pnl_usd,
                    current_fills=current.fills,
                    current_realized_events=current.realized_events,
                    prior_folds_seen=len(prior),
                    prior_active_folds=sum(1 for row in prior if row.active),
                    prior_realized_events=prior_events,
                    prior_wins=prior_wins,
                    prior_losses=prior_losses,
                    prior_win_rate=prior_win_rate,
                    prior_realized_pnl_usd=sum(row.realized_pnl_usd for row in prior),
                    allowed=allowed,
                    decision_reason=reason,
                )
            )
            observed_contribution_by_key[(fold, symbol)] = (
                current
                if allowed
                else _zero_contribution(
                    fold=fold,
                    fold_return_pct=current.fold_return_pct,
                    symbol=symbol,
                )
            )

    fold_rows = tuple(_summarize_fold(fold, tuple(row for row in rows if row.fold == fold)) for fold in sorted_folds)
    return FoldSymbolEvidenceReport(
        policy=gate_policy,
        rows=tuple(rows),
        fold_rows=fold_rows,
    )


def sweep_fold_symbol_evidence_policies(
    *,
    attribution_csv: str | Path,
    folds_csv: str | Path,
    symbols: tuple[str, ...] | list[str] | None = None,
    lookback_folds_values: tuple[int, ...] = (1, 2, 3),
    min_prior_pnl_usd_values: tuple[float, ...] = (-1_000.0, 0.0, 250.0, 1_000.0),
    min_prior_win_rate_values: tuple[float, ...] = (0.0, 0.5),
    min_prior_active_folds: int = 1,
    min_prior_realized_events: int = 1,
    allow_without_history: bool = True,
) -> FoldSymbolEvidenceSweepReport:
    candidates: list[FoldSymbolEvidenceSweepCandidate] = []
    for lookback_folds in lookback_folds_values:
        for min_prior_pnl_usd in min_prior_pnl_usd_values:
            for min_prior_win_rate in min_prior_win_rate_values:
                policy = FoldSymbolEvidencePolicy(
                    lookback_folds=lookback_folds,
                    min_prior_active_folds=min_prior_active_folds,
                    min_prior_realized_events=min_prior_realized_events,
                    min_prior_pnl_usd=min_prior_pnl_usd,
                    min_prior_win_rate=min_prior_win_rate,
                    allow_without_history=allow_without_history,
                )
                report = build_fold_symbol_evidence_report(
                    attribution_csv=attribution_csv,
                    folds_csv=folds_csv,
                    symbols=symbols,
                    policy=policy,
                )
                candidates.append(
                    FoldSymbolEvidenceSweepCandidate(
                        rank=0,
                        policy=policy,
                        ungated_realized_pnl_usd=report.ungated_realized_pnl_usd,
                        gated_realized_pnl_usd=report.gated_realized_pnl_usd,
                        simulated_delta_usd=report.simulated_delta_usd,
                        avoided_loss_usd=report.avoided_loss_usd,
                        missed_gain_usd=report.missed_gain_usd,
                        allowed_fraction=report.allowed_fraction,
                    )
                )
    ranked = tuple(
        FoldSymbolEvidenceSweepCandidate(
            rank=rank,
            policy=candidate.policy,
            ungated_realized_pnl_usd=candidate.ungated_realized_pnl_usd,
            gated_realized_pnl_usd=candidate.gated_realized_pnl_usd,
            simulated_delta_usd=candidate.simulated_delta_usd,
            avoided_loss_usd=candidate.avoided_loss_usd,
            missed_gain_usd=candidate.missed_gain_usd,
            allowed_fraction=candidate.allowed_fraction,
        )
        for rank, candidate in enumerate(
            sorted(candidates, key=lambda candidate: candidate.rank_key, reverse=True),
            start=1,
        )
    )
    return FoldSymbolEvidenceSweepReport(candidates=ranked)


def write_fold_symbol_evidence_detail_csv(
    report: FoldSymbolEvidenceReport,
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_DETAIL_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow(
                {
                    "fold": row.fold,
                    "fold_return_pct": row.fold_return_pct,
                    "symbol": row.symbol,
                    "current_realized_pnl_usd": row.current_realized_pnl_usd,
                    "current_fills": row.current_fills,
                    "current_realized_events": row.current_realized_events,
                    "prior_folds_seen": row.prior_folds_seen,
                    "prior_active_folds": row.prior_active_folds,
                    "prior_realized_events": row.prior_realized_events,
                    "prior_wins": row.prior_wins,
                    "prior_losses": row.prior_losses,
                    "prior_win_rate": row.prior_win_rate,
                    "prior_realized_pnl_usd": row.prior_realized_pnl_usd,
                    "allowed": row.allowed,
                    "decision_reason": row.decision_reason,
                    "gated_realized_pnl_usd": row.gated_realized_pnl_usd,
                    "avoided_loss_usd": row.avoided_loss_usd,
                    "missed_gain_usd": row.missed_gain_usd,
                    "kept_gain_usd": row.kept_gain_usd,
                    "kept_loss_usd": row.kept_loss_usd,
                }
            )


def write_fold_symbol_evidence_summary_csv(
    report: FoldSymbolEvidenceReport,
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "scope": "TOTAL",
                "fold": "",
                "fold_return_pct": "",
                "symbols": len({row.symbol for row in report.rows}),
                "allowed_symbols": sum(1 for row in report.rows if row.allowed),
                "blocked_symbols": sum(1 for row in report.rows if not row.allowed),
                "allowed_fraction": report.allowed_fraction,
                "ungated_realized_pnl_usd": report.ungated_realized_pnl_usd,
                "gated_realized_pnl_usd": report.gated_realized_pnl_usd,
                "simulated_delta_usd": report.simulated_delta_usd,
                "avoided_loss_usd": report.avoided_loss_usd,
                "missed_gain_usd": report.missed_gain_usd,
                "kept_gain_usd": sum(row.kept_gain_usd for row in report.rows),
                "kept_loss_usd": sum(row.kept_loss_usd for row in report.rows),
            }
        )
        for row in report.fold_rows:
            writer.writerow(
                {
                    "scope": "FOLD",
                    "fold": row.fold,
                    "fold_return_pct": row.fold_return_pct,
                    "symbols": row.symbols,
                    "allowed_symbols": row.allowed_symbols,
                    "blocked_symbols": row.blocked_symbols,
                    "allowed_fraction": row.allowed_symbols / row.symbols if row.symbols else 0.0,
                    "ungated_realized_pnl_usd": row.ungated_realized_pnl_usd,
                    "gated_realized_pnl_usd": row.gated_realized_pnl_usd,
                    "simulated_delta_usd": row.simulated_delta_usd,
                    "avoided_loss_usd": row.avoided_loss_usd,
                    "missed_gain_usd": row.missed_gain_usd,
                    "kept_gain_usd": row.kept_gain_usd,
                    "kept_loss_usd": row.kept_loss_usd,
                }
            )


def write_fold_symbol_evidence_sweep_csv(
    report: FoldSymbolEvidenceSweepReport,
    output: str | Path,
) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SWEEP_FIELDS)
        writer.writeheader()
        for candidate in report.candidates:
            policy = candidate.policy
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "lookback_folds": policy.lookback_folds,
                    "min_prior_active_folds": policy.min_prior_active_folds,
                    "min_prior_realized_events": policy.min_prior_realized_events,
                    "min_prior_pnl_usd": policy.min_prior_pnl_usd,
                    "min_prior_win_rate": policy.min_prior_win_rate,
                    "allow_without_history": policy.allow_without_history,
                    "ungated_realized_pnl_usd": candidate.ungated_realized_pnl_usd,
                    "gated_realized_pnl_usd": candidate.gated_realized_pnl_usd,
                    "simulated_delta_usd": candidate.simulated_delta_usd,
                    "avoided_loss_usd": candidate.avoided_loss_usd,
                    "missed_gain_usd": candidate.missed_gain_usd,
                    "allowed_fraction": candidate.allowed_fraction,
                }
            )


def _read_symbol_contributions(path: str | Path) -> tuple[FoldSymbolContribution, ...]:
    grouped: dict[tuple[int, str], dict[str, float | int]] = {}
    fold_return_by_key: dict[tuple[int, str], float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "fold",
            "fold_return_pct",
            "symbol",
            "fills",
            "realized_events",
            "wins",
            "losses",
            "realized_pnl_usd",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"attribution CSV missing required columns: {sorted(missing)}")
        for csv_row in reader:
            key = (int(csv_row["fold"]), csv_row["symbol"])
            bucket = grouped.setdefault(
                key,
                {
                    "fills": 0,
                    "realized_events": 0,
                    "wins": 0,
                    "losses": 0,
                    "realized_pnl_usd": 0.0,
                },
            )
            bucket["fills"] = int(bucket["fills"]) + int(csv_row["fills"])
            bucket["realized_events"] = int(bucket["realized_events"]) + int(
                csv_row["realized_events"]
            )
            bucket["wins"] = int(bucket["wins"]) + int(csv_row["wins"])
            bucket["losses"] = int(bucket["losses"]) + int(csv_row["losses"])
            bucket["realized_pnl_usd"] = float(bucket["realized_pnl_usd"]) + float(
                csv_row["realized_pnl_usd"]
            )
            fold_return_by_key[key] = float(csv_row["fold_return_pct"])
    return tuple(
        FoldSymbolContribution(
            fold=fold,
            fold_return_pct=fold_return_by_key[(fold, symbol)],
            symbol=symbol,
            fills=int(values["fills"]),
            realized_events=int(values["realized_events"]),
            wins=int(values["wins"]),
            losses=int(values["losses"]),
            realized_pnl_usd=float(values["realized_pnl_usd"]),
        )
        for (fold, symbol), values in grouped.items()
    )


def _read_fold_returns(path: str | Path) -> dict[int, float]:
    fold_return_by_fold: dict[int, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"fold", "return_pct"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"fold CSV missing required columns: {sorted(missing)}")
        for row in reader:
            fold_return_by_fold[int(row["fold"])] = float(row["return_pct"])
    if not fold_return_by_fold:
        raise ValueError("fold CSV has no rows")
    return fold_return_by_fold


def _complete_contributions(
    *,
    contributions: tuple[FoldSymbolContribution, ...],
    fold_return_by_fold: dict[int, float],
    symbols: tuple[str, ...],
) -> tuple[FoldSymbolContribution, ...]:
    contribution_by_key = {
        (row.fold, row.symbol): row
        for row in contributions
    }
    completed: list[FoldSymbolContribution] = []
    for fold, fold_return in sorted(fold_return_by_fold.items()):
        for symbol in symbols:
            completed.append(
                contribution_by_key.get(
                    (fold, symbol),
                    FoldSymbolContribution(
                        fold=fold,
                        fold_return_pct=fold_return,
                        symbol=symbol,
                        realized_pnl_usd=0.0,
                        fills=0,
                        realized_events=0,
                        wins=0,
                        losses=0,
                    ),
                )
            )
    return tuple(completed)


def _zero_contribution(
    *,
    fold: int,
    fold_return_pct: float,
    symbol: str,
) -> FoldSymbolContribution:
    return FoldSymbolContribution(
        fold=fold,
        fold_return_pct=fold_return_pct,
        symbol=symbol,
        realized_pnl_usd=0.0,
        fills=0,
        realized_events=0,
        wins=0,
        losses=0,
    )


def _decision_for_symbol(
    prior: tuple[FoldSymbolContribution, ...],
    policy: FoldSymbolEvidencePolicy,
) -> tuple[bool, str]:
    active_folds = sum(1 for row in prior if row.active)
    if active_folds == 0:
        if policy.allow_without_history:
            return True, "allowed: no prior active history"
        return False, "blocked: no prior active history"

    prior_events = sum(row.realized_events for row in prior)
    prior_wins = sum(row.wins for row in prior)
    prior_pnl = sum(row.realized_pnl_usd for row in prior)
    prior_win_rate = prior_wins / prior_events if prior_events > 0 else 0.0
    if active_folds < policy.min_prior_active_folds:
        return (
            False,
            (
                f"blocked: prior active folds {active_folds} < "
                f"{policy.min_prior_active_folds}"
            ),
        )
    if prior_events < policy.min_prior_realized_events:
        return (
            False,
            (
                f"blocked: prior events {prior_events} < "
                f"{policy.min_prior_realized_events}"
            ),
        )
    if prior_pnl < policy.min_prior_pnl_usd:
        return (
            False,
            (
                f"blocked: prior pnl {prior_pnl:.2f} < "
                f"{policy.min_prior_pnl_usd:.2f}"
            ),
        )
    if prior_win_rate < policy.min_prior_win_rate:
        return (
            False,
            (
                f"blocked: prior win rate {prior_win_rate:.1%} < "
                f"{policy.min_prior_win_rate:.1%}"
            ),
        )
    return (
        True,
        (
            f"allowed: prior pnl {prior_pnl:.2f}, events {prior_events}, "
            f"win rate {prior_win_rate:.1%}"
        ),
    )


def _summarize_fold(
    fold: int,
    rows: tuple[FoldSymbolEvidenceRow, ...],
) -> FoldSymbolEvidenceFoldRow:
    if not rows:
        raise ValueError("cannot summarize empty fold evidence")
    ungated = sum(row.current_realized_pnl_usd for row in rows)
    gated = sum(row.gated_realized_pnl_usd for row in rows)
    return FoldSymbolEvidenceFoldRow(
        fold=fold,
        fold_return_pct=rows[0].fold_return_pct,
        symbols=len(rows),
        allowed_symbols=sum(1 for row in rows if row.allowed),
        blocked_symbols=sum(1 for row in rows if not row.allowed),
        ungated_realized_pnl_usd=ungated,
        gated_realized_pnl_usd=gated,
        simulated_delta_usd=gated - ungated,
        avoided_loss_usd=sum(row.avoided_loss_usd for row in rows),
        missed_gain_usd=sum(row.missed_gain_usd for row in rows),
        kept_gain_usd=sum(row.kept_gain_usd for row in rows),
        kept_loss_usd=sum(row.kept_loss_usd for row in rows),
    )


_DETAIL_FIELDS = (
    "fold",
    "fold_return_pct",
    "symbol",
    "current_realized_pnl_usd",
    "current_fills",
    "current_realized_events",
    "prior_folds_seen",
    "prior_active_folds",
    "prior_realized_events",
    "prior_wins",
    "prior_losses",
    "prior_win_rate",
    "prior_realized_pnl_usd",
    "allowed",
    "decision_reason",
    "gated_realized_pnl_usd",
    "avoided_loss_usd",
    "missed_gain_usd",
    "kept_gain_usd",
    "kept_loss_usd",
)

_SUMMARY_FIELDS = (
    "scope",
    "fold",
    "fold_return_pct",
    "symbols",
    "allowed_symbols",
    "blocked_symbols",
    "allowed_fraction",
    "ungated_realized_pnl_usd",
    "gated_realized_pnl_usd",
    "simulated_delta_usd",
    "avoided_loss_usd",
    "missed_gain_usd",
    "kept_gain_usd",
    "kept_loss_usd",
)

_SWEEP_FIELDS = (
    "rank",
    "lookback_folds",
    "min_prior_active_folds",
    "min_prior_realized_events",
    "min_prior_pnl_usd",
    "min_prior_win_rate",
    "allow_without_history",
    "ungated_realized_pnl_usd",
    "gated_realized_pnl_usd",
    "simulated_delta_usd",
    "avoided_loss_usd",
    "missed_gain_usd",
    "allowed_fraction",
)
