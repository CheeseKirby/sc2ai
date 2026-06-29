"""Signal-derived offline risk critic for strategy candidate actions."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rl.strategy_outcome_diagnostics import _StrategyOutcomeRow
from rl.strategy_replay_candidate import classify_replay_context, classify_threat_state
from rl.strategy_signal_dataset import StrategySignalDataset, StrategySignalRecord


DEFAULT_CRITIC_MIN_SAMPLES = 3
DEFAULT_CRITIC_MAX_BAD_RATE = 0.40
DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE = 0.0


@dataclass(frozen=True)
class StrategySignalRiskGroup:
    """Aggregated risk statistics for one action/context/threat group."""

    action: str
    context: str
    threat_state: str
    samples: int
    bad_records: int
    veto_negative_records: int
    drop_non_executable_records: int
    accept_positive_records: int
    bad_rate: float
    veto_negative_rate: float
    accept_positive_rate: float


@dataclass(frozen=True)
class StrategySignalCriticDecision:
    """Critic decision for one candidate action in one row context."""

    candidate_action: str
    context: str
    threat_state: str
    hard_veto: bool
    reasons: list[str]
    matched_group: StrategySignalRiskGroup | None


@dataclass(frozen=True)
class StrategySignalRiskCritic:
    """Lookup critic built from row-level strategy signal labels."""

    groups: dict[tuple[str, str, str], StrategySignalRiskGroup]
    min_samples: int
    max_bad_rate: float
    max_veto_negative_rate: float

    def decision_for(
        self,
        row: _StrategyOutcomeRow,
        candidate_action: str,
    ) -> StrategySignalCriticDecision:
        """Return a conservative risk decision for a candidate action."""
        context = classify_replay_context(row, candidate_action)
        threat_state = classify_threat_state(row)
        group = self.groups.get((candidate_action, context, threat_state))
        reasons: list[str] = []
        if group is not None and group.samples >= self.min_samples:
            if group.veto_negative_rate > self.max_veto_negative_rate:
                reasons.append(_risk_reason(group))
            elif group.bad_rate > self.max_bad_rate:
                reasons.append(_risk_reason(group))
        return StrategySignalCriticDecision(
            candidate_action=candidate_action,
            context=context,
            threat_state=threat_state,
            hard_veto=bool(reasons),
            reasons=reasons,
            matched_group=group,
        )


def build_strategy_signal_risk_critic(
    dataset: StrategySignalDataset,
    *,
    min_samples: int = DEFAULT_CRITIC_MIN_SAMPLES,
    max_bad_rate: float = DEFAULT_CRITIC_MAX_BAD_RATE,
    max_veto_negative_rate: float = DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
) -> StrategySignalRiskCritic:
    """Build an exact-context risk critic from recorded signal labels."""
    records_by_key: dict[tuple[str, str, str], list[StrategySignalRecord]] = {}
    for record in dataset.records:
        if record.candidate_source != "recorded":
            continue
        key = (record.candidate_action, record.context, record.threat_state)
        records_by_key.setdefault(key, []).append(record)

    groups = {
        key: _group_for(key, records)
        for key, records in records_by_key.items()
    }
    return StrategySignalRiskCritic(
        groups=groups,
        min_samples=int(min_samples),
        max_bad_rate=float(max_bad_rate),
        max_veto_negative_rate=float(max_veto_negative_rate),
    )


def _group_for(
    key: tuple[str, str, str],
    records: list[StrategySignalRecord],
) -> StrategySignalRiskGroup:
    action, context, threat_state = key
    training_use_counts = Counter(record.recommended_training_use for record in records)
    bad_records = sum(1 for record in records if record.label_quality == "bad")
    samples = len(records)
    return StrategySignalRiskGroup(
        action=action,
        context=context,
        threat_state=threat_state,
        samples=samples,
        bad_records=bad_records,
        veto_negative_records=int(training_use_counts["veto_negative"]),
        drop_non_executable_records=int(training_use_counts["drop_non_executable"]),
        accept_positive_records=int(training_use_counts["accept_positive"]),
        bad_rate=_ratio(bad_records, samples),
        veto_negative_rate=_ratio(training_use_counts["veto_negative"], samples),
        accept_positive_rate=_ratio(training_use_counts["accept_positive"], samples),
    )


def _risk_reason(group: StrategySignalRiskGroup) -> str:
    return (
        "risk:"
        f"action={group.action},"
        f"context={group.context},"
        f"threat={group.threat_state}"
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
