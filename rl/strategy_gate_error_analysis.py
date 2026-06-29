"""Error analysis for strategy checkpoint promotion gate blockers."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl.experiments import read_json


ISSUE_ACTION_CRITIC_FALLBACK = "action_critic_fallback"
ISSUE_PREDICTED_NON_EXECUTABLE = "predicted_non_executable"
ISSUE_DROP_NON_EXECUTABLE_MATCH = "predicted_matches_drop_non_executable_labels"
ISSUE_VETO_NEGATIVE_MATCH = "predicted_matches_veto_negative_labels"
ISSUE_ACTION_SPACE_EXHAUSTED_MATCH = (
    "predicted_matches_action_space_exhausted_labels"
)


@dataclass(frozen=True)
class StrategyGateErrorExample:
    """One representative row that blocks or weakens strategy promotion."""

    audit_path: str
    checkpoint_path: str
    source_path: str
    step: int
    game_time: float
    recorded_action: str
    raw_predicted_action: str
    predicted_action: str
    recorded_training_use: str
    recorded_label_quality: str
    context: str
    threat_state: str
    issues: list[str]
    predicted_blocker: str | None
    action_critic_selected_unsafe_probability: float | None
    critic_vetoed_actions: list[str]
    critic_veto_reasons: list[str]


@dataclass(frozen=True)
class StrategyGateErrorAnalysis:
    """Aggregated error clusters for one or more checkpoint signal audits."""

    audit_paths: list[str]
    audits: int
    rows: int
    issue_rows: int
    issue_counts: dict[str, int]
    veto_negative_matches: int
    action_space_exhausted_matches: int
    drop_non_executable_matches: int
    predicted_non_executable_rows: int
    action_critic_fallback_rows: int
    fallback_and_veto_negative_rows: int
    fallback_and_accept_positive_rows: int
    action_critic_selected_unsafe_probability_avg: float | None
    action_critic_selected_unsafe_probability_max: float | None
    issue_by_predicted_action: dict[str, int]
    issue_by_recorded_action: dict[str, int]
    issue_by_training_use: dict[str, int]
    issue_by_threat_state: dict[str, int]
    issue_by_context: dict[str, int]
    fallback_by_predicted_action: dict[str, int]
    fallback_by_training_use: dict[str, int]
    fallback_by_threat_state: dict[str, int]
    fallback_by_context: dict[str, int]
    fallback_by_candidate_action_count: dict[str, int]
    fallback_by_candidate_action_set: dict[str, int]
    fallback_single_candidate_action: dict[str, int]
    veto_match_by_predicted_action: dict[str, int]
    veto_match_by_recorded_action: dict[str, int]
    veto_match_by_threat_state: dict[str, int]
    veto_match_by_context: dict[str, int]
    veto_match_by_candidate_action_set: dict[str, int]
    action_space_match_by_predicted_action: dict[str, int]
    action_space_match_by_threat_state: dict[str, int]
    action_space_match_by_candidate_action_set: dict[str, int]
    non_executable_by_blocker: dict[str, int]
    critic_veto_action_counts: dict[str, int]
    critic_veto_reason_counts: dict[str, int]
    warning_counts: dict[str, int]
    examples: list[StrategyGateErrorExample]


def analyze_strategy_gate_errors(
    audit_paths: list[str | Path],
    *,
    max_examples: int = 12,
) -> StrategyGateErrorAnalysis:
    """Aggregate promotion-gate error clusters from checkpoint audit JSON files."""
    if max_examples < 0:
        raise ValueError("max_examples must be >= 0")

    audit_path_strings: list[str] = []
    decisions: list[tuple[str, str, dict[str, Any]]] = []
    warning_counter: Counter[str] = Counter()

    for path in audit_paths:
        audit_path = str(path)
        audit = read_json(path)
        audit_path_strings.append(audit_path)
        warning_counter.update(str(warning) for warning in audit.get("warnings") or [])
        checkpoint_path = str(audit.get("checkpoint_path", ""))
        for decision in audit.get("decisions") or []:
            decisions.append((audit_path, checkpoint_path, decision))

    issue_rows = [
        (audit_path, checkpoint_path, decision, _decision_issues(decision))
        for audit_path, checkpoint_path, decision in decisions
    ]
    issue_rows = [
        row
        for row in issue_rows
        if row[3]
    ]
    issue_counts = Counter(
        issue
        for _, _, _, issues in issue_rows
        for issue in issues
    )
    selected_probabilities = [
        _float_or_none(decision.get("action_critic_selected_unsafe_probability"))
        for _, _, decision, issues in issue_rows
        if ISSUE_ACTION_CRITIC_FALLBACK in issues
    ]
    selected_probabilities = [
        probability
        for probability in selected_probabilities
        if probability is not None
    ]
    fallback_rows = [
        decision
        for _, _, decision, issues in issue_rows
        if ISSUE_ACTION_CRITIC_FALLBACK in issues
    ]
    veto_rows = [
        decision
        for _, _, decision, issues in issue_rows
        if ISSUE_VETO_NEGATIVE_MATCH in issues
    ]
    action_space_rows = [
        decision
        for _, _, decision, issues in issue_rows
        if ISSUE_ACTION_SPACE_EXHAUSTED_MATCH in issues
    ]
    non_executable_rows = [
        decision
        for _, _, decision, issues in issue_rows
        if ISSUE_PREDICTED_NON_EXECUTABLE in issues
    ]

    return StrategyGateErrorAnalysis(
        audit_paths=audit_path_strings,
        audits=len(audit_path_strings),
        rows=len(decisions),
        issue_rows=len(issue_rows),
        issue_counts=_sorted_counter(issue_counts),
        veto_negative_matches=issue_counts[ISSUE_VETO_NEGATIVE_MATCH],
        action_space_exhausted_matches=issue_counts[
            ISSUE_ACTION_SPACE_EXHAUSTED_MATCH
        ],
        drop_non_executable_matches=issue_counts[ISSUE_DROP_NON_EXECUTABLE_MATCH],
        predicted_non_executable_rows=issue_counts[ISSUE_PREDICTED_NON_EXECUTABLE],
        action_critic_fallback_rows=issue_counts[ISSUE_ACTION_CRITIC_FALLBACK],
        fallback_and_veto_negative_rows=sum(
            1
            for _, _, _, issues in issue_rows
            if ISSUE_ACTION_CRITIC_FALLBACK in issues
            and ISSUE_VETO_NEGATIVE_MATCH in issues
        ),
        fallback_and_accept_positive_rows=sum(
            1
            for decision in fallback_rows
            if bool(decision.get("prediction_matches_accept_positive", False))
        ),
        action_critic_selected_unsafe_probability_avg=_mean_or_none(
            selected_probabilities
        ),
        action_critic_selected_unsafe_probability_max=_max_or_none(
            selected_probabilities
        ),
        issue_by_predicted_action=_count_decisions(
            issue_rows,
            "predicted_action",
        ),
        issue_by_recorded_action=_count_decisions(
            issue_rows,
            "recorded_action",
        ),
        issue_by_training_use=_count_decisions(
            issue_rows,
            "recorded_training_use",
        ),
        issue_by_threat_state=_count_decisions(
            issue_rows,
            "threat_state",
        ),
        issue_by_context=_count_decisions(issue_rows, "context"),
        fallback_by_predicted_action=_count_values(
            decision.get("predicted_action", "<unknown>")
            for decision in fallback_rows
        ),
        fallback_by_training_use=_count_values(
            decision.get("recorded_training_use", "<unknown>")
            for decision in fallback_rows
        ),
        fallback_by_threat_state=_count_values(
            decision.get("threat_state", "<unknown>")
            for decision in fallback_rows
        ),
        fallback_by_context=_count_values(
            decision.get("context", "<unknown>")
            for decision in fallback_rows
        ),
        fallback_by_candidate_action_count=_count_values(
            str(len(decision.get("action_critic_candidate_actions") or []))
            for decision in fallback_rows
        ),
        fallback_by_candidate_action_set=_count_values(
            _candidate_action_set(decision)
            for decision in fallback_rows
        ),
        fallback_single_candidate_action=_count_values(
            _single_candidate_action(decision)
            for decision in fallback_rows
            if len(decision.get("action_critic_candidate_actions") or []) == 1
        ),
        veto_match_by_predicted_action=_count_values(
            decision.get("predicted_action", "<unknown>")
            for decision in veto_rows
        ),
        veto_match_by_recorded_action=_count_values(
            decision.get("recorded_action", "<unknown>")
            for decision in veto_rows
        ),
        veto_match_by_threat_state=_count_values(
            decision.get("threat_state", "<unknown>")
            for decision in veto_rows
        ),
        veto_match_by_context=_count_values(
            decision.get("context", "<unknown>")
            for decision in veto_rows
        ),
        veto_match_by_candidate_action_set=_count_values(
            _candidate_action_set(decision)
            for decision in veto_rows
        ),
        action_space_match_by_predicted_action=_count_values(
            decision.get("predicted_action", "<unknown>")
            for decision in action_space_rows
        ),
        action_space_match_by_threat_state=_count_values(
            decision.get("threat_state", "<unknown>")
            for decision in action_space_rows
        ),
        action_space_match_by_candidate_action_set=_count_values(
            _candidate_action_set(decision)
            for decision in action_space_rows
        ),
        non_executable_by_blocker=_count_values(
            decision.get("predicted_blocker") or "<none>"
            for decision in non_executable_rows
        ),
        critic_veto_action_counts=_count_values(
            action
            for _, _, decision, issues in issue_rows
            if ISSUE_ACTION_CRITIC_FALLBACK in issues
            for action in decision.get("critic_vetoed_actions") or []
        ),
        critic_veto_reason_counts=_count_values(
            reason
            for _, _, decision, issues in issue_rows
            if ISSUE_ACTION_CRITIC_FALLBACK in issues
            for reason in decision.get("critic_veto_reasons") or []
        ),
        warning_counts=_sorted_counter(warning_counter),
        examples=_examples(issue_rows, max_examples=max_examples),
    )


def _decision_issues(decision: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if bool(decision.get("prediction_matches_veto_negative", False)):
        issues.append(ISSUE_VETO_NEGATIVE_MATCH)
    if bool(decision.get("prediction_matches_drop_non_executable", False)):
        issues.append(ISSUE_DROP_NON_EXECUTABLE_MATCH)
    if bool(decision.get("prediction_matches_action_space_exhausted", False)):
        issues.append(ISSUE_ACTION_SPACE_EXHAUSTED_MATCH)
    if not bool(decision.get("predicted_immediate_executable", True)):
        issues.append(ISSUE_PREDICTED_NON_EXECUTABLE)
    if bool(decision.get("action_critic_fallback_selected", False)):
        issues.append(ISSUE_ACTION_CRITIC_FALLBACK)
    return issues


def _examples(
    issue_rows: list[tuple[str, str, dict[str, Any], list[str]]],
    *,
    max_examples: int,
) -> list[StrategyGateErrorExample]:
    ranked = sorted(issue_rows, key=_example_sort_key)
    return [
        StrategyGateErrorExample(
            audit_path=audit_path,
            checkpoint_path=checkpoint_path,
            source_path=str(decision.get("path", "")),
            step=int(decision.get("step", 0) or 0),
            game_time=float(decision.get("game_time", 0.0) or 0.0),
            recorded_action=str(decision.get("recorded_action", "")),
            raw_predicted_action=str(decision.get("raw_predicted_action", "")),
            predicted_action=str(decision.get("predicted_action", "")),
            recorded_training_use=str(decision.get("recorded_training_use", "")),
            recorded_label_quality=str(decision.get("recorded_label_quality", "")),
            context=str(decision.get("context", "")),
            threat_state=str(decision.get("threat_state", "")),
            issues=issues,
            predicted_blocker=decision.get("predicted_blocker"),
            action_critic_selected_unsafe_probability=_float_or_none(
                decision.get("action_critic_selected_unsafe_probability")
            ),
            critic_vetoed_actions=[
                str(action) for action in decision.get("critic_vetoed_actions") or []
            ],
            critic_veto_reasons=[
                str(reason) for reason in decision.get("critic_veto_reasons") or []
            ],
        )
        for audit_path, checkpoint_path, decision, issues in ranked[:max_examples]
    ]


def _example_sort_key(
    row: tuple[str, str, dict[str, Any], list[str]],
) -> tuple[float, str, int]:
    audit_path, _, decision, issues = row
    severity = 0.0
    if ISSUE_VETO_NEGATIVE_MATCH in issues:
        severity += 1000.0
    if ISSUE_ACTION_SPACE_EXHAUSTED_MATCH in issues:
        severity += 900.0
    if ISSUE_DROP_NON_EXECUTABLE_MATCH in issues:
        severity += 700.0
    if ISSUE_PREDICTED_NON_EXECUTABLE in issues:
        severity += 500.0
    if ISSUE_ACTION_CRITIC_FALLBACK in issues:
        severity += 100.0
    probability = _float_or_none(
        decision.get("action_critic_selected_unsafe_probability")
    )
    severity += probability or 0.0
    return (-severity, audit_path, int(decision.get("step", 0) or 0))


def _count_decisions(
    rows: list[tuple[str, str, dict[str, Any], list[str]]],
    key: str,
) -> dict[str, int]:
    return _count_values(str(decision.get(key, "<unknown>")) for _, _, decision, _ in rows)


def _count_values(values) -> dict[str, int]:
    return _sorted_counter(Counter(str(value) for value in values))


def _candidate_action_set(decision: dict[str, Any]) -> str:
    actions = [
        str(action)
        for action in decision.get("action_critic_candidate_actions") or []
    ]
    if not actions:
        return "<none>"
    return "+".join(sorted(actions))


def _single_candidate_action(decision: dict[str, Any]) -> str:
    actions = [
        str(action)
        for action in decision.get("action_critic_candidate_actions") or []
    ]
    return actions[0] if actions else "<none>"


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in counter.items()))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))
