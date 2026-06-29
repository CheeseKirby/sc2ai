"""Offline signal audit for trained strategy checkpoints."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bot.managers.rl_strategy_policy import RLStrategyPolicy as _RLStrategyPolicy
from rl.normalization import ObservationNormalizer
from rl.strategy_action_critic import (
    StrategyActionCriticNetwork,
    action_critic_feature_vector_from_observation,
    load_strategy_action_critic_checkpoint,
)
from rl.strategy_actions import STRATEGY_ACTION_NAMES
from rl.strategy_checkpoints import load_strategy_policy_checkpoint
from rl.strategy_datasets import (
    StrategyTrajectoryPathInput,
    discover_strategy_trajectory_files,
)
from rl.strategy_observations import (
    STRATEGY_OBSERVATION_FIELDS,
    STRATEGY_OBSERVATION_SCHEMA_VERSION,
    strategy_observation_dict_to_vector,
)
from rl.strategy_outcome_diagnostics import (
    _StrategyOutcomeRow,
    _iter_valid_strategy_rows,
    _source_for_file,
)
from rl.strategy_replay_candidate import (
    candidate_executability,
    classify_replay_context,
    classify_threat_state,
)
from rl.strategy_signal_dataset import (
    StrategySignalRecord,
    build_strategy_signal_dataset,
)
from rl.strategy_signal_critic import (
    DEFAULT_CRITIC_MAX_BAD_RATE,
    DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    DEFAULT_CRITIC_MIN_SAMPLES,
    StrategySignalRiskCritic,
    build_strategy_signal_risk_critic,
)


MAX_PREDICTED_NON_EXECUTABLE_RATIO = 0.05
PREDICTION_MODES: tuple[str, ...] = (
    "raw",
    "executable-mask",
    "signal-risk-mask",
    "action-critic-mask",
)
ACTION_CRITIC_FALLBACK_POLICIES: tuple[str, ...] = (
    "lowest-risk",
    "signal-risk",
    "first-executable",
    "rule-safe",
    "rule-risk",
    "threat-risk",
    "mixed-threat-risk",
)
_ = _RLStrategyPolicy


@dataclass(frozen=True)
class StrategyCheckpointSignalDecision:
    """One checkpoint prediction compared with the row-level strategy signal."""

    path: str
    source: str
    step: int
    game_time: float
    recorded_action: str
    raw_predicted_action: str
    predicted_action: str
    recorded_training_use: str
    recorded_label_quality: str
    recorded_immediate_executable: bool
    recorded_candidate_blocker: str | None
    raw_predicted_immediate_executable: bool
    raw_predicted_blocker: str | None
    predicted_immediate_executable: bool
    predicted_blocker: str | None
    prediction_was_masked: bool
    critic_vetoed_actions: list[str]
    critic_veto_reasons: list[str]
    action_critic_candidate_actions: list[str]
    action_critic_vetoed_probabilities: list[float]
    action_critic_selected_unsafe_probability: float | None
    action_critic_fallback_selected: bool
    action_critic_fallback_policy_used: str | None
    prediction_matches_recorded: bool
    prediction_matches_accept_positive: bool
    prediction_matches_bad_recorded: bool
    prediction_matches_veto_negative: bool
    prediction_matches_drop_non_executable: bool
    prediction_matches_action_space_exhausted: bool
    context: str
    threat_state: str


@dataclass(frozen=True)
class StrategyCheckpointSignalAudit:
    """Dataset-level checkpoint signal quality audit."""

    inputs: list[str]
    checkpoint_path: str
    prediction_mode: str
    action_critic_checkpoint_path: str | None
    action_critic_threshold: float | None
    action_critic_fallback_policy: str | None
    files: int
    rows: int
    signal_healthy: bool
    blocking_reasons: list[str]
    warnings: list[str]
    recorded_training_use_counts: dict[str, int]
    recorded_label_quality_counts: dict[str, int]
    recorded_action_counts_by_name: dict[str, int]
    raw_predicted_action_counts_by_name: dict[str, int]
    predicted_action_counts_by_name: dict[str, int]
    prediction_matches_recorded: int
    prediction_match_ratio: float
    raw_predicted_non_executable_rows: int
    raw_predicted_non_executable_ratio: float
    raw_predicted_blocker_counts: dict[str, int]
    predicted_executable_rows: int
    predicted_non_executable_rows: int
    predicted_non_executable_ratio: float
    predicted_blocker_counts: dict[str, int]
    predicted_non_executable_by_recorded_training_use: dict[str, int]
    masked_prediction_changes: int
    masked_prediction_change_ratio: float
    critic_vetoed_candidates: int
    critic_veto_reason_counts: dict[str, int]
    critic_veto_action_counts: dict[str, int]
    action_critic_selected_unsafe_probability_avg: float | None
    action_critic_selected_unsafe_probability_max: float | None
    action_critic_vetoed_probability_avg: float | None
    action_critic_vetoed_probability_max: float | None
    action_critic_fallback_rows: int
    action_critic_fallback_policy_counts: dict[str, int]
    accept_positive_rows: int
    accept_positive_prediction_matches: int
    accept_positive_prediction_match_ratio: float
    bad_recorded_rows: int
    bad_recorded_prediction_matches: int
    bad_recorded_prediction_match_ratio: float
    veto_negative_rows: int
    veto_negative_prediction_matches: int
    veto_negative_prediction_match_ratio: float
    drop_non_executable_rows: int
    drop_non_executable_prediction_matches: int
    drop_non_executable_prediction_match_ratio: float
    action_space_exhausted_rows: int
    action_space_exhausted_prediction_matches: int
    action_space_exhausted_prediction_match_ratio: float
    decisions: list[StrategyCheckpointSignalDecision]


def audit_strategy_checkpoint_signals(
    paths: StrategyTrajectoryPathInput,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    prediction_mode: str = "raw",
    critic_min_samples: int = DEFAULT_CRITIC_MIN_SAMPLES,
    critic_max_bad_rate: float = DEFAULT_CRITIC_MAX_BAD_RATE,
    critic_max_veto_negative_rate: float = DEFAULT_CRITIC_MAX_VETO_NEGATIVE_RATE,
    action_critic_checkpoint_path: str | Path | None = None,
    action_critic_threshold: float = 0.5,
    action_critic_fallback_policy: str = "lowest-risk",
    max_predicted_non_executable_ratio: float = MAX_PREDICTED_NON_EXECUTABLE_RATIO,
) -> StrategyCheckpointSignalAudit:
    """Audit checkpoint predictions against row-level strategy signals."""
    if prediction_mode not in PREDICTION_MODES:
        names = ", ".join(PREDICTION_MODES)
        raise ValueError(f"Unknown prediction_mode {prediction_mode!r}; expected {names}")
    if not 0.0 <= action_critic_threshold <= 1.0:
        raise ValueError("action_critic_threshold must be in [0.0, 1.0]")
    if action_critic_fallback_policy not in ACTION_CRITIC_FALLBACK_POLICIES:
        names = ", ".join(ACTION_CRITIC_FALLBACK_POLICIES)
        raise ValueError(
            "Unknown action_critic_fallback_policy "
            f"{action_critic_fallback_policy!r}; expected {names}"
        )
    if prediction_mode == "action-critic-mask" and action_critic_checkpoint_path is None:
        raise ValueError(
            "action_critic_checkpoint_path is required for action-critic-mask"
        )
    input_paths = _input_paths(paths)
    input_strings = [str(path) for path in input_paths]
    files = discover_strategy_trajectory_files(paths)
    signal_dataset = build_strategy_signal_dataset(
        paths,
        include_before_filter_candidates=False,
    )
    signal_by_key = {
        _record_key(record.path, record.step, record.recorded_action): record
        for record in signal_dataset.records
        if record.candidate_source == "recorded"
    }
    critic = (
        build_strategy_signal_risk_critic(
            signal_dataset,
            min_samples=critic_min_samples,
            max_bad_rate=critic_max_bad_rate,
            max_veto_negative_rate=critic_max_veto_negative_rate,
        )
        if prediction_mode == "signal-risk-mask"
        or (
            prediction_mode == "action-critic-mask"
            and action_critic_fallback_policy
            in {"signal-risk", "rule-risk", "threat-risk"}
        )
        else None
    )

    torch_device = torch.device(device)
    action_critic = (
        _load_action_critic_scorer(
            action_critic_checkpoint_path,
            device=torch_device,
            threshold=action_critic_threshold,
        )
        if prediction_mode == "action-critic-mask"
        else None
    )
    loaded = load_strategy_policy_checkpoint(checkpoint_path, map_location=torch_device)
    model = loaded.model.to(torch_device)
    model.eval()
    normalizer = (
        ObservationNormalizer.from_dict(
            loaded.metadata.normalizer,
            expected_fields=STRATEGY_OBSERVATION_FIELDS,
            expected_schema_version=STRATEGY_OBSERVATION_SCHEMA_VERSION,
        )
        if loaded.metadata.normalizer is not None
        else None
    )

    decisions: list[StrategyCheckpointSignalDecision] = []
    missing_signal_rows = 0
    for path in files:
        source = _source_for_file(path, input_paths)
        for row in _iter_valid_strategy_rows(path, source=source):
            if row.done:
                continue
            signal = signal_by_key.get(_row_key(row))
            if signal is None:
                missing_signal_rows += 1
                continue
            prediction = _predict_checkpoint_action_for_row(
                model,
                row,
                normalizer=normalizer,
                device=torch_device,
                prediction_mode=prediction_mode,
                critic=critic,
                action_critic=action_critic,
                action_critic_fallback_policy=action_critic_fallback_policy,
            )
            raw_predicted_executable, raw_predicted_blocker = candidate_executability(
                row,
                prediction.raw_predicted_action,
            )
            predicted_executable, predicted_blocker = candidate_executability(
                row,
                prediction.predicted_action,
            )
            decisions.append(
                _decision(
                    row=row,
                    signal=signal,
                    raw_predicted_action=prediction.raw_predicted_action,
                    predicted_action=prediction.predicted_action,
                    raw_predicted_executable=raw_predicted_executable,
                    raw_predicted_blocker=raw_predicted_blocker,
                    predicted_executable=predicted_executable,
                    predicted_blocker=predicted_blocker,
                    prediction_was_masked=prediction.prediction_was_masked,
                    critic_vetoed_actions=list(prediction.critic_vetoed_actions),
                    critic_veto_reasons=list(prediction.critic_veto_reasons),
                    action_critic_candidate_actions=list(
                        prediction.action_critic_candidate_actions
                    ),
                    action_critic_vetoed_probabilities=list(
                        prediction.action_critic_vetoed_probabilities
                    ),
                    action_critic_selected_unsafe_probability=(
                        prediction.action_critic_selected_unsafe_probability
                    ),
                    action_critic_fallback_selected=(
                        prediction.action_critic_fallback_selected
                    ),
                    action_critic_fallback_policy_used=(
                        prediction.action_critic_fallback_policy_used
                    ),
                )
            )

    warnings: list[str] = []
    if missing_signal_rows:
        warnings.append(f"missing_signal_rows:{missing_signal_rows}")
    if len(signal_by_key) != len(decisions) + missing_signal_rows:
        warnings.append("signal_record_row_count_mismatch")

    return _summarize(
        input_strings=input_strings,
        checkpoint_path=checkpoint_path,
        prediction_mode=prediction_mode,
        action_critic_checkpoint_path=action_critic_checkpoint_path,
        action_critic_threshold=(
            action_critic_threshold
            if prediction_mode == "action-critic-mask"
            else None
        ),
        action_critic_fallback_policy=(
            action_critic_fallback_policy
            if prediction_mode == "action-critic-mask"
            else None
        ),
        files=len(files),
        decisions=decisions,
        warnings=warnings,
        max_predicted_non_executable_ratio=max_predicted_non_executable_ratio,
    )


@dataclass(frozen=True)
class _CheckpointPrediction:
    raw_predicted_action: str
    predicted_action: str
    prediction_was_masked: bool
    critic_vetoed_actions: tuple[str, ...]
    critic_veto_reasons: tuple[str, ...]
    action_critic_candidate_actions: tuple[str, ...]
    action_critic_vetoed_probabilities: tuple[float, ...]
    action_critic_selected_unsafe_probability: float | None
    action_critic_fallback_selected: bool
    action_critic_fallback_policy_used: str | None


@dataclass(frozen=True)
class _ActionCriticScorer:
    model: StrategyActionCriticNetwork
    normalizer: ObservationNormalizer | None
    feature_fields: tuple[str, ...]
    device: torch.device
    threshold: float

    @torch.no_grad()
    def unsafe_probability(
        self,
        row: _StrategyOutcomeRow,
        candidate_action: str,
    ) -> float:
        features = action_critic_feature_vector_from_observation(
            row.observation,
            candidate_action,
            feature_fields=self.feature_fields,
        )
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return float(self.model.predict_unsafe_probability(tensor).item())


def _load_action_critic_scorer(
    path: str | Path | None,
    *,
    device: torch.device,
    threshold: float,
) -> _ActionCriticScorer:
    if path is None:
        raise ValueError("action critic checkpoint path is required")
    loaded = load_strategy_action_critic_checkpoint(path, map_location=device)
    model = loaded.model.to(device)
    model.eval()
    normalizer = (
        ObservationNormalizer.from_dict(
            loaded.metadata.normalizer,
            expected_fields=loaded.metadata.feature_fields,
            expected_schema_version=loaded.metadata.feature_schema_version,
        )
        if loaded.metadata.normalizer is not None
        else None
    )
    return _ActionCriticScorer(
        model=model,
        normalizer=normalizer,
        feature_fields=loaded.metadata.feature_fields,
        device=device,
        threshold=threshold,
    )


@torch.no_grad()
def _predict_checkpoint_action_for_row(
    model: torch.nn.Module,
    row: _StrategyOutcomeRow,
    *,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
    prediction_mode: str,
    critic: StrategySignalRiskCritic | None,
    action_critic: _ActionCriticScorer | None,
    action_critic_fallback_policy: str,
) -> _CheckpointPrediction:
    logits = _checkpoint_logits(
        model,
        row.observation,
        normalizer=normalizer,
        device=device,
    )
    ranked_action_ids = [
        int(action_id)
        for action_id in torch.argsort(logits, descending=True).detach().cpu().tolist()
    ]
    raw_predicted_action = _action_name(ranked_action_ids[0])
    if prediction_mode == "raw":
        return _prediction(raw_predicted_action, raw_predicted_action)
    if prediction_mode in {
        "executable-mask",
        "signal-risk-mask",
        "action-critic-mask",
    }:
        first_executable_action: str | None = None
        risk_vetoed_candidates: list[tuple[float, str]] = []
        action_critic_candidates: list[tuple[float, str]] = []
        critic_vetoed_actions: list[str] = []
        critic_veto_reasons: list[str] = []
        action_critic_vetoed_probabilities: list[float] = []
        for action_id in ranked_action_ids:
            candidate_action = _action_name(action_id)
            executable, _ = candidate_executability(row, candidate_action)
            if not executable:
                continue
            if first_executable_action is None:
                first_executable_action = candidate_action
            if prediction_mode == "signal-risk-mask" and critic is not None:
                decision = critic.decision_for(row, candidate_action)
                if decision.hard_veto:
                    critic_vetoed_actions.append(candidate_action)
                    critic_veto_reasons.extend(decision.reasons)
                    risk_vetoed_candidates.append(
                        (_critic_risk_score(decision), candidate_action)
                    )
                    continue
            if prediction_mode == "action-critic-mask":
                if action_critic is None:
                    raise ValueError("action_critic is required for action-critic-mask")
                unsafe_probability = action_critic.unsafe_probability(
                    row,
                    candidate_action,
                )
                action_critic_candidates.append((unsafe_probability, candidate_action))
                if unsafe_probability >= action_critic.threshold:
                    critic_vetoed_actions.append(candidate_action)
                    critic_veto_reasons.append("action_critic_unsafe_probability_high")
                    action_critic_vetoed_probabilities.append(unsafe_probability)
                    continue
                return _prediction(
                    raw_predicted_action,
                    candidate_action,
                    critic_vetoed_actions=critic_vetoed_actions,
                    critic_veto_reasons=critic_veto_reasons,
                    action_critic_candidate_actions=[
                        action for _, action in action_critic_candidates
                    ],
                    action_critic_vetoed_probabilities=(
                        action_critic_vetoed_probabilities
                    ),
                    action_critic_selected_unsafe_probability=unsafe_probability,
                    action_critic_fallback_selected=False,
                )
            return _prediction(
                raw_predicted_action,
                candidate_action,
                critic_vetoed_actions=critic_vetoed_actions,
                critic_veto_reasons=critic_veto_reasons,
            )
        if first_executable_action is not None:
            fallback_probability: float | None = None
            if prediction_mode == "action-critic-mask" and action_critic_candidates:
                fallback_action = _action_critic_fallback_action(
                    row=row,
                    ranked_action_ids=ranked_action_ids,
                    first_executable_action=first_executable_action,
                    action_critic_candidates=action_critic_candidates,
                    fallback_policy=action_critic_fallback_policy,
                    critic=critic,
                )
                fallback_probability = {
                    action: probability
                    for probability, action in action_critic_candidates
                }.get(fallback_action)
            else:
                fallback_action = (
                    min(risk_vetoed_candidates)[1]
                    if risk_vetoed_candidates
                    else first_executable_action
                )
            return _prediction(
                raw_predicted_action,
                fallback_action,
                critic_vetoed_actions=critic_vetoed_actions,
                critic_veto_reasons=critic_veto_reasons,
                action_critic_candidate_actions=[
                    action for _, action in action_critic_candidates
                ],
                action_critic_vetoed_probabilities=(
                    action_critic_vetoed_probabilities
                ),
                action_critic_selected_unsafe_probability=fallback_probability,
                action_critic_fallback_selected=(
                    prediction_mode == "action-critic-mask"
                    and bool(action_critic_candidates)
                ),
                action_critic_fallback_policy_used=(
                    action_critic_fallback_policy
                    if prediction_mode == "action-critic-mask"
                    and bool(action_critic_candidates)
                    else None
                ),
            )
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def _action_critic_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    ranked_action_ids: list[int],
    first_executable_action: str,
    action_critic_candidates: list[tuple[float, str]],
    fallback_policy: str,
    critic: StrategySignalRiskCritic | None,
) -> str:
    if fallback_policy == "lowest-risk":
        return min(action_critic_candidates)[1]
    if fallback_policy == "first-executable":
        return first_executable_action
    if fallback_policy == "signal-risk":
        return _signal_risk_fallback_action(
            row=row,
            ranked_action_ids=ranked_action_ids,
            first_executable_action=first_executable_action,
            critic=critic,
        )
    if fallback_policy == "rule-safe":
        return _rule_safe_fallback_action(
            row=row,
            first_executable_action=first_executable_action,
            action_critic_candidates=action_critic_candidates,
        )
    if fallback_policy == "rule-risk":
        return _rule_risk_fallback_action(
            row=row,
            ranked_action_ids=ranked_action_ids,
            first_executable_action=first_executable_action,
            action_critic_candidates=action_critic_candidates,
            critic=critic,
        )
    if fallback_policy == "threat-risk":
        return _threat_risk_fallback_action(
            row=row,
            action_critic_candidates=action_critic_candidates,
            critic=critic,
        )
    if fallback_policy == "mixed-threat-risk":
        return _mixed_threat_risk_fallback_action(
            row=row,
            action_critic_candidates=action_critic_candidates,
        )
    raise ValueError(f"Unknown action critic fallback policy: {fallback_policy}")


def _rule_safe_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    first_executable_action: str,
    action_critic_candidates: list[tuple[float, str]],
) -> str:
    candidate_actions = {action for _, action in action_critic_candidates}
    if classify_threat_state(row) == "no_threat":
        return first_executable_action

    for action in _rule_safe_threat_fallback_order(row):
        if action in candidate_actions:
            return action
    return first_executable_action


def _rule_safe_threat_fallback_order(row: _StrategyOutcomeRow) -> tuple[str, ...]:
    threat_state = classify_threat_state(row)
    if threat_state == "air_threat":
        return (
            "BUILD_STATIC_DEFENSE",
            "PRODUCE_ARMY",
            "TECH_ROBO",
            "ADD_GATEWAYS",
            "FORGE_UPGRADES",
            "STAY_COURSE",
            "BOOST_WORKERS",
            "EXPAND",
        )
    if threat_state == "ground_threat":
        return (
            "BUILD_STATIC_DEFENSE",
            "PRODUCE_ARMY",
            "ADD_GATEWAYS",
            "FORGE_UPGRADES",
            "TECH_ROBO",
            "STAY_COURSE",
            "BOOST_WORKERS",
            "EXPAND",
        )
    return (
        "BUILD_STATIC_DEFENSE",
        "PRODUCE_ARMY",
        "ADD_GATEWAYS",
        "TECH_ROBO",
        "FORGE_UPGRADES",
        "STAY_COURSE",
        "BOOST_WORKERS",
        "EXPAND",
    )


def _rule_risk_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    ranked_action_ids: list[int],
    first_executable_action: str,
    action_critic_candidates: list[tuple[float, str]],
    critic: StrategySignalRiskCritic | None,
) -> str:
    candidate_actions = {action for _, action in action_critic_candidates}
    if classify_threat_state(row) == "no_threat":
        ordered_actions = tuple(
            _action_name(action_id)
            for action_id in ranked_action_ids
            if _action_name(action_id) in candidate_actions
        )
    else:
        ordered_actions = _rule_safe_threat_fallback_order(row)

    risk_vetoed_candidates: list[tuple[float, str]] = []
    for action in ordered_actions:
        if action not in candidate_actions:
            continue
        if critic is not None:
            decision = critic.decision_for(row, action)
            if decision.hard_veto:
                risk_vetoed_candidates.append((_critic_risk_score(decision), action))
                continue
        return action

    if risk_vetoed_candidates:
        return min(risk_vetoed_candidates)[1]
    return first_executable_action


def _threat_risk_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    action_critic_candidates: list[tuple[float, str]],
    critic: StrategySignalRiskCritic | None,
) -> str:
    lowest_risk_action = min(action_critic_candidates)[1]
    if classify_threat_state(row) == "no_threat" or lowest_risk_action != "STAY_COURSE":
        return lowest_risk_action

    risk_vetoed_candidates: list[tuple[float, str]] = []
    for unsafe_probability, action in sorted(action_critic_candidates):
        if action == "STAY_COURSE":
            continue
        if critic is not None:
            decision = critic.decision_for(row, action)
            if decision.hard_veto:
                risk_vetoed_candidates.append((_critic_risk_score(decision), action))
                continue
        return action

    if risk_vetoed_candidates:
        return min(risk_vetoed_candidates)[1]
    return lowest_risk_action


def _mixed_threat_risk_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    action_critic_candidates: list[tuple[float, str]],
) -> str:
    lowest_risk_action = min(action_critic_candidates)[1]
    if (
        classify_threat_state(row) != "air_and_ground_threat"
        or lowest_risk_action != "STAY_COURSE"
    ):
        return lowest_risk_action

    for _, action in sorted(action_critic_candidates):
        if action != "STAY_COURSE":
            return action
    return lowest_risk_action


def _signal_risk_fallback_action(
    *,
    row: _StrategyOutcomeRow,
    ranked_action_ids: list[int],
    first_executable_action: str,
    critic: StrategySignalRiskCritic | None,
) -> str:
    if critic is None:
        return first_executable_action
    risk_vetoed_candidates: list[tuple[float, str]] = []
    for action_id in ranked_action_ids:
        candidate_action = _action_name(action_id)
        executable, _ = candidate_executability(row, candidate_action)
        if not executable:
            continue
        decision = critic.decision_for(row, candidate_action)
        if decision.hard_veto:
            risk_vetoed_candidates.append(
                (_critic_risk_score(decision), candidate_action)
            )
            continue
        return candidate_action
    if risk_vetoed_candidates:
        return min(risk_vetoed_candidates)[1]
    return first_executable_action


def _critic_risk_score(decision) -> float:
    group = decision.matched_group
    if group is None:
        return 0.0
    return (2.0 * group.veto_negative_rate) + group.bad_rate


def _prediction(
    raw_predicted_action: str,
    predicted_action: str,
    *,
    critic_vetoed_actions: list[str] | None = None,
    critic_veto_reasons: list[str] | None = None,
    action_critic_candidate_actions: list[str] | None = None,
    action_critic_vetoed_probabilities: list[float] | None = None,
    action_critic_selected_unsafe_probability: float | None = None,
    action_critic_fallback_selected: bool = False,
    action_critic_fallback_policy_used: str | None = None,
) -> _CheckpointPrediction:
    return _CheckpointPrediction(
        raw_predicted_action=raw_predicted_action,
        predicted_action=predicted_action,
        prediction_was_masked=predicted_action != raw_predicted_action,
        critic_vetoed_actions=tuple(critic_vetoed_actions or ()),
        critic_veto_reasons=tuple(critic_veto_reasons or ()),
        action_critic_candidate_actions=tuple(
            action_critic_candidate_actions or ()
        ),
        action_critic_vetoed_probabilities=tuple(
            action_critic_vetoed_probabilities or ()
        ),
        action_critic_selected_unsafe_probability=(
            action_critic_selected_unsafe_probability
        ),
        action_critic_fallback_selected=action_critic_fallback_selected,
        action_critic_fallback_policy_used=action_critic_fallback_policy_used,
    )


@torch.no_grad()
def _checkpoint_logits(
    model: torch.nn.Module,
    observation: dict[str, float],
    *,
    normalizer: ObservationNormalizer | None,
    device: torch.device,
) -> torch.Tensor:
    vector = strategy_observation_dict_to_vector(observation)
    if normalizer is not None:
        vector = normalizer.transform(vector)
    tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return model(tensor).squeeze(0)


def _action_name(action_id: int) -> str:
    return STRATEGY_ACTION_NAMES.get(action_id, f"<unknown:{action_id}>")


def _decision(
    *,
    row: _StrategyOutcomeRow,
    signal: StrategySignalRecord,
    raw_predicted_action: str,
    predicted_action: str,
    raw_predicted_executable: bool,
    raw_predicted_blocker: str | None,
    predicted_executable: bool,
    predicted_blocker: str | None,
    prediction_was_masked: bool,
    critic_vetoed_actions: list[str],
    critic_veto_reasons: list[str],
    action_critic_candidate_actions: list[str],
    action_critic_vetoed_probabilities: list[float],
    action_critic_selected_unsafe_probability: float | None,
    action_critic_fallback_selected: bool,
    action_critic_fallback_policy_used: str | None,
) -> StrategyCheckpointSignalDecision:
    prediction_matches_recorded = predicted_action == signal.recorded_action
    return StrategyCheckpointSignalDecision(
        path=str(row.path),
        source=row.source,
        step=row.step,
        game_time=row.game_time,
        recorded_action=signal.recorded_action,
        raw_predicted_action=raw_predicted_action,
        predicted_action=predicted_action,
        recorded_training_use=signal.recommended_training_use,
        recorded_label_quality=signal.label_quality,
        recorded_immediate_executable=signal.immediate_executable,
        recorded_candidate_blocker=signal.candidate_blocker,
        raw_predicted_immediate_executable=raw_predicted_executable,
        raw_predicted_blocker=raw_predicted_blocker,
        predicted_immediate_executable=predicted_executable,
        predicted_blocker=predicted_blocker,
        prediction_was_masked=prediction_was_masked,
        critic_vetoed_actions=critic_vetoed_actions,
        critic_veto_reasons=critic_veto_reasons,
        action_critic_candidate_actions=action_critic_candidate_actions,
        action_critic_vetoed_probabilities=action_critic_vetoed_probabilities,
        action_critic_selected_unsafe_probability=(
            action_critic_selected_unsafe_probability
        ),
        action_critic_fallback_selected=action_critic_fallback_selected,
        action_critic_fallback_policy_used=action_critic_fallback_policy_used,
        prediction_matches_recorded=prediction_matches_recorded,
        prediction_matches_accept_positive=(
            signal.recommended_training_use == "accept_positive"
            and prediction_matches_recorded
        ),
        prediction_matches_bad_recorded=(
            signal.label_quality == "bad" and prediction_matches_recorded
        ),
        prediction_matches_veto_negative=(
            signal.recommended_training_use == "veto_negative"
            and prediction_matches_recorded
        ),
        prediction_matches_drop_non_executable=(
            signal.recommended_training_use == "drop_non_executable"
            and prediction_matches_recorded
        ),
        prediction_matches_action_space_exhausted=(
            signal.recommended_training_use == "action_space_exhausted"
            and prediction_matches_recorded
        ),
        context=classify_replay_context(row, predicted_action),
        threat_state=classify_threat_state(row),
    )


def _summarize(
    *,
    input_strings: list[str],
    checkpoint_path: str | Path,
    prediction_mode: str,
    action_critic_checkpoint_path: str | Path | None,
    action_critic_threshold: float | None,
    action_critic_fallback_policy: str | None,
    files: int,
    decisions: list[StrategyCheckpointSignalDecision],
    warnings: list[str],
    max_predicted_non_executable_ratio: float,
) -> StrategyCheckpointSignalAudit:
    rows = len(decisions)
    prediction_matches = sum(
        1 for decision in decisions if decision.prediction_matches_recorded
    )
    predicted_executable_rows = sum(
        1 for decision in decisions if decision.predicted_immediate_executable
    )
    predicted_non_executable_rows = rows - predicted_executable_rows
    raw_predicted_non_executable_rows = sum(
        1 for decision in decisions if not decision.raw_predicted_immediate_executable
    )
    masked_prediction_changes = sum(
        1 for decision in decisions if decision.prediction_was_masked
    )
    critic_vetoed_candidates = sum(
        len(decision.critic_vetoed_actions) for decision in decisions
    )
    selected_action_critic_probabilities = [
        decision.action_critic_selected_unsafe_probability
        for decision in decisions
        if decision.action_critic_selected_unsafe_probability is not None
    ]
    vetoed_action_critic_probabilities = [
        probability
        for decision in decisions
        for probability in decision.action_critic_vetoed_probabilities
    ]
    action_critic_fallback_rows = sum(
        1 for decision in decisions if decision.action_critic_fallback_selected
    )
    action_critic_fallback_policy_counts = _count(
        decision.action_critic_fallback_policy_used or "none"
        for decision in decisions
        if decision.action_critic_fallback_selected
    )
    accept_positive_rows = sum(
        1
        for decision in decisions
        if decision.recorded_training_use == "accept_positive"
    )
    accept_positive_matches = sum(
        1 for decision in decisions if decision.prediction_matches_accept_positive
    )
    bad_rows = sum(
        1 for decision in decisions if decision.recorded_label_quality == "bad"
    )
    bad_matches = sum(
        1 for decision in decisions if decision.prediction_matches_bad_recorded
    )
    veto_negative_rows = sum(
        1 for decision in decisions if decision.recorded_training_use == "veto_negative"
    )
    veto_negative_matches = sum(
        1 for decision in decisions if decision.prediction_matches_veto_negative
    )
    drop_non_executable_rows = sum(
        1
        for decision in decisions
        if decision.recorded_training_use == "drop_non_executable"
    )
    drop_non_executable_matches = sum(
        1 for decision in decisions if decision.prediction_matches_drop_non_executable
    )
    action_space_exhausted_rows = sum(
        1
        for decision in decisions
        if decision.recorded_training_use == "action_space_exhausted"
    )
    action_space_exhausted_matches = sum(
        1
        for decision in decisions
        if decision.prediction_matches_action_space_exhausted
    )
    predicted_non_executable_ratio = _ratio(predicted_non_executable_rows, rows)
    blocking_reasons = _blocking_reasons(
        rows=rows,
        predicted_non_executable_ratio=predicted_non_executable_ratio,
        max_predicted_non_executable_ratio=max_predicted_non_executable_ratio,
        veto_negative_matches=veto_negative_matches,
        drop_non_executable_matches=drop_non_executable_matches,
        action_space_exhausted_matches=action_space_exhausted_matches,
        action_critic_fallback_rows=action_critic_fallback_rows,
    )

    return StrategyCheckpointSignalAudit(
        inputs=input_strings,
        checkpoint_path=str(checkpoint_path),
        prediction_mode=prediction_mode,
        action_critic_checkpoint_path=(
            str(action_critic_checkpoint_path)
            if action_critic_checkpoint_path is not None
            else None
        ),
        action_critic_threshold=action_critic_threshold,
        action_critic_fallback_policy=action_critic_fallback_policy,
        files=files,
        rows=rows,
        signal_healthy=not blocking_reasons,
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        recorded_training_use_counts=_count(
            decision.recorded_training_use for decision in decisions
        ),
        recorded_label_quality_counts=_count(
            decision.recorded_label_quality for decision in decisions
        ),
        recorded_action_counts_by_name=_count(
            decision.recorded_action for decision in decisions
        ),
        raw_predicted_action_counts_by_name=_count(
            decision.raw_predicted_action for decision in decisions
        ),
        predicted_action_counts_by_name=_count(
            decision.predicted_action for decision in decisions
        ),
        prediction_matches_recorded=prediction_matches,
        prediction_match_ratio=_ratio(prediction_matches, rows),
        raw_predicted_non_executable_rows=raw_predicted_non_executable_rows,
        raw_predicted_non_executable_ratio=_ratio(
            raw_predicted_non_executable_rows,
            rows,
        ),
        raw_predicted_blocker_counts=_count(
            decision.raw_predicted_blocker or "none"
            for decision in decisions
            if not decision.raw_predicted_immediate_executable
        ),
        predicted_executable_rows=predicted_executable_rows,
        predicted_non_executable_rows=predicted_non_executable_rows,
        predicted_non_executable_ratio=predicted_non_executable_ratio,
        predicted_blocker_counts=_count(
            decision.predicted_blocker or "none"
            for decision in decisions
            if not decision.predicted_immediate_executable
        ),
        predicted_non_executable_by_recorded_training_use=_count(
            decision.recorded_training_use
            for decision in decisions
            if not decision.predicted_immediate_executable
        ),
        masked_prediction_changes=masked_prediction_changes,
        masked_prediction_change_ratio=_ratio(masked_prediction_changes, rows),
        critic_vetoed_candidates=critic_vetoed_candidates,
        critic_veto_reason_counts=_count(
            reason
            for decision in decisions
            for reason in decision.critic_veto_reasons
        ),
        critic_veto_action_counts=_count(
            action
            for decision in decisions
            for action in decision.critic_vetoed_actions
        ),
        action_critic_selected_unsafe_probability_avg=_mean_or_none(
            selected_action_critic_probabilities
        ),
        action_critic_selected_unsafe_probability_max=_max_or_none(
            selected_action_critic_probabilities
        ),
        action_critic_vetoed_probability_avg=_mean_or_none(
            vetoed_action_critic_probabilities
        ),
        action_critic_vetoed_probability_max=_max_or_none(
            vetoed_action_critic_probabilities
        ),
        action_critic_fallback_rows=action_critic_fallback_rows,
        action_critic_fallback_policy_counts=action_critic_fallback_policy_counts,
        accept_positive_rows=accept_positive_rows,
        accept_positive_prediction_matches=accept_positive_matches,
        accept_positive_prediction_match_ratio=_ratio(
            accept_positive_matches,
            accept_positive_rows,
        ),
        bad_recorded_rows=bad_rows,
        bad_recorded_prediction_matches=bad_matches,
        bad_recorded_prediction_match_ratio=_ratio(bad_matches, bad_rows),
        veto_negative_rows=veto_negative_rows,
        veto_negative_prediction_matches=veto_negative_matches,
        veto_negative_prediction_match_ratio=_ratio(
            veto_negative_matches,
            veto_negative_rows,
        ),
        drop_non_executable_rows=drop_non_executable_rows,
        drop_non_executable_prediction_matches=drop_non_executable_matches,
        drop_non_executable_prediction_match_ratio=_ratio(
            drop_non_executable_matches,
            drop_non_executable_rows,
        ),
        action_space_exhausted_rows=action_space_exhausted_rows,
        action_space_exhausted_prediction_matches=action_space_exhausted_matches,
        action_space_exhausted_prediction_match_ratio=_ratio(
            action_space_exhausted_matches,
            action_space_exhausted_rows,
        ),
        decisions=decisions,
    )


def _blocking_reasons(
    *,
    rows: int,
    predicted_non_executable_ratio: float,
    max_predicted_non_executable_ratio: float,
    veto_negative_matches: int,
    drop_non_executable_matches: int,
    action_space_exhausted_matches: int,
    action_critic_fallback_rows: int,
) -> list[str]:
    reasons: list[str] = []
    if rows <= 0:
        reasons.append("no_rows")
        return reasons
    if predicted_non_executable_ratio > max_predicted_non_executable_ratio:
        reasons.append("predicted_non_executable_ratio_high")
    if veto_negative_matches > 0:
        reasons.append("predicted_matches_veto_negative_labels")
    if drop_non_executable_matches > 0:
        reasons.append("predicted_matches_drop_non_executable_labels")
    if action_space_exhausted_matches > 0:
        reasons.append("predicted_matches_action_space_exhausted_labels")
    if action_critic_fallback_rows > 0:
        reasons.append("action_critic_all_executable_candidates_vetoed")
    return reasons


def _row_key(row: _StrategyOutcomeRow) -> tuple[str, int, str]:
    return _record_key(row.path, row.step, row.action_name)


def _record_key(path: str | Path, step: int, action_name: str) -> tuple[str, int, str]:
    return (str(Path(path).resolve()), int(step), str(action_name))


def _input_paths(paths: StrategyTrajectoryPathInput) -> tuple[str | Path, ...]:
    if isinstance(paths, (str, Path)):
        return (paths,)
    if not isinstance(paths, IterableABC):
        raise TypeError("paths must be a path or an iterable of paths")
    return tuple(paths)


def _count(values) -> dict[str, int]:
    return dict(sorted((name, int(count)) for name, count in Counter(values).items()))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))
