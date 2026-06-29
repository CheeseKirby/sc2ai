"""One-shot offline strategy data readiness pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rl.experiments import write_json
from rl.strategy_emergency_action_analysis import analyze_strategy_emergency_actions
from rl.strategy_observation_detail_gate import (
    StrategyObservationDetailGateConfig,
    evaluate_strategy_observation_detail_payload,
)
from rl.strategy_policy_explanation_gate import (
    StrategyPolicyExplanationGateConfig,
    evaluate_strategy_policy_explanation_gate,
)
from rl.strategy_training_readiness import (
    StrategyTrainingReadinessConfig,
    evaluate_strategy_training_readiness,
)
from rl.strategy_trajectory_detail_gate import (
    StrategyTrajectoryDetailGateConfig,
    evaluate_strategy_trajectory_detail_gate,
)
from rl.strategy_datasets import StrategyTrajectoryPathInput


@dataclass(frozen=True)
class StrategyDataReadinessPipelineResult:
    """Summary of all offline data-readiness artifacts."""

    recommendation: str
    training_ready: bool
    trajectory_detail_ready: bool
    policy_explanation_ready: bool
    observation_detail_ready: bool
    promotion_ready: bool | None
    blocking_reasons: list[str]
    inputs: list[str]
    artifacts: dict[str, str]


def run_strategy_data_readiness_pipeline(
    inputs: StrategyTrajectoryPathInput,
    *,
    output_dir: str | Path,
    prefix: str = "strategy_data_readiness",
    max_examples: int = 12,
    promotion_gate_path: str | Path | None = None,
    trajectory_detail_config: StrategyTrajectoryDetailGateConfig | None = None,
    policy_explanation_config: StrategyPolicyExplanationGateConfig | None = None,
    observation_detail_config: StrategyObservationDetailGateConfig | None = None,
) -> StrategyDataReadinessPipelineResult:
    """Run raw trajectory, emergency, observation, and readiness checks."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(out_dir, prefix)

    trajectory_gate = evaluate_strategy_trajectory_detail_gate(
        inputs,
        config=trajectory_detail_config,
    )
    write_json(artifacts["trajectory_detail_gate"], asdict(trajectory_gate))

    explanation_gate = evaluate_strategy_policy_explanation_gate(
        inputs,
        config=policy_explanation_config,
    )
    write_json(artifacts["policy_explanation_gate"], asdict(explanation_gate))

    emergency_analysis = analyze_strategy_emergency_actions(
        inputs,
        max_examples=max_examples,
    )
    emergency_payload = asdict(emergency_analysis)
    write_json(artifacts["emergency_analysis"], emergency_payload)

    observation_gate = evaluate_strategy_observation_detail_payload(
        emergency_payload,
        analysis_path=str(artifacts["emergency_analysis"]),
        config=observation_detail_config or StrategyObservationDetailGateConfig(),
    )
    write_json(artifacts["observation_detail_gate"], asdict(observation_gate))

    readiness = evaluate_strategy_training_readiness(
        artifacts["observation_detail_gate"],
        config=StrategyTrainingReadinessConfig(
            expected_inputs=tuple(trajectory_gate.inputs),
            trajectory_detail_gate_path=str(artifacts["trajectory_detail_gate"]),
            policy_explanation_gate_path=str(artifacts["policy_explanation_gate"]),
            promotion_gate_path=(
                str(promotion_gate_path) if promotion_gate_path is not None else None
            ),
        ),
    )
    write_json(artifacts["training_readiness"], asdict(readiness))

    result = StrategyDataReadinessPipelineResult(
        recommendation=readiness.recommendation,
        training_ready=readiness.training_ready,
        trajectory_detail_ready=trajectory_gate.ready,
        policy_explanation_ready=explanation_gate.ready,
        observation_detail_ready=observation_gate.ready,
        promotion_ready=readiness.promotion_ready,
        blocking_reasons=readiness.blocking_reasons,
        inputs=trajectory_gate.inputs,
        artifacts={name: str(path) for name, path in artifacts.items()},
    )
    write_json(artifacts["summary"], asdict(result))
    return result


def _artifact_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    safe_prefix = prefix.strip() or "strategy_data_readiness"
    return {
        "trajectory_detail_gate": output_dir
        / f"{safe_prefix}_trajectory_detail_gate.json",
        "policy_explanation_gate": output_dir
        / f"{safe_prefix}_policy_explanation_gate.json",
        "emergency_analysis": output_dir
        / f"{safe_prefix}_emergency_action_analysis.json",
        "observation_detail_gate": output_dir
        / f"{safe_prefix}_observation_detail_gate.json",
        "training_readiness": output_dir
        / f"{safe_prefix}_training_readiness.json",
        "summary": output_dir / f"{safe_prefix}_summary.json",
    }
