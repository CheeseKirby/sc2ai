from __future__ import annotations

import json

import pytest

from rl.strategy_lab import (
    HeuristicStrategyLabPolicy,
    StayCourseStrategyLabPolicy,
    benchmark_strategy_policies,
)


@pytest.mark.unit
def test_strategy_lab_benchmark_emits_comparable_metrics_and_traces(tmp_path) -> None:
    trace_path = tmp_path / "decisions.jsonl"

    report = benchmark_strategy_policies(
        {
            "heuristic": HeuristicStrategyLabPolicy(),
            "stay_course": StayCourseStrategyLabPolicy(),
        },
        episodes_per_scenario=2,
        seed=17,
        trace_path=trace_path,
    )

    assert report["schema_version"] == "strategy_lab_v1"
    assert report["scenario_count"] >= 4
    assert report["policies"]["heuristic"]["episodes"] == (
        report["scenario_count"] * 2
    )
    assert report["policies"]["heuristic"]["mean_reward"] > (
        report["policies"]["stay_course"]["mean_reward"]
    )
    assert 0.0 <= report["policies"]["heuristic"]["blocked_action_rate"] <= 1.0
    assert report["policies"]["heuristic"]["latency_ms"]["p95"] >= 0.0

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows
    assert {
        "policy",
        "scenario",
        "step",
        "action",
        "reward",
        "reward_components",
        "latency_ms",
    } <= rows[0].keys()

class BrokenPolicy:
    def decide(self, observation: dict[str, float]) -> None:
        del observation
        raise TimeoutError("planner timed out")


@pytest.mark.unit
def test_strategy_lab_contains_policy_failures_with_safe_fallback(tmp_path) -> None:
    report = benchmark_strategy_policies(
        {"broken": BrokenPolicy()},
        episodes_per_scenario=1,
        max_steps=2,
        trace_path=tmp_path / "fallback.jsonl",
    )

    metrics = report["policies"]["broken"]
    assert metrics["fallback_count"] == metrics["decisions"]
    assert metrics["action_distribution"]["STAY_COURSE"] == metrics["decisions"]
