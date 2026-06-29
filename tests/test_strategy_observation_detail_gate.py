from __future__ import annotations

from dataclasses import asdict

import pytest

from rl.experiments import write_json
from rl.strategy_observation_detail_gate import (
    StrategyObservationDetailGateConfig,
    evaluate_strategy_observation_detail_gate,
)
from scripts.gate_strategy_observation_details import (
    format_strategy_observation_detail_gate,
)


def _analysis(**overrides) -> dict:
    payload = {
        "inputs": ["data/trajectories/new_strategy_details"],
        "rows": 10,
        "observation_detail_rows": 10,
        "observation_detail_ratio": 1.0,
        "threatened_only_stay_course_rows": 2,
        "threatened_only_stay_course_detail_rows": 2,
        "threatened_only_stay_course_detail_ratio": 1.0,
        "air_threat_only_stay_course_rows": 1,
        "air_threat_only_stay_course_detail_rows": 1,
        "air_threat_only_stay_course_detail_ratio": 1.0,
        "unaddressed_air_defense_gap_by_reason": {},
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_observation_detail_gate_promotes_complete_detail_coverage(tmp_path) -> None:
    analysis_path = tmp_path / "emergency_analysis.json"
    write_json(analysis_path, _analysis())

    result = evaluate_strategy_observation_detail_gate(analysis_path)

    assert result.recommendation == "ready"
    assert result.ready is True
    assert result.blocking_reasons == []
    assert result.analysis_path == str(analysis_path)
    assert result.analysis_inputs == ["data/trajectories/new_strategy_details"]
    assert result.observation_detail_rows == 10
    assert result.air_threat_only_stay_course_detail_ratio == 1.0
    assert asdict(result)["static_defense_type_ambiguous_rows"] == 0


@pytest.mark.unit
def test_observation_detail_gate_holds_old_or_ambiguous_data(tmp_path) -> None:
    analysis_path = tmp_path / "old_analysis.json"
    write_json(
        analysis_path,
        _analysis(
            rows=330,
            observation_detail_rows=0,
            observation_detail_ratio=0.0,
            threatened_only_stay_course_rows=38,
            threatened_only_stay_course_detail_rows=0,
            threatened_only_stay_course_detail_ratio=0.0,
            air_threat_only_stay_course_rows=14,
            air_threat_only_stay_course_detail_rows=0,
            air_threat_only_stay_course_detail_ratio=0.0,
            unaddressed_air_defense_gap_by_reason={
                "static_defense_type_ambiguous": 11,
                "no_observed_anti_air_assets": 2,
            },
        ),
    )

    result = evaluate_strategy_observation_detail_gate(analysis_path)

    assert result.recommendation == "hold"
    assert result.ready is False
    assert result.blocking_reasons == [
        "observation_detail_coverage_low",
        "threatened_only_stay_detail_coverage_low",
        "air_threat_only_stay_detail_coverage_low",
        "static_defense_type_ambiguous_rows_high",
    ]
    assert result.static_defense_type_ambiguous_rows == 11


@pytest.mark.unit
def test_observation_detail_gate_ignores_empty_relevant_slices(tmp_path) -> None:
    analysis_path = tmp_path / "no_threat_analysis.json"
    write_json(
        analysis_path,
        _analysis(
            rows=5,
            observation_detail_rows=5,
            observation_detail_ratio=1.0,
            threatened_only_stay_course_rows=0,
            threatened_only_stay_course_detail_rows=0,
            threatened_only_stay_course_detail_ratio=0.0,
            air_threat_only_stay_course_rows=0,
            air_threat_only_stay_course_detail_rows=0,
            air_threat_only_stay_course_detail_ratio=0.0,
        ),
    )

    result = evaluate_strategy_observation_detail_gate(analysis_path)

    assert result.recommendation == "ready"
    assert result.blocking_reasons == []


@pytest.mark.unit
def test_observation_detail_gate_respects_custom_thresholds(tmp_path) -> None:
    analysis_path = tmp_path / "partial_analysis.json"
    write_json(
        analysis_path,
        _analysis(
            observation_detail_rows=8,
            observation_detail_ratio=0.8,
            threatened_only_stay_course_detail_rows=1,
            threatened_only_stay_course_detail_ratio=0.5,
            air_threat_only_stay_course_detail_rows=1,
            air_threat_only_stay_course_detail_ratio=1.0,
        ),
    )

    result = evaluate_strategy_observation_detail_gate(
        analysis_path,
        config=StrategyObservationDetailGateConfig(
            min_observation_detail_ratio=0.75,
            min_threatened_only_stay_course_detail_ratio=0.5,
            min_air_threat_only_stay_course_detail_ratio=1.0,
        ),
    )

    assert result.recommendation == "ready"
    assert result.ready is True


@pytest.mark.unit
def test_format_observation_detail_gate_includes_thresholds(tmp_path) -> None:
    analysis_path = tmp_path / "old_analysis.json"
    write_json(
        analysis_path,
        _analysis(observation_detail_rows=0, observation_detail_ratio=0.0),
    )

    report = format_strategy_observation_detail_gate(
        evaluate_strategy_observation_detail_gate(analysis_path)
    )

    assert "Strategy observation detail gate" in report
    assert "recommendation: hold" in report
    assert "observation_details: 0/10 ratio=0.000" in report
    assert "inputs: data/trajectories/new_strategy_details" in report
    assert "min_observation_detail_ratio=1.000" in report
