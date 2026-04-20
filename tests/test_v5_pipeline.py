from __future__ import annotations

from scripts.run_v5_sample import build_sample_bundle
from src.v5 import run_v5_pipeline


def test_v5_pipeline_smoke() -> None:
    outputs = run_v5_pipeline(build_sample_bundle())
    assert "v5_topics" in outputs
    assert "v5_signal_units" in outputs
    assert len(outputs["v5_topics"]) > 0
    assert len(outputs["v5_signal_units"]) >= len(outputs["v5_comment_context"])
    assert not outputs["v5_topic_insight_actions"].empty
    required = {"topic_id", "insight_text", "action_text", "priority_level", "evidence_strength", "supporting_comment_ids"}
    assert required.issubset(set(outputs["v5_topic_insight_actions"].columns))
