"""Test generation gates and citation validation without a running LLM."""

import io
import json
from unittest.mock import patch

import pytest

from maintenance_copilot.copilot import (
    generate_report,
    request_ollama,
    validate_answer,
)


SOURCE = {
    "chunk_id": "DEMO-COOLING:s1:c1:example",
    "document_id": "DEMO-COOLING",
    "document_type": "synthetic_demo",
    "equipment_scope": "hydraulic_test_rig_demo",
    "text": "Compare recorded temperature trends with the nominal-cooler reference.",
}


def make_context(alert=True, with_sources=True):
    sources = [SOURCE.copy()] if with_sources else []
    status = (
        "no_alert"
        if not alert
        else "context_found" if sources else "no_relevant_context"
    )
    return {
        "schema_version": 1,
        "equipment_scope": "hydraulic_test_rig_demo",
        "status": status,
        "assessment": {
            "schema_version": 1,
            "event_type": "cycle_assessment",
            "machine_id": "hydraulic-test-rig",
            "cycle_id": 283,
            "component_scope": "cooler",
            "operating_conditions": "stable",
            "detectors": {
                "isolation_forest": {"is_anomaly": alert},
                "ts1_mean_threshold": {"is_anomaly": alert},
            },
        },
        "sources": sources,
    }


def make_answer():
    return {
        "explanations": [],
        "review_suggestions": [
            {
                "text": "Review temperature trends against the nominal-cooler reference.",
                "source_id": SOURCE["chunk_id"],
                "quote": SOURCE["text"],
            }
        ],
    }


def test_valid_answer_preserves_assessment_and_sources():
    context = make_context()

    with patch(
        "maintenance_copilot.copilot.request_ollama",
        return_value=make_answer(),
    ):
        report = generate_report(context)

    assert report["status"] == "generated"
    assert report["assessment"] == context["assessment"]
    assert report["sources"] == context["sources"]
    assert report["root_cause_status"] == "not_assessed"
    assert any("synthetic" in text for text in report["limitations"])


@pytest.mark.parametrize(
    ("alert", "with_sources", "expected_status"),
    [
        (False, False, "no_alert"),
        (True, False, "no_relevant_context"),
    ],
)
def test_generation_gate_skips_ollama(alert, with_sources, expected_status):
    with patch("maintenance_copilot.copilot.request_ollama") as request:
        report = generate_report(make_context(alert, with_sources))

    request.assert_not_called()
    assert report["status"] == expected_status
    assert report["requested_model"] is None


def test_unknown_source_is_rejected():
    answer = make_answer()
    answer["review_suggestions"][0]["source_id"] = "INVENTED-SOURCE"

    with pytest.raises(ValueError, match="Unknown source"):
        validate_answer(answer, [SOURCE])


def test_fabricated_quote_is_rejected():
    answer = make_answer()
    answer["review_suggestions"][0]["quote"] = "Replace the cooling unit immediately."

    with pytest.raises(ValueError, match="Quote not found"):
        validate_answer(answer, [SOURCE])


def test_unexpected_response_fields_are_rejected():
    answer = make_answer()
    answer["failure_probability"] = 0.84

    with pytest.raises(ValueError, match="unexpected fields"):
        validate_answer(answer, [SOURCE])


def test_conflicting_context_status_is_rejected():
    context = make_context()
    context["status"] = "no_alert"

    with pytest.raises(ValueError, match="conflicts"):
        generate_report(context)


def test_empty_supported_answer_reports_insufficient_evidence():
    with patch(
        "maintenance_copilot.copilot.request_ollama",
        return_value={"explanations": [], "review_suggestions": []},
    ):
        report = generate_report(make_context())

    assert report["status"] == "insufficient_evidence"


def test_ollama_response_is_parsed():
    response = {
        "done": True,
        "done_reason": "stop",
        "message": {"content": json.dumps(make_answer())},
    }
    stream = io.BytesIO(json.dumps(response).encode("utf-8"))

    with patch("maintenance_copilot.copilot.urlopen", return_value=stream):
        answer = request_ollama({"model": "example", "stream": False})

    assert answer == make_answer()


def test_truncated_ollama_response_is_rejected():
    response = {"done": True, "done_reason": "length"}
    stream = io.BytesIO(json.dumps(response).encode("utf-8"))

    with patch("maintenance_copilot.copilot.urlopen", return_value=stream):
        with pytest.raises(ValueError, match="generation limit"):
            request_ollama({"model": "example"})