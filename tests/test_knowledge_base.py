"""Regression checks for the repository's synthetic demo knowledge base."""

import unittest
from pathlib import Path

from maintenance_copilot.retrieval import (
    DocumentRetriever,
    load_chunks,
    retrieve_for_assessment,
)


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "docs" / "knowledge_base"


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks(KNOWLEDGE_DIR)
        cls.retriever = DocumentRetriever(cls.chunks)

    def test_required_demo_sections_are_present(self):
        expected = {
            ("DEMO-COOLING", "Elevated temperature"),
            ("DEMO-COOLING", "Additional cooling evidence"),
            ("DEMO-COOLING", "Machine-specific limits"),
            ("DEMO-SENSORS", "Sensor plausibility"),
            ("DEMO-SENSORS", "Reference and operating conditions"),
        }
        actual = {(chunk.document_id, chunk.section) for chunk in self.chunks}
        self.assertFalse(expected - actual, f"Missing demo sections: {expected - actual}")

    def test_cooling_context_is_found_for_elevated_temperature(self):
        event = {
            "schema_version": 1,
            "event_type": "cycle_assessment",
            "machine_id": "hydraulic-test-rig",
            "component_scope": "cooler",
            "operating_conditions": "stable",
            "detectors": {
                "isolation_forest": {"is_anomaly": True},
                "ts1_mean_threshold": {"is_anomaly": True},
            },
            "temperature_evidence": [{"deviation_c": 18.0} for _ in range(4)],
        }
        result = retrieve_for_assessment(event, self.retriever)
        found = {(item["document_id"], item["section"]) for item in result["sources"]}

        self.assertEqual(result["status"], "context_found")
        self.assertIn(("DEMO-COOLING", "Elevated temperature"), found)

    def test_reference_comparison_passage_is_retrievable(self):
        results = self.retriever.search(
            "compare complete temperature cycles reference stable operating conditions",
            equipment_scope="hydraulic_test_rig_demo",
        )
        found = {(item["document_id"], item["section"]) for item in results}
        self.assertIn(("DEMO-SENSORS", "Reference and operating conditions"), found)

    def test_sensor_quality_passage_is_retrievable(self):
        results = self.retriever.search(
            "sensor plausibility missing values abrupt changes channels change together",
            equipment_scope="hydraulic_test_rig_demo",
        )
        found = {(item["document_id"], item["section"]) for item in results}
        self.assertIn(("DEMO-SENSORS", "Sensor plausibility"), found)


if __name__ == "__main__":
    unittest.main()