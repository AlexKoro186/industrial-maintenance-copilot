"""Tests for passage provenance, retrieval scope, and abstention."""

import tempfile
import unittest
from pathlib import Path

from maintenance_copilot.retrieval import (
    DocumentRetriever,
    load_chunks,
    retrieve_for_assessment,
)


def write_document(
    directory,
    name,
    document_id,
    body,
    scope="hydraulic_test_rig_demo",
):
    path = Path(directory) / name

    path.write_text(
        f"---\n"
        f"document_id: {document_id}\n"
        f"title: Reference example\n"
        f"document_type: synthetic_demo\n"
        f"equipment_scope: {scope}\n"
        f'revision: "1"\n'
        f"---\n"
        f"# Reference example\n\n"
        f"## Review\n"
        f"{body}\n",
        encoding="utf-8",
    )

    return path


def make_event(alert=True):
    return {
        "schema_version": 1,
        "event_type": "cycle_assessment",
        "machine_id": "hydraulic-test-rig",
        "component_scope": "cooler",
        "operating_conditions": "stable",
        "detectors": {
            "isolation_forest": {
                "is_anomaly": alert,
            },
            "ts1_mean_threshold": {
                "is_anomaly": alert,
            },
        },
        "temperature_evidence": [
            {"deviation_c": 10.0}
            for _ in range(4)
        ],
    }


class RetrievalTests(unittest.TestCase):
    def test_relevant_passage_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "cooler.md",
                "COOLER",
                "Elevated cooler temperature and cooling flow.",
            )

            write_document(
                directory,
                "other.md",
                "OTHER",
                "Warehouse inventory and packaging labels.",
            )

            search = DocumentRetriever(
                load_chunks(directory)
            )

            result = search.search(
                "elevated cooler temperature",
                equipment_scope="hydraulic_test_rig_demo",
            )

            self.assertEqual(
                result[0]["document_id"], "COOLER"
            )
            self.assertEqual(
                result[0]["section"], "Review"
            )
            self.assertEqual(
                result[0]["source_file"], "cooler.md"
            )
            self.assertEqual(
                result[0]["document_type"], "synthetic_demo"
            )
            self.assertEqual(
                len(result[0]["source_sha256"]), 64
            )

    def test_unrelated_query_returns_no_passages(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "cooler.md",
                "COOLER",
                "Cooler temperature.",
            )

            search = DocumentRetriever(
                load_chunks(directory)
            )

            result = search.search(
                "astronomy galaxies",
                equipment_scope="hydraulic_test_rig_demo",
            )

            self.assertEqual(result, [])

    def test_wrong_equipment_scope_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "other.md",
                "OTHER",
                "Elevated cooler temperature.",
                scope="other_machine",
            )

            search = DocumentRetriever(
                load_chunks(directory)
            )

            result = search.search(
                "cooler temperature",
                equipment_scope="hydraulic_test_rig_demo",
            )

            self.assertEqual(result, [])

    def test_document_edit_changes_chunk_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_document(
                directory,
                "cooler.md",
                "COOLER",
                "Cooler temperature.",
            )

            before = load_chunks(directory)[0]

            path.write_text(
                path.read_text() + "Additional evidence.\n",
                encoding="utf-8",
            )

            after = load_chunks(directory)[0]

            self.assertNotEqual(
                before.chunk_id,
                after.chunk_id,
            )
            self.assertNotEqual(
                before.source_sha256,
                after.source_sha256,
            )

    def test_chunk_windows_preserve_overlap_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            words = [
                f"word{i}"
                for i in range(23)
            ]

            write_document(
                directory,
                "long.md",
                "LONG",
                " ".join(words),
            )

            chunks = load_chunks(
                directory,
                max_words=10,
                overlap=2,
            )

            self.assertEqual(len(chunks), 3)

            self.assertEqual(
                chunks[0].text.split()[-2:],
                chunks[1].text.split()[:2],
            )

            self.assertEqual(
                chunks[-1].text.split()[-1],
                "word22",
            )

    def test_duplicate_document_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "first.md",
                "SAME",
                "Cooler temperature.",
            )

            write_document(
                directory,
                "second.md",
                "SAME",
                "Sensor readings.",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Duplicate",
            ):
                load_chunks(directory)

    def test_no_alert_does_not_produce_troubleshooting_context(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "cooler.md",
                "COOLER",
                "Cooler temperature.",
            )

            search = DocumentRetriever(
                load_chunks(directory)
            )

            result = retrieve_for_assessment(
                make_event(alert=False),
                search,
            )

            self.assertEqual(
                result["status"],
                "no_alert",
            )

            self.assertEqual(
                result["sources"],
                [],
            )

    def test_missing_context_does_not_create_a_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            write_document(
                directory,
                "other.md",
                "OTHER",
                "Warehouse inventory and packaging.",
            )

            search = DocumentRetriever(
                load_chunks(directory)
            )

            result = retrieve_for_assessment(
                make_event(),
                search,
            )

            self.assertEqual(
                result["status"],
                "no_relevant_context",
            )

            self.assertEqual(
                result["root_cause_status"],
                "not_assessed",
            )


if __name__ == "__main__":
    unittest.main()