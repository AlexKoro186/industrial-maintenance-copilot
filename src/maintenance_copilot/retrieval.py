"""Section-aware retrieval baseline for the demo knowledge base."""

import hashlib
import re

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer


EQUIPMENT_SCOPES = {
    "hydraulic-test-rig": "hydraulic_test_rig_demo",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    source_file: str
    source_sha256: str
    document_type: str
    equipment_scope: str
    revision: str
    text: str


def load_chunks(
    directory: str | Path,
    max_words=120,
    overlap=20,
) -> list[Chunk]:
    """Read YAML metadata and split each Markdown section into word windows."""
    if not 0 <= overlap < max_words:
        raise ValueError(
            "Require 0 <= overlap < max_words."
        )

    directory = Path(directory)
    chunks = []
    seen_ids = set()

    fields = (
        "document_id",
        "title",
        "document_type",
        "equipment_scope",
        "revision",
    )

    for path in sorted(directory.rglob("*.md")):
        raw = path.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")

        match = re.match(
            r"\A---\n(.*?)\n---\n",
            text,
            flags=re.DOTALL,
        )

        if match is None:
            raise ValueError(
                f"Missing YAML metadata: {path.name}"
            )

        metadata = yaml.safe_load(match.group(1))

        if not isinstance(metadata, dict) or any(
            not isinstance(metadata.get(key), str)
            or not metadata[key].strip()
            for key in fields
        ):
            raise ValueError(
                f"Invalid document metadata: {path.name}"
            )

        document_id = metadata["document_id"]

        if document_id in seen_ids:
            raise ValueError(
                f"Duplicate document_id: {document_id}"
            )

        seen_ids.add(document_id)

        checksum = hashlib.sha256(raw).hexdigest()

        sections = re.split(
            r"(?m)^##[ \t]+([^\n]+)\n",
            text[match.end():],
        )

        count_before = len(chunks)

        for offset in range(1, len(sections), 2):
            section = sections[offset].strip()
            words = sections[offset + 1].split()

            for part, start in enumerate(
                range(0, len(words), max_words - overlap),
                1,
            ):
                excerpt = " ".join(
                    words[start:start + max_words]
                )

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document_id}:"
                            f"s{(offset + 1) // 2}:"
                            f"c{part}:{checksum[:12]}"
                        ),
                        section=section,
                        source_file=(
                            path.relative_to(directory).as_posix()
                        ),
                        source_sha256=checksum,
                        text=excerpt,
                        **{
                            key: metadata[key]
                            for key in fields
                        },
                    )
                )

                if start + max_words >= len(words):
                    break

        if len(chunks) == count_before:
            raise ValueError(
                f"No nonempty ## sections found: {path.name}"
            )

    if not chunks:
        raise ValueError(
            f"No knowledge-base documents found in {directory}."
        )

    return chunks


class DocumentRetriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError(
                "At least one document chunk is required."
            )

        self.chunks = list(chunks)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            [
                f"{chunk.title}\n{chunk.section}\n{chunk.text}"
                for chunk in self.chunks
            ]
        )

    def search(
        self,
        query: str,
        *,
        equipment_scope: str,
        top_k=3,
        min_score=0.08,
    ) -> list[dict]:
        """Return scoped passages; similarity scores are not confidence probabilities."""
        if top_k < 1 or not 0 < min_score <= 1:
            raise ValueError(
                "Require top_k >= 1 and 0 < min_score <= 1."
            )

        if not query.strip():
            return []

        vector = self.vectorizer.transform([query])

        if vector.nnz == 0:
            return []

        scores = (
            self.matrix @ vector.T
        ).toarray().ravel()

        results = []

        for index in np.argsort(-scores, kind="stable"):
            chunk = self.chunks[index]

            if (
                chunk.equipment_scope != equipment_scope
                or scores[index] < min_score
            ):
                continue

            results.append(
                {
                    **asdict(chunk),
                    "retrieval_score": float(scores[index]),
                }
            )

            if len(results) == top_k:
                break

        return results


def retrieve_for_assessment(
    event: dict,
    retriever: DocumentRetriever,
) -> dict:
    """Build document context for the existing cooler assessment format."""
    if (
        event.get("schema_version") != 1
        or event.get("event_type") != "cycle_assessment"
        or event.get("component_scope") != "cooler"
        or event.get("operating_conditions") != "stable"
    ):
        raise ValueError(
            "Expected a stable, version 1 cooler assessment."
        )

    scope = EQUIPMENT_SCOPES.get(
        event.get("machine_id")
    )

    if scope is None:
        raise ValueError(
            "No document scope configured for this machine_id."
        )

    try:
        flags = [
            event["detectors"][name]["is_anomaly"]
            for name in (
                "isolation_forest",
                "ts1_mean_threshold",
            )
        ]

        deltas = np.array(
            [
                float(row["deviation_c"])
                for row in event["temperature_evidence"]
            ]
        )

    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid detector results or temperature evidence."
        ) from exc

    if (
        any(type(flag) is not bool for flag in flags)
        or len(deltas) != 4
        or not np.isfinite(deltas).all()
    ):
        raise ValueError(
            "Expected boolean alerts and four finite temperature deviations."
        )

    output = {
        "schema_version": 1,
        "assessment": event,
        "equipment_scope": scope,
        "retrieval_method": "tfidf_cosine",
        "query": None,
        "status": "no_alert",
        "sources": [],
        "root_cause_status": "not_assessed",
    }

    if not any(flags):
        return output

    query = (
        "cooler temperature deviation sensor plausibility "
        "operating conditions"
    )

    if (deltas > 0).any():
        query += " elevated temperature cooling heat load"

    sources = retriever.search(
        query,
        equipment_scope=scope,
    )

    output.update(
        query=query,
        sources=sources,
        status=(
            "context_found"
            if sources
            else "no_relevant_context"
        ),
    )

    return output