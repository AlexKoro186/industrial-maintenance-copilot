"""Retrieve source passages for a saved cycle assessment."""

import argparse
import json
from pathlib import Path

import yaml

from maintenance_copilot.retrieval import (
    DocumentRetriever,
    load_chunks,
    retrieve_for_assessment,
)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--assessment",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=root / "docs/knowledge_base",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args(argv)

    try:
        event = json.loads(
            args.assessment.read_text(encoding="utf-8")
        )

        if not isinstance(event, dict):
            raise ValueError(
                "The assessment must be a JSON object."
            )

        retriever = DocumentRetriever(
            load_chunks(args.knowledge_dir)
        )

        result = retrieve_for_assessment(
            event,
            retriever,
        )

        text = json.dumps(
            result,
            indent=2,
            allow_nan=False,
        ) + "\n"

        if args.output is not None:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.output.write_text(
                text,
                encoding="utf-8",
            )

    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())