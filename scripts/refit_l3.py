"""Operator command to fit / refit Layer 3 clustering."""

from __future__ import annotations

import argparse
import logging

from mh_rag.config import get_settings
from mh_rag.layers.l3 import Layer3Service
from mh_rag.logging_setup import configure_logging
from mh_rag.store import FalkorStore, apply_schema

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--if-needed`` evaluates triggers; ``--force`` always fits."""
    parser = argparse.ArgumentParser(description="Fit or refit Layer-3 clusters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--if-needed",
        action="store_true",
        help="Fit only when needs_refit() is true, or bootstrap-check otherwise",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Force a fit even below bootstrap threshold / without triggers",
    )
    parser.add_argument("--graph", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.graph:
        settings = settings.model_copy(update={"graph_name": args.graph})
    configure_logging(settings.log_level)

    store = FalkorStore(settings)
    apply_schema(store)
    service = Layer3Service(store, settings)

    if args.force:
        report = service.fit(force=True)
        print(
            f"status={report.status} chunks={report.chunk_count} "
            f"clusters={report.cluster_count} version={report.cluster_version} "
            f"noise={report.noise_count}"
        )
        return 0

    # --if-needed
    decision = service.needs_refit()
    manifest_missing = decision.reason == "no_manifest"
    if manifest_missing:
        report = service.fit(force=False)
        print(
            f"status={report.status} chunks={report.chunk_count} "
            f"clusters={report.cluster_count} version={report.cluster_version} "
            f"reason={report.reason or decision.reason}"
        )
        return 0

    if not decision.needed:
        # Still assign any pending EXTRACTED chunks
        assign = service.assign_pending()
        print(
            f"status=UP_TO_DATE reason={decision.reason} "
            f"assign_status={assign.status} chunks_assigned={assign.chunks_assigned} "
            f"entities_assigned={assign.entities_assigned}"
        )
        return 0

    report = service.fit(force=False)
    print(
        f"status={report.status} reason={decision.reason} "
        f"chunks={report.chunk_count} clusters={report.cluster_count} "
        f"version={report.cluster_version} noise={report.noise_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
