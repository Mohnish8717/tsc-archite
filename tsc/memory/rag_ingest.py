"""
RAG Ingestion CLI + File Watcher
=================================
Handles:
  • Bulk initial ingestion of company documents
  • Incremental rebuild when new files are added (auto-trigger)
  • LazyGraphRAG community index rebuild (triggered on doc add)

Usage:
    # Bulk ingest a folder
    python -m tsc.memory.rag_ingest bulk --dir ./company_docs

    # Watch folder for new files (auto-ingest + graph rebuild)
    python -m tsc.memory.rag_ingest watch --dir ./company_docs

    # Build / rebuild LazyGraphRAG community index
    python -m tsc.memory.rag_ingest build-graph
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".json", ".docx"}

# Map file-name patterns → doc_type for collection routing
_DOC_TYPE_HINTS = {
    "competitor": "competitor",
    "regulation": "regulation",
    "policy": "regulation",
    "compliance": "regulation",
    "hipaa": "regulation",
    "gdpr": "regulation",
}


def _infer_doc_type(path: Path) -> str:
    stem = path.stem.lower()
    for kw, dtype in _DOC_TYPE_HINTS.items():
        if kw in stem:
            return dtype
    return "document"


# ---------------------------------------------------------------------------
# Bulk ingestion
# ---------------------------------------------------------------------------
async def bulk_ingest(directory: str) -> None:
    from tsc.memory.world_rag import WorldRAGEngine

    engine = WorldRAGEngine()
    await engine.initialize("global")

    dir_path = Path(directory)
    files = [
        f for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported files found in %s", directory)
        return

    logger.info("Found %d files — starting bulk ingestion...", len(files))
    total_chunks = 0
    failed = 0

    for i, f in enumerate(files, 1):
        doc_type = _infer_doc_type(f)
        try:
            n = await engine.ingest_company_doc(
                file_path=str(f),
                doc_type=doc_type,
                rebuild_graph=True,  # incremental on each doc
            )
            total_chunks += n
            logger.info("[%d/%d] ✅ %s → %d chunks (%s)", i, len(files), f.name, n, doc_type)
        except Exception as exc:
            failed += 1
            logger.error("[%d/%d] ❌ %s — %s", i, len(files), f.name, exc)

    logger.info(
        "Bulk ingestion complete: %d chunks from %d files (%d failed)",
        total_chunks, len(files) - failed, failed,
    )

    # MIGRATED: LightRAG builds its graph automatically during insert().
    # No separate community index build step needed.
    # Original: await build_lazy_graphrag_index(directory)



# ---------------------------------------------------------------------------
# LazyGraphRAG community index builder
# MIGRATED: This function is now a no-op stub.
# LightRAG builds its graph during insert(); no external index step needed.
# Original subprocess call kept here as a comment for reference.
# ---------------------------------------------------------------------------
async def build_lazy_graphrag_index(input_dir: str) -> None:
    """
    STUB: LazyGraphRAG replaced by LightRAG.
    LightRAG builds the knowledge graph automatically on every insert().
    This function is kept as a no-op so any remaining callers do not crash.

    ORIGINAL IMPLEMENTATION (commented, not deleted):
    # Build / rebuild the Microsoft LazyGraphRAG community index.
    # Uses Leiden partitioning (no LLM calls at index time — lazy mode).
    # index_dir = Path("graphrag_index")
    # index_dir.mkdir(parents=True, exist_ok=True)
    # result = subprocess.run(
    #     [sys.executable, "-m", "graphrag.index",
    #      "--root", str(index_dir),
    #      "--method", "local",
    #      "--input-dir", input_dir,
    #      "--skip-validation"],
    #     capture_output=True, text=True, timeout=600,
    # )
    """
    logger.debug("build_lazy_graphrag_index: no-op (LightRAG handles graph building).")



# ---------------------------------------------------------------------------
# File watcher — auto-ingest + incremental graph rebuild on new files
# ---------------------------------------------------------------------------
class FileWatcher:
    """
    Watches a directory for new or modified files.
    Triggers incremental ingestion + graph rebuild automatically.
    This is the best approach: rebuild happens on every doc add,
    keeping the graph and community index always up to date.
    """

    def __init__(self, directory: str, poll_interval: int = 30):
        self.dir = Path(directory)
        self.interval = poll_interval
        self._seen: dict[str, float] = {}  # path → mtime

    async def start(self) -> None:
        from tsc.memory.world_rag import WorldRAGEngine

        engine = WorldRAGEngine()
        await engine.initialize("global")

        logger.info("Watching %s for new/modified files (poll every %ds)...", self.dir, self.interval)

        # Seed seen files without processing
        for f in self._scan():
            self._seen[str(f)] = f.stat().st_mtime

        while True:
            new_files = self._detect_changes()
            if new_files:
                logger.info("Detected %d new/modified file(s):", len(new_files))
                for f in new_files:
                    doc_type = _infer_doc_type(f)
                    try:
                        n = await engine.ingest_company_doc(
                            file_path=str(f),
                            doc_type=doc_type,
                            rebuild_graph=True,  # incremental entity extraction
                        )
                        logger.info("  ✅ %s → %d chunks (%s)", f.name, n, doc_type)
                    except Exception as exc:
                        logger.error("  ❌ %s — %s", f.name, exc)

                # MIGRATED: LightRAG rebuilds graph automatically on insert().
                # Original: await build_lazy_graphrag_index(str(self.dir))

            else:
                logger.debug("No changes detected in %s", self.dir)

            await asyncio.sleep(self.interval)

    def _scan(self) -> list[Path]:
        return [
            f for f in self.dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def _detect_changes(self) -> list[Path]:
        changed: list[Path] = []
        for f in self._scan():
            mtime = f.stat().st_mtime
            key = str(f)
            if key not in self._seen or self._seen[key] != mtime:
                changed.append(f)
                self._seen[key] = mtime
        return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="WorldRAGEngine ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # bulk
    p_bulk = sub.add_parser("bulk", help="Bulk ingest a directory of documents")
    p_bulk.add_argument("--dir", required=True, help="Directory of company documents")

    # watch
    p_watch = sub.add_parser("watch", help="Watch directory and auto-ingest new files")
    p_watch.add_argument("--dir", required=True, help="Directory to watch")
    p_watch.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

    # build-graph
    p_graph = sub.add_parser("build-graph", help="Rebuild LazyGraphRAG community index")
    p_graph.add_argument("--dir", required=True, help="Input documents directory")

    args = parser.parse_args()

    if args.command == "bulk":
        asyncio.run(bulk_ingest(args.dir))
    elif args.command == "watch":
        watcher = FileWatcher(args.dir, poll_interval=args.interval)
        asyncio.run(watcher.start())
    elif args.command == "build-graph":
        # MIGRATED: build-graph is now a no-op — LightRAG builds on insert().
        # Original: asyncio.run(build_lazy_graphrag_index(args.dir))
        logger.info("build-graph: LightRAG auto-builds graph on insert. No action needed.")



if __name__ == "__main__":
    main()
