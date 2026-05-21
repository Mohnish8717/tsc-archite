#!/usr/bin/env bash
# ============================================================
# start_rag.sh — WorldRAGEngine one-command startup
# ============================================================
# Usage:
#   ./start_rag.sh             → start infra only
#   ./start_rag.sh --ingest    → start infra + bulk ingest ./company_docs
#   ./start_rag.sh --watch     → start infra + watch for new docs (auto-ingest)
#   ./start_rag.sh --smoke     → start infra + run smoke test
#   ./start_rag.sh --eval      → start infra + run full RAGAS evaluation
# ============================================================
set -euo pipefail

DOCS_DIR="${RAG_DOCS_DIR:-./company_docs}"
RUN_ID="${RAG_RUN_ID:-smoke-$(date +%s)}"

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[RAG]${NC} $*"; }
warn()  { echo -e "${YELLOW}[RAG]${NC} $*"; }
error() { echo -e "${RED}[RAG]${NC} $*"; exit 1; }

# ── 1. Start Qdrant + Neo4j ───────────────────────────────────
info "Starting Qdrant + Neo4j via docker-compose..."
if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Install Docker Desktop: https://docker.com/products/docker-desktop"
fi

docker compose up -d qdrant neo4j
info "Waiting for services to be healthy..."

wait_for() {
    local name="$1" url="$2" retries=30
    for i in $(seq 1 $retries); do
        if curl -sf "$url" &>/dev/null; then
            info "$name ✅ ready"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    warn "$name not reachable after ${retries}×2s — continuing anyway"
}

wait_for "Qdrant"  "http://localhost:6333/healthz"
wait_for "Neo4j"   "http://localhost:7474"

# ── 2. Create Neo4j full-text index (idempotent) ─────────────
info "Ensuring Neo4j full-text entity search index..."
python - <<'PY' || warn "Neo4j index creation skipped (not yet connected)"
import asyncio, os
async def _create_ft_index():
    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "changeme"))
    )
    async with driver.session() as s:
        await s.run(
            "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
            "FOR (n:Company|Feature|Competitor|Regulation|Risk|CustomerSegment|Market|Persona) "
            "ON EACH [n.name]"
        )
    await driver.close()
    print("Full-text index ready")
asyncio.run(_create_ft_index())
PY

# ── 3. Mode selection ─────────────────────────────────────────
MODE="${1:-}"

case "$MODE" in
    --ingest)
        info "Bulk ingesting documents from $DOCS_DIR..."
        mkdir -p "$DOCS_DIR"
        python -m tsc.memory.rag_ingest bulk --dir "$DOCS_DIR"
        info "Bulk ingestion complete ✅"
        ;;

    --watch)
        info "Starting file watcher on $DOCS_DIR (Ctrl+C to stop)..."
        mkdir -p "$DOCS_DIR"
        # Ingest existing docs first
        python -m tsc.memory.rag_ingest bulk --dir "$DOCS_DIR"
        # Then watch for new files
        python -m tsc.memory.rag_ingest watch --dir "$DOCS_DIR" --interval 30
        ;;

    --smoke)
        info "Running 7-checkpoint smoke test..."
        python scratch/test_world_rag.py
        ;;

    --eval)
        info "Running full RAG evaluation (RAGAS-aligned)..."
        python -m tsc.memory.rag_eval \
            --run-id "$RUN_ID" \
            --output "rag_eval_results.json"
        ;;

    --build-graph)
        info "Building LazyGraphRAG community index from $DOCS_DIR..."
        python -m tsc.memory.rag_ingest build-graph --dir "$DOCS_DIR"
        ;;

    "")
        info "Infrastructure started. Available modes:"
        echo "  ./start_rag.sh --ingest      Bulk ingest ./company_docs"
        echo "  ./start_rag.sh --watch       Watch for new docs (auto-ingest)"
        echo "  ./start_rag.sh --smoke       Run smoke test (7 checkpoints)"
        echo "  ./start_rag.sh --eval        Full RAGAS evaluation"
        echo "  ./start_rag.sh --build-graph Rebuild LazyGraphRAG index"
        ;;

    *)
        error "Unknown mode: $MODE"
        ;;
esac

info "Done ✅"
