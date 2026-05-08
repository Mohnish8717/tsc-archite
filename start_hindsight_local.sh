#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Hindsight Local Self-Hosted Launcher
# ═══════════════════════════════════════════════════════════════════════════
# Starts the Hindsight Docker container with runtime LLM provider selection.
# Reads HINDSIGHT_LLM_PROVIDER from .env to determine which backend to use.
#
# Usage:
#   ./start_hindsight_local.sh              # Uses provider from .env (default: gemini)
#   ./start_hindsight_local.sh gemini       # Override: use Gemini
#   ./start_hindsight_local.sh ollama       # Override: use Ollama (fully local)
#   ./start_hindsight_local.sh groq         # Override: use Groq
#   ./start_hindsight_local.sh stop         # Stop the container
#   ./start_hindsight_local.sh status       # Check status
#   ./start_hindsight_local.sh logs         # View container logs
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="hindsight-local"
HINDSIGHT_IMAGE="ghcr.io/vectorize-io/hindsight:latest"
DATA_DIR="$HOME/.hindsight-docker"

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}ℹ${NC}  $*"; }
ok()    { echo -e "${GREEN}✅${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠️${NC}  $*"; }
err()   { echo -e "${RED}❌${NC} $*" >&2; }

# ── Load .env ───────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Handle commands ─────────────────────────────────────────────────────
CMD="${1:-start}"

case "$CMD" in
    stop)
        info "Stopping Hindsight container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null && ok "Stopped." || warn "Not running."
        exit 0
        ;;
    logs)
        docker logs -f "$CONTAINER_NAME" 2>/dev/null || warn "Container not running."
        exit 0
        ;;
    status)
        if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Status}}' | grep -q .; then
            STATUS=$(docker ps --filter "name=$CONTAINER_NAME" --format '{{.Status}}')
            ok "Hindsight is running: $STATUS"
            echo -e "   ${BOLD}API:${NC}      http://localhost:8888"
            echo -e "   ${BOLD}Web UI:${NC}   http://localhost:9999"
            HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/health 2>/dev/null || echo "000")
            if [ "$HEALTH" = "200" ]; then
                echo -e "   ${BOLD}Health:${NC}   ${GREEN}OK${NC}"
            else
                echo -e "   ${BOLD}Health:${NC}   ${YELLOW}Starting...${NC}"
            fi
        else
            warn "Hindsight is not running. Use: ./start_hindsight_local.sh"
        fi
        exit 0
        ;;
    purge)
        info "Purging all simulation memory banks from Hindsight..."
        HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/health 2>/dev/null || echo "000")
        if [ "$HEALTH" != "200" ]; then
            err "Hindsight API not reachable at http://localhost:8888. Start Hindsight first."
            exit 1
        fi
        # Fetch all bank IDs and filter for simulation-related patterns
        BANKS=$(curl -s http://localhost:8888/api/v1/banks 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    banks = data if isinstance(data, list) else data.get('banks', data.get('items', []))
    for b in banks:
        bid = b.get('bank_id', b.get('id', ''))
        if any(bid.startswith(p) for p in ['oasis-', 'pre-', 'boardroom-']):
            print(bid)
except Exception:
    pass
" 2>/dev/null)
        COUNT=0
        for BANK_ID in $BANKS; do
            HTTP=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "http://localhost:8888/api/v1/banks/$BANK_ID" 2>/dev/null)
            if [ "$HTTP" = "200" ] || [ "$HTTP" = "204" ]; then
                COUNT=$((COUNT + 1))
                echo -e "   ${GREEN}✓${NC} Deleted: $BANK_ID"
            else
                echo -e "   ${YELLOW}⚠${NC} Failed to delete: $BANK_ID (HTTP $HTTP)"
            fi
        done
        if [ "$COUNT" -gt 0 ]; then
            ok "Purged $COUNT simulation banks."
        else
            info "No simulation banks found to purge."
        fi
        exit 0
        ;;
    gemini|ollama|groq|llamacpp)
        PROVIDER="$CMD"
        ;;
    start)
        PROVIDER="${HINDSIGHT_LLM_PROVIDER:-gemini}"
        ;;
    *)
        err "Unknown command: $CMD"
        echo "Usage: $0 [start|stop|status|logs|purge|gemini|ollama|groq|llamacpp]"
        exit 1
        ;;
esac

# ── Resolve provider config ────────────────────────────────────────────
declare -a DOCKER_ENV_ARGS=()

case "$PROVIDER" in
    gemini)
        API_KEY="${HINDSIGHT_GEMINI_API_KEY:-${GEMINI_API_KEY:-}}"
        MODEL="${HINDSIGHT_GEMINI_MODEL:-gemini-2.5-flash}"
        if [ -z "$API_KEY" ]; then
            err "GEMINI_API_KEY not set in .env"
            exit 1
        fi
        DOCKER_ENV_ARGS=(
            -e "HINDSIGHT_API_LLM_PROVIDER=gemini"
            -e "HINDSIGHT_API_LLM_API_KEY=$API_KEY"
            -e "HINDSIGHT_API_LLM_MODEL=$MODEL"
        )
        info "Provider: ${BOLD}Gemini${NC} (model: $MODEL)"
        ;;
    ollama)
        MODEL="${HINDSIGHT_OLLAMA_MODEL:-llama3.2:latest}"

        # Check if Ollama is running on host
        if ! curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
            warn "Ollama not detected at http://localhost:11434"
            if command -v ollama &>/dev/null; then
                info "Starting Ollama..."
                ollama serve &>/dev/null &
                sleep 3
            else
                err "Ollama not installed. Install: brew install ollama"
                exit 1
            fi
        fi

        # Ensure model is pulled
        if ! ollama list 2>/dev/null | grep -q "$MODEL"; then
            info "Pulling model $MODEL (first time only, ~8GB)..."
            ollama pull "$MODEL"
        fi

        DOCKER_ENV_ARGS=(
            -e "HINDSIGHT_API_LLM_PROVIDER=ollama"
            -e "HINDSIGHT_API_LLM_BASE_URL=http://host.docker.internal:11434/v1"
            -e "HINDSIGHT_API_LLM_MODEL=$MODEL"
            -e "HINDSIGHT_WORKER_CONCURRENCY=2"
        )
        info "Provider: ${BOLD}Ollama${NC} (model: $MODEL, fully local, \$0)"
        ;;
    groq)
        API_KEY="${HINDSIGHT_GROQ_API_KEY:-${GROQ_API_KEY:-}}"
        MODEL="${HINDSIGHT_GROQ_MODEL:-llama-3.3-70b-versatile}"
        if [ -z "$API_KEY" ]; then
            err "GROQ_API_KEY not set in .env"
            exit 1
        fi
        DOCKER_ENV_ARGS=(
            -e "HINDSIGHT_API_LLM_PROVIDER=groq"
            -e "HINDSIGHT_API_LLM_API_KEY=$API_KEY"
            -e "HINDSIGHT_API_LLM_MODEL=$MODEL"
            -e "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS=8192"
        )
        info "Provider: ${BOLD}Groq${NC} (model: $MODEL, cloud-fast)"
        ;;
    llamacpp)
        DOCKER_ENV_ARGS=(
            -e "HINDSIGHT_API_LLM_PROVIDER=llamacpp"
        )
        info "Provider: ${BOLD}llamacpp${NC} (built-in Gemma 4 E2B, ~3.5GB auto-download)"
        ;;
esac

# ── Preflight ───────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    err "Docker not found. Install: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    err "Docker daemon not running. Start Docker Desktop first."
    exit 1
fi

# Check if image exists
if ! docker images "$HINDSIGHT_IMAGE" --format '{{.ID}}' | grep -q .; then
    info "Hindsight image not found locally. Pulling (this takes 5-10 min on first run)..."
    docker pull "$HINDSIGHT_IMAGE"
fi

# ── Stop existing container ─────────────────────────────────────────────
if docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    info "Stopping existing Hindsight container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# ── Create data directory ───────────────────────────────────────────────
mkdir -p "$DATA_DIR"

# ── Launch ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     🧠 Hindsight Local Self-Hosted — Starting...     ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

docker run -d \
    --name "$CONTAINER_NAME" \
    -p 8888:8888 \
    -p 9999:9999 \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "$DATA_DIR:/home/hindsight/.pg0" \
    "$HINDSIGHT_IMAGE"

# ── Health check ────────────────────────────────────────────────────────
info "Waiting for Hindsight API..."
READY=false
for i in $(seq 1 45); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        READY=true
        break
    fi
    sleep 2
    printf "."
done
echo ""

if $READY; then
    echo ""
    ok "Hindsight is running!"
    echo ""
    echo -e "   ${BOLD}API Server:${NC}     http://localhost:8888"
    echo -e "   ${BOLD}Control Plane:${NC}  http://localhost:9999"
    echo -e "   ${BOLD}LLM Provider:${NC}   $PROVIDER"
    echo -e "   ${BOLD}Data Dir:${NC}       $DATA_DIR"
    echo ""
    echo -e "   ${CYAN}Your TSC .env is configured:${NC}"
    echo -e "     HINDSIGHT_URL=http://localhost:8888"
    echo ""
    echo -e "   ${CYAN}Switch providers anytime:${NC}"
    echo -e "     ./start_hindsight_local.sh gemini"
    echo -e "     ./start_hindsight_local.sh ollama"
    echo -e "     ./start_hindsight_local.sh groq"
    echo ""
    echo -e "   ${CYAN}Other commands:${NC}"
    echo -e "     ./start_hindsight_local.sh status"
    echo -e "     ./start_hindsight_local.sh logs"
    echo -e "     ./start_hindsight_local.sh stop"
    echo ""
else
    warn "Hindsight may still be initializing (embedded PostgreSQL + model downloads)."
    warn "This is normal on first run. Check progress with:"
    echo "  docker logs -f $CONTAINER_NAME"
fi
