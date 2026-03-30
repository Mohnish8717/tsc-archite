#!/bin/bash

# clean_start.sh - Purge OASIS simulation environment on macOS
# This script force-kills any existing python3 processes and removes stale lock files.

echo "🧹 Cleaning OASIS simulation environment..."

# 1. Force-kill all python3 processes related to the simulation
# We use pkill for thoroughness
if pgrep -f "python3" > /dev/null; then
    echo "🚨 Killing existing python3 processes..."
    pkill -9 -f "python3"
    sleep 1
else
    echo "✅ No running python3 processes found."
fi

# 2. Purge stale SQLite journal and lock files in the staging directory
SIM_DIR="/tmp/oasis_runs"
if [ -d "$SIM_DIR" ]; then
    echo "📂 Purging stale lock files in $SIM_DIR..."
    find "$SIM_DIR" -name "*.sqlite-journal" -delete
    find "$SIM_DIR" -name "*.sqlite-shm" -delete
    find "$SIM_DIR" -name "*.sqlite-wal" -delete
    find "$SIM_DIR" -name "commands.json" -delete
else
    echo "✅ Workspace directory $SIM_DIR clean."
fi

# 3. Verify ulimit for file descriptors
FD_LIMIT=$(ulimit -n)
if [ "$FD_LIMIT" -lt 1024 ]; then
    echo "⚠️  File descriptor limit is low ($FD_LIMIT). Raising to 1024..."
    ulimit -n 1024
fi

echo "✨ Environment is ready for a fresh OASIS simulation."
