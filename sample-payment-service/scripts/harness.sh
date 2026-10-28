#!/usr/bin/env bash
# Switch the harness off or on, to compare agent runs with and without it.
#   scripts/harness.sh off   moves CLAUDE.md and .claude/settings.json into .harness-off/
#   scripts/harness.sh on    moves them back
#   scripts/harness.sh       shows the current state
# Restart Claude Code after switching: it reads these files at session start.
set -euo pipefail
cd "$(dirname "$0")/.."

files=(CLAUDE.md .claude/settings.json)

case "${1:-status}" in
  off)
    for f in "${files[@]}"; do
      if [ -f "$f" ]; then mkdir -p ".harness-off/$(dirname "$f")"; mv "$f" ".harness-off/$f"; fi
    done
    echo "Harness OFF. Restart Claude Code to get a session with no instructions, rules or hooks."
    ;;
  on)
    for f in "${files[@]}"; do
      if [ -f ".harness-off/$f" ]; then mkdir -p "$(dirname "$f")"; mv ".harness-off/$f" "$f"; fi
    done
    rm -rf .harness-off
    echo "Harness ON. Restart Claude Code to load it."
    ;;
  status)
    if [ -f CLAUDE.md ] && [ -f .claude/settings.json ]; then echo "Harness is ON"; else echo "Harness is OFF (or partly off)"; fi
    ;;
  *)
    echo "usage: scripts/harness.sh [on|off|status]" >&2; exit 1 ;;
esac
