#!/usr/bin/env bash
# Layer 3 (PreToolUse): block agent edits to protected code.
# Claude Code sends the pending tool call as JSON on stdin.
file=$(jq -r '.tool_input.file_path // empty')

case "$file" in
  */src/main/java/*/ledger/*|*/src/main/resources/db/migration/*)
    echo "Blocked: $file is protected. Ledger posting and DB migrations are changed by humans only. Put new payment rules in the validation package." >&2
    exit 2   # exit code 2 = cancel the call and show this message to the agent
    ;;
esac
exit 0
