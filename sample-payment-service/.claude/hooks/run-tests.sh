#!/usr/bin/env bash
# Layer 4 (PostToolUse): run the unit tests after every Java edit.
file=$(jq -r '.tool_input.file_path // empty')
[[ "$file" == *.java ]] || exit 0

cd "$CLAUDE_PROJECT_DIR" || exit 0
if ! output=$(./mvnw -q test 2>&1); then
  echo "Tests failed after editing $file:" >&2
  "$CLAUDE_PROJECT_DIR"/.claude/hooks/summarize-maven.sh <<< "$output" >&2
  exit 2   # the edit already happened; exit 2 sends this output to the agent
fi
exit 0
