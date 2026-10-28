#!/usr/bin/env bash
# Layer 5 (Stop): don't let the agent finish while the build is red.
input=$(cat)

# If this hook already sent the agent back once, let it stop.
# Otherwise a build it cannot fix would loop forever.
if [ "$(echo "$input" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR" || exit 0

# Nothing changed in this project (e.g. the agent only answered a question):
# skip the full build. Note this also skips it once changes are committed.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -z "$(git status --porcelain -- .)" ]; then
  exit 0
fi

if ! output=$(./mvnw -q verify 2>&1); then
  echo "You can't finish yet: ./mvnw verify fails. Fix it, or tell the user why you can't." >&2
  "$CLAUDE_PROJECT_DIR"/.claude/hooks/summarize-maven.sh <<< "$output" >&2
  exit 2   # exit code 2 = keep working instead of stopping
fi
exit 0
