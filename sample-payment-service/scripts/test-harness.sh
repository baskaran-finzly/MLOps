#!/usr/bin/env bash
# Tests the harness itself, without starting Claude Code.
# Each check feeds a hook the same JSON Claude Code would send on stdin,
# then checks the exit code: 0 = allow, 2 = block or send feedback.
#
#   scripts/test-harness.sh          all checks (runs Maven, about a minute)
#   scripts/test-harness.sh --quick  skip the checks that run Maven
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
hooks="$root/.claude/hooks"
quick=false; [ "${1:-}" = "--quick" ] && quick=true
pass=0; fail=0

check() {  # check <description> <expected exit> <actual exit>
  if [ "$2" = "$3" ]; then echo "  PASS  $1"; pass=$((pass + 1))
  else echo "  FAIL  $1 (expected exit $2, got $3)"; fail=$((fail + 1)); fi
}
edit_json() { printf '{"hook_event_name":"%s","tool_name":"Edit","tool_input":{"file_path":"%s"}}' "$1" "$2"; }

echo "Setup"
command -v jq >/dev/null; check "jq is installed" 0 $?
jq -e . "$root/.claude/settings.json" >/dev/null 2>&1; check "settings.json is valid JSON" 0 $?
for h in "$hooks"/*.sh; do [ -x "$h" ]; check "$(basename "$h") is executable" 0 $?; done

echo "Layer 2: permission rules are present"
for rule in 'Read(./.env)' 'Bash(git push *)'; do
  jq -e --arg r "$rule" '.permissions.deny | index($r)' "$root/.claude/settings.json" >/dev/null; check "deny contains $rule" 0 $?
done

echo "Layer 3: protect-ledger.sh"
edit_json PreToolUse "$root/src/main/java/com/acme/payments/ledger/LedgerPoster.java" | "$hooks/protect-ledger.sh" 2>/dev/null
check "blocks an edit to ledger/LedgerPoster.java" 2 $?
edit_json PreToolUse "$root/src/main/java/com/acme/payments/validation/NewRule.java" | "$hooks/protect-ledger.sh" 2>/dev/null
check "allows an edit to validation/NewRule.java" 0 $?

echo "Layer 4: run-tests.sh"
edit_json PostToolUse "$root/README.md" | CLAUDE_PROJECT_DIR="$root" "$hooks/run-tests.sh" 2>/dev/null
check "ignores a non-Java edit" 0 $?

echo "Layer 5: verify-before-stop.sh"
echo '{"hook_event_name":"Stop","stop_hook_active":true}' | CLAUDE_PROJECT_DIR="$root" "$hooks/verify-before-stop.sh" 2>/dev/null
check "lets the agent stop on the second attempt (no endless loop)" 0 $?

if ! $quick; then
  echo "Maven-backed checks (slow)"
  edit_json PostToolUse "$root/src/main/java/com/acme/payments/service/PaymentService.java" | CLAUDE_PROJECT_DIR="$root" "$hooks/run-tests.sh" 2>/dev/null
  check "run-tests.sh passes on the current, green code" 0 $?

  # Break a copy of the project and confirm the hooks notice. This is the
  # false-negative check: a hook that never fails is not protecting anything.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  (cd "$root" && tar --exclude=./target --exclude=./.git -cf - .) | (cd "$tmp" && tar -xf -)
  sed -i 's/compareTo(BigDecimal.ZERO) <= 0/compareTo(BigDecimal.ZERO) < 0/' \
    "$tmp/src/main/java/com/acme/payments/validation/PositiveAmountRule.java"
  edit_json PostToolUse "$tmp/src/main/java/com/acme/payments/validation/PositiveAmountRule.java" | CLAUDE_PROJECT_DIR="$tmp" "$hooks/run-tests.sh" 2>/dev/null
  check "run-tests.sh reports a deliberately broken rule" 2 $?
  echo '{"hook_event_name":"Stop","stop_hook_active":false}' | CLAUDE_PROJECT_DIR="$tmp" "$hooks/verify-before-stop.sh" 2>/dev/null
  check "verify-before-stop.sh refuses to finish on the broken copy" 2 $?
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
