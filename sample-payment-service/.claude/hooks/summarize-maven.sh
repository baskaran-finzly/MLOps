#!/usr/bin/env bash
# Shared by the hooks: reduce Maven output on stdin to the lines an agent can act on
# (failing tests, assertion messages, compile errors), dropping Maven's boilerplate.
output=$(cat)
summary=$(grep -E "Tests run:.*(Failures|Errors): [1-9]|<<< (FAILURE|ERROR)|expected|(Exception|Error): |\.java:\[[0-9]+|COMPILATION ERROR" <<< "$output" \
  | grep -vE "Please refer|Help 1|-e switch|-X switch|For more information|Failed to execute goal| (WARN|INFO|DEBUG) |^[[:space:]]+at " \
  | awk '!seen[$0]++' \
  | head -25)
# Unknown failure shape: fall back to the end of the log rather than say nothing.
[ -n "$summary" ] || summary=$(tail -25 <<< "$output")
echo "$summary"
