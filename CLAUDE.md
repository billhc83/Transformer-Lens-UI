# Learngentic — Workflow Rules

Every session is a data point. These rules apply here as they do globally.

1. Call `record_task` **first** — before reading files, running commands, or writing code.
2. Use `run_local_task` for all local model work — never `curl` or raw HTTP.
3. Call `report_local_result` immediately after evaluating `run_local_task` output.
4. Call `report_outcome` before declaring any task done.

Skip only for single read-only lookups or one-turn explanations with no tool calls.
If `record_task` fails due to a server error, proceed and note the failure.
