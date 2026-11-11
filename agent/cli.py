"""Command-line chat with the MLOps agent (same orchestrator as the API).

    python -m agent.cli "Is the prediction API healthy?"
    python -m agent.cli --json "..."      # full structured response

Pending state-changing actions are shown at the end; you are asked to confirm each one.
"""

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from agent.llm import GeminiLLM
from agent.orchestrator import Agent


def main() -> None:
    # Windows consoles default to cp1252, which cannot print characters like "≥".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    load_dotenv()
    p = argparse.ArgumentParser(description="Ask the MLOps agent a question.")
    p.add_argument("message")
    p.add_argument("--json", action="store_true", help="print the full structured response")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print structured JSON logs (tool/LLM calls) to stderr",
    )
    args = p.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    agent = Agent(GeminiLLM())
    resp = agent.run(args.message)
    if args.json:
        print(resp.model_dump_json(indent=2))
    else:
        print(resp.answer)
        print(
            f"\n--- request {resp.request_id} | {resp.model} | {resp.steps} LLM turns | "
            f"{resp.duration_ms / 1000:.1f}s"
        )
        for c in resp.tool_calls:
            print(
                f"  {c.status.value:22} {c.duration_ms:8.1f} ms  {c.tool} {json.dumps(c.arguments)}"
            )
        for skipped in resp.llm_fallbacks:
            print(f"  model skipped -> {skipped}")
        if resp.error:
            print(f"  LLM error: {resp.error}")

    for action in resp.pending_actions:
        print(f"\nPending action {action.action_id}: {action.tool} {json.dumps(action.arguments)}")
        if not sys.stdin.isatty():
            print("  (not confirmed: no interactive terminal)")
            continue
        if input("  Execute it? [y/N] ").strip().lower() == "y":
            result = agent.execute_action(action.action_id)
            print(f"  {result.status}: {result.result.model_dump_json() if result.result else ''}")
        else:
            agent.reject_action(action.action_id)
            print("  rejected")


if __name__ == "__main__":
    main()
