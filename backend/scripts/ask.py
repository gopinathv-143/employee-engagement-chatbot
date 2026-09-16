"""
Step 4 test/CLI: talk to the agent directly from the terminal, no FastAPI
needed. Useful for fast manual iteration and for watching the tool-calling
+ verification trace for a given question.

Run from the project root (with the venv active), AFTER scripts/build_db.py
and (for retrieval/sentiment questions) scripts/build_index.py:

    python scripts/ask.py "How many survey responses are there?"
    python scripts/ask.py --trace "What percentage of responses are dissatisfied?"
    python scripts/ask.py            # interactive loop if no question is given
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows terminals default stdout to cp1252, which can't encode every
# character an LLM might produce (e.g. U+202F narrow no-break space in
# formatted numbers). Force UTF-8 so answers never crash the print.
sys.stdout.reconfigure(encoding="utf-8")

from app.agent import chat


def ask_once(question: str, show_trace: bool) -> None:
    result = chat(question)
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(result.answer)
    print(f"\n(iterations used: {result.iterations_used}, gave_up: {result.gave_up})")
    if show_trace:
        print("\nTOOL TRACE:")
        print(json.dumps(result.tool_trace, indent=2, default=str))
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--trace", action="store_true", help="Print the full tool-call trace.")
    args = parser.parse_args()

    if args.question:
        ask_once(args.question, args.trace)
        return

    print("Interactive mode. Type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        ask_once(question, args.trace)


if __name__ == "__main__":
    main()
