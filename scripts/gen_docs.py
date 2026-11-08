"""Regenerate docs/data_dictionary.md from the data contract. `--check` fails if stale (CI)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tax_rag.catalog.contract import load_contract  # noqa: E402

TARGET = ROOT / "docs" / "data_dictionary.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = load_contract().to_markdown()
    if args.check:
        if not TARGET.exists() or TARGET.read_text("utf-8") != rendered:
            print(f"{TARGET.relative_to(ROOT)} is stale; run: python scripts/gen_docs.py")
            return 1
        print("data dictionary up to date")
        return 0
    TARGET.write_text(rendered, "utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
