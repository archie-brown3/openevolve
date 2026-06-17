#!/usr/bin/env python3

"""Extract the raw code payload from an OpenEvolve program JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_program_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level")

    return data


def extract_code(data: dict[str, Any]) -> str:
    code = data.get("code", "")
    if not isinstance(code, str):
        raise ValueError("Expected 'code' to be a string")
    return code


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".code.txt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the code section from an OpenEvolve program JSON file."
    )
    parser.add_argument("input", type=Path, help="Path to the program JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write the extracted code to. Defaults to <input>.code.txt",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (args.output or default_output_path(input_path)).expanduser().resolve()

    program_data = load_program_payload(input_path)
    code = extract_code(program_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code, encoding="utf-8")

    print(f"Wrote code payload to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())