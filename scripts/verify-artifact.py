#!/usr/bin/env python3
"""Validate one JSON result produced by an AEGIS artifact profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

PROFILES = ("kick-the-tires", "scaled-reproduction")


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an AEGIS artifact result and its CPU/profile contract."
    )
    parser.add_argument("result", type=Path, help="Path to the artifact JSON result.")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        help="Require the result to have this artifact profile.",
    )
    return parser


def verify(path: Path, expected_profile: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"result is not a readable file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error

    try:
        payload = json.loads(text, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error

    try:
        from aegis.artifact import validate_result
    except ImportError as error:
        raise ValueError(
            "the aegis package is unavailable; run scripts/install-artifact.sh first"
        ) from error

    try:
        validate_result(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"artifact schema validation failed: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("artifact result must be a JSON object")
    if expected_profile is not None and payload.get("profile") != expected_profile:
        raise ValueError(
            f"profile mismatch: expected {expected_profile!r}, got {payload.get('profile')!r}"
        )
    if payload.get("device") != "cpu":
        raise ValueError(f"evaluator result must use device 'cpu', got {payload.get('device')!r}")
    return payload


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        payload = verify(arguments.result, arguments.profile)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"Verified {arguments.result}: "
        f"profile={payload['profile']}, device={payload['device']}, schema={payload['schema']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
