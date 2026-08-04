"""Command-line interface for AEGIS studies and artifact profiles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn, TextIO

from aegis.studies import RunSettings, StudyDefinition, load_study

_ARTIFACT_PROFILE_NAMES = (
    "quick-test",
    "kick-the-tires",
    "scaled-reproduction",
    "scaled",
)


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_set_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"{raw!r} is not KEY=JSON; the assignment separator is missing"
        )
    key, encoded = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"{raw!r} is missing a KEY before '='")
    if not encoded:
        raise argparse.ArgumentTypeError(f"{raw!r} is missing JSON after '='")
    try:
        value = json.loads(encoded, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"{raw!r} does not contain strict JSON after '=': {error}"
        ) from error
    if isinstance(value, float) and not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{raw!r} contains a non-finite JSON number")
    return key, value


def _add_artifact_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifact-results"),
        help="Directory for the atomically written JSON result (default: artifact-results).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the saved-result line (script-friendly).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis",
        description=(
            "Run AEGIS artifact profiles (evaluator) or external study configurations "
            "(research harness)."
        ),
        epilog=(
            "Evaluator tip: run 'aegis artifact' for an interactive menu, or "
            "'aegis artifact quick-test' / 'aegis artifact scaled-reproduction'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="Execute one external study configuration.",
    )
    run.add_argument("config_path", type=Path, help="Path to a study JSON file.")
    run.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the study model list.",
    )
    run.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override the study dataset list.",
    )
    run.add_argument(
        "--defenses",
        nargs="+",
        default=None,
        help="Override the study defense list.",
    )
    run.add_argument(
        "--attacks",
        nargs="+",
        default=None,
        help="Override the study attack list.",
    )
    run.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        type=_parse_set_assignment,
        metavar="KEY=JSON",
        help="Override one study parameter with a strict JSON value. Repeatable.",
    )
    run.add_argument("--seed", type=int, default=42, help="Scientific random seed.")
    run.add_argument(
        "--quick",
        action="store_true",
        help="Apply the study quick_defaults before other overrides.",
    )
    run.add_argument(
        "--validate",
        action="store_true",
        help="Apply the study validation_defaults before other overrides.",
    )
    run.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Root containing local tiny_gpt2/gpt2_medium_weights directories.",
    )
    run.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face model and dataset cache root.",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for atomically written JSON results (default: results).",
    )
    run.add_argument(
        "--output-name",
        default=None,
        help="Override the study output filename.",
    )
    run.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:1, mps, or cpu (auto by default).",
    )

    describe = commands.add_parser(
        "describe",
        help="Print one external study definition as JSON.",
    )
    describe.add_argument("config_path", type=Path, help="Path to a study JSON file.")

    artifact_common = argparse.ArgumentParser(add_help=False)
    _add_artifact_options(artifact_common)
    artifact = commands.add_parser(
        "artifact",
        parents=[artifact_common],
        help="Run a self-contained artifact profile (interactive if no profile is given).",
    )
    artifact_commands = artifact.add_subparsers(
        dest="artifact_command",
        required=False,
    )
    helps = {
        "quick-test": "Friendly alias for kick-the-tires (smoke / quick test).",
        "kick-the-tires": "Minimal smoke profile: hooks + channel perturbation.",
        "scaled-reproduction": "Scaled privacy reproduction: open vs AEGIS token-set recall.",
        "scaled": "Friendly alias for scaled-reproduction.",
    }
    for profile in _ARTIFACT_PROFILE_NAMES:
        artifact_commands.add_parser(
            profile,
            parents=[artifact_common],
            help=helps[profile],
        )
    return parser


def _merge_parameters(study: StudyDefinition, arguments: argparse.Namespace) -> dict[str, Any]:
    parameters = dict(study.defaults)
    if arguments.quick:
        parameters.update(study.quick_defaults)
    if arguments.validate:
        parameters.update(study.validation_defaults)
    for key, value in arguments.set_values:
        parameters[key] = value
    return parameters


def _resolve_list(
    override: list[str] | None,
    configured: tuple[str, ...],
    parameter_key: str,
    parameters: dict[str, Any],
) -> tuple[str, ...]:
    if override is not None:
        return tuple(override)
    value = parameters.get(parameter_key)
    # Non-empty defaults.* lists select a runtime subset; empty lists fall back
    # to the study's top-level catalog so shipped configs remain runnable.
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    return configured


def _build_run_settings(arguments: argparse.Namespace) -> RunSettings:
    study = load_study(arguments.config_path)
    parameters = _merge_parameters(study, arguments)
    models = _resolve_list(arguments.models, study.models, "models", parameters)
    datasets = _resolve_list(arguments.datasets, study.datasets, "datasets", parameters)
    defenses = _resolve_list(arguments.defenses, study.defenses, "defenses", parameters)
    attacks = _resolve_list(arguments.attacks, study.attacks, "attacks", parameters)

    if study.behavior.get("dataset_required") and not datasets:
        raise ValueError(f"{study.name} requires at least one dataset")

    return RunSettings(
        study=study,
        models=models,
        datasets=datasets,
        defenses=defenses,
        attacks=attacks,
        seed=arguments.seed,
        quick=arguments.quick,
        validate=arguments.validate,
        model_dir=arguments.model_dir,
        cache_dir=arguments.cache_dir,
        output_dir=arguments.output_dir,
        device=arguments.device,
        parameters=parameters,
        output_name=arguments.output_name,
    )


def _prompt_artifact_profile(
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> str | None:
    from aegis.artifact.report import interactive_menu_text, resolve_profile

    stdout.write(interactive_menu_text())
    stdout.flush()
    while True:
        stdout.write("Select [1/2/q]: ")
        stdout.flush()
        line = stdin.readline()
        if line == "":
            stdout.write("\nNo selection received; exiting.\n")
            return None
        choice = line.strip().lower()
        if choice in {"q", "quit", "exit"}:
            stdout.write("Exiting without running a profile.\n")
            return None
        try:
            return resolve_profile(choice)
        except ValueError:
            stdout.write("Please enter 1, 2, or q.\n")


def _run_artifact_command(arguments: argparse.Namespace) -> int:
    from aegis.artifact import (
        format_artifact_report,
        resolve_profile,
        run_artifact_profile,
        write_result,
    )

    selected = arguments.artifact_command
    if selected is None:
        if not sys.stdin.isatty():
            print(
                "error: no artifact profile selected. Use "
                "'aegis artifact quick-test' or 'aegis artifact scaled-reproduction', "
                "or run in a terminal for the interactive menu.",
                file=sys.stderr,
            )
            return 2
        selected = _prompt_artifact_profile()
        if selected is None:
            return 0
    else:
        selected = resolve_profile(selected)

    quiet = bool(getattr(arguments, "quiet", False))
    if not quiet:
        title = "Quick test" if selected == "kick-the-tires" else "Scaled reproduction"
        print(f"Running AEGIS {title} ({selected}) on {arguments.device}")
        print("Building a local micro GPT-2 (offline; no downloads)...")

    payload = run_artifact_profile(
        selected,
        device=arguments.device,
        show_progress=not quiet,
    )
    output_path = write_result(payload, arguments.output_dir)

    if not quiet:
        print(format_artifact_report(payload), end="")
    print(f"Artifact result saved -> {output_path}")
    if not quiet:
        print(
            "Next: validate with "
            f"scripts/verify-artifact.py --profile {selected} {output_path}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "describe":
        load_study(arguments.config_path)
        # Print the validated source document so describe is a pure view of the file.
        print(arguments.config_path.read_text(encoding="utf-8"), end="")
        return 0

    if arguments.command == "artifact":
        return _run_artifact_command(arguments)

    try:
        settings = _build_run_settings(arguments)
    except ValueError as error:
        parser.error(str(error))

    from aegis.harness.runner import run_study

    result = run_study(settings)
    print(f"Results saved -> {result.output_path}")
    if settings.validate:
        passed = result.payload.get("passed")
        if isinstance(passed, bool):
            return 0 if passed else 1
        print(
            "error: --validate was set but the study produced no boolean "
            "'passed' verdict",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "main"]
