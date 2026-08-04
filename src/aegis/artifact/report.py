"""Human-readable reports for artifact profile runs."""

from __future__ import annotations

from typing import Any

PROFILE_CHOICES = (
    (
        "1",
        "kick-the-tires",
        "Quick test",
        "Smoke check: AEGIS hooks + channel perturbation (usually under 2 minutes).",
    ),
    (
        "2",
        "scaled-reproduction",
        "Scaled reproduction",
        "Privacy claim: open vs AEGIS content-token embed-norm recall (~5 min, ETA).",
    ),
)

PROFILE_ALIASES = {
    "1": "kick-the-tires",
    "quick-test": "kick-the-tires",
    "kick-the-tires": "kick-the-tires",
    "2": "scaled-reproduction",
    "scaled": "scaled-reproduction",
    "scaled-reproduction": "scaled-reproduction",
}


def resolve_profile(name: str) -> str:
    profile = PROFILE_ALIASES.get(name.strip().lower())
    if profile is None:
        known = ", ".join(sorted(set(PROFILE_ALIASES.values())))
        raise ValueError(f"unknown artifact profile {name!r}; choose one of: {known}")
    return profile


def interactive_menu_text() -> str:
    lines = [
        "",
        "=" * 72,
        "AEGIS NDSS 2027 Artifact Runner",
        "=" * 72,
        "",
        "This package ships a scaled-down CPU evaluator for the Artifacts Reproduced",
        "badge. No external models, tokenizers, or datasets are downloaded.",
        "",
        "Choose a profile:",
        "",
    ]
    for key, profile, title, blurb in PROFILE_CHOICES:
        lines.append(f"  [{key}] {title}  ({profile})")
        lines.append(f"      {blurb}")
        lines.append("")
    lines.extend(
        [
            "  [q] Quit",
            "",
            "Tip: non-interactive equivalents are:",
            "  aegis artifact quick-test",
            "  aegis artifact scaled-reproduction",
            "",
        ]
    )
    return "\n".join(lines)


def format_artifact_report(payload: dict[str, Any]) -> str:
    profile = str(payload["profile"])
    title = "Quick test" if profile == "kick-the-tires" else "Scaled reproduction"
    utility = payload["utility"]
    channels = payload["channels"]
    lines = [
        "",
        "=" * 72,
        f"AEGIS Artifact — {title} ({profile})",
        "=" * 72,
        "",
        "What this run checks",
    ]
    if profile == "kick-the-tires":
        lines.extend(
            [
                "  Smoke test that default AegisDefense hooks into a tiny offline GPT-2,",
                "  perturbs attention / embedding / MLP gradients, and keeps training losses",
                "  finite. No external models or datasets are downloaded.",
            ]
        )
    else:
        lines.extend(
            [
                "  Scaled Channel-2 style recovery: embedding-gradient row norms recover the",
                "  content-token set (pad/BOS/EOS excluded) on an undefended micro-model, then",
                "  the same probe is repeated under default AegisDefense.",
            ]
        )

    lines.extend(
        [
            "",
            "Results",
            f"  Device:   {payload['device']}",
            f"  Runtime:  {float(payload['runtime']):.3f} s",
            "  Utility:",
            (
                f"    loss_before={float(utility['loss_before']):.8f}  "
                f"loss_after={float(utility['loss_after']):.8f}  "
                f"delta={float(utility['loss_delta']):.8f}"
            ),
            "",
            "  Gradient-channel perturbation (relative change vs undefended):",
        ]
    )
    for name in ("attention", "embedding", "mlp"):
        check = channels[name]
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"    {name:<10}  {float(check['value']):.6f}  "
            f">= {float(check['threshold']):.1f}   {status}"
        )

    reproduction = payload.get("reproduction")
    if isinstance(reproduction, dict):
        lines.extend(
            [
                "",
                "Privacy reproduction (content-token embed-norm recall)",
                f"  Attack:              {reproduction['attack']}",
                (
                    f"  Open recall:         {float(reproduction['open_token_recall']):.3f}  "
                    f"(threshold >= {float(reproduction['open_threshold']):.1f})"
                ),
                (
                    f"  Protected recall:    {float(reproduction['protected_token_recall']):.3f}  "
                    f"(threshold <= {float(reproduction['protect_threshold']):.2f})"
                ),
                f"  Relative reduction:  {float(reproduction['relative_reduction']):.3f}",
                f"  Claim: {reproduction['claim']}",
                f"  Verdict:             {'PASS' if reproduction['passed'] else 'FAIL'}",
            ]
        )

    lines.extend(["", "Interpretation"])
    if profile == "kick-the-tires":
        lines.extend(
            [
                "  PASS means AEGIS is wired correctly and measurably alters all three",
                "  gradient channels while protected training remains numerically finite.",
            ]
        )
    else:
        lines.extend(
            [
                "  PASS means the undefended model leaks the content-token set and AEGIS",
                "  suppresses that leakage, supporting the paper's directional privacy claim",
                "  on a scaled CPU micro-model (not a full-table ROUGE reproduction).",
            ]
        )
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


__all__ = [
    "PROFILE_ALIASES",
    "PROFILE_CHOICES",
    "format_artifact_report",
    "interactive_menu_text",
    "resolve_profile",
]
