#!/usr/bin/env python3
"""Emit a safe, static plan for the blocked contour A/B experiment.

The default plan mode parses source text only: it does not import Ziwago, load a
checkpoint, construct an organism, train, or run runtime probes. Execution mode
is deliberately locked until a real contour distinction and J acceptance exist.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "ziwago.ab_contour_plan.v1"
RAM_REQUIREMENT_GB = 10.0
USER_REPORTED_FREE_RAM_GB = 8.35
COMPOSE_CONDITIONS = (
    "replay_literal_prefix",
    "compose_prefix",
    "compose_question",
    "compose_question_prefix",
)


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "not_measured", "value": None, "reason": reason}


def _name_mentions(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(item, ast.Name) and item.id in names for item in ast.walk(node))


def _attribute_mentions(node: ast.AST, attributes: set[str]) -> bool:
    return any(
        isinstance(item, ast.Attribute) and item.attr in attributes
        for item in ast.walk(node)
    )


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_method(cls: ast.ClassDef | None, name: str) -> ast.FunctionDef | None:
    if cls is None:
        return None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _calls_in_method(method: ast.FunctionDef | None, callee: str) -> list[ast.Call]:
    if method is None:
        return []
    return [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == callee
    ]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def detect_contour_identity(
    organism_source: str,
    flat_source: str,
) -> dict[str, Any]:
    """Statically determine whether ``integrated`` selects a distinct contour."""
    organism_tree = ast.parse(organism_source)
    flat_tree = ast.parse(flat_source)
    organism_cls = _find_class(organism_tree, "ZiwagoOrganism")
    organism_init = _find_method(organism_cls, "__init__")
    flat_cls = _find_class(flat_tree, "ZiwagoPureSynapticDialogueOrganismV14")
    flat_init = _find_method(flat_cls, "__init__")

    if organism_init is None or flat_init is None:
        return {
            "status": "undetermined",
            "reason": "Could not statically locate the organism constructors.",
            "selection_based_on_integrated": None,
            "evidence": [],
        }

    parameter_names = {
        argument.arg for argument in (
            list(organism_init.args.posonlyargs)
            + list(organism_init.args.args)
            + list(organism_init.args.kwonlyargs)
        )
    }
    has_integrated_parameter = "integrated" in parameter_names
    guarded_uses = [
        node for node in ast.walk(organism_init)
        if isinstance(node, (ast.If, ast.IfExp))
        and (
            _name_mentions(node.test, {"integrated"})
            or _attribute_mentions(node.test, {"_integrated"})
        )
    ]
    argument_uses = [
        node for node in ast.walk(organism_init)
        if isinstance(node, ast.Call)
        and any(
            _name_mentions(argument, {"integrated"})
            or _attribute_mentions(argument, {"_integrated"})
            for argument in list(node.args) + [keyword.value for keyword in node.keywords]
        )
    ]
    compatibility_assignments = [
        node for node in ast.walk(organism_init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_integrated"
            for target in ast.walk(node)
        )
        and _name_mentions(node.value, {"integrated"})
    ]
    core_enable_calls = _calls_in_method(flat_init, "enable_integrated_organism")
    unconditional_enable = bool(core_enable_calls) and not any(
        isinstance(parent, ast.If)
        for call in core_enable_calls
        for parent in ast.walk(flat_init)
        if isinstance(parent, ast.If) and call in ast.walk(parent)
    )
    integrated_field_reads = [
        node for node in ast.walk(organism_cls or organism_init)
        if isinstance(node, ast.Attribute)
        and node.attr == "_integrated"
        and isinstance(node.ctx, ast.Load)
    ]

    # ``bool(integrated)`` is only a compatibility-field assignment. Any other
    # use as a branch/call argument or a later field read could route behavior.
    behavior_uses = bool(guarded_uses) or any(
        call not in [
            nested
            for assignment in compatibility_assignments
            for nested in ast.walk(assignment.value)
            if isinstance(nested, ast.Call)
        ]
        for call in argument_uses
    ) or bool(integrated_field_reads)
    same_contour = (
        has_integrated_parameter
        and bool(compatibility_assignments)
        and not behavior_uses
        and unconditional_enable
    )
    schema_dispatches = [
        node for node in ast.walk(organism_init)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Constant)
            and child.value == "v20_schema_version"
            for child in ast.walk(node.test)
        )
        and any(
            isinstance(child, ast.Call)
            and _call_name(child) in {"_load_v19_payload", "_load_v20_payload"}
            for child in ast.walk(node)
        )
    ]

    evidence: list[dict[str, Any]] = []
    for assignment in compatibility_assignments:
        evidence.append({
            "source": "ziwago/organism_core.py",
            "line": assignment.lineno,
            "finding": "integrated is stored as compatibility metadata",
        })
    for call in core_enable_calls:
        evidence.append({
            "source": "ziwago/organism/flat.py",
            "line": call.lineno,
            "finding": "integrated biological setup is called during core construction",
            "unconditional_in_constructor": unconditional_enable,
        })
    for dispatch in schema_dispatches:
        evidence.append({
            "source": "ziwago/organism_core.py",
            "line": dispatch.lineno,
            "finding": "v19/v20 schema key dispatches checkpoint payload loaders",
        })
    if organism_cls is not None:
        for node in ast.walk(organism_cls):
            if (
                isinstance(node, ast.If)
                and _name_mentions(node.test, {"integrated"})
            ):
                evidence.append({
                    "source": "ziwago/organism_core.py",
                    "line": node.lineno,
                    "finding": "constructor branch depends on integrated",
                })

    return {
        "status": "single_unified_contour" if same_contour else "distinct_or_undetermined",
        "selection_based_on_integrated": False if same_contour else None,
        "integrated_parameter_present": has_integrated_parameter,
        "integrated_argument_selects_biology": False if same_contour else None,
        "unconditional_integrated_setup": unconditional_enable,
        "interpretation": (
            "ZiwagoOrganism(integrated=True) and ZiwagoOrganism(integrated=False) "
            "construct the same unified biological contour; the argument is retained "
            "only as compatibility metadata."
            if same_contour else
            "Static inspection did not establish that both integrated argument values "
            "construct the same contour."
        ),
        "checkpoint_version_semantics": {
            "status": (
                "checkpoint_formats_not_biological_contours"
                if same_contour and schema_dispatches else "not_verified"
            ),
            "explanation": (
                "v19/v20 select checkpoint payload/schema handling; they are not "
                "old/new biological contour selectors."
            ),
            "source": "ziwago/organism_core.py checkpoint payload dispatch",
        },
        "declared_unified_contour_features": {
            "status": "source_declared_runtime_calls_not_measured",
            "items": [
                "thalamus",
                "TRN",
                "AST/GW",
                "criticality",
                "interoception",
                "limbics",
                "hippocampus",
            ],
        },
        "evidence": evidence,
    }


def build_plan_report(root: Path = ROOT) -> dict[str, Any]:
    """Build the no-model, no-training plan and honest unavailable statuses."""
    organism_source = (root / "ziwago" / "organism_core.py").read_text(encoding="utf-8")
    flat_source = (root / "ziwago" / "organism" / "flat.py").read_text(encoding="utf-8")
    identity = detect_contour_identity(organism_source, flat_source)
    block_reason = (
        "Not measured: J has not passed production restore, user-reported free RAM "
        f"is {USER_REPORTED_FREE_RAM_GB:.2f} GB (< {RAM_REQUIREMENT_GB:.0f} GB project requirement), "
        "and the current organism has no distinct old/new biological contours."
    )
    acceptance = {
        "status": "not_measured",
        "reason": block_reason,
        "closed_book": _unavailable(block_reason),
        "open_book": _unavailable(block_reason),
        "overall_acceptance": _unavailable(block_reason),
        "evaluator": "ziwago/evaluation/gates.py:evaluate_acceptance_suite",
        "planned_fields": [
            "passed",
            "available",
            "case_count",
            "echo",
            "collapse",
            "refusal_rate",
            "split_hash",
        ],
    }
    compose: dict[str, Any] = {
        "status": "not_measured",
        "reason": block_reason,
        "case_set": _unavailable(
            "No fixture set was selected or evaluated in static plan mode."
        ),
        "controls": [
            "Use identical cases, tokenizer, seed, and stopping criteria in every arm.",
            "Keep replay_literal_prefix distinct from generated compose conditions.",
            "Do not compare checkpoint schema labels as contour labels.",
        ],
        "conditions": {},
    }
    for condition in COMPOSE_CONDITIONS:
        compose["conditions"][condition] = {
            "status": "not_measured",
            "reason": block_reason,
            "case_count": None,
            "passes": None,
            "rate": None,
            "prefix_restated": None,
            "degenerate": None,
            "learned_eos": None,
            "conflict_mean": None,
            "conflict_max": None,
        }

    phase_fields = {
        phase: {
            "status": "not_measured",
            "call_count": None,
            "unique_functions": None,
            "reason": "No organism was constructed and no live phase was profiled.",
        }
        for phase in ("train", "chat", "generate")
    }
    configs = {
        "trained_candidate_old_contour": "trained candidate; old contour",
        "trained_candidate_new_contour": "trained candidate; new contour",
        "fresh_small_split_old_contour": "fresh organism; 1–2 waves on the small split; old contour",
        "fresh_small_split_new_contour": "fresh organism; 1–2 waves on the small split; new contour",
    }
    experiments = {
        key: {
            "condition": label,
            "status": "not_measured",
            "reason": block_reason,
        }
        for key, label in configs.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "K-track contour A/B instrumentation-only plan",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "static_plan",
        "execution": {
            "performed": False,
            "model_loaded": False,
            "organism_constructed": False,
            "training_performed": False,
            "heavy_probes_run": False,
        },
        "scope": {
            "edited_ziwago_source": False,
            "written_files": [
                "tools/probes/ab_contour_probe.py",
                "tests/test_k_ab_contour_probe.py",
                "D:\\Ziwago18-storage\\Produce\\ab_contour_plan_20261114.json",
                "D:\\Ziwago18-storage\\Produce\\ab_contour_plan_20261114.md",
            ],
        },
        "readiness": {
            "j_production_restore": {
                "status": "blocked",
                "reason": "J has not yet passed production restore, per task context.",
            },
            "free_ram_gb": {
                "status": "user_reported_below_requirement",
                "value": USER_REPORTED_FREE_RAM_GB,
                "source": "User-provided task context; probe did not sample RAM.",
                "required_gb": RAM_REQUIREMENT_GB,
            },
            "ab_possible": False,
            "reason": block_reason,
        },
        "contour_identity": identity,
        "experiments": experiments,
        "acceptance": acceptance,
        "compose": compose,
        "memory_profile": {
            "status": "not_measured",
            "reason": "No process, organism, or checkpoint was loaded in plan mode.",
            "peak_commit_bytes": None,
            "peak_private_bytes": None,
            "baseline_private_bytes": None,
            "sample_interval_ms": None,
        },
        "module_call_map": {
            "status": "not_measured",
            "reason": "module_map_probe was not invoked; it constructs an organism and runs train/chat/generate phases.",
            "phases": phase_fields,
            "object_graph": _unavailable("No organism object graph was created."),
            "checkpoint_component_map": _unavailable("No checkpoint was opened."),
        },
        "metric_status_table": [
            {
                "metric": "acceptance.closed_book",
                "old_contour": None,
                "new_contour": None,
                "delta": None,
                "status": "not_measured",
                "reason": block_reason,
            },
            {
                "metric": "acceptance.open_book",
                "old_contour": None,
                "new_contour": None,
                "delta": None,
                "status": "not_measured",
                "reason": block_reason,
            },
            *[
                {
                    "metric": f"compose.{condition}.rate",
                    "old_contour": None,
                    "new_contour": None,
                    "delta": None,
                    "status": "not_measured",
                    "reason": block_reason,
                }
                for condition in COMPOSE_CONDITIONS
            ],
            {
                "metric": "memory.peak_commit_bytes",
                "old_contour": None,
                "new_contour": None,
                "delta": None,
                "status": "not_measured",
                "reason": "No process was started.",
            },
            {
                "metric": "memory.peak_private_bytes",
                "old_contour": None,
                "new_contour": None,
                "delta": None,
                "status": "not_measured",
                "reason": "No process was started.",
            },
            *[
                {
                    "metric": f"module_calls.{phase}",
                    "old_contour": None,
                    "new_contour": None,
                    "delta": None,
                    "status": "not_measured",
                    "reason": "No live phase was profiled.",
                }
                for phase in ("train", "chat", "generate")
            ],
        ],
        "reproducible_commands": [
            {
                "purpose": "Verify J production restore only after at least 10 GB free RAM is available.",
                "command": (
                    "python tools/limited_run.py --ram-free-pct 85 --cpu-pct 70 "
                    "--log D:\\Ziwago18-storage\\Produce\\logs\\j_restore.log -- "
                    "python tools/probes/ckpt_memory_profile.py --checkpoint "
                    "<trained-candidate.pt> --restore --out <j-production-restore.json>"
                ),
            },
            {
                "purpose": "Run the guarded K experiment after J passes and a real old/new contour selector exists.",
                "command": (
                    "python tools/limited_run.py --ram-free-pct 85 --cpu-pct 70 "
                    "--log D:\\Ziwago18-storage\\Produce\\logs\\ab_contour.log -- "
                    "python tools/probes/ab_contour_probe.py --mode execute "
                    "--j-acceptance-report <j-acceptance.json> "
                    "--checkpoint <trained-candidate.pt> "
                    "--smoke-split <balanced-smoke-split.jsonl> "
                    "--out D:\\Ziwago18-storage\\Produce\\ab_contour_YYYYMMDD.json"
                ),
            },
            {
                "purpose": "Project regression suite.",
                "command": "python -m pytest tests -q",
            },
        ],
        "execution_mode_note": (
            "--mode execute is opt-in and currently refuses before loading anything: "
            "this instrumentation-only branch has no valid contour A/B until a genuine "
            "old/new biological contour is available and J production restore passes."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# K-track contour A/B — instrumentation-only plan",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        "- Mode: `static_plan` (no model, organism, training, or heavy probe)",
        f"- A/B status: **not measured** — {report['readiness']['reason']}",
        "",
        "## Contour identity",
        "",
        f"- Detection: **{report['contour_identity']['status']}**",
        f"- `integrated=True/False` selects biology: **{report['contour_identity']['integrated_argument_selects_biology']}**",
        f"- v19/v20 semantics: **{report['contour_identity']['checkpoint_version_semantics']['status']}**",
        f"- Interpretation: {report['contour_identity']['interpretation']}",
        "- Source-declared biology is not a runtime call measurement; the module-call map remains unavailable.",
        "",
        "## Metric status",
        "",
        "| Metric | Old contour | New contour | Δ | Status / reason |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["metric_status_table"]:
        reason = row["reason"].replace("|", "\\|")
        lines.append(
            f"| `{row['metric']}` | — | — | — | **{row['status']}**: {reason} |"
        )
    lines.extend([
        "",
        "All unavailable numeric fields are JSON `null`; no values were inferred or copied from prior runs.",
        "",
        "## Reproducible commands for later",
        "",
        "Run only after J production restore passes and at least 10 GB RAM is free. The K execute command remains guarded until a genuine old/new biological contour exists.",
        "",
    ])
    for item in report["reproducible_commands"]:
        lines.extend([f"### {item['purpose']}", "", "```powershell", item["command"], "```", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan", "execute"),
        default="plan",
        help="plan is static and safe; execute is currently refused until prerequisites exist",
    )
    parser.add_argument("--j-acceptance-report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke-split", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="write JSON here and a sibling Markdown table with the same basename",
    )
    args = parser.parse_args(argv)
    if args.mode == "execute":
        print(
            "Execution refused before loading any model: J production restore is blocked "
            "and the static contour detector finds one unified contour, not an old/new "
            "biological A/B. Use --mode plan for the instrumentation report.",
            file=sys.stderr,
        )
        return 2

    report = build_plan_report()
    markdown = render_markdown(report)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        md_path = args.out.with_suffix(".md")
        md_path.write_text(markdown + "\n", encoding="utf-8")
        print(f"JSON: {args.out}")
        print(f"Markdown: {md_path}")
    else:
        print(rendered, end="")
        print("\n--- MARKDOWN METRIC-STATUS TABLE ---\n")
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
