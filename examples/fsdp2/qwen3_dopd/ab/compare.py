"""Combine the five full GSM8K runs after verifying that the input pairs match."""

import argparse
import json
from pathlib import Path


GROUPS = {
    "original-ar": ("ar", None),
    "init-ar": ("ar", None),
    "init-diffusion": ("diffusion", "response"),
    "baseline50": ("diffusion", "response"),
    "combined50": ("diffusion", "prompt"),
}
BUDGETS = ("256", "1024")
PAIRS = (
    ("original-ar", "init-ar"),
    ("init-ar", "init-diffusion"),
    ("init-diffusion", "baseline50"),
    ("baseline50", "combined50"),
)


def load_run(root, group, mode, alignment):
    directory = root / group
    if (directory / ".run-in-progress").exists():
        raise ValueError(f"Evaluation is incomplete: {directory}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "samples.jsonl").open(encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream]
    ids = [row["doc_id"] for row in samples]
    if len(ids) != 200 or sorted(set(ids)) != ids or summary["doc_ids"] != ids:
        raise ValueError(f"Expected the complete, ordered 200-item result: {directory}")
    if (summary["mode"], summary["block_alignment"]) != (mode, alignment):
        raise ValueError(f"Unexpected model mode or alignment: {directory}")
    if summary["enable_thinking"] is not False or summary["max_new_tokens"] != 1024 or summary["eos_token_id"] != 151645:
        raise ValueError(f"Unexpected thinking, budget, or EOS settings: {directory}")
    if summary["sample_sha256"] != "9b397825208d0033cebe67b27e15120e6bf8e17098b37091da1c649c5d709502":
        raise ValueError(f"Unexpected GSM8K sample hash: {directory}")
    for budget in BUDGETS:
        metrics = summary["metrics"][budget]
        if metrics["n"] != len(ids) or metrics["correct_count"] != sum(
            row["scores"][budget]["correct"] for row in samples
        ):
            raise ValueError(f"Summary and item scores disagree: {directory} ({budget})")
    return summary, samples


def compare(root):
    runs = {group: load_run(root, group, *settings) for group, settings in GROUPS.items()}
    anchor, anchor_rows = runs["original-ar"]
    shared_fields = (
        "sample_sha256", "sample_source_revision", "tokenizer", "dtype", "device_type",
        "math_verify_version", "transformers_version", "torch_version", "scorer", "prompt_suffix",
    )
    for group, (summary, rows) in runs.items():
        if any(summary[field] != anchor[field] for field in shared_fields):
            raise ValueError(f"Incompatible evaluation settings: {group}")
        for reference, row in zip(anchor_rows, rows):
            if any(row[field] != reference[field] for field in ("doc_id", "gold", "prompt_sha256", "prompt_ids")):
                raise ValueError(f"Input mismatch for {group}, doc_id={reference['doc_id']}")

    comparisons = {}
    for left, right in PAIRS:
        left_rows, right_rows = runs[left][1], runs[right][1]
        counts = {}
        for budget in BUDGETS:
            gains, losses = [], []
            for a, b in zip(left_rows, right_rows):
                a_correct, b_correct = a["scores"][budget]["correct"], b["scores"][budget]["correct"]
                if a_correct != b_correct:
                    (gains if b_correct else losses).append(a["doc_id"])
            counts[budget] = {
                "gained_doc_ids": gains,
                "lost_doc_ids": losses,
                "net_accuracy_delta_percentage_points": 100 * (len(gains) - len(losses)) / len(left_rows),
            }
        comparisons[f"{left} -> {right}"] = counts

    ar_differences = [
        a["doc_id"] for a, b in zip(anchor_rows, runs["init-ar"][1])
        if a["raw_response_ids"] != b["raw_response_ids"] or a["stop_reason"] != b["stop_reason"]
    ]
    return {
        "sample_sha256": anchor["sample_sha256"],
        "n": len(anchor_rows),
        "metrics": {group: summary["metrics"] for group, (summary, _) in runs.items()},
        "paired_accuracy_changes": comparisons,
        "ar_parity": {
            "exact_raw_response_and_stop_match_count": len(anchor_rows) - len(ar_differences),
            "mismatched_doc_ids": ar_differences,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.eval_root)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)

    print(f"{'group':<16} {'acc@256':>8} {'acc@1024':>9} {'EOS@1024':>9} {'acc|EOS':>8} {'med tokens':>10}")
    for group, budgets in result["metrics"].items():
        full = budgets["1024"]
        conditional = full["conditional_accuracy_given_normal_end"]
        conditional_text = f"{conditional:.1%}" if conditional is not None else "n/a"
        print(
            f"{group:<16} {budgets['256']['accuracy']:>8.1%} {full['accuracy']:>9.1%} "
            f"{full['normal_end_rate']:>9.1%} {conditional_text:>8} {full['generated_tokens_median']:>10g}"
        )
    print(f"AR exact output/stop matches: {result['ar_parity']['exact_raw_response_and_stop_match_count']}/{result['n']}")
    print(f"Paired results: {args.output}")


if __name__ == "__main__":
    main()
