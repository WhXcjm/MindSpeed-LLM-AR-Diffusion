"""Prepare Qwen3 checkpoint views and inspect the full DCP used by the A/B run."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


MASK = "<|MASK|>"
MASK_ID = 151669
THINK_SUFFIX = "<think>\n\n</think>\n\n"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weight_files(source):
    index = source / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(read_json(index)["weight_map"].values()))
        if not names or any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError(f"Invalid weight map in {index}")
        files = [index, *(source / name for name in names)]
    else:
        files = [source / "model.safetensors"]
    if any(not path.is_file() or path.stat().st_size <= 16 for path in files):
        raise ValueError(f"Incomplete safetensors model in {source}")
    return files


def linked_weights(source_files, destination):
    for path in source_files:
        target = destination / path.name
        if path.suffix == ".json":
            shutil.copy2(path, target)
        else:
            try:
                os.link(path.resolve(), target)
            except OSError:
                # The same model mount must be visible in both containers.
                target.symlink_to(path.resolve())


def prepare(source, destination):
    from transformers import AutoTokenizer

    source = source.resolve(strict=True)
    destination = destination.resolve()
    config = read_json(source / "config.json")
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != ("qwen3", 2048, 28):
        raise ValueError(f"Expected the original dense Qwen3-1.7B checkpoint: {source}")
    if config.get("eos_token_id") != 151645 or config.get("vocab_size", 0) <= MASK_ID:
        raise ValueError("Expected the post-trained Qwen3-1.7B vocabulary and assistant EOS (151645)")
    files = weight_files(source)
    assets = [path for path in source.iterdir() if path.is_file() and (
        path.name.startswith(("tokenizer", "chat_template"))
        or path.name in {"config.json", "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}
    )]
    manifest = {
        "source": str(source),
        "asset_sha256": {path.name: sha256_file(path) for path in sorted(assets)},
        "weight_files": {path.name: path.stat().st_size for path in files},
        "mask_token_id": MASK_ID,
    }
    if destination.exists():
        if read_json(destination / "manifest.json") != manifest:
            raise ValueError(f"Prepared assets do not match the source model: {destination}")
        for name in ("a2d-init", "a2d-init-hf"):
            for path in files:
                target = destination / name / path.name
                valid = sha256_file(target) == sha256_file(path) if path.suffix == ".json" else path.samefile(target)
                if not valid:
                    raise ValueError(f"Prepared weight changed: {target}")
        print(f"Reusing validated A2D views: {destination}")
        return

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    messages = [{"role": "user", "content": "What is 1 + 1? Put the final answer in \\boxed{}."}]
    prompt_args = dict(add_generation_prompt=True, enable_thinking=False)
    before = tokenizer.apply_chat_template(messages, tokenize=False, **prompt_args)
    if not before.endswith(THINK_SUFFIX) or tokenizer.eos_token_id != 151645:
        raise ValueError("Tokenizer does not render the expected Qwen3 no-thinking suffix/EOS")
    original_ids = tokenizer.encode(before, add_special_tokens=False)
    tokenizer.add_special_tokens({"mask_token": MASK})
    if tokenizer.mask_token_id != MASK_ID or len(tokenizer) > config["vocab_size"]:
        raise ValueError(f"MASK must use existing reserved weight row {MASK_ID}; got {tokenizer.mask_token_id}")
    if tokenizer.encode(before, add_special_tokens=False) != original_ids:
        raise ValueError("Registering MASK changed normal prompt tokenization")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dopd50-models-", dir=destination.parent) as temp:
        staging = Path(temp) / "models"
        staging.mkdir()
        for name, model_type, arch in (
            ("a2d-init", "a2d-qwen3", "A2DQwen3LMHeadModel"),
            ("a2d-init-hf", "qwen3", "Qwen3ForCausalLM"),
        ):
            view = staging / name
            view.mkdir()
            view_config = {**config, "model_type": model_type, "architectures": [arch], "mask_token_id": MASK_ID}
            view_config.pop("auto_map", None)
            write_json(view / "config.json", view_config)
            if (source / "generation_config.json").is_file():
                shutil.copy2(source / "generation_config.json", view / "generation_config.json")
            tokenizer.save_pretrained(view)
            linked_weights(files, view)
            reloaded = AutoTokenizer.from_pretrained(view, local_files_only=True, trust_remote_code=False)
            if reloaded.mask_token_id != MASK_ID or reloaded.apply_chat_template(messages, tokenize=False, **prompt_args) != before:
                raise ValueError(f"Saved tokenizer changed chat rendering or MASK ID: {view}")
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    print(f"Prepared train and HF evaluation views with the same source weights: {destination}")


def checkpoint_info(checkpoint, expected_iteration, loss_mask_mode, block_alignment):
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    checkpoint = checkpoint.resolve(strict=True)
    metadata = checkpoint / ".metadata"
    extra = checkpoint / "extra_state" / "extra_state_rank_0.pt"
    if not metadata.is_file() or any(
        not (extra.parent / f"extra_state_rank_{rank}.pt").is_file() for rank in range(8)
    ):
        raise ValueError(f"Full DCP metadata and eight rank states required: {checkpoint}")
    dcp = FileSystemReader(checkpoint).read_metadata()
    if not all(any(key.startswith(prefix) for key in dcp.state_dict_metadata) for prefix in ("model.", "optimizer.")):
        raise ValueError(f"Missing model or optimizer state in {metadata}")
    if not dcp.storage_data or any(
        not (checkpoint / entry.relative_path).is_file() for entry in dcp.storage_data.values()
    ):
        raise ValueError(f"Missing DCP shards in {checkpoint}")
    # This reader is only used on the experiment's own trusted checkpoint.
    state = torch.load(extra, map_location="cpu", weights_only=False)
    iteration = int(state["global_step"])
    optimizer_step = int(state["optimizer_step"])
    if iteration != expected_iteration or optimizer_step != expected_iteration * 16:
        raise ValueError(f"Expected iteration {expected_iteration} with {expected_iteration * 16} updates; got {iteration}/{optimizer_step}")
    if not isinstance(state.get("lr_scheduler"), dict):
        raise ValueError(f"No scheduler state in {extra}")
    settings = {
        "loss_mask_mode": loss_mask_mode,
        "block_alignment": block_alignment,
        "max_steps": 483,
        "inner_steps_per_iteration": 16,
    }
    if any(state.get(key) != expected for key, expected in settings.items()):
        raise ValueError(f"Unexpected training settings in {extra}: expected {settings}")
    return {
        "checkpoint": str(checkpoint),
        "global_step": iteration,
        "optimizer_step": optimizer_step,
        **settings,
        "metadata_sha256": sha256_file(metadata),
        "rank0_extra_state_sha256": sha256_file(extra),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("destination", type=Path)
    ckpt = commands.add_parser("checkpoint-info")
    ckpt.add_argument("checkpoint", type=Path)
    ckpt.add_argument("--iteration", type=int, default=50)
    ckpt.add_argument("--loss-mask-mode", required=True, choices=("newly_unmasked", "remaining_mask"))
    ckpt.add_argument("--block-alignment", required=True, choices=("response", "prompt"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.destination)
    else:
        info = checkpoint_info(args.checkpoint, args.iteration, args.loss_mask_mode, args.block_alignment)
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
