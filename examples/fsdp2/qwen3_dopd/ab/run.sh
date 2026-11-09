#!/usr/bin/env bash
set -euo pipefail

AB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
TRAIN_REPO=$(cd -- "$AB_DIR/../../../.." && pwd -P)

fail() {
    printf '%s\n' "$*" >&2
    exit 2
}

usage() {
    cat <<'EOF'
Usage: bash run.sh PATHS.env COMMAND [ARGS]

Training container:
  prepare
  train baseline|combined
  export baseline|combined
  resume baseline|combined [iteration (default 50)]

Evaluation container:
  eval original-ar|init-ar|init-diffusion|baseline50|combined50 [--limit N] [--block-alignment response|prompt]
  compare
EOF
}

if [[ $# -eq 0 || ${1:-} == --help ]]; then
    usage
    exit 0
fi
[[ $# -ge 2 ]] || fail "Provide paths.env and a command. See --help."
PATHS=$1
COMMAND=$2
shift 2
[[ -r "$PATHS" ]] || fail "Cannot read paths file: $PATHS"
# shellcheck disable=SC1090
source "$(realpath -- "$PATHS")"
[[ ${QWEN3_MODEL:-} == /* ]] || fail "QWEN3_MODEL must be an absolute path."
[[ ${OPDLM_TRAIN_DATA:-} == /* ]] || fail "OPDLM_TRAIN_DATA must be an absolute path."
[[ ${RUN_ROOT:-} == /* ]] || fail "RUN_ROOT must be an absolute path."
EVAL_REPO=${EVAL_REPO:-"$TRAIN_REPO/../lm-eval-harness"}
EVAL_NPUS=${EVAL_NPUS:-8}
MODELS="$RUN_ROOT/models"
export PYTHONUNBUFFERED=1

train_group() {
    group=$1
    case "$group" in
        baseline) loss_mode=newly_unmasked; alignment=response ;;
        combined) loss_mode=remaining_mask; alignment=prompt ;;
        *) fail "Expected training group baseline or combined." ;;
    esac
    train_dir="$RUN_ROOT/train/$group"
    ckpt50="$train_dir/checkpoints/global_step_50"
}

inspect_checkpoint() {
    python "$AB_DIR/assets.py" checkpoint-info "$1" --iteration "$2" \
        --loss-mask-mode "$loss_mode" --block-alignment "$alignment"
}

run_training() {
    local action=$1 iteration=${2:-50} checkpoint existing step log
    local -a stopping
    [[ -r "$MODELS/a2d-init/config.json" ]] || fail "Run prepare in the training container first."
    [[ -r "$OPDLM_TRAIN_DATA" ]] || fail "Cannot read OPDLM training data: $OPDLM_TRAIN_DATA"
    mkdir -p -- "$RUN_ROOT/train"
    exec 9>"$RUN_ROOT/train/.$group.lock"
    flock -n 9 || fail "Training group is already running: $group"
    if [[ "$action" == train ]]; then
        [[ ! -e "$train_dir" ]] || fail "Training directory already exists: $train_dir"
        stopping=(--dopd.stop_after_iteration=50)
    else
        [[ "$iteration" =~ ^[1-9][0-9]{0,2}$ ]] && (( iteration >= 50 && iteration < 483 && iteration % 50 == 0 )) \
            || fail "Resume iteration must be one of 50, 100, ..., 450."
        checkpoint="$train_dir/checkpoints/global_step_$iteration"
        inspect_checkpoint "$checkpoint" "$iteration"
        for existing in "$train_dir"/checkpoints/global_step_*; do
            step=${existing##*/global_step_}
            step=${step%_final}
            if [[ "$step" =~ ^[0-9]+$ ]] && (( step > iteration )); then
                fail "A newer checkpoint exists: $existing. Resume from the latest complete checkpoint."
            fi
        done
        stopping=("--training.resume_from_checkpoint=$checkpoint")
    fi
    mkdir -p -- "$train_dir/logs"
    log=$(mktemp "$train_dir/logs/$action.XXXXXX")
    cd -- "$TRAIN_REPO"
    export PYTHONPATH="${PYTHONPATH:-}"
    source examples/fsdp2/env_config.sh
    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        train_dopd_fsdp2.py examples/fsdp2/qwen3_dopd/dopd_qwen3_1.7b_fsdp2_A2.yaml \
        "--model.model_name_or_path=$MODELS/a2d-init" \
        "--dopd.teacher_model_name_or_path=$QWEN3_MODEL" \
        "--dopd.train_data_path=$OPDLM_TRAIN_DATA" \
        "--training.output_dir=$train_dir" \
        "--dopd.loss_mask_mode=$loss_mode" "--dopd.block_alignment=$alignment" \
        "${stopping[@]}" 2>&1 | tee "$log"
    if [[ "$action" == train ]]; then
        inspect_checkpoint "$ckpt50" 50
    fi
}

case "$COMMAND" in
    prepare)
        [[ $# -eq 0 ]] || fail "prepare takes no arguments."
        python "$AB_DIR/assets.py" prepare "$QWEN3_MODEL" "$MODELS"
        ;;
    train|resume)
        [[ $# -ge 1 && $# -le 2 ]] || fail "Provide a group. Only resume accepts a checkpoint iteration."
        [[ "$COMMAND" == resume || $# -eq 1 ]] || fail "train takes a group only."
        train_group "$1"
        run_training "$COMMAND" "${2:-50}"
        ;;
    export)
        [[ $# -eq 1 ]] || fail "export takes a training group."
        train_group "$1"
        export_root="$RUN_ROOT/exports"
        destination="$export_root/${group}50"
        [[ -r "$EVAL_REPO/scripts/prepare_dopd_hf_checkpoint.py" ]] || fail "Evaluation checkout not found: $EVAL_REPO"
        [[ -r "$MODELS/a2d-init-hf/config.json" ]] || fail "Run prepare first."
        [[ ! -e "$destination" ]] || fail "Export already exists: $destination"
        mkdir -p -- "$export_root"
        stage=$(mktemp -d "$export_root/.${group}50.XXXXXX")
        inspect_checkpoint "$ckpt50" 50 > "$stage/source_dcp.json"
        cd -- "$TRAIN_REPO"
        python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
            --load-dir "$ckpt50" --save-dir "$stage"
        python "$EVAL_REPO/scripts/prepare_dopd_hf_checkpoint.py" \
            --source-model-dir "$MODELS/a2d-init-hf" --output-dir "$stage" --mask-token-id 151669
        mv -T -- "$stage" "$destination"
        printf 'Exported %s\n' "$destination"
        ;;
    eval)
        [[ $# -ge 1 ]] || fail "Provide an evaluation group."
        group=$1
        shift
        mode=diffusion
        alignment=response
        case "$group" in
            original-ar) model=$QWEN3_MODEL; mode=ar ;;
            init-ar) model="$MODELS/a2d-init-hf"; mode=ar ;;
            init-diffusion) model="$MODELS/a2d-init-hf" ;;
            baseline50) model="$RUN_ROOT/exports/baseline50" ;;
            combined50) model="$RUN_ROOT/exports/combined50"; alignment=prompt ;;
            *) fail "Unknown evaluation group: $group" ;;
        esac
        override=
        limit=
        while [[ $# -gt 0 ]]; do
            [[ $# -ge 2 ]] || fail "Missing value for $1."
            case "$1" in
                --limit)
                    [[ "$2" =~ ^[1-9][0-9]{0,2}$ ]] && (( $2 <= 200 )) || fail "Limit must be between 1 and 200."
                    limit=$2 ;;
                --block-alignment)
                    [[ "$mode" == diffusion && ( "$2" == response || "$2" == prompt ) ]] \
                        || fail "Block-alignment overrides apply only to diffusion and must be response or prompt."
                    override=$2 ;;
                *) fail "Unknown evaluation option: $1" ;;
            esac
            shift 2
        done
        output_root="$RUN_ROOT/eval"
        extra=()
        if [[ -n "$override" ]]; then
            output_root="$RUN_ROOT/eval-cross"
            group="$group-$override"
            alignment=$override
        fi
        if [[ -n "$limit" ]]; then
            output_root="$output_root-smoke-$limit"
            extra+=(--limit "$limit")
        fi
        [[ "$EVAL_NPUS" =~ ^[1-9][0-9]*$ ]] || fail "EVAL_NPUS must be a positive integer."
        [[ -r "$EVAL_REPO/scripts/dopd50_eval.py" ]] || fail "Evaluation checkout not found: $EVAL_REPO"
        [[ -r "$model/config.json" ]] || fail "Prepared model/export not found: $model"
        cd -- "$EVAL_REPO"
        export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
        export TOKENIZERS_PARALLELISM=false
        mkdir -p -- "$RUN_ROOT/eval-logs"
        log=$(mktemp "$RUN_ROOT/eval-logs/$group.XXXXXX")
        python -m torch.distributed.run --standalone --nproc_per_node="$EVAL_NPUS" \
            scripts/dopd50_eval.py --model "$model" --tokenizer "$QWEN3_MODEL" --mode "$mode" --device npu \
            --block-alignment "$alignment" --output-dir "$output_root/$group" \
            "${extra[@]}" 2>&1 | tee "$log"
        ;;
    compare)
        [[ $# -eq 0 ]] || fail "compare takes no arguments."
        python "$AB_DIR/compare.py" "$RUN_ROOT/eval" "$RUN_ROOT/comparison.json"
        ;;
    *) fail "Unknown command: $COMMAND. See --help." ;;
esac
