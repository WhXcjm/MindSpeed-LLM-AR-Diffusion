"""Pure tensor utilities for the d-OPD reproduction.

The future-aware reweighting follows the public d-OPD implementation while
the attention/state helpers are written for MindSpeed's Qwen3 diffusion model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class IntermediateState:
    """One independently sampled denoising state per response block."""

    noised_response: torch.LongTensor
    loss_mask: torch.BoolTensor


def active_rank_training_schedule(
    training_order: list[int],
    *,
    rank: int,
    world_size: int,
    active_ranks: int,
    accumulation: int,
) -> list[tuple[int, bool, bool]]:
    """为全 rank 或轮换 rank batch 构造集合通信对齐的优化器任务。"""

    if not 0 < active_ranks <= world_size or world_size % active_ranks:
        raise ValueError("active_ranks must be a positive divisor of world_size")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    if active_ranks < world_size:
        if accumulation != 1:
            raise ValueError("rotating active-rank cohorts require accumulation=1")
        cohort_count = world_size // active_ranks
        rank_cohort = rank // active_ranks
        return [
            (trajectory_index, cohort == rank_cohort, True)
            for trajectory_index in training_order
            for cohort in range(cohort_count)
        ]
    return [
        (trajectory_index, True, (micro_step + 1) % accumulation == 0)
        for micro_step, trajectory_index in enumerate(training_order)
    ]


def cosine_response_length(
    step: int,
    minimum: int,
    maximum: int,
    schedule_steps: int,
    block_size: int,
) -> int:
    """Cosine-ramp rollout length, rounded up to a complete block."""

    if step < 1:
        raise ValueError("step must be at least 1")
    progress = min(float(step - 1) / float(max(1, schedule_steps - 1)), 1.0)
    fraction = 0.5 * (1.0 - math.cos(math.pi * progress))
    raw = minimum + (maximum - minimum) * fraction
    aligned = int(math.ceil(raw / block_size) * block_size)
    return min(maximum, max(minimum, aligned))


def prompt_tail_length(prompt_length: int, block_size: int, block_alignment: str) -> int:
    if block_alignment == "response":
        return 0
    if block_alignment == "prompt":
        return prompt_length % block_size
    raise ValueError("block_alignment must be response or prompt")


def response_block_end(
    position: int, response_length: int, block_size: int, prompt_tail: int
) -> int:
    """Exclusive response-relative end of the physical block containing position."""
    physical_end = ((position + prompt_tail) // block_size + 1) * block_size
    return min(physical_end - prompt_tail, response_length)


def response_block_spans(response_length: int, block_size: int, prompt_tail: int):
    start = 0
    while start < response_length:
        end = response_block_end(start, response_length, block_size, prompt_tail)
        yield start, end
        start = end


def build_dopd_attention_mask(
    prompt_length: int,
    response_length: int,
    block_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    block_alignment: str = "response",
) -> torch.Tensor:
    """Build attention for [clean sequence, noised prompt tail and response].

    Each noised block reads clean history and its own noised copy. With prompt
    alignment the two copies of the prompt tail therefore see different first
    response blocks. Response alignment has no duplicated prompt tail.
    """

    if prompt_length <= 0 or response_length <= 0 or block_size <= 0:
        raise ValueError("prompt_length, response_length, and block_size must be positive")
    prompt_tail = prompt_tail_length(prompt_length, block_size, block_alignment)
    prefill_length = prompt_length - prompt_tail
    clean_length = prompt_length + response_length
    total = clean_length + prompt_tail + response_length
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)

    # Match the reference implementation's right-aligned block partition for
    # the prompt. Prompt logits are not trained, but these states provide KV
    # context to both response branches.
    for end in range(prefill_length, 0, -block_size):
        start = max(0, end - block_size)
        allowed[start:end, :end] = True

    for start, end in response_block_spans(response_length, block_size, prompt_tail):
        clean_start = prefill_length if start == 0 else prompt_length + start
        clean_end = prompt_length + end
        allowed[clean_start:clean_end, :clean_end] = True

        noised_start = clean_length + clean_start - prefill_length
        noised_end = clean_length + clean_end - prefill_length
        allowed[noised_start:noised_end, :clean_start] = True
        allowed[noised_start:noised_end, noised_start:noised_end] = True
    mask = torch.full(
        (1, 1, total, total),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    mask[0, 0][allowed] = 0
    return mask


def sample_intermediate_state(
    response_ids: torch.LongTensor,
    first_unmask_steps: torch.LongTensor,
    block_size: int,
    mask_token_id: int,
    rng: random.Random,
    *,
    prompt_length: int = 0,
    block_alignment: str = "response",
    loss_mask_mode: str = "newly_unmasked",
) -> IntermediateState:
    """Uniformly select one valid rollout round independently for each block.

    At reveal step ``s``, tokens revealed before ``s`` are visible, while tokens
    first revealed at or after ``s`` remain masked. The loss can supervise all
    these positions or the positions revealed in round ``s`` only.
    """

    if response_ids.ndim != 1 or first_unmask_steps.ndim != 1:
        raise ValueError("response_ids and first_unmask_steps must be one-dimensional")
    if response_ids.shape != first_unmask_steps.shape:
        raise ValueError("response_ids and first_unmask_steps must have the same shape")
    if (first_unmask_steps <= 0).any().item():
        raise ValueError("every generated token must have a positive first-unmask step")
    if loss_mask_mode not in {"newly_unmasked", "remaining_mask"}:
        raise ValueError("loss_mask_mode must be newly_unmasked or remaining_mask")

    noised = response_ids.clone()
    loss_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    prompt_tail = prompt_tail_length(prompt_length, block_size, block_alignment)
    for start, end in response_block_spans(response_ids.numel(), block_size, prompt_tail):
        block_steps = first_unmask_steps[start:end]
        choices = torch.unique(block_steps).detach().cpu().tolist()
        chosen = int(rng.choice(choices))
        remaining = block_steps >= chosen
        noised[start:end][remaining] = mask_token_id
        loss_mask[start:end] = (
            remaining if loss_mask_mode == "remaining_mask" else block_steps == chosen
        )
    if not loss_mask.any().item():
        raise RuntimeError("sampled trajectory state contains no loss positions")
    return IntermediateState(noised_response=noised, loss_mask=loss_mask)


def response_valid_mask(
    response_ids: torch.LongTensor,
    eos_token_id: int | None,
    exclude_eos: bool,
) -> torch.BoolTensor:
    """Return positions before the first EOS, optionally including EOS itself."""

    valid = torch.ones_like(response_ids, dtype=torch.bool)
    if eos_token_id is None:
        return valid
    eos = response_ids.eq(int(eos_token_id))
    if not eos.any().item():
        return valid
    first = int(eos.nonzero(as_tuple=True)[0][0].item())
    valid[first + (0 if exclude_eos else 1) :] = False
    return valid


def apply_future_reweighting(
    base_log_probs: torch.Tensor,
    candidate_ids: torch.LongTensor,
    relative_future_scores: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply sparse future likelihood ratios and normalize over the full vocab."""

    if base_log_probs.ndim != 1:
        raise ValueError("base_log_probs must be one-dimensional")
    if candidate_ids.ndim != 1 or relative_future_scores.ndim != 1:
        raise ValueError("candidate_ids and relative_future_scores must be one-dimensional")
    if candidate_ids.numel() != relative_future_scores.numel():
        raise ValueError("candidate ids and future scores must have equal length")
    if strength == 0 or candidate_ids.numel() == 0:
        return base_log_probs
    finite = torch.isfinite(relative_future_scores)
    if not finite.any().item():
        return base_log_probs
    candidate_ids = candidate_ids[finite]
    relative_future_scores = relative_future_scores[finite]
    corrected = base_log_probs.clone()
    corrected[candidate_ids] += float(strength) * relative_future_scores.to(corrected.dtype)
    normalizer = torch.logsumexp(corrected, dim=-1)
    if not torch.isfinite(normalizer).item():
        return base_log_probs
    return corrected - normalizer


def topk_forward_kl(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Compute the public implementation's teacher-top-k forward KL objective."""

    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("teacher and student log-probabilities must have equal shapes")
    k = min(int(top_k), teacher_log_probs.shape[-1])
    values, indices = torch.topk(teacher_log_probs, k=k, dim=-1)
    student_values = student_log_probs.gather(-1, indices)
    finite = torch.isfinite(values)
    terms = torch.where(
        finite,
        values.exp() * (values - student_values),
        torch.zeros_like(values),
    )
    return terms.sum(dim=-1)

