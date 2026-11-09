"""On-policy block-diffusion rollout with first-reveal trajectory capture."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mindspeed_llm.fsdp2.models.qwen3.utils import set_diffusion_lm


def _root_forward(model, *, lm_positions: torch.LongTensor | None = None, **kwargs):
    """Run the FSDP root while projecting only requested hidden positions."""

    if lm_positions is None:
        return model(**kwargs)

    def select_hidden(_module, inputs):
        hidden_states = inputs[0].index_select(1, lm_positions)
        return (hidden_states, *inputs[1:])

    handle = model.lm_head.register_forward_pre_hook(select_hidden)
    try:
        return model(**kwargs)
    finally:
        handle.remove()


def _crop_cache(cache, length: int) -> None:
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    for layer_index in range(len(cache.key_cache)):
        cache.key_cache[layer_index] = cache.key_cache[layer_index][:, :, :length]
        cache.value_cache[layer_index] = cache.value_cache[layer_index][:, :, :length]
    cache._seen_tokens = length


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        logits = torch.full_like(logits, -torch.inf).scatter(-1, sorted_indices, sorted_logits)
    return logits


def _sample(logits, temperature, top_k, top_p, mask_token_id, generators):
    logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
    if 0 <= mask_token_id < logits.shape[-1]:
        logits[..., mask_token_id] = -torch.inf
    raw_probabilities = F.softmax(logits, dim=-1)
    if temperature == 0 or top_k == 1:
        proposed = logits.argmax(dim=-1)
    else:
        sample_logits = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
        probabilities = F.softmax(sample_logits, dim=-1)
        if isinstance(generators, (list, tuple)):
            rows = []
            for row, generator in zip(probabilities, generators):
                sampled = torch.multinomial(
                    row.reshape(-1, row.shape[-1]),
                    num_samples=1,
                    generator=generator,
                ).reshape(row.shape[:-1])
                rows.append(sampled)
            proposed = torch.stack(rows)
        else:
            proposed = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]),
                num_samples=1,
                generator=generators,
            ).reshape(probabilities.shape[:-1])
    confidence = raw_probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
    return proposed, confidence


def _transfer_schedule(block_length: int, steps: int) -> list[int]:
    base, remainder = divmod(block_length, steps)
    return [base + (1 if index < remainder else 0) for index in range(steps)]


def _prompt_block_attention_mask(
    prompt_attention_mask: torch.BoolTensor,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build JetEngine's start-aligned block-causal prompt prefill mask."""
    batch_size, padded_length = prompt_attention_mask.shape
    allowed = torch.zeros(
        (batch_size, padded_length, padded_length),
        dtype=torch.bool,
        device=prompt_attention_mask.device,
    )
    for row in range(batch_size):
        valid = prompt_attention_mask[row].nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            allowed[row].fill_diagonal_(True)
            continue
        valid_start = int(valid[0].item())
        if not prompt_attention_mask[row, valid_start:].all().item():
            raise ValueError("rollout prompts must use contiguous left padding")
        # Keep padded query rows numerically safe; padded keys remain invisible
        # to every valid query and to all subsequent response blocks.
        if valid_start:
            pad_rows = torch.arange(valid_start, device=allowed.device)
            allowed[row, pad_rows, pad_rows] = True
        for start in range(valid_start, padded_length, block_size):
            end = min(start + block_size, padded_length)
            allowed[row, start:end, valid_start:end] = True
    mask = torch.full(
        (batch_size, 1, padded_length, padded_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=prompt_attention_mask.device,
    )
    mask[:, 0].masked_fill_(allowed, 0)
    return mask


def _split_prompt_for_rollout(prompt_ids, prompt_lengths, block_size, mask_token_id):
    """Left-pad complete prefix blocks and keep each row's tail in block zero."""
    device = prompt_ids.device
    prompt_width = prompt_ids.shape[1]
    tail_lengths = prompt_lengths.remainder(block_size)
    prefix_lengths = prompt_lengths - tail_lengths
    # Keep one masked dummy cache column when all local rows have empty prefill.
    prefix_width = max(1, int(prefix_lengths.max().item()))
    prefix_offsets = torch.arange(prefix_width, device=device) + (
        prefix_lengths - prefix_width
    ).unsqueeze(1)
    prefix_valid = prefix_offsets.ge(0)
    source_positions = (prompt_width - prompt_lengths).unsqueeze(1) + prefix_offsets
    prefix_ids = prompt_ids.gather(1, source_positions.clamp(0, prompt_width - 1))

    columns = torch.arange(block_size, device=device)
    tail_positions = (prompt_width - tail_lengths).unsqueeze(1) + columns
    first_mutable = columns >= tail_lengths.unsqueeze(1)
    first_block = prompt_ids.gather(1, tail_positions.clamp(max=prompt_width - 1))
    first_block.masked_fill_(first_mutable, mask_token_id)
    return prefix_ids, prefix_valid, prefix_lengths, tail_lengths, first_block, first_mutable


@torch.no_grad()
def block_diffusion_rollout(
    model,
    prompt_ids: torch.LongTensor,
    *,
    response_tokens: int,
    block_size: int,
    denoise_steps: int,
    mask_token_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    remasking_strategy: str,
    dynamic_threshold: float,
    seed: int,
    eos_token_id: int | None = None,
    prompt_attention_mask: torch.BoolTensor | None = None,
    sample_seeds: list[int] | None = None,
    block_alignment: str = "response",
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Generate responses and record each token's first reveal step.

    Ranks stop only when every trajectory across the FSDP group has emitted EOS,
    so all ranks execute the same number of student forwards. The configured
    response length remains a hard curriculum cap. Prompt alignment may decode
    extra positions in the final physical block and discards them on return.
    """

    from transformers.cache_utils import DynamicCache

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] < 1:
        raise ValueError("prompt_ids must have shape [local_rollout_batch, prompt_length]")
    if response_tokens <= 0:
        raise ValueError("response_tokens must be positive")
    if block_alignment not in {"response", "prompt"}:
        raise ValueError("block_alignment must be response or prompt")
    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    else:
        prompt_attention_mask = prompt_attention_mask.to(device=device, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt_attention_mask must have the same shape as prompt_ids")
    prompt_lengths = prompt_attention_mask.sum(dim=-1)
    if (prompt_lengths <= 0).any().item():
        raise ValueError("every rollout prompt must contain at least one non-padding token")
    prompt_aligned = block_alignment == "prompt"
    if prompt_aligned:
        valid_columns = torch.arange(prompt_length, device=device) >= (
            prompt_length - prompt_lengths
        ).unsqueeze(1)
        if not torch.equal(valid_columns, prompt_attention_mask):
            raise ValueError("rollout prompts must use contiguous left padding")
        (
            prompt_ids,
            prompt_attention_mask,
            prompt_lengths,
            tail_lengths,
            first_block,
            first_mutable,
        ) = _split_prompt_for_rollout(prompt_ids, prompt_lengths, block_size, mask_token_id)
        prompt_length = prompt_ids.shape[1]
        max_tail = tail_lengths.max().to(dtype=torch.int32)
        if distributed:
            dist.all_reduce(max_tail, op=dist.ReduceOp.MAX)
        max_physical_tokens = response_tokens + int(max_tail.item())
    else:
        max_physical_tokens = response_tokens
    if sample_seeds is not None:
        if len(sample_seeds) != batch_size:
            raise ValueError("sample_seeds must contain one seed per rollout prompt")
        generators = [
            torch.Generator(device=device).manual_seed(int(sample_seed))
            for sample_seed in sample_seeds
        ]
    else:
        generators = torch.Generator(device=device).manual_seed(int(seed))
    was_training = model.training
    model.eval()

    # Official JetEngine prefill is block-causal (staircase), not token-causal.
    set_diffusion_lm(model, True)
    cache = DynamicCache()
    prompt_positions = prompt_attention_mask.long().cumsum(dim=-1) - 1
    prompt_positions.masked_fill_(~prompt_attention_mask, 0)
    prompt_cache_positions = torch.arange(prompt_length, device=device)
    model_dtype = next(model.parameters()).dtype
    # Call the Qwen3Diffusion root rather than its backbone directly.  The root
    # owns the composable FSDP2 hooks for parameters not covered by child plans.
    prefill = _root_forward(
        model,
        lm_positions=torch.tensor([prompt_length - 1], device=device),
        input_ids=prompt_ids,
        attention_mask=_prompt_block_attention_mask(
            prompt_attention_mask,
            block_size,
            model_dtype,
        ),
        position_ids=prompt_positions,
        cache_position=prompt_cache_positions,
        past_key_values=cache,
        use_cache=True,
    )
    cache = prefill.past_key_values
    del prefill

    response_blocks = []
    reveal_blocks = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated = 0
    global_denoise_step = 0
    num_blocks = math.ceil(max_physical_tokens / block_size)
    for block_index in range(num_blocks):
        current_length = (
            block_size if prompt_aligned else min(block_size, response_tokens - generated)
        )
        if prompt_aligned and block_index == 0:
            block, mutable = first_block, first_mutable
        else:
            block = torch.full(
                (batch_size, current_length), mask_token_id, dtype=prompt_ids.dtype, device=device
            )
            mutable = torch.ones_like(block, dtype=torch.bool)
        first_reveal = torch.zeros_like(block)
        steps = min(denoise_steps, current_length)
        schedule = _transfer_schedule(current_length, steps)
        block_start = prompt_length + generated

        for step_index in range(steps):
            masked = mutable & block.eq(mask_token_id)
            has_masked = bool(masked.any().item())
            if not has_masked and not distributed:
                break
            prefix_cache_length = cache.get_seq_length()
            logical_offsets = torch.arange(
                generated, generated + current_length, device=device
            )
            positions = prompt_lengths.unsqueeze(1) + logical_offsets.unsqueeze(0)
            cache_positions = torch.arange(
                block_start, block_start + current_length, device=device
            )
            key_valid = torch.cat(
                [
                    prompt_attention_mask,
                    torch.ones(
                        (batch_size, generated + current_length),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.full(
                (batch_size, 1, current_length, key_valid.shape[1]),
                torch.finfo(model_dtype).min,
                dtype=model_dtype,
                device=device,
            )
            attention_mask.masked_fill_(key_valid[:, None, None, :], 0)
            # FSDP ranks must enter the same root forwards even when a local
            # batch completes its denoising rounds ahead of the other ranks.
            denoise = _root_forward(
                model,
                lm_positions=None if has_masked else torch.zeros(1, device=device, dtype=torch.long),
                input_ids=block,
                attention_mask=attention_mask,
                position_ids=positions,
                cache_position=cache_positions,
                past_key_values=cache,
                use_cache=True,
            )
            cache = denoise.past_key_values
            _crop_cache(cache, prefix_cache_length)
            if not has_masked:
                del denoise
                continue
            logits = denoise.logits
            del denoise
            proposed, confidence = _sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                mask_token_id=mask_token_id,
                generators=generators,
            )
            update = torch.zeros_like(masked)
            masked_confidence = confidence.masked_fill(~masked, -torch.inf)
            if remasking_strategy == "low_confidence_dynamic":
                update = masked & (confidence > dynamic_threshold)
            # Enforce the paper's per-sequence transfer schedule.  Rollouts are
            # batched locally so 8 ranks generate 128 trajectories without
            # serializing 16 independent decode loops on every rank.
            for row in range(batch_size):
                minimum = min(schedule[step_index], int(masked[row].sum().item()))
                selected_count = int(update[row].sum().item())
                if selected_count < minimum:
                    needed = minimum - selected_count
                    remaining = masked_confidence[row].masked_fill(update[row], -torch.inf)
                    selected = torch.topk(remaining, k=needed).indices
                    update[row].scatter_(0, selected, True)
            if step_index + 1 == steps:
                update |= masked
            global_denoise_step += 1
            first_reveal[update & first_reveal.eq(0)] = global_denoise_step
            block[update] = proposed[update]

        if (mutable & (block.eq(mask_token_id) | first_reveal.eq(0))).any().item():
            raise RuntimeError(f"rollout block {block_index} did not fully denoise")
        response_blocks.append(block)
        reveal_blocks.append(first_reveal)
        generated += current_length

        globally_finished = False
        if eos_token_id is not None:
            eos_in_response = block.eq(int(eos_token_id)) & mutable
            if prompt_aligned:
                response_offsets = torch.arange(
                    generated - current_length, generated, device=device
                ) - tail_lengths.unsqueeze(1)
                eos_in_response &= response_offsets.lt(response_tokens)
            finished |= eos_in_response.any(dim=1)
            all_finished = finished.all().to(dtype=torch.int32)
            if distributed:
                dist.all_reduce(all_finished, op=dist.ReduceOp.MIN)
            globally_finished = bool(all_finished.item())

        if block_index + 1 < num_blocks and not globally_finished:
            cache_length = cache.get_seq_length()
            logical_offsets = torch.arange(
                generated - current_length,
                generated,
                device=device,
            )
            positions = prompt_lengths.unsqueeze(1) + logical_offsets.unsqueeze(0)
            refresh_key_valid = torch.cat(
                [
                    prompt_attention_mask,
                    torch.ones(
                        (batch_size, generated), dtype=torch.bool, device=device
                    ),
                ],
                dim=1,
            )
            refresh_attention_mask = torch.full(
                (batch_size, 1, current_length, refresh_key_valid.shape[1]),
                torch.finfo(model_dtype).min,
                dtype=model_dtype,
                device=device,
            )
            refresh_attention_mask.masked_fill_(
                refresh_key_valid[:, None, None, :], 0
            )
            refresh = _root_forward(
                model,
                lm_positions=torch.tensor([current_length - 1], device=device),
                input_ids=block,
                attention_mask=refresh_attention_mask,
                position_ids=positions,
                cache_position=torch.arange(
                    cache_length, cache_length + current_length, device=device
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache = refresh.past_key_values
            del refresh
        if globally_finished:
            break

    if was_training:
        model.train()
    responses = torch.cat(response_blocks, dim=1)
    reveals = torch.cat(reveal_blocks, dim=1)
    if prompt_aligned:
        width = min(response_tokens, generated)
        indices = tail_lengths.unsqueeze(1) + torch.arange(width, device=device)
        valid = indices.lt(generated)
        indices = indices.clamp(max=generated - 1)
        responses, reveals = responses.gather(1, indices), reveals.gather(1, indices)
        if globally_finished:
            # Every row already has a real EOS. Rectangular padding is removed
            # at the first EOS by the trainer before any target is built.
            responses.masked_fill_(~valid, int(eos_token_id))
    return responses, reveals

