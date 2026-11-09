import random
from types import SimpleNamespace

import torch

from mindspeed_llm.experimental.dopd import rollout, scoring
from mindspeed_llm.experimental.dopd.core import (
    active_rank_training_schedule,
    apply_future_reweighting,
    build_dopd_attention_mask,
    cosine_response_length,
    response_block_end,
    response_block_spans,
    response_valid_mask,
    sample_intermediate_state,
    topk_forward_kl,
)
from mindspeed_llm.experimental.dopd.rollout import _prompt_block_attention_mask


def test_two_node_rotation_retains_global_optimizer_batch_eight():
    schedules = [
        active_rank_training_schedule(
            list(range(8)), rank=rank, world_size=16, active_ranks=8, accumulation=1
        )
        for rank in range(16)
    ]
    assert all(len(schedule) == 16 for schedule in schedules)
    for update in range(16):
        assert sum(int(schedule[update][1]) for schedule in schedules) == 8
    for schedule in schedules:
        active_trajectories = [trajectory for trajectory, active, _ in schedule if active]
        assert active_trajectories == list(range(8))


def test_cosine_response_curriculum_hits_endpoints_and_block_alignment():
    values = [cosine_response_length(i, 100, 4000, 100, 4) for i in range(1, 101)]
    assert values[0] == 100
    assert values[-1] == 4000
    assert values == sorted(values)
    assert all(value % 4 == 0 for value in values)


def test_dopd_mask_uses_clean_history_and_current_noised_block():
    mask = build_dopd_attention_mask(3, 4, 2, dtype=torch.float32, device="cpu")[0, 0]
    assert mask[0, 0] == 0 and mask[0, 1] < 0
    assert mask[1, :3].eq(0).all()
    assert mask[3, :5].eq(0).all() and mask[3, 5:].lt(0).all()
    assert mask[7, :3].eq(0).all() and mask[7, 3:7].lt(0).all()
    assert mask[7, 7:9].eq(0).all() and mask[7, 9:].lt(0).all()
    assert mask[9, :5].eq(0).all() and mask[9, 5:9].lt(0).all()
    assert mask[9, 9:].eq(0).all()


def test_rollout_prefill_mask_is_start_aligned_block_causal_with_left_padding():
    valid = torch.tensor(
        [[False, True, True, True, True], [False, False, True, True, True]]
    )
    mask = _prompt_block_attention_mask(valid, block_size=2, dtype=torch.float32)

    # Four-token prompt: [1,2] and [3,4] are bidirectional blocks.
    assert mask[0, 0, 1:3, 1:3].eq(0).all()
    assert mask[0, 0, 1:3, 3:].lt(0).all()
    assert mask[0, 0, 3:5, 1:5].eq(0).all()
    # Three-token prompt is start-aligned: [2,3] then [4].
    assert mask[1, 0, 2:4, 2:4].eq(0).all()
    assert mask[1, 0, 2:4, 4].lt(0).all()
    assert mask[1, 0, 4, 2:5].eq(0).all()
    # Left-padding keys never leak into valid queries.
    assert mask[0, 0, 1:, 0].lt(0).all()
    assert mask[1, 0, 2:, :2].lt(0).all()


def test_sample_intermediate_state_masks_current_and_later_reveal_groups():
    response = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
    steps = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    state = sample_intermediate_state(response, steps, 4, 99, random.Random(7))
    for start in (0, 4):
        selected_step = steps[start : start + 4][state.loss_mask[start : start + 4]][0]
        expected_mask = steps[start : start + 4] >= selected_step
        assert torch.equal(state.noised_response[start : start + 4].eq(99), expected_mask)


def test_remaining_mask_changes_supervision_without_changing_sampled_input():
    response = torch.tensor([10, 11, 12, 13])
    steps = torch.tensor([1, 2, 3, 4])
    rng = SimpleNamespace(choice=lambda choices: 2)
    old = sample_intermediate_state(response, steps, 4, 99, rng)
    new = sample_intermediate_state(
        response, steps, 4, 99, rng, loss_mask_mode="remaining_mask"
    )
    assert old.noised_response.tolist() == new.noised_response.tolist() == [10, 99, 99, 99]
    assert old.loss_mask.tolist() == [False, True, False, False]
    assert new.loss_mask.tolist() == [False, True, True, True]
    valid = response_valid_mask(response, eos_token_id=12, exclude_eos=True)
    assert (new.loss_mask & valid).tolist() == [False, True, False, False]


def test_prompt_aligned_training_keeps_clean_and_noised_prompt_tail_separate():
    # P=6, R=5, B=4: [p0:p4], [p4:p6, r0:r2], [r2:r5].
    assert list(response_block_spans(5, 4, 2)) == [(0, 2), (2, 5)]
    assert [response_block_end(i, 5, 4, 2) for i in range(5)] == [2, 2, 5, 5, 5]
    mask = build_dopd_attention_mask(
        6, 5, 4, dtype=torch.float32, device="cpu", block_alignment="prompt"
    )[0, 0]
    assert mask.shape == (18, 18)
    assert mask[:4, :4].eq(0).all() and mask[:4, 4:].lt(0).all()
    # The clean prompt tail reads clean r0:r2; the noised prompt tail only
    # reads the complete prompt prefix and its own noised first block.
    assert mask[4:8, :8].eq(0).all() and mask[4:8, 8:].lt(0).all()
    assert mask[11:15, :4].eq(0).all()
    assert mask[11:15, 4:11].lt(0).all()
    assert mask[11:15, 11:15].eq(0).all() and mask[11:15, 15:].lt(0).all()
    # Later noised response reads the completed mixed block, not its noised copy.
    assert mask[15:, :8].eq(0).all() and mask[15:, 8:15].lt(0).all()
    assert mask[15:, 15:].eq(0).all()
    state = sample_intermediate_state(
        torch.tensor([20, 21, 22, 23, 24]),
        torch.tensor([2, 1, 3, 4, 3]),
        4,
        99,
        SimpleNamespace(choice=lambda choices: choices[-1]),
        prompt_length=6,
        block_alignment="prompt",
        loss_mask_mode="remaining_mask",
    )
    assert state.noised_response.tolist() == [99, 21, 22, 99, 24]


class _RolloutProbe(torch.nn.Module):
    """Record model inputs and predict the deterministic token 10 + RoPE position."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.lm_head = torch.nn.Identity()
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, cache_position, past_key_values, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone(), position_ids.clone()))
        kv = input_ids[:, None, :, None].float()
        past_key_values.update(kv, kv, 0, {"cache_position": cache_position})
        logits = torch.full((*input_ids.shape, 128), -20.0)
        logits.scatter_(2, (position_ids + 10).unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=self.lm_head(logits), past_key_values=past_key_values)


def test_prompt_aligned_rollout_batches_all_remainders_and_caps_response(monkeypatch):
    monkeypatch.setattr(rollout, "set_diffusion_lm", lambda model, value: None)
    model = _RolloutProbe()
    # Logical prompt lengths 4, 5, 6, 3: all four remainders, including an
    # empty prefill row. Token 99 is immutable here and must not be resampled.
    prompts = torch.tensor([
        [0, 0, 31, 32, 33, 34],
        [0, 41, 42, 43, 44, 99],
        [51, 52, 53, 54, 55, 56],
        [0, 0, 0, 61, 62, 63],
    ])
    output, reveals = rollout.block_diffusion_rollout(
        model,
        prompts,
        response_tokens=5,
        block_size=4,
        denoise_steps=4,
        mask_token_id=99,
        temperature=0,
        top_k=1,
        top_p=1.0,
        remasking_strategy="low_confidence_static",
        dynamic_threshold=0.9,
        seed=42,
        prompt_attention_mask=prompts.ne(0),
        block_alignment="prompt",
    )
    assert output.tolist() == [
        [14, 15, 16, 17, 18],
        [15, 16, 17, 18, 19],
        [16, 17, 18, 19, 20],
        [13, 14, 15, 16, 17],
    ]
    assert reveals.gt(0).all()
    first_ids, first_mask, first_positions = model.calls[1]
    assert first_ids.tolist() == [
        [99, 99, 99, 99], [99, 99, 99, 99], [55, 56, 99, 99], [61, 62, 63, 99]
    ]
    assert first_positions.tolist() == [[4, 5, 6, 7]] * 3 + [[0, 1, 2, 3]]
    assert first_mask[3, 0, :, :4].lt(0).all()
    assert first_mask[3, 0, :, 4:].eq(0).all()


def test_rollout_keeps_forward_count_when_local_rank_finishes_first(monkeypatch):
    monkeypatch.setattr(rollout, "set_diffusion_lm", lambda model, value: None)
    monkeypatch.setattr(rollout.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rollout.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        rollout.dist, "all_reduce",
        lambda tensor, op: tensor.fill_(3) if op == rollout.dist.ReduceOp.MAX else None,
    )
    calls = []
    for prompt in (torch.tensor([[31, 32, 33, 34]]), torch.tensor([[31, 32, 33]])):
        model = _RolloutProbe()
        output, reveals = rollout.block_diffusion_rollout(
            model, prompt, response_tokens=5, block_size=4, denoise_steps=4,
            mask_token_id=99, temperature=0, top_k=1, top_p=1.0,
            remasking_strategy="low_confidence_static", dynamic_threshold=0.9,
            seed=42, block_alignment="prompt",
        )
        assert output.shape == reveals.shape == (1, 5)
        assert reveals.gt(0).all()
        calls.append(len(model.calls))
    # Prefill + two physical blocks of four forwards + one inter-block save.
    assert calls == [10, 10]


def test_prompt_tail_eos_does_not_stop_generation(monkeypatch):
    monkeypatch.setattr(rollout, "set_diffusion_lm", lambda model, value: None)
    model = _RolloutProbe()
    output, reveals = rollout.block_diffusion_rollout(
        model, torch.tensor([[31, 15, 33]]), response_tokens=2, block_size=4,
        denoise_steps=4, mask_token_id=99, temperature=0, top_k=1, top_p=1.0,
        remasking_strategy="low_confidence_static", dynamic_threshold=0.9,
        seed=42, eos_token_id=15, block_alignment="prompt",
    )
    # The first block produces 13. The second produces 14 and then EOS (15),
    # but 15 is past the response cap and the prompt EOS is immutable.
    assert output.tolist() == [[13, 14]]
    assert reveals.gt(0).all()
    assert model.calls[0][1].shape == (1, 1, 1, 1)
    assert model.calls[1][1][0, 0, :, 0].lt(0).all()


def test_prompt_aligned_loss_projects_response_and_scores_same_block(monkeypatch):
    monkeypatch.setattr(scoring, "set_diffusion_lm", lambda model, value: None)
    student = _RolloutProbe()
    student.mask_token_id = 99
    student.lm_head = torch.nn.Linear(128, 128, bias=False)
    original_forward = student.forward

    def student_forward(*, use_cache, **kwargs):
        from transformers.cache_utils import DynamicCache

        return original_forward(
            **kwargs, cache_position=None, past_key_values=DynamicCache()
        )

    student.forward = student_forward
    teacher = SimpleNamespace(
        model=lambda input_ids, **kwargs: SimpleNamespace(
            last_hidden_state=torch.zeros((*input_ids.shape, 128)), past_key_values=None
        ),
        lm_head=torch.nn.Identity(),
    )
    scored = []

    def score(*args, position, future_end, branch_token_ids, **kwargs):
        scored.append((position, future_end))
        return torch.arange(branch_token_ids.numel(), dtype=torch.float32)

    monkeypatch.setattr(scoring, "score_future_branches", score)
    loss, stats = scoring.compute_dopd_loss(
        student, teacher, torch.tensor([[31, 32, 33, 34, 35, 36]]),
        torch.tensor([[21, 22, 23, 24, 25]]),
        torch.tensor([[99, 22, 99, 24, 25]]),
        torch.tensor([[True, False, True, False, False]]),
        block_size=4, candidate_top_k=2, loss_top_k=2, correction_strength=0.5,
        require_visible_future=True, correction_allowed=True,
        score_chunk_size=8, cache_max_bytes=1024,
        block_alignment="prompt",
    )
    assert scored == [(6, 8), (8, 11)]
    assert stats.loss_tokens == stats.eligible_positions == 2
    ids, mask, positions = student.calls[0]
    assert ids[0, 11:].tolist() == [35, 36, 99, 22, 99, 24, 25]
    assert positions.tolist() == [list(range(11)) + list(range(4, 11))]
    assert mask[0, 0, 13, 4:11].lt(0).all()
    loss.backward()
    assert student.lm_head.weight.grad.abs().sum() > 0


def test_future_reweighting_normalizes_and_topk_kl_backpropagates():
    teacher = torch.tensor([0.5, 0.3, 0.2]).log()
    corrected = apply_future_reweighting(
        teacher, torch.tensor([0, 2]), torch.tensor([-1.0, 1.0]), strength=0.5
    )
    torch.testing.assert_close(corrected.exp().sum(), torch.tensor(1.0))
    assert corrected[2] > teacher[2]

    logits = torch.randn(1, 3, requires_grad=True)
    student = logits.log_softmax(-1)
    loss = topk_forward_kl(corrected.unsqueeze(0), student, top_k=2).mean()
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

