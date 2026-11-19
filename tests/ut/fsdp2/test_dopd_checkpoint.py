import json

import pytest
import torch
from filelock import FileLock, Timeout
from torch.distributed import checkpoint as dcp

from mindspeed_llm.experimental.dopd.checkpoint import (
    CHECKPOINT_META,
    checkpoint_plan,
    open_checkpoint,
    publish_checkpoint,
    stage_checkpoint,
    validate_dcp,
)


@pytest.mark.parametrize(
    "step,stop,history,latest,expected",
    [
        (483, 483, 0, None, []),
        (50, 483, 50, 0, ["checkpoints"]),
        (483, 483, 50, 0, ["latest"]),
        (23, 23, 50, 10, ["latest"]),
        (20, 23, 50, 10, ["latest"]),
        (150, 150, 50, 10, ["checkpoints", "latest"]),
    ],
)
def test_checkpoint_schedule(step, stop, history, latest, expected):
    assert checkpoint_plan(step, stop, history, latest) == expected


def _write_staged(root, step, only_model):
    stage = stage_checkpoint(root, step)
    state = {"model": {"weight": torch.tensor([step])}}
    if not only_model:
        state["optimizer"] = {"state": torch.tensor([step * 2])}
        extra_dir = stage / "extra_state"
        extra_dir.mkdir()
        torch.save({"global_step": step}, extra_dir / "extra_state_rank_0.pt")
    dcp.save(state, checkpoint_id=stage)
    (stage / CHECKPOINT_META).write_text(
        json.dumps({"global_step": step, "only_model": only_model}), encoding="utf-8"
    )
    return stage


def test_rolling_publish_retains_old_on_failure_and_leaves_history_untouched(tmp_path):
    history = tmp_path / "checkpoints"
    latest = tmp_path / "latest"
    step_50 = history / "global_step_50"
    old = latest / "global_step_50"
    publish_checkpoint(_write_staged(history, 50, True), step_50, only_model=True, world_size=1, keep=10)
    publish_checkpoint(_write_staged(latest, 50, False), old, only_model=False, world_size=1, keep=1)

    broken = _write_staged(latest, 90, False)
    next_path = latest / "global_step_100"
    shard = next(broken.glob("*.distcp"))
    shard.write_bytes(b"x")
    with pytest.raises(ValueError, match="Missing or truncated"):
        publish_checkpoint(broken, latest / "global_step_90", only_model=False, world_size=1, keep=1)
    with open_checkpoint(latest) as current:
        assert current == old

    publish_checkpoint(_write_staged(latest, 100, False), next_path, only_model=False, world_size=1, keep=1)
    assert not old.exists()
    assert step_50.is_dir()
    with open_checkpoint(latest) as current:
        state = {"model": {"weight": torch.empty(1, dtype=torch.long)}}
        dcp.load(state, checkpoint_id=current)
        assert state["model"]["weight"].item() == 100
        validate_dcp(current, require_training=True, world_size=1)
    with pytest.raises(ValueError, match="Complete optimizer"):
        validate_dcp(step_50, require_training=True, world_size=1)
    with pytest.raises(FileExistsError):
        stage_checkpoint(history, 50)


def test_latest_resolution_after_interrupted_prune(tmp_path):
    latest = tmp_path / "latest"
    for step in [90, 100]:
        publish_checkpoint(
            _write_staged(latest, step, True),
            latest / f"global_step_{step}",
            only_model=True,
            world_size=1,
            keep=None,
        )
    pending = _write_staged(latest, 110, True)
    with pytest.raises(ValueError, match="unpublished"):
        with open_checkpoint(pending):
            pytest.fail("A staged checkpoint was exposed to a reader")
    with open_checkpoint(latest) as current:
        assert current.name == "global_step_100"
        validate_dcp(current)
        with pytest.raises(ValueError, match="Complete optimizer"):
            validate_dcp(current, require_training=True, world_size=1)


def test_latest_reader_holds_publication_lock(tmp_path):
    latest = tmp_path / "latest"
    path = latest / "global_step_50"
    publish_checkpoint(_write_staged(latest, 50, True), path, only_model=True, world_size=1, keep=1)
    # A lock file may be absent between acquisitions, as on Windows.
    (latest / ".checkpoint.lock").unlink(missing_ok=True)
    with open_checkpoint(latest) as current:
        assert current == path
        with pytest.raises(Timeout):
            with FileLock(str(latest / ".checkpoint.lock"), timeout=0):
                pytest.fail("A writer could remove a checkpoint while it was being read")
