"""Publish and resolve d-OPD checkpoints on a shared filesystem."""

import os
import re
import shutil
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

import torch.distributed as dist
from filelock import FileLock
from torch.distributed.checkpoint import FileSystemReader


CHECKPOINT_META = "dopd_checkpoint_meta.json"
_LOCK = ".checkpoint.lock"
_STEP_NAME = re.compile(r"global_step_([0-9]+)(_final)?")


def checkpoint_plan(step, stop, history_interval, latest_interval):
    """Return the save destinations due at the end of this outer iteration."""
    destinations = []
    if history_interval and step % history_interval == 0:
        destinations.append("checkpoints")
    if latest_interval is not None and (
        step == stop or (latest_interval and step % latest_interval == 0)
    ):
        destinations.append("latest")
    return destinations


def _published(root):
    checkpoints = []
    for path in root.iterdir():
        match = _STEP_NAME.fullmatch(path.name)
        if match and not path.is_symlink() and (path / ".metadata").is_file():
            checkpoints.append((int(match[1]), bool(match[2]), path))
    return [path for _, _, path in sorted(checkpoints)]


def stage_checkpoint(root, step):
    if root.is_symlink():
        raise ValueError(f"Cannot write new checkpoint versions through a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _sync_directory(root.parent)
    with FileLock(str(root / _LOCK)):
        target = root / f"global_step_{step}"
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Checkpoint already exists: {target}")
        staging = root / f".pending-{uuid4().hex}"
        staging.mkdir()
        return staging


def validate_dcp(path, *, require_training=False, world_size=None):
    metadata = FileSystemReader(path).read_metadata()
    keys = metadata.state_dict_metadata
    if not any(key.startswith("model.") for key in keys):
        raise ValueError(f"Missing model state in {path}")
    sizes = {}
    for entry in (metadata.storage_data or {}).values():
        name = path / entry.relative_path
        sizes[name] = max(sizes.get(name, 0), entry.offset + entry.length)
    if not sizes or any(not name.is_file() or name.stat().st_size < end for name, end in sizes.items()):
        raise ValueError(f"Missing or truncated DCP shards in {path}")
    files = [path / ".metadata", *sizes]
    if require_training:
        states = [path / "extra_state" / f"extra_state_rank_{rank}.pt" for rank in range(world_size)]
        if not any(key.startswith("optimizer.") for key in keys) or any(
            not file.is_file() or not file.stat().st_size for file in states
        ):
            raise ValueError(f"Complete optimizer and per-rank training state required: {path}")
        files.extend(states)
    return files


def _sync_directory(path):
    if hasattr(os, "O_DIRECTORY"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish_checkpoint(staging, target, *, only_model, world_size, keep):
    files = validate_dcp(staging, require_training=not only_model, world_size=world_size)
    for path in [*files, staging / CHECKPOINT_META]:
        with path.open("rb+") as stream:
            os.fsync(stream.fileno())
    if not only_model:
        _sync_directory(staging / "extra_state")
    _sync_directory(staging)

    root = target.parent
    with FileLock(str(root / _LOCK)):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Checkpoint already exists: {target}")
        existing = _published(root)
        if root.name == "latest" and existing:
            current_step = int(_STEP_NAME.fullmatch(existing[-1].name)[1])
            next_step = int(_STEP_NAME.fullmatch(target.name)[1])
            if current_step >= next_step:
                raise ValueError(f"Use a new output directory or resume from {existing[-1]}")
        staging.rename(target)
        _sync_directory(root)
        if keep and keep > 0:
            for old in _published(root)[:-keep]:
                try:
                    shutil.rmtree(old)
                except OSError as error:
                    warnings.warn(f"Could not remove old checkpoint {old}: {error}", stacklevel=2)


@contextmanager
def open_checkpoint(path):
    """Pin the selected checkpoint until a reader has finished loading it."""
    path = Path(path).absolute()
    if path.name.startswith(".pending-") and path.parent.name in ("latest", "checkpoints"):
        raise ValueError(f"Cannot load an unpublished checkpoint: {path}")
    with ExitStack() as stack:
        root = path if path.name == "latest" else path.parent
        if root.name in ("latest", "checkpoints") and root.is_dir() and not root.is_symlink():
            stack.enter_context(FileLock(str(root / _LOCK)))
        if path.name == "latest" and not path.is_symlink() and path.is_dir():
            versions = _published(path)
            if versions:
                path = versions[-1]
        if not (path / ".metadata").is_file():
            raise FileNotFoundError(f"No published DCP checkpoint at {path}")
        yield path.resolve()


def rank_zero_result(action):
    """Run filesystem work on rank 0 and communicate its outcome to all ranks."""
    payload = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (action(), None)
        except Exception as error:
            payload[0] = (None, f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(payload, src=0)
    result, error = payload[0]
    if error is not None:
        raise RuntimeError(error)
    return result


@contextmanager
def open_distributed_checkpoint(path):
    with ExitStack() as stack:
        resolved = rank_zero_result(lambda: stack.enter_context(open_checkpoint(path)))
        yield resolved
        dist.barrier()
