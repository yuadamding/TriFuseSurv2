"""Canonical, fail-closed experiment settings for checkpoint continuation."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


# These change I/O or reporting only. New options are immutable by default.
# In particular, epochs, selection metrics, precision and dropout are NOT here.
RESUME_MUTABLE_SETTINGS = frozenset({
    "resume", "workers", "eval_workers", "prefetch_factor",
    "cache_volumes", "volume_cache_size", "log_every_batches",
    "roi_focus_every_batches", "print_trainable_backbone_params",
})


def canonical_experiment_spec(args, *, runtime: Mapping[str, Any] | None = None) -> dict:
    settings = {name: value for name, value in vars(args).items()
                if name not in RESUME_MUTABLE_SETTINGS
                and not (name in {"amp_init_scale", "routing_intervention", "numerical_failure_policy"} and value is None)}
    value = {"schema_version": 1, "settings": settings, "runtime": dict(runtime or {}),
             "resume_mutable_settings": sorted(RESUME_MUTABLE_SETTINGS)}
    # Round-trip gives one JSON representation for tuples/lists and rejects NaN.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def experiment_spec_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def changed_contract_fields(previous: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(previous, dict) and isinstance(current, dict):
        changed = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                changed.append(path)
            else:
                changed.extend(changed_contract_fields(previous[key], current[key], path))
        return changed
    return [] if previous == current else [prefix]


def scheduler_t_max(args) -> int:
    epochs = int(args.epochs)
    if int(args.roi_focus_warmup_epochs) > 0 and float(args.roi_focus_warmup_survival_weight) <= 0:
        return max(1, epochs - int(args.roi_focus_warmup_epochs))
    return epochs


def execution_topology(device, *, ddp_world_size: int | None = None, data_parallel_devices: int | None = None) -> dict:
    """Use one shared specification for every rank; device ordinals are local."""
    mode, count = "single", 1
    if ddp_world_size is not None:
        mode, count = "ddp", int(ddp_world_size)
    elif data_parallel_devices is not None:
        mode, count = "data_parallel", int(data_parallel_devices)
    if count < 1:
        raise ValueError("Execution topology requires a positive world size")
    return {"device_type": device.type, "execution_mode": mode, "effective_world_size": count}
