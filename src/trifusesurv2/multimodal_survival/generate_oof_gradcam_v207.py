#!/usr/bin/env python3
"""Generate OOF Grad-CAM packages for TriFuseSurv2 2.0.7 v2 checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from trifusesurv2.encoders.clinical import SemanticClinicalTokenEncoder
from trifusesurv2.encoders.radiomics import HabitatRadiomicsTokenEncoder
from trifusesurv2.explain.ablation import compute_v2_ablation_package
from trifusesurv2.explain.attention_export import attention_aux_to_jsonable
from trifusesurv2.explain.gradcam_v2_core import (
    MODEL_CLASS,
    SOFTWARE_VERSION,
    TARGET_COMMIT_SHA,
    append_manifest,
    assert_v207_v2_checkpoint,
    cam_from_features,
    cam_mass_summary,
    checkpoint_args_to_dict,
    image_habitat_names,
    isolate_image_habitat_gradient,
    make_overlay_png,
    normalize_component,
    normalized_state_dict,
    pid_slug,
    risk_vector,
    save_nifti,
    select_survival_target,
    split_signed_cam,
    supports_from_backbone_aux,
    write_json,
)
from trifusesurv2.models.contour_habitat_survival import ContourAwareHabitatSurvivalModel
from trifusesurv2.models.swinunetr_shared_roi_token_backbone import ContourAwareROITokenBackbone
from trifusesurv2.schema import (
    CLINICAL_TOKEN_GROUPS,
    ENDPOINT_MAP,
    IMAGE_HABITATS,
    NODE_TOPOLOGY_FEATURES,
    RADIOLOGY_HABITATS,
    SURVIVAL_ENDPOINTS,
    TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS,
)
from trifusesurv2.utils.data import PreprocessedHabitatOOFDataset, resolve_preprocessed_case_path


def _checkpoint_for_fold(run_dir: Path, fold: int, checkpoint: str) -> Path:
    candidates = [
        run_dir / f"fold_{fold:02d}" / f"{checkpoint}.pt",
        run_dir / f"fold{fold:02d}" / f"{checkpoint}.pt",
        run_dir / f"{run_dir.name}_fold{fold:02d}" / f"fold_{fold:02d}" / f"{checkpoint}.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find fold {fold} checkpoint={checkpoint} under {run_dir}")


def _relative_path(path: str | Path | None) -> str:
    """Return a cwd-relative path string for manifests/status files."""

    if path is None or str(path) == "":
        return ""
    try:
        return os.path.relpath(os.fspath(path), start=os.getcwd())
    except Exception:
        return str(path)


def _load_splits(splits_dir: Path, fold: int) -> dict[str, list[str]]:
    fold_dir = splits_dir / f"fold_{fold:02d}"
    if fold_dir.is_dir():
        out = {}
        for split in ("train", "val", "test"):
            path = fold_dir / f"{split}_ids.txt"
            out[split] = [x.strip() for x in path.read_text().splitlines() if x.strip()]
        return out
    csv_path = splits_dir / "splits.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        fdf = df[df["fold"].astype(int) == int(fold)] if "fold" in df.columns else df
        return {
            split: fdf.loc[fdf["split"].astype(str).str.lower() == split, "patient_id"].astype(str).tolist()
            for split in ("train", "val", "test")
        }
    raise FileNotFoundError(f"Could not find split files for fold {fold} in {splits_dir}")


def _select_ids(df: pd.DataFrame, ids: list[str], id_col: str) -> pd.DataFrame:
    wanted = {str(x) for x in ids}
    return df[df[id_col].astype(str).isin(wanted)].reset_index(drop=True)


def _linear_in_dim(state: dict[str, torch.Tensor], suffix: str) -> int:
    for key, value in state.items():
        if key.endswith(suffix) and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[1])
    return 0


def _clinical_groups_from_state(state: dict[str, torch.Tensor]) -> OrderedDict[str, tuple[str, ...]]:
    names = []
    prefix = "habitat_model.clinical_proj."
    for key in state:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            name = rest.split(".", 1)[0]
            if name not in names:
                names.append(name)
    base = TREATMENT_AWARE_CLINICAL_TOKEN_GROUPS if "treatment" in names else CLINICAL_TOKEN_GROUPS
    return OrderedDict((name, tuple(base.get(name, ()))) for name in names) if names else OrderedDict(CLINICAL_TOKEN_GROUPS)


def _radiomics_habitats_from_state(state: dict[str, torch.Tensor]) -> tuple[str, ...]:
    names = []
    prefix = "habitat_model.radiomics_proj."
    for key in state:
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            name = rest.split(".", 1)[0]
            if name not in names:
                names.append(name)
    return tuple(names) if names else tuple(RADIOLOGY_HABITATS)


def _backbone_cfg(ck_args: dict[str, Any], image_token_dim: int) -> dict[str, Any]:
    return dict(
        img_size=tuple(int(x) for x in ck_args.get("img_size", (128, 256, 256))),
        feature_size=int(ck_args.get("feature_size", 96)),
        depths=tuple(int(x) for x in ck_args.get("depths", (2, 2, 18, 2))),
        num_heads=tuple(int(x) for x in ck_args.get("num_heads", (3, 6, 12, 24))),
        drop_rate=float(ck_args.get("drop_rate", 0.0)),
        attn_drop_rate=float(ck_args.get("attn_drop_rate", 0.0)),
        dropout_path_rate=float(ck_args.get("dropout_path_rate", 0.0)),
        normalize=True,
        use_checkpoint=bool(ck_args.get("use_checkpoint", False)),
        token_dim=int(image_token_dim or ck_args.get("img_token_dim", ck_args.get("fused_dim", 512))),
        token_mlp_dropout=float(ck_args.get("token_mlp_dropout", 0.0)),
        token_mlp_hidden_dim=int(ck_args.get("token_mlp_hidden_dim", 0)),
        attn_mask_bias=float(ck_args.get("attn_mask_bias", 2.0)),
        use_multiscale=bool(ck_args.get("use_multiscale", False)),
        mask_interp=str(ck_args.get("mask_interp", "nearest")),
        min_roi_frac=float(ck_args.get("min_roi_frac", 1e-5)),
        min_roi_voxels_deep=int(ck_args.get("min_roi_voxels_deep", 8)),
        token_dropout=0.0,
        pt_shell_radius=int(ck_args.get("pt_shell_radius", 3)),
        ln_shell_radius=int(ck_args.get("ln_shell_radius", 3)),
        shell_body_from_ct=bool(ck_args.get("shell_body_from_ct", False)),
        body_ct_thr=str(ck_args.get("body_ct_thr", "auto")),
        body_ct_thr_hu=float(ck_args.get("body_ct_thr_hu", -500.0)),
        body_close_r=int(ck_args.get("body_close_r", 2)),
        body_max_frac=float(ck_args.get("body_max_frac", 0.995)),
        strict_swinvit_layout=bool(ck_args.get("strict_swinvit_layout", True)),
        debug_swinvit_layout=False,
        force_presence_from_raw_masks=bool(ck_args.get("force_presence_from_raw_masks", False)),
        raw_mask_threshold=float(ck_args.get("raw_mask_threshold", 0.5)),
        fallback_peri_to_intra=bool(ck_args.get("fallback_peri_to_intra", True)),
    )


def _build_model_from_checkpoint(ck: dict[str, Any], state: dict[str, torch.Tensor]) -> ContourAwareHabitatSurvivalModel:
    ck_args = checkpoint_args_to_dict(ck)
    image_token_dim = _linear_in_dim(state, "habitat_model.image_proj.weight")
    model_dim = int(state["habitat_model.image_proj.weight"].shape[0])
    radiomics_token_dim = _linear_in_dim(state, "habitat_model.radiomics_proj.pt_intra.weight")
    if radiomics_token_dim <= 0:
        radiomics_token_dim = _linear_in_dim(state, "habitat_model.radiomics_proj.ln_intra.weight")
    clinical_token_dim = _linear_in_dim(state, "habitat_model.clinical_proj.biology.weight")
    if clinical_token_dim <= 0:
        clinical_token_dim = _linear_in_dim(state, "habitat_model.clinical_proj.host.weight")
    node_token_dim = _linear_in_dim(state, "habitat_model.node_proj.weight")
    topology_dim = _linear_in_dim(state, "habitat_model.topology_proj.weight")
    num_time_bins = int(ck.get("num_time_bins", state["habitat_model.survival_heads.shared_logits.weight"].shape[0]))
    layer_ids = {
        int(k.split(".")[3])
        for k in state
        if k.startswith("habitat_model.sequence_encoder.layers.") and k.split(".")[3].isdigit()
    }
    backbone = ContourAwareROITokenBackbone(**_backbone_cfg(ck_args, image_token_dim))
    model = ContourAwareHabitatSurvivalModel(
        backbone=backbone,
        num_time_bins=num_time_bins,
        time_bin_width_days=float(ck_args.get("time_bin_width_days", 180.0)),
        radiomics_token_dim=int(radiomics_token_dim),
        clinical_token_dim=int(clinical_token_dim),
        node_token_dim=int(node_token_dim),
        topology_dim=int(topology_dim),
        model_dim=int(model_dim),
        num_heads=int(ck_args.get("v2_num_heads", 8)),
        transformer_layers=max(layer_ids) + 1 if layer_ids else int(ck_args.get("v2_transformer_layers", 2)),
        dropout=0.0,
        radiomics_habitats=_radiomics_habitats_from_state(state),
        clinical_groups=_clinical_groups_from_state(state),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"V2 checkpoint did not load cleanly: missing={missing[:10]} unexpected={unexpected[:10]}")
    return model


@torch.no_grad()
def _apply_checkpoint_weight_variant(model: torch.nn.Module, ck: dict[str, Any], weights: str) -> str:
    variant = str(weights).strip().lower()
    if variant in ("", "best", "last", "model_state"):
        return "model_state"
    if variant not in ("ema", "swa"):
        raise ValueError(f"Unsupported weights variant: {weights}")
    state = ck.get(variant)
    if not isinstance(state, dict) or not isinstance(state.get("shadow"), dict) or not state["shadow"]:
        raise RuntimeError(f"Requested --weights {variant}, but checkpoint does not contain {variant} shadow weights.")
    if variant == "swa" and int(state.get("n_averaged", 0)) <= 0:
        raise RuntimeError("Requested --weights swa, but checkpoint has no averaged SWA weights.")
    params = dict(model.named_parameters())
    applied = 0
    for name, tensor in state["shadow"].items():
        clean = str(name)
        if clean.startswith("module."):
            clean = clean[len("module.") :]
        if clean not in params:
            continue
        p = params[clean]
        if tuple(p.shape) != tuple(tensor.shape):
            continue
        p.copy_(tensor.to(device=p.device, dtype=p.dtype))
        applied += 1
    if applied <= 0:
        raise RuntimeError(f"Requested --weights {variant}, but no shadow tensors matched model parameters.")
    return f"{variant}_shadow_applied_{applied}"


def _oof_lookup(df: Optional[pd.DataFrame], pid: str, *, endpoint: str, horizon_days: float) -> tuple[Optional[float], str]:
    if df is None:
        return None, "unchecked"
    rows = df[df["patient_id"].astype(str) == str(pid)]
    if rows.empty:
        raise RuntimeError(f"OOF CSV has no row for patient_id={pid}")
    if "risk_endpoint" in rows.columns:
        rows = rows[rows["risk_endpoint"].astype(str).str.upper() == str(endpoint).upper()]
    if "risk_horizon_days" in rows.columns:
        h = rows["risk_horizon_days"].astype(float)
        rows = rows[(h - float(horizon_days)).abs() <= 1e-3]
    if rows.empty:
        raise RuntimeError(f"OOF CSV has no endpoint/horizon-matched row for patient_id={pid}")
    if len(rows) > 1 and len(rows["risk_score"].astype(float).round(12).unique()) > 1:
        raise RuntimeError(f"OOF lookup ambiguous for patient_id={pid}: matched rows={len(rows)}")
    notes = []
    for col in ("fold", "checkpoint", "weights", "model_version", "commit_sha"):
        if col not in rows.columns:
            notes.append(f"oof_missing_{col}")
    return float(rows.iloc[0]["risk_score"]), ";".join(notes)


def _config_tag(args: argparse.Namespace, *, endpoint: str, horizon: float) -> str:
    raw = "|".join([
        str(endpoint).upper(),
        f"{float(horizon):.6f}",
        str(args.target),
        str(args.checkpoint),
        str(args.weights),
        str(args.support_source),
        str(args.display_contours),
        str(int(bool(args.native_use_masks))),
        f"{float(args.teacher_force_alpha):.6f}",
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return pid_slug(f"{endpoint}_h{int(round(float(horizon)))}_{args.target}_{args.weights}_{digest}")


def _find_ct_path(meta_by_pid: dict[str, dict[str, Any]], pid: str, data_root: Path, ct_col: str) -> Optional[Path]:
    row = meta_by_pid.get(str(pid))
    if row is None:
        return None
    resolved = resolve_preprocessed_case_path(str(row.get(ct_col, "")), data_root=str(data_root), patient_id=str(pid))
    path = Path(resolved)
    return path if path.is_file() else None


def _to_device(batch: dict[str, Any], device: torch.device, key: str) -> Optional[torch.Tensor]:
    value = batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    return None


def run_fold(args: argparse.Namespace, fold: int, manifest_path: Path) -> dict[str, Any]:
    ck_path = _checkpoint_for_fold(args.run_dir, fold, args.checkpoint)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert_v207_v2_checkpoint(ck, checkpoint_path=_relative_path(ck_path), require_commit=bool(args.require_commit_sha))
    ck_args = checkpoint_args_to_dict(ck)
    state = normalized_state_dict(ck)
    model = _build_model_from_checkpoint(ck, state)
    device = torch.device(args.device)
    model.to(device).eval()
    loaded_weight_variant = _apply_checkpoint_weight_variant(model, ck, str(args.weights))

    endpoint = str(args.endpoint).upper()
    horizon = float(args.risk_horizon_days)
    time_col, event_col = ENDPOINT_MAP[endpoint]
    id_col = str(ck_args.get("id_col", "patient_id"))
    ct_col = str(ck_args.get("ct_col", "ct_out_path"))
    mask_pt_col = str(ck_args.get("mask_pt_col", "mask_primary_out_path"))
    mask_ln_col = str(ck_args.get("mask_ln_col", "mask_nodal_out_path"))

    meta = pd.read_csv(args.meta_csv, dtype={id_col: str})
    meta_by_pid = {str(getattr(r, id_col)): r._asdict() for r in meta.itertuples(index=False)}
    split = _load_splits(args.splits_dir, fold)
    tr_df = _select_ids(meta, split["train"], id_col)
    te_df = _select_ids(meta, split["test"], id_col)
    if int(args.max_model_patients) > 0:
        te_df = te_df.head(int(args.max_model_patients)).reset_index(drop=True)

    clinical_groups = _clinical_groups_from_state(state)
    clinical_token_dim = _linear_in_dim(state, "habitat_model.clinical_proj.biology.weight") or _linear_in_dim(
        state,
        "habitat_model.clinical_proj.host.weight",
    )
    radiomics_token_dim = _linear_in_dim(state, "habitat_model.radiomics_proj.pt_intra.weight") or _linear_in_dim(
        state,
        "habitat_model.radiomics_proj.ln_intra.weight",
    )
    clinical_encoder = SemanticClinicalTokenEncoder.fit(tr_df, clinical_groups)
    radiomics_encoder = None
    if model.habitat_model.radiomics_proj is not None and args.radiomics_csv:
        all_ids = pd.concat([tr_df[id_col], te_df[id_col]]).astype(str).unique().tolist()
        radiomics_encoder = HabitatRadiomicsTokenEncoder.fit_from_wide_csv(
            radiomics_csv=args.radiomics_csv,
            train_ids=tr_df[id_col].astype(str).tolist(),
            all_ids=all_ids,
            total_pcs_per_group=int(radiomics_token_dim or args.radiomics_pcs_per_group),
            require_presence_columns=bool(args.require_radiomics_presence_columns),
            random_state=int(ck_args.get("seed", args.seed)),
        )

    dataset = PreprocessedHabitatOOFDataset(
        te_df,
        id_col=id_col,
        time_col=time_col,
        event_col=event_col,
        multi_time_cols=tuple(ENDPOINT_MAP[ep][0] for ep in SURVIVAL_ENDPOINTS),
        multi_event_cols=tuple(ENDPOINT_MAP[ep][1] for ep in SURVIVAL_ENDPOINTS),
        ct_col=ct_col,
        mask_pt_col=mask_pt_col,
        mask_ln_col=mask_ln_col,
        clinical_token_encoder=clinical_encoder,
        radiomics_token_encoder=radiomics_encoder,
        use_radiomics=radiomics_encoder is not None,
        node_topology_dir=str(args.node_topology_dir or ""),
        max_nodes=int(args.max_nodes),
        node_token_dim=_linear_in_dim(state, "habitat_model.node_proj.weight"),
        clinical_token_dim=int(clinical_token_dim),
        radiomics_token_dim=int(radiomics_token_dim),
        strict_files=True,
        expected_dhw=tuple(int(x) for x in ck_args.get("img_size", (128, 256, 256))),
        data_root=str(args.meta_csv.parent),
        mode="eval",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.workers))
    oof_df = pd.read_csv(args.oof_predictions_csv, dtype={"patient_id": str}) if args.oof_predictions_csv else None

    config_tag = _config_tag(args, endpoint=endpoint, horizon=horizon)
    rows: list[dict[str, Any]] = []
    wrote = 0
    for batch in loader:
        pid = str(batch["pid"][0])
        x = _to_device(batch, device, "x")
        mask_pt_native = _to_device(batch, device, "mask_pt")
        mask_ln_native = _to_device(batch, device, "mask_ln")
        clinical_tokens = _to_device(batch, device, "clinical_tokens")
        clinical_presence = _to_device(batch, device, "clinical_presence")
        radiomics_tokens = _to_device(batch, device, "radiomics_tokens")
        radiomics_presence = _to_device(batch, device, "radiomics_presence")
        if radiomics_tokens is not None and radiomics_tokens.numel() == 0:
            radiomics_tokens = None
            radiomics_presence = None
        node_tokens = _to_device(batch, device, "node_tokens")
        node_presence = _to_device(batch, device, "node_presence")
        if node_tokens is not None and node_tokens.numel() == 0:
            node_tokens = None
            node_presence = None
        topology_token = _to_device(batch, device, "topology_token")
        topology_presence = _to_device(batch, device, "topology_presence")
        if topology_token is not None and topology_token.numel() == 0:
            topology_token = None
            topology_presence = None

        use_native = bool(args.native_use_masks)
        mask_pt_model = mask_pt_native if use_native else None
        mask_ln_model = mask_ln_native if use_native else None

        model.zero_grad(set_to_none=True)
        image_tokens, image_presence, bb_aux = model.backbone(
            x,
            mask_pt=mask_pt_model,
            mask_ln=mask_ln_model,
            teacher_force_alpha=float(args.teacher_force_alpha if use_native else 0.0),
            return_aux=True,
            return_cam_features=True,
        )
        pred_supports_t = supports_from_backbone_aux(bb_aux)
        supports_t = pred_supports_t
        if str(args.support_source) == "native":
            body_t = supports_t["body"]
            pt_native_t = mask_pt_native.detach().clamp(0, 1)
            ln_native_t = mask_ln_native.detach().clamp(0, 1)
            pt_shell_t = model.backbone._soft_shell(pt_native_t, int(getattr(model.backbone, "pt_shell_radius", 3))).detach()
            ln_shell_t = model.backbone._soft_shell(ln_native_t, int(getattr(model.backbone, "ln_shell_radius", 3))).detach()
            if bool(getattr(model.backbone, "shell_body_from_ct", False)):
                pt_shell_t = pt_shell_t * body_t
                ln_shell_t = ln_shell_t * body_t
            habitat_union_t = (pt_native_t + pt_shell_t + ln_native_t + ln_shell_t).clamp(0, 1)
            supports_t = {
                "full_volume": torch.ones_like(body_t),
                "body": body_t,
                "pt_intra": pt_native_t,
                "pt_peri": pt_shell_t.clamp(0, 1),
                "ln_intra": ln_native_t,
                "ln_peri": ln_shell_t.clamp(0, 1),
                "pt_ln_union": (pt_native_t + ln_native_t).clamp(0, 1),
                "habitat_union": habitat_union_t,
                "off_habitat_body": (body_t - habitat_union_t).clamp(0, 1),
            }
        pred_supports_np = {name: value.detach().cpu().numpy()[0, 0].astype(np.float32) for name, value in pred_supports_t.items()}
        supports_np = {name: value.detach().cpu().numpy()[0, 0].astype(np.float32) for name, value in supports_t.items()}
        ct_np = x.detach().cpu().numpy()[0, 0].astype(np.float32)
        native_pt_np = mask_pt_native.detach().cpu().numpy()[0, 0].astype(np.float32)
        native_ln_np = mask_ln_native.detach().cpu().numpy()[0, 0].astype(np.float32)

        with torch.no_grad():
            logits_full, aux_full = model.forward_from_image_tokens(
                image_tokens=image_tokens.detach(),
                image_presence=image_presence.detach(),
                clinical_tokens=None if clinical_tokens is None else clinical_tokens.detach(),
                clinical_presence=None if clinical_presence is None else clinical_presence.detach(),
                radiomics_tokens=None if radiomics_tokens is None else radiomics_tokens.detach(),
                radiomics_presence=None if radiomics_presence is None else radiomics_presence.detach(),
                node_tokens=None if node_tokens is None else node_tokens.detach(),
                node_presence=None if node_presence is None else node_presence.detach(),
                topology_token=None if topology_token is None else topology_token.detach(),
                topology_presence=None if topology_presence is None else topology_presence.detach(),
                return_aux=True,
                return_attention=bool(args.save_attention),
            )
            risk_full = float(risk_vector(model, logits_full, endpoint=endpoint, horizon_days=horizon).detach().cpu()[0])
        saved, oof_notes = _oof_lookup(oof_df, pid, endpoint=endpoint, horizon_days=horizon)
        risk_absdiff = "" if saved is None else abs(risk_full - float(saved))
        if saved is not None and float(risk_absdiff) > float(args.risk_match_tol):
            raise RuntimeError(
                f"OOF risk mismatch for {pid}: gradcam={risk_full:.8f} saved={float(saved):.8f} "
                f"absdiff={float(risk_absdiff):.8g} tol={float(args.risk_match_tol):.8g}"
            )

        attention_rel = Path(f"fold_{fold:02d}") / pid_slug(pid) / config_tag / "attention_context.json"
        if args.save_attention:
            write_json(args.out_dir / attention_rel, attention_aux_to_jsonable(aux_full))
        ablation_rel = Path(f"fold_{fold:02d}") / pid_slug(pid) / config_tag / "token_ablation.json"
        ablations = {}
        if args.save_ablations:
            ablations = compute_v2_ablation_package(
                model,
                image_tokens=image_tokens.detach(),
                image_presence=image_presence.detach(),
                clinical_tokens=None if clinical_tokens is None else clinical_tokens.detach(),
                clinical_presence=None if clinical_presence is None else clinical_presence.detach(),
                radiomics_tokens=None if radiomics_tokens is None else radiomics_tokens.detach(),
                radiomics_presence=None if radiomics_presence is None else radiomics_presence.detach(),
                node_tokens=None if node_tokens is None else node_tokens.detach(),
                node_presence=None if node_presence is None else node_presence.detach(),
                topology_token=None if topology_token is None else topology_token.detach(),
                topology_presence=None if topology_presence is None else topology_presence.detach(),
                endpoint=endpoint,
                horizon_days=horizon,
            )
            write_json(args.out_dir / ablation_rel, ablations)

        map_specs = [("full_model", None), *[(name, idx) for idx, name in enumerate(image_habitat_names())]]
        for map_i, (habitat_name, habitat_idx) in enumerate(map_specs):
            model.zero_grad(set_to_none=True)
            for feat in bb_aux["cam_features"]:
                if feat.grad is not None:
                    feat.grad.detach_()
                    feat.grad.zero_()
            tokens_for_grad = image_tokens if habitat_idx is None else isolate_image_habitat_gradient(image_tokens, int(habitat_idx))
            logits, _aux = model.forward_from_image_tokens(
                image_tokens=tokens_for_grad,
                image_presence=image_presence,
                clinical_tokens=clinical_tokens,
                clinical_presence=clinical_presence,
                radiomics_tokens=radiomics_tokens,
                radiomics_presence=radiomics_presence,
                node_tokens=node_tokens,
                node_presence=node_presence,
                topology_token=topology_token,
                topology_presence=topology_presence,
                return_aux=True,
                return_attention=False,
            )
            target = select_survival_target(
                model,
                logits,
                endpoint=endpoint,
                horizon_days=horizon,
                target_type=str(args.target),
            )
            target.backward(retain_graph=(map_i < len(map_specs) - 1))
            signed_cam, scale_cams, scale_shapes = cam_from_features(bb_aux["cam_features"], tuple(int(x) for x in x.shape[2:]))
            pos_raw, neg_raw = split_signed_cam(signed_cam)
            focus_name = "habitat_union" if habitat_name == "full_model" else ("body" if habitat_name == "global" else habitat_name)
            support_np = supports_np[focus_name]
            pos_focus = normalize_component(pos_raw, support=support_np)
            neg_focus = normalize_component(neg_raw, support=support_np)
            pos_raw_norm = normalize_component(pos_raw, support=None)
            neg_raw_norm = normalize_component(neg_raw, support=None)
            mass = cam_mass_summary(pos_raw, supports_np, denominator="body")

            case_dir_rel = Path(f"fold_{fold:02d}") / pid_slug(pid) / config_tag / pid_slug(habitat_name)
            signed_rel = case_dir_rel / "signed_raw_gradcam.nii.gz"
            pos_rel = case_dir_rel / f"{pid_slug(focus_name)}_focus_risk_increasing_gradcam.nii.gz"
            neg_rel = case_dir_rel / f"{pid_slug(focus_name)}_focus_risk_decreasing_gradcam.nii.gz"
            pos_raw_rel = case_dir_rel / "raw_risk_increasing_gradcam.nii.gz"
            neg_raw_rel = case_dir_rel / "raw_risk_decreasing_gradcam.nii.gz"
            overlay_rel = case_dir_rel / f"{pid_slug(focus_name)}_focus_risk_increasing_overlay.png"
            ct_path = _find_ct_path(meta_by_pid, pid, args.meta_csv.parent, ct_col)
            geom = save_nifti(signed_cam, args.out_dir / signed_rel, ct_path, clip=False)
            save_nifti(pos_focus, args.out_dir / pos_rel, ct_path, clip=True)
            save_nifti(neg_focus, args.out_dir / neg_rel, ct_path, clip=True)
            save_nifti(pos_raw_norm, args.out_dir / pos_raw_rel, ct_path, clip=True)
            save_nifti(neg_raw_norm, args.out_dir / neg_raw_rel, ct_path, clip=True)
            if args.save_per_scale_cams:
                for scale_i, scale_cam in enumerate(scale_cams):
                    save_nifti(scale_cam, args.out_dir / case_dir_rel / f"scale_{scale_i:02d}_signed_raw_gradcam.nii.gz", ct_path, clip=False)
            overlay_pt = native_pt_np if str(args.display_contours) == "native" else pred_supports_np["pt_intra"]
            overlay_ln = native_ln_np if str(args.display_contours) == "native" else pred_supports_np["ln_intra"]
            make_overlay_png(
                ct_np,
                pos_focus,
                args.out_dir / overlay_rel,
                title=f"{pid} {endpoint} {horizon:.0f}d {habitat_name} {focus_name} risk={risk_full:.4f}",
                mask_pt=overlay_pt,
                mask_ln=overlay_ln,
                body_mask=supports_np["body"],
                max_slices=int(args.max_slices),
            )
            row = {
                "software_version": SOFTWARE_VERSION,
                "commit_sha": TARGET_COMMIT_SHA,
                "model_class": MODEL_CLASS,
                "checkpoint_path": _relative_path(ck_path),
                "weights_type": str(args.weights),
                "loaded_weight_variant": loaded_weight_variant,
                "fold": int(fold),
                "patient_id": pid,
                "endpoint": endpoint,
                "horizon_days": float(horizon),
                "target_type": str(args.target),
                "risk_gradcam_forward": float(risk_full),
                "risk_saved_oof": "" if saved is None else float(saved),
                "risk_absdiff": risk_absdiff,
                "oof_validation_notes": str(oof_notes),
                "model_mask_source": "native" if use_native else "predicted",
                "support_source": str(args.support_source),
                "display_contour_source": str(args.display_contours),
                "teacher_force_alpha": float(args.teacher_force_alpha if use_native else 0.0),
                "image_habitat": habitat_name,
                "gradient_path_isolated": bool(habitat_idx is not None),
                "raw_signed_cam_path": str(signed_rel),
                "risk_increasing_cam_path": str(pos_rel),
                "risk_decreasing_cam_path": str(neg_rel),
                "risk_increasing_raw_cam_path": str(pos_raw_rel),
                "risk_decreasing_raw_cam_path": str(neg_raw_rel),
                "overlay_png_path": str(overlay_rel),
                "attention_json_path": str(attention_rel) if args.save_attention else "",
                "ablation_json_path": str(ablation_rel) if args.save_ablations else "",
                "activation_shapes": ";".join(scale_shapes),
                "support_is_empty": bool(float((support_np > 0.05).sum()) == 0.0),
                "geometry_copied": bool(geom["geometry_copied"]),
                "cam_shape_zyx": geom["cam_shape_zyx"],
                "cam_size_xyz": geom["cam_size_xyz"],
                "reference_size_xyz": geom["reference_size_xyz"],
                "delta_risk_without_image_habitat": ablations.get(f"delta_risk_without_image_{habitat_name}", ""),
                "delta_risk_without_radiomics_habitat": ablations.get(f"delta_risk_without_radiomics_{habitat_name}", ""),
                "delta_risk_without_clinical_group_biology": ablations.get("delta_risk_without_clinical_biology", ""),
                "delta_risk_without_clinical_group_burden": ablations.get("delta_risk_without_clinical_burden", ""),
                "delta_risk_without_clinical_group_host": ablations.get("delta_risk_without_clinical_host", ""),
                "delta_risk_without_topology": ablations.get("delta_risk_without_topology", ""),
                "delta_risk_without_node_tokens": ablations.get("delta_risk_without_node_tokens", ""),
            }
            row.update(mass)
            rows.append(row)
        append_manifest(manifest_path, rows)
        rows.clear()
        wrote += 1

    return {
        "fold": int(fold),
        "checkpoint": _relative_path(ck_path),
        "endpoint": endpoint,
        "horizon_days": horizon,
        "wrote": int(wrote),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    p.add_argument("--weights", type=str, default="ema", choices=["best", "ema", "last", "swa"])
    p.add_argument("--meta_csv", type=Path, required=True)
    p.add_argument("--splits_dir", type=Path, required=True)
    p.add_argument("--radiomics_csv", type=Path, default=None)
    p.add_argument("--node_topology_dir", type=Path, default=None)
    p.add_argument("--oof_predictions_csv", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--endpoint", type=str, required=True, choices=[*SURVIVAL_ENDPOINTS, "ALL"])
    p.add_argument("--risk_horizon_days", type=float, required=True)
    p.add_argument("--target", type=str, default="cumulative_risk", choices=["cumulative_risk", "hazard_logit"])
    p.add_argument("--support_source", type=str, default="predicted", choices=["predicted", "native"])
    p.add_argument("--display_contours", type=str, default="native", choices=["native", "predicted"])
    p.add_argument("--native_use_masks", action="store_true")
    p.add_argument("--teacher_force_alpha", type=float, default=1.0)
    p.add_argument("--save_attention", action="store_true")
    p.add_argument("--save_ablations", action="store_true")
    p.add_argument("--save_per_scale_cams", action="store_true")
    p.add_argument("--require_commit_sha", action="store_true")
    p.add_argument("--require_radiomics_presence_columns", action="store_true")
    p.add_argument("--radiomics_pcs_per_group", type=int, default=16)
    p.add_argument("--max_nodes", type=int, default=16)
    p.add_argument("--risk_match_tol", type=float, default=1e-4)
    p.add_argument("--max_model_patients", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max_slices", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    if not (0.0 <= float(args.teacher_force_alpha) <= 1.0):
        raise ValueError("--teacher_force_alpha must be in [0,1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()
    manifest = args.out_dir / "gradcam_v207_manifest.csv"
    statuses = []
    endpoints = tuple(SURVIVAL_ENDPOINTS) if str(args.endpoint).upper() == "ALL" else (str(args.endpoint).upper(),)
    for endpoint in endpoints:
        endpoint_args = copy.copy(args)
        endpoint_args.endpoint = endpoint
        for fold in [int(x) for x in args.folds]:
            statuses.append(run_fold(endpoint_args, fold, manifest))
    status = {
        "kind": "v207_oof_gradcam",
        "software_version": SOFTWARE_VERSION,
        "commit_sha": TARGET_COMMIT_SHA,
        "model_class": MODEL_CLASS,
        "endpoints": list(endpoints),
        "risk_horizon_days": float(args.risk_horizon_days),
        "target": str(args.target),
        "manifest": _relative_path(manifest),
        "fold_status": statuses,
    }
    write_json(args.out_dir / "gradcam_v207_status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
