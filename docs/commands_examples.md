# Command Examples

```bash
cd .

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src python3 -m trifusesurv.segmentation.train \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --splits_dir runs/opscc_splits_os_seed1 \
  --cv_folds 4 --debug_fold 0 --strict_splits \
  --out_dir runs/seg_pretrain_swinunetr_from_preprocessed \
  --img_size 256 256 128 \
  --epochs 50 --workers 8 --amp \
  --device cuda:0
```

```bash
python - <<'PY'
import os, glob
import numpy as np
import pandas as pd

# ---- EDIT THESE ----
EXP_DIR = "runs/moe_discrete_swinunetr/cv4_ptln_tokens_globalPretrain_withClin"
META_CSV = "OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv"
ID_COL = "patient_id"
TIME_COL = "OS.TIME"
EVENT_COL = "OS.EVENT"
RISK_HORIZON_DAYS = 3*365.0  # keep consistent with your training run
PREFER = "swa"              # "ema" or "last"
# --------------------

# Import helper metrics from the packaged project
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from trifusesurv.utils import survival as H

fold_dirs = sorted(glob.glob(os.path.join(EXP_DIR, "fold_*")))
if not fold_dirs:
    raise RuntimeError(f"No fold_* dirs found under {EXP_DIR}")

dfs = []
used_files = []
for fd in fold_dirs:
    f_ema = os.path.join(fd, "test_risks_ema.csv")
    f_last = os.path.join(fd, "test_risks_last.csv")
    f_pref = f_ema if PREFER.lower() == "ema" else f_last
    f = f_pref if os.path.isfile(f_pref) else (f_last if os.path.isfile(f_last) else (f_ema if os.path.isfile(f_ema) else None))
    if f is None:
        raise RuntimeError(f"Missing risk file in {fd}: expected test_risks_ema.csv or test_risks_last.csv")
    df = pd.read_csv(f, dtype={ID_COL: str})
    if ID_COL not in df.columns:
        raise RuntimeError(f"{f} missing id col {ID_COL}. Columns={list(df.columns)}")
    if "risk_score" not in df.columns:
        raise RuntimeError(f"{f} missing 'risk_score'. Columns={list(df.columns)}")
    df = df[[ID_COL, "risk_score"]].copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df["fold_dir"] = os.path.basename(fd)
    dfs.append(df)
    used_files.append(f)

pool = pd.concat(dfs, axis=0, ignore_index=True)
pool["risk_score"] = pd.to_numeric(pool["risk_score"], errors="coerce")
pool = pool.dropna(subset=["risk_score"]).copy()

# Duplicate check: each patient should appear once across folds (OOF)
dup = pool[ID_COL].duplicated(keep=False)
if dup.any():
    bad = pool.loc[dup, ID_COL].value_counts().head(20)
    raise RuntimeError(f"Duplicate patient_id across folds (first 20):\n{bad}")

# Write pooled cohort OOF risks
out_csv = os.path.join(EXP_DIR, f"cv_test_risks_{PREFER}_pooled.csv")
pool[[ID_COL, "risk_score"]].to_csv(out_csv, index=False)
print(f"[OK] wrote pooled OOF risks: {out_csv}")
print(f"[OK] pooled N={len(pool)} from files:")
for f in used_files:
    print("   ", f)

# Compute pooled metrics on the pooled IDs
meta = pd.read_csv(META_CSV, dtype={ID_COL: str})
meta[ID_COL] = meta[ID_COL].astype(str)
meta[TIME_COL] = pd.to_numeric(meta[TIME_COL], errors="coerce")
meta[EVENT_COL] = pd.to_numeric(meta[EVENT_COL], errors="coerce")
meta = meta.dropna(subset=[TIME_COL, EVENT_COL]).copy()
meta[EVENT_COL] = meta[EVENT_COL].astype(int)

m = meta.set_index(ID_COL)
ids = [pid for pid in pool[ID_COL].tolist() if pid in m.index]
missing = len(pool) - len(ids)

times = m.loc[ids, TIME_COL].to_numpy(float)
events = m.loc[ids, EVENT_COL].to_numpy(float)
risks = pool.set_index(ID_COL).loc[ids, "risk_score"].to_numpy(float)

c_index = H.concordance_index(times, events, risks)
auc_h = H.ipcw_auc_at_horizon_from_risk(times, events, risks, float(RISK_HORIZON_DAYS))

print(f"[POOLED] C-index={c_index:.4f} | IPCW_AUC@{int(round(RISK_HORIZON_DAYS))}d={auc_h:.4f} | N={len(ids)} | missing_in_meta={missing}")
PY
```
