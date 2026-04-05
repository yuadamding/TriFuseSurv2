# Two-Stage Pipeline

TriFuseSurv follows a two-stage training process.

## Preparation

Preprocess the cohort and then generate patient-level split files.

- Export CT and RTSTRUCT-derived masks: `trifusesurv-preprocess-export-swinunetr`
- Generate split files: `trifusesurv-make-cv-splits`

## Stage 1

Train a SwinViT / SwinUNETR encoder on tumor segmentation.

- Train: `trifusesurv-stage1-train-seg`
- Evaluate: `trifusesurv-stage1-eval-seg`
- Visualize: `trifusesurv-stage1-gradcam-seg`

## Stage 2

Load the pretrained image encoder and integrate three modalities for survival prediction.

- Train: `trifusesurv-stage2-train-survival`
- Train LoRA: `trifusesurv-stage2-train-survival-lora`
- Token SHAP: `trifusesurv-stage2-shap-tokens`
- Grouped SHAP export: `trifusesurv-stage2-shap-grouped`
- Grouped SHAP export v2: `trifusesurv-stage2-shap-grouped-v2`
