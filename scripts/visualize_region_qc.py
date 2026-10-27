#!/usr/bin/env python3
"""Render one provenance-validated region manifest as a standalone QC PNG."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trifusesurv2.preprocessing.region_io import load_region_arrays, load_region_manifest
from trifusesurv2.schema import RADIOLOGY_HABITATS, REGION_STATUS_CODES


COLORS = ("#d74d55", "#ffbb52", "#227ab8", "#77cfca")
LABELS = ("PT interior", "PT 0–10 mm shell", "LN interior", "LN 0–10 mm shell")
OVERLAP_COLOR = "#b47bea"


def render(manifest_path: Path, output_path: Path, *, synthetic: bool = False) -> None:
    manifest = load_region_manifest(manifest_path)
    ct_path = manifest_path.resolve().parent / manifest["image"]["path"]
    masks, support_valid, states, _ = load_region_arrays(manifest, ct_path=ct_path)
    image = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(image)
    masks = masks.astype(bool)
    spacing = image.GetSpacing()
    overlap = masks[1] & masks[3]
    state_names = {value: key for key, value in REGION_STATUS_CODES.items()}

    def slice_index(support):
        area = support.sum(axis=(1, 2))
        return int(np.argmax(area)) if area.any() else int(ct.shape[0] // 2)

    panels = (
        ("Primary tumor compartments", slice_index(masks[0] | masks[1]), (0, 1)),
        ("Nodal compartments", slice_index(masks[2] | masks[3]), (2, 3)),
        ("Shared shell overlap", slice_index(overlap if overlap.any() else masks.any(axis=0)), tuple(range(4))),
    )
    fig = plt.figure(figsize=(15, 10.5), facecolor="white")
    title = "Shared anatomical support — region QC"
    fig.suptitle(title, fontsize=22, fontweight="bold", x=0.045, y=0.975, ha="left", color="#17263b")
    subtitle = "SYNTHETIC ENGINEERING EXAMPLE · NO CLINICAL QUALIFICATION" if synthetic else "Validated source manifest and exact model-grid support"
    fig.text(0.045, 0.93, subtitle, fontsize=11, color="#854515" if synthetic else "#40526b", fontweight="bold")
    fig.text(0.045, 0.901, f"Case {manifest['patient_id']}  |  Scan {manifest['scan_id']}  |  "
             f"Model spacing x/y/z: {spacing[0]:.3g} / {spacing[1]:.3g} / {spacing[2]:.3g} mm", fontsize=10, color="#40526b")
    grid = fig.add_gridspec(1, 3, left=0.045, right=0.98, bottom=0.48, top=0.85, wspace=0.2)
    extent = (-spacing[0] / 2, (ct.shape[2] - .5) * spacing[0],
              -spacing[1] / 2, (ct.shape[1] - .5) * spacing[1])
    for index, (title, z, compartments) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(ct[z], cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1, interpolation="nearest")
        for compartment in compartments:
            rgba = np.zeros((*ct[z].shape, 4))
            rgba[masks[compartment, z]] = to_rgba(COLORS[compartment], .62 if compartment % 2 else .85)
            ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        if index == 2:
            rgba = np.zeros((*ct[z].shape, 4))
            rgba[overlap[z]] = to_rgba(OVERLAP_COLOR, .9)
            ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(f"{title}\nGrid z = {z}", fontsize=12, color="#17263b", pad=10)
        ax.set_xlabel("Grid x (mm)", fontsize=9)
        ax.set_ylabel("Grid y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
    handles = [Patch(color=color, label=label) for color, label in zip(COLORS, LABELS)]
    handles.append(Patch(color=OVERLAP_COLOR, label="PT/LN shell overlap"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.51, .439), ncol=5, frameon=False, fontsize=10)

    table_ax = fig.add_axes((.045, .22, .935, .17))
    table_ax.axis("off")
    rows = []
    for index, habitat in enumerate(RADIOLOGY_HABITATS):
        record = manifest["regions"][habitat]
        crop = record["crop_completeness"]
        reference_volume = float(record["physical_volume_mm3"])
        image_volume = int(masks[index].sum()) * float(np.prod(spacing))
        ratio = f"{image_volume / reference_volume:.3f}" if reference_volume else "n/a"
        boundary = record["reference_boundary_truncated"]
        rows.append([habitat, record["status"], state_names[int(states[index])],
                     "yes" if record["feature_validity"]["image"] else "no",
                     f"{reference_volume:,.1f}", f"{image_volume:,.1f}",
                     f"{100 * crop['fraction']:.1f}%", ratio,
                     "n/a" if not habitat.endswith("_peri") else
                     ("unknown" if boundary is None else ("FLAG" if boundary else "clear"))])
    headers = ["Region", "Reference\nstate", "Model\nstate", "Image\nfeatures valid", "Reference\nvolume (mm³)",
               "Model\nvolume (mm³)", "Reference\nretained", "Model/ref\nvolume", "Shell/FOV\ntruncation"]
    table = table_ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center",
                          colWidths=[.105, .105, .105, .115, .13, .13, .1, .1, .11])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.95)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dce3eb")
        if row == 0:
            cell.set_facecolor("#eaf0f6")
            cell.set_text_props(weight="bold", color="#17263b")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 else "white")
            if column == 8 and rows[row - 1][column] == "FLAG":
                cell.set_text_props(color="#a04716", weight="bold")
    overlap_volume = int(overlap.sum()) * float(np.prod(spacing))
    notes = [
        f"CT bytes, mask checksums, full geometry and physical-feature provenance validated.  "
        f"Model shell overlap: {overlap_volume:,.1f} mm³ (allowed and measured).",
        "Reference retention describes crop coverage only. A source-FOV flag means the requested shell reaches the scan boundary; missing coverage is not restored.",
        "Shells are generated on the reference grid by voxel-center Euclidean distance, exclude both tumor interiors, then use the declared valid-tissue mask.",
        "Model supports use nearest-neighbor resampling; volume is measured rather than assumed conserved. Grid axes are not patient anatomical directions.",
    ]
    for index, note in enumerate(notes):
        fig.text(.045, .168 - .027 * index, note, fontsize=9, color="#40526b", va="top")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white", metadata={"Description": subtitle})
    plt.close(fig)
    print(f"Wrote {output_path} | verified regions: {int(support_valid.sum())}/4 | model shell overlap: {overlap_volume:.1f} mm3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="One persisted region manifest; reads only its bound CT and masks.")
    parser.add_argument("--output", required=True, type=Path, help="Destination standalone PNG.")
    parser.add_argument("--synthetic", action="store_true", help="Label the output as a synthetic engineering example.")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must have a .png suffix")
    render(args.manifest, args.output, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
