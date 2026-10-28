# Common-grid development proposal and qualification boundary

The common-grid implementation fixes the crop-dependent sampling mechanism.
The numeric configuration below is a development candidate for the available
local inputs, not a frozen clinical measurement policy. The execution contract
uses **one Kubernetes H100** with the approved transferred repository and inputs.

As recorded on **October 27, 2026, at 9:03 PM CDT**, executable source `614d7fb`
had completed seven synthetic sampling regressions inside an H100 worker,
actual PyRadiomics measurement checks, tiny F2/F3 CUDA forward/backward checks,
and six real representative larger-canvas comparisons. Development geometry
review retained five cases with shell/source-FOV flags and one disappearing tiny
nodal interior marked invalid. The subsequent cohort worker was Running/Ready
in input preparation; paired training had not started. These dated observations
are bound in the [evidence summary](development_qualification_summary.json).
The [implementation and validation guide](implementation_and_validation.md)
explains what each result establishes. Full-cohort qualification and clinical
performance remain unestablished.

## Available input evidence

The [local input audit](local_input_sampling_audit.json) records CSV declarations
and two preprocessed CT header/range checks. It contains no outcome analysis or
radiomics qualification.

| Observation | Recorded evidence | Consequence |
|---|---|---|
| Inventory | 613 rows, 613 unique patient IDs and nonmissing declared acquisition IDs | Identity availability does not prove image/acquisition provenance |
| Declared original x/y spacing | 0.3125–0.607422 mm; median 0.488281 mm | These are upstream metadata declarations, not measurements of every current CT |
| Declared original z spacing | 0.784661–5.43103 mm; median 1 mm | 251 cases exceed 2 mm and 50 exceed 3 mm |
| Prior preparation | All rows declare target spacing 0.5 × 0.5 × 1 mm; 612 `union_bbox` crops and one `ct_foreground` crop | The working inputs already passed through resampling and cropping |
| Two inspected working CTs | x/y/z size 128 × 256 × 256, spacing 0.5 × 0.5 × 1 mm, intensities in [0,1] | Two checks do not qualify all 613 images or reconstruct original sampling |
| Original source paths | None of the 613 declared original patient directories or RTSTRUCT paths resolved locally | Original CT/contour provenance and tissue outside the prepared crop cannot be recovered from these inputs alone |
| Existing contour QC declarations | One primary alignment failure, one nodal alignment failure and 97 empty nodal declarations | Failed or empty inputs require annotation-availability review; these strings do not establish known anatomical absence |

There is no qualified HU provenance for this local working set. Existing
`cohort_preprocessed.csv`, `cohort_preprocessed_stage2.csv` and the historical
radiomics CSV are not ready verified inputs. The patient-key alias `Patient_ID`
can resolve `patient_id`; it must never supply a missing acquisition identity.
Do not manufacture contour, tissue or intensity provenance to pass a loader.

The available prepared image is the source FOV for new processing. A shell flag
relative to that FOV does not establish coverage in the unavailable original CT.
Padding can preserve a tensor canvas but cannot restore cropped-away tissue.

The development adapter proposes support from stored normalized CT values
greater than 0.02. This is an explicitly recorded candidate intensity threshold,
with no HU interpretation or tissue-validation claim. Legacy derived contours
remain unreviewed: failed QC masks are unavailable, and nodal QC `empty` means
unknown/missing annotation rather than asserted node-negative disease. This
adapter supports the scoped H100 development checks; it does not promote the
working set into a principal clinical cohort. If another development run
explicitly chooses `full_fov`, that includes all imaged voxels and is likewise
not a tissue-segmentation claim.

## Candidate settings

| Component | Development candidate |
|---|---|
| Physical lattice | [`common_grid_candidate.json`](../configs/common_grid_candidate.json): x/y/z spacing 1 × 1 × 3 mm, identity LPS direction, origin/phase (0,0,0) mm |
| Tensor canvas | D/H/W 96 × 160 × 96; x/y/z physical extent 96 × 160 × 288 mm |
| Extent handling | Integer-offset crop/pad; a larger required extent fails and needs a larger canvas/bucket at the same lattice |
| CT handling | Source intensity values mapped once with linear float32 interpolation; no new intensity normalization is inferred |
| Compartment handling | Authoritative source compartments mapped once with nearest neighbor; exact same persisted masks serve CT pooling and radiomics |
| Serialization | Compressed `.mha` preserves physical geometry; staged outputs must pass geometry readback before publication |
| Radiomics | [`radiomics_normalized_candidate.json`](../configs/radiomics_normalized_candidate.json): `Original` image type; first-order, shape, GLCM, GLRLM, GLSZM, GLDM and NGTDM feature classes; bin width 0.0125 in normalized intensity units |
| Extraction transforms | No added normalization, resampling, resegmentation or mask correction; three-dimensional extraction, distance setting [1], label 1 |

The 1 mm in-plane candidate reduces sampling density relative to the prepared
0.5 mm working images and the declared submillimeter original acquisitions.
The 3 mm z candidate acknowledges substantial variation in declared slice
spacing while reducing the development canvas. It does not restore through-plane
detail: 50 declared acquisitions are coarser than 3 mm. These considerations
motivate a candidate for representative-case testing, not an optimal resolution
or a claim of image-quality equivalence. Runtime memory and throughput still
need measurement in the actual environment.

The extraction candidate concerns normalized intensities. A bin width of 0.0125
has no HU interpretation. Actual software/version, supported features, errors
and numerical behavior must be recorded from the selected PyRadiomics
installation. The configuration is an input to qualification, not its result.

## Worker commands

After the approved image and project mount pass their environment checks, run
these commands inside the single-H100 worker from the bound repository root.
Both commands check the actual visible GPU before reading their inputs. Use new
output directories for each attempt and retain failed receipts.

```bash
python scripts/qualify_h100.py \
  --grid-config-json configs/common_grid_candidate.json \
  --radiomics-config-json configs/radiomics_normalized_candidate.json \
  --out-dir /approved/run/synthetic-qualification

python scripts/prepare_development_inputs.py \
  --source-csv /approved/run/local-inputs/OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv \
  --data-root /approved/run/local-inputs/OPSCC_preprocessed_128 \
  --grid-config-json configs/common_grid_candidate.json \
  --radiomics-config-json configs/radiomics_normalized_candidate.json \
  --output-shape 96 160 96 \
  --out-dir /approved/run/representative-development-inputs
```

The first command checks synthetic measurements with actual PyRadiomics and
performs F2/F3 CUDA forward/backward checks. The second selects outcome-free
metadata representatives, builds candidate tissue/support files, extracts real
features, compares them with a larger canvas, and writes the manifest audit,
QC overlays and a development receipt. Inspect the overlays before accepting
the measurements. `--ids-file` supplies an explicit representative inventory;
`--all-cases` is a separate complete-cohort preparation step. Neither preparation
command creates evaluation splits or starts cohort training.

For sequential F2 then F3 training, `qualify_h100.py --training-plan-json`
accepts a separately frozen plan binding the source, full resolved semantic
settings, epochs, seeds, folds, input hashes, warm-start selections and separate
output directories. Its grid and radiomics configuration must match the
qualification inputs. A passing synthetic receipt is not real-input promotion
or a completed paired cohort experiment.

## Implemented engineering acceptance

[`test_common_grid.py`](../tests/test_common_grid.py) holds the source image and
primary compartment fixed while adding a distant node outside the primary
shell. The primary compartment retains identical canonical lattice indices,
binary support and sampled float32 CT/HU values. Integer crop shifts, padding
and larger canvases preserve those measurements. Other cases cover rotated,
reflected and anisotropic sources, small-region disappearance, full-source FOV
enclosure and overflow rejection. These are synthetic test spacings, not a
selection of the cohort resolution.

The full lattice is derived only from the explicit policy and source geometry.
The crop envelope includes mapped support and projected source voxel centers,
including a small region that vanishes during resampling. Consequently, source
center retention can be 100% while image features are invalid due to resampling
loss. Both facts are recorded separately. Source and model identities, spacing,
direction, phase, integer offsets, padding, checksums, volumes and disappearance
descriptors are validated when persisted manifests are read.

The preprocessing integration also checks the actual CT/mask files and binds
the radiomics sampling policy. Principal F0–F4 comparisons require every case
to use the same validated common-grid policy ID. `legacy_resize` remains an
explicit diagnostic for the historical mechanism and cannot pass that gate.
The gate establishes a software contract; it does not establish that the
candidate policy is clinically adequate.

## Remaining cohort and clinical qualification

For a principal clinical study, qualify representative coverage without consulting
outcomes: no nodal involvement with qualified negative annotation, spatially
separated lesions, small regions, boundary lesions, varied acquisition
spacing/protocols, and empty/invalid shells.
For each case, resolve annotation and intensity provenance, inspect overlays,
audit volume/retention/disappearance/FOV facts and retain extraction failures.
Known failed contours must remain distinguishable from known absence. The six
development representatives do not supply qualified negative nodal annotation,
expert contour review, or protocol/institution coverage that the source delivery
does not contain.

The completed source-bound H100 checks record the actual software/hardware and
measurement invariance evidence for this candidate configuration. Reuse them
only while their source, configuration and environment bindings remain valid;
changed keys need corresponding qualification. Common lattice sampling does not
by itself guarantee invariance under arbitrary crop-dependent filtering or
intensity normalization. Model prediction
invariance is not required: additional nodal disease can legitimately change a
patient-level prediction.

Freeze the resulting input eligibility, physical grid, support definition,
intensity/discretization recipe, feature set, software versions and configuration
hashes only after the evidence supports them. If original CT or RTSTRUCT inputs
remain unavailable, any study of the prepared working set must retain that
limitation rather than being relabeled as original-acquisition qualification.

Then begin the controlled F2/F3 contrast with identical eligible patients,
splits, initialization, fitted preprocessing, objective, checkpoint rule and
training budget. Include clinical-only C0 and clinical plus physical
burden/geometry C1 baselines in the study plan; baseline results do not yet exist.
Predefine endpoint/time origin, ascertainment, supported evaluation horizons,
censoring conventions, discrimination and calibration metrics, paired comparisons
and uncertainty. Separate variation across training seeds from patient-sampling
uncertainty, and retain explored folds as development data. No confirmatory
survival benefit is established by the implementation or an engineering pilot.
