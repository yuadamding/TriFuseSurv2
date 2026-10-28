# Corrected baseline candidate B — October 28, 2026

**A corrected C0/C1 package is prepared locally; execution remains blocked by
budget acceptance and the principal-study freeze.** Candidate B binds source
`038296466f5ac0baa9e5d84c9dc0e078049d65a5`, whose executable source matches
correction commit `3298bf017c667ea44ae8ad621dc8cfaa95961aa6`. It preserves the
[accepted baseline statistical recipe](../configs/baseline_study_spec.json),
including deterministic C0/C1 fits, validation-IBS penalty selection and the
fixed evaluation grid. The old baseline-A package remains unchanged.

The copied metadata, outcomes, 365/77/77 split files and geometry inventory
retain their exact hashes. Whole-file equality establishes that all 519
eligible metadata rows were copied without conversion. A separate development
audit compared **4,862 clinical cells across 442 training/validation patients**:
all 11 approved fields matched the original eligible CSV literally. AGE had
two blank cells and no invalid nonblank values. No held-out clinical values were
compared or summarized, and no outcome-dependent analysis was performed.

The [corrected reader](tabular_baselines.md) preserves raw clinical strings
before categorical handling or strict AGE validation. This package adds no
vocabulary fitting or imputation during preparation. It retains the same
16 pinned wheels, totaling 85,772,036 bytes, for a prospective isolated
`baseline-v2` environment; no environment was installed.

Local verification checked **179 payload files**, parsed **101 Python files**
and passed four focused metadata/guard tests:

- Development-only literal comparison accepts true blank AGE cells.
- Changes from literal `NA` to blank, `01` to `1`, or trimmed nonblank whitespace
  are rejected.
- Nonblank invalid numeric AGE values are rejected.
- The provisional worker stops before output creation, environment setup or fitting;
  changing the plan status alone cannot release it.

The earlier **11-test/16-subtest** corrected-reader result was reused only after
matching its recorded source and test hashes against the candidate and verifying
the retained test-log hash. Those model fixtures were not rerun during packaging.
Independent read-only review found no blocking candidate defect. These checks
establish local package integrity, not worker qualification or clinical performance.

Evidence is retained under
`/storage/TriFuseSurv/runs/study-20261028/baseline-development-b`, with the bundle
in `sealed/baseline-development-b` and paths/lineage in `HANDOFF.json`.

| Artifact | SHA-256 |
| --- | --- |
| Executable source tree | `c40782445575be0e3338971c03f31764080d3ad77db60e5b9b6ec60ed7603806` |
| Frozen statistical specification | `a6d129c091666955ef997fc950c218fcabc64a5566ebfdb9183357dee850bf7b` |
| Original and candidate eligible CSV | `a7121f3b4258f33bd1189931c2a70f7e13b036017c1fe5235714105ff362292e` |
| `BUNDLE_BINDING.json` | `2db5122e7006f7c36555747b2ecd5d55fc882358685341c9a8bddeab1e39381c` |
| `PLAN.json` | `c32c0bd4f2eeb55ea0d2ec90674a26054499a836baceef364e9de279460f1925` |
| `manifest.json` | `4e620c5fb3e0c69367c696721c3559709fb5ddd1ce5bf253d3a985d61d36c7c0` |
| `CLINICAL_CELL_IDENTITY.json` | `f831982d1c1eb55750726ed1f39f28d05e2af2cacd500d41f1f0f5a66d107ef8` |
| `LOCAL_CHECKS.json` | `93f76daa4d02394e79fa0ecb1c68790a3efbfd2a837c36b00c1bfae42e8a25ac` |
| `HANDOFF.json` | `4a5efdb1bcbfbef9fa61882fe56a78b10a4bc17be588d21ca76ca3684df0affe` |

The candidate is read-only, its publication allowlist is provisional, and its
worker has an unconditional execution guard. Preserve it as evidence and create
a fresh reviewed release after budget acceptance and principal freeze. That
release must bind the latest exact lifecycle cleanup, current source and
controller contracts, remote admission and storage checks, and same-image
dependency/reader qualification. Historical pilot receipts establish lineage
only. The prospective institutional template reserves one H100 with CUDA masked
for CPU Cox fitting; this preparation authorized and performed no remote work,
GPU allocation, environment installation or model fit.
