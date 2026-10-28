# Prospective neural clinical type policy

New v2 runs can select `--clinical_type_policy approved_categorical_v1` alongside
an explicit `--clinical_cols` list. This is separate from selecting which source
columns are allowed. The default `legacy` policy preserves the existing neural
encoding, including ordinal staging and its derived raw-stage features.

The prospective policy assigns these types:

| Source | Representation |
| --- | --- |
| AGE | Numeric, standardized using training-only mean and sample standard deviation |
| PATHOLOGY, SEX, RACE, KFCF, SMOKE, ALCOHOL, HPV | Literal categorical |
| T, N, M | Literal categorical, with no additional ordinal or derived `*_RAW` channels |

`NSTAGE`, `TX`, derived aliases and unapproved source columns are rejected if
requested. Their values cannot affect fitting, token values or token presence.
The existing biology, burden and host group identities remain unchanged.
Production configuration supplies all 11 approved sources; a smaller explicit
subset is permitted for fabricated fixtures. This policy requires model v2.

The encoder API is:

```python
encoder = SemanticClinicalTokenEncoder.fit(
    training_frame,
    columns=["AGE", "HPV", "T", "N", "M"],
    type_policy="approved_categorical_v1",
)
```

Categorical vocabulary comes only from the supplied training rows. Strings are
not case-folded, numerically parsed, stage-normalized or stripped when nonblank.
For example, `nd`, `ND`, `NA`, `01`, `1` and `1.0` retain their literal identities.
Actual `None`, pandas/NumPy missing values, empty strings and whitespace-only
strings are missing. Literal text such as `NA` or `NaN` remains categorical.

Each categorical feature has separate missing and unseen slots following its
sorted training vocabulary. Missing values do not make a group present; nonempty
unseen categories map to the unseen slot and do make it present. An entirely
missing or absent training column still retains these two slots. Transforming
validation or test rows never expands the vocabulary or refits preprocessing.

AGE uses a standardized value plus a separate missing indicator. Missing AGE is
encoded as standardized zero and indicator one. Constant, single-observation or
near-constant training AGE (standard deviation at most `1e-6`) uses scale one.
If training has no AGE observations, the declared fallback is mean zero and
scale one. Nonempty unparseable or infinite AGE values fail explicitly rather
than becoming categories or silently changing the cohort.

The prospective full-cohort CSV reader preserves selected clinical cells before
pandas numeric/NA inference. Endpoint columns retain the existing numeric
validation. Development-only execution already retains raw train/validation
CSV records before endpoint conversion, with held-out records excluded. Input
view preparation must also preserve source cell strings: the encoder cannot
recover literal distinctions lost by an earlier CSV conversion.

The fitted encoder stores its full `type_policy` descriptor and learned feature
specifications. New prospective model contracts store `clinical_type_policy`;
the experiment contract records both the named CLI option and the full runtime
descriptor. These include source types, literal-category rules, missing/unseen
slots, presence behavior and training-only fitting. The default legacy mode
adds no policy field to historical contract shapes and omits its default CLI
option from canonical experiment settings. Old encoder pickles without the new
attribute keep legacy behavior.

The changed token dimensions and representation require a fresh source-bound
attempt. This correction neither changes the completed pilot's saved encoders
nor explains the separately retained fabricated CUDA-token failure. Synthetic
checks in `tests/test_clinical_type_policy.py`, `test_development_training.py`
and `test_verified_training.py` verify the prospective path, literal CSV input,
training-only vocabulary, checkpoint persistence and legacy training/resume.
