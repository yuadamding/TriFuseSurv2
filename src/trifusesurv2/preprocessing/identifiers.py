"""Literal identifier handling shared by cohort and radiomics CSV readers.

Only empty/whitespace-only identifier cells are missing. Strings such as ``NA``
and ``001`` are identifiers, while non-identifier columns retain pandas' usual
numeric and missing-value inference. Column aliases never equate acquisitions
with patients or rename the source table.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def _spelling(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def resolve_identifier_column(columns: Iterable[str], requested: str, *, required: bool = True) -> str | None:
    columns = list(columns)
    matches = [name for name in columns if _spelling(name) == _spelling(requested)]
    if not matches and not required:
        return None
    if len(matches) != 1:
        detail = f"ambiguous matches {matches}" if matches else "no matching column"
        raise ValueError(f"Identifier column {requested!r}: {detail}.")
    return matches[0]


def is_identifier_column(name: str) -> bool:
    lowered = str(name).casefold()
    canonical = _spelling(name)
    if canonical in {"patientid", "caseid", "scanid", "radiomicsscanid", "contourversion", "segmenterid"}:
        return True
    return (
        lowered.endswith(("_id", "_ids", "_uid", "_sha256", "_id_norm"))
        or lowered.startswith(("region_id__", "support_sha256__"))
    )


def _literal_identifier(value: str):
    return pd.NA if not value.strip() else value


def require_identifier(value, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonmissing literal identifier string.")
    return value


def read_identifier_csv(path, *, identifier_columns: Iterable[str] = (), **kwargs) -> pd.DataFrame:
    """Read a CSV path without numeric/NA coercion of identifier columns.

Caller-supplied dtype/converter settings for non-identifiers remain supported;
identifier conversion is always controlled by this contract.
"""
    header_options = {key: value for key, value in kwargs.items()
                      if key not in {"dtype", "converters", "nrows", "chunksize", "iterator"}}
    position = path.tell() if hasattr(path, "tell") else None
    columns = pd.read_csv(path, nrows=0, **header_options).columns
    if position is not None:
        path.seek(position)
    identifiers = {name for name in columns if is_identifier_column(name)}
    for requested in identifier_columns:
        actual = resolve_identifier_column(columns, requested, required=False)
        if actual is not None:
            identifiers.add(actual)
    options = dict(kwargs)
    converters = dict(options.pop("converters", {}) or {})
    converters.update({name: _literal_identifier for name in identifiers})
    dtype = options.get("dtype")
    if isinstance(dtype, dict):
        options["dtype"] = {name: value for name, value in dtype.items() if name not in identifiers}
    frame = pd.read_csv(path, converters=converters, **options)
    for name in identifiers:
        if name in frame:
            frame[name] = frame[name].astype("string")
    return frame
