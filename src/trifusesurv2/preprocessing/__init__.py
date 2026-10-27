"""Shared physical-region preprocessing for image and radiomics inputs."""

from .compartments import (
    ReferenceGeometry,
    assert_same_geometry,
    build_compartments,
    load_region_manifest,
    mask_checksum,
    physical_geometry_checksum,
    physical_shell,
    region_identity,
    validate_region_manifest,
    write_region_manifest,
)

__all__ = [
    "ReferenceGeometry", "assert_same_geometry", "build_compartments",
    "load_region_manifest", "mask_checksum", "physical_geometry_checksum", "physical_shell", "region_identity",
    "validate_region_manifest", "write_region_manifest",
]
