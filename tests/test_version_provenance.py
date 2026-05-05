"""Version/provenance consistency tests."""

from __future__ import annotations

import importlib.util
import pathlib
import re
import unittest

from trifusesurv2 import __version__
from trifusesurv2.explain.gradcam_v2_core import SOFTWARE_VERSION, TARGET_COMMIT_SHA


class VersionProvenanceTest(unittest.TestCase):
    def test_package_and_gradcam_versions_match(self):
        pyproject = pathlib.Path(__file__).parents[1] / "pyproject.toml"
        text = pyproject.read_text()
        match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, flags=re.MULTILINE)
        self.assertIsNotNone(match)
        project_version = match.group(1)
        self.assertEqual(project_version, "2.0.11")
        self.assertEqual(__version__, project_version)
        self.assertEqual(SOFTWARE_VERSION, project_version)
        self.assertEqual(TARGET_COMMIT_SHA, "daaaa363020b7e27b93981f62dfa17821489e1ea")

    def test_current_gradcam_cli_module_exists_and_old_module_is_absent(self):
        self.assertIsNotNone(importlib.util.find_spec("trifusesurv2.multimodal_survival.generate_oof_gradcam_v211"))
        old_names = [
            "trifusesurv2.multimodal_survival.generate_oof_gradcam_" + suffix
            for suffix in ("v20" + "7", "v20" + "8", "v20" + "9", "v21" + "0")
        ]
        for old_name in old_names:
            self.assertIsNone(importlib.util.find_spec(old_name))


if __name__ == "__main__":
    unittest.main()
