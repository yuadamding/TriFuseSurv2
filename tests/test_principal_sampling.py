"""A principal comparison cannot mix physical measurement policies."""
from types import SimpleNamespace
import unittest

from trifusesurv2.multimodal_survival.train import _validate_principal_sampling


class PrincipalSamplingTests(unittest.TestCase):
    def test_principal_requires_verified_common_grid_for_every_patient(self):
        args = SimpleNamespace(fusion_mode="F3", comparison_recipe="principal")
        for manifests in (None, {}, {"a": {}}, {"a": {"sampling": {"policy": "legacy_resize"}}}):
            with self.subTest(manifests=manifests), self.assertRaisesRegex(ValueError, "common_grid"):
                _validate_principal_sampling(args, manifests)
        valid = {"sampling": {"policy": "common_grid", "policy_id": "same"}}
        _validate_principal_sampling(args, {"a": valid, "b": valid})
        with self.assertRaisesRegex(ValueError, "one common_grid policy_id"):
            _validate_principal_sampling(args, {"a": valid, "b": {
                "sampling": {"policy": "common_grid", "policy_id": "different"}}})

    def test_explicit_robustness_and_legacy_remain_diagnostic_routes(self):
        for mode, recipe in (("F3", "robustness"), ("legacy", "principal")):
            _validate_principal_sampling(SimpleNamespace(fusion_mode=mode, comparison_recipe=recipe), None)


if __name__ == "__main__":
    unittest.main()
