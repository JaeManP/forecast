import os
import unittest
from unittest import mock

from verily.forecast import guardrails


class GuardrailTests(unittest.TestCase):
    def test_wandb_disabled_is_always_allowed(self):
        guardrails.validate_external_logging(False, synthetic_mode=False)
        guardrails.validate_external_logging(False, synthetic_mode=True)

    def test_wandb_requires_synthetic_mode(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic-data"):
            guardrails.validate_external_logging(True, synthetic_mode=False)

    def test_wandb_blocked_in_aou_workspace_even_for_synthetic_mode(self):
        with mock.patch.dict(os.environ, {"WORKSPACE_CDR": "project.dataset"}):
            with self.assertRaisesRegex(RuntimeError, "synthetic-data"):
                guardrails.validate_external_logging(True, synthetic_mode=True)

    def test_wandb_allowed_for_declared_synthetic_mode_outside_workspace(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            guardrails.validate_external_logging(True, synthetic_mode=True)

    def test_default_mixed_precision_uses_fp16_only_on_cuda(self):
        self.assertEqual(guardrails.default_mixed_precision(True), "fp16")
        self.assertEqual(guardrails.default_mixed_precision(False), "no")


if __name__ == "__main__":
    unittest.main()
