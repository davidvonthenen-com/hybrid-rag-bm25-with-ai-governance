"""Tests for configuration defaults."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from common.config import load_settings


class TestConfigDefaults(unittest.TestCase):
    """Validate platform-specific defaults for the LLaMA settings."""

    def test_darwin_defaults_use_gpu_layers(self) -> None:
        """Apple Silicon defaults to full GPU offload when unset."""

        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ), patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.llama_n_gpu_layers, -1)

    def test_env_override_preserves_gpu_layers(self) -> None:
        """Environment overrides take precedence over platform defaults."""

        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ), patch.dict(os.environ, {"LLAMA_N_GPU_LAYERS": "12"}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.llama_n_gpu_layers, 12)


if __name__ == "__main__":
    unittest.main()
