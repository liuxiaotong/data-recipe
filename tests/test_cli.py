"""Unit tests for CLI validation and utility functions.

Tests validate_output_path() and other CLI-level utilities
without invoking actual analysis pipelines.
"""

import unittest
from pathlib import Path

from datarecipe.cli import validate_output_path


class TestValidateOutputPath(unittest.TestCase):
    """Test validate_output_path() security function."""

    def test_valid_normal_path(self):
        result = validate_output_path("/tmp/datarecipe_output")
        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_absolute())

    def test_valid_relative_path_resolves(self):
        result = validate_output_path("output/test")
        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_absolute())

    def test_base_dir_within(self):
        result = validate_output_path("/tmp/output/sub", base_dir=Path("/tmp/output"))
        self.assertIsInstance(result, Path)

    def test_base_dir_outside_raises(self):
        with self.assertRaises(ValueError) as cm:
            validate_output_path("/other/path", base_dir=Path("/tmp/output"))
        self.assertIn("outside allowed directory", str(cm.exception))

    def test_blocked_usr(self):
        with self.assertRaises(ValueError):
            validate_output_path("/usr/local/share/data")

    def test_blocked_bin(self):
        # /bin may resolve to /usr/bin on some systems, both are blocked
        with self.assertRaises(ValueError):
            validate_output_path("/usr/bin/something")

    def test_blocked_root(self):
        with self.assertRaises(ValueError):
            validate_output_path("/root/.config")

    def test_home_dir_allowed(self):
        # Home directory should be allowed
        result = validate_output_path("/Users/test/output")
        self.assertIsInstance(result, Path)

    def test_tmp_dir_allowed(self):
        result = validate_output_path("/tmp/some/nested/path")
        self.assertIsInstance(result, Path)


if __name__ == "__main__":
    unittest.main()
