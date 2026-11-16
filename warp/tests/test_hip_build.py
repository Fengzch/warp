# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from warp._src import build_dll


class TestHipBuild(unittest.TestCase):
    def test_explicit_sdk(self):
        with tempfile.TemporaryDirectory(prefix="hip sdk ") as directory:
            compiler = Path(directory) / "bin" / "hipcc"
            compiler.parent.mkdir()
            compiler.touch()
            self.assertEqual(build_dll.find_rocm_sdk(directory), directory)

    def test_invalid_explicit_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "HIP compiler not found"):
                build_dll.find_rocm_sdk(directory)

    def test_dtk_root(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / "bin" / "hipcc"
            compiler.parent.mkdir()
            compiler.touch()
            with patch.dict(os.environ, {"DTK_ROOT": directory}, clear=True):
                self.assertEqual(build_dll.find_rocm_sdk(), directory)

    def test_arches(self):
        args = argparse.Namespace(hip_arch="gfx936;gfx942 gfx936,gfx90a:xnack-")
        self.assertEqual(
            build_dll.hip_arch_flags(args),
            "--offload-arch=gfx936 --offload-arch=gfx942 --offload-arch=gfx90a:xnack-",
        )

    def test_arch_environment(self):
        with patch.dict(os.environ, {"HIP_ARCH": "gfx936"}, clear=True):
            self.assertEqual(build_dll.hip_arch_flags(argparse.Namespace()), "--offload-arch=gfx936")

    def test_no_assumed_architecture(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no GPU architecture is assumed"):
                build_dll.hip_arch_flags(argparse.Namespace())

    def test_invalid_architecture(self):
        for value in ("gfx936;touch /tmp/unexpected", "sm_90", ";;;"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "Invalid HIP target"):
                    build_dll.hip_arch_flags(argparse.Namespace(hip_arch=value))


if __name__ == "__main__":
    unittest.main()
