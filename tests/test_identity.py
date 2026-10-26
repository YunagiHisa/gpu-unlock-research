"""Synthetic secrets only; never read the operator's real key in CPU tests."""
import contextlib
import hashlib
import hmac
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from benchmarks import gemm
from benchmarks.identity import IdentityMismatch, gpu_pseudonym


class TestIdentity(unittest.TestCase):
    def setup_key(self, root):
        directory = Path(root) / "config"
        directory.mkdir(mode=0o700)
        key = directory / "hmac.key"
        key.write_bytes(b"A" * 32)  # Deliberately synthetic public test fixture.
        key.chmod(0o600)
        return key

    def test_hmac_is_stable_and_never_exposes_input(self):
        with tempfile.TemporaryDirectory() as root:
            key = self.setup_key(root)
            uuid = b"GPU-test-private-uuid"
            expected = hmac.new(key.read_bytes(), b"gpu-unlock-research/gpu-id/v1\0" + uuid, hashlib.sha256).hexdigest()
            a = gpu_pseudonym(lambda: uuid, key)
            b = gpu_pseudonym(lambda: uuid, key)
            self.assertEqual(a, b)
            self.assertEqual(a["value"], "hmac-sha256:" + expected)
            self.assertNotIn(uuid.decode(), json.dumps(a))
            self.assertNotIn(key.read_text(), json.dumps(a))
            key.write_bytes(b"B" * 32)
            self.assertNotEqual(a["value"], gpu_pseudonym(lambda: uuid, key)["value"])

    def test_permissions_owner_and_bad_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            key = self.setup_key(root)
            for target, bad_mode in ((key, 0o644), (key.parent, 0o755)):
                target.chmod(bad_mode)
                result = gpu_pseudonym(lambda: "GPU-test-uuid", key)
                self.assertEqual(result["status"], "unavailable")
                target.chmod(0o600 if target == key else 0o700)
            with patch("benchmarks.identity.os.getuid", return_value=-1):
                self.assertEqual(gpu_pseudonym(lambda: "GPU-test-uuid", key)["status"], "unavailable")
            key.write_bytes(b"short-secret")
            self.assertEqual(gpu_pseudonym(lambda: "GPU-test-uuid", key)["reason"], "key_requires_exactly_32_bytes")

    def test_missing_key_does_not_generate_or_read_uuid(self):
        with tempfile.TemporaryDirectory() as root:
            key = Path(root) / "missing" / "hmac.key"
            reader = Mock()
            result = gpu_pseudonym(reader, key)
            self.assertEqual(result["reason"], "key_missing")
            self.assertIn("README", result["initialization_help"])
            self.assertFalse(key.exists())
            reader.assert_not_called()

    def test_symlink_hardlink_and_in_repo_key_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            key = self.setup_key(root)
            link = key.parent / "link"
            link.symlink_to(key)
            self.assertEqual(gpu_pseudonym(lambda: "GPU-test-uuid", link)["status"], "unavailable")
            os.link(key, key.parent / "hardlink")
            self.assertEqual(gpu_pseudonym(lambda: "GPU-test-uuid", key)["status"], "unavailable")
        result = gpu_pseudonym(lambda: "GPU-test-uuid", Path(gemm.__file__).parent / "hmac.key")
        self.assertEqual(result["reason"], "key_must_be_outside_repository")

    def test_nvml_error_text_is_not_exported(self):
        with tempfile.TemporaryDirectory() as root:
            key = self.setup_key(root)
            result = gpu_pseudonym(Mock(side_effect=RuntimeError("private-key-or-uuid")), key)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("private-key-or-uuid", json.dumps(result))

    def test_identity_mismatch_saves_diagnostic_with_dedicated_exit(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "diagnostic.json"
            with patch.object(gemm, "run", side_effect=IdentityMismatch()), contextlib.redirect_stdout(io.StringIO()):
                code = gemm.main(["--mode", "fp16", "--output", str(target)])
            self.assertEqual(code, 4)
            result = json.loads(target.read_text())
            self.assertFalse(result["identity_verified"])
            self.assertFalse(result["baseline_eligible"])
            self.assertNotIn("timings_ms", result)
            self.assertIn("physical_gpu_identity_mismatch", result["baseline_ineligibility_reasons"])

    def test_identity_contradiction_stops_before_tensor_allocation(self):
        torch = MagicMock()
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="test")
        args = gemm.parse_args(["--mode", "fp16"])
        with patch.dict("sys.modules", {"torch": torch}), \
             patch.object(gemm, "configure_mode", return_value={}), \
             patch.object(gemm, "cuda_pci_bus_id", return_value="0000:01:00.0"), \
             patch.object(gemm, "state_evidence", return_value={}), \
             patch.object(gemm, "NvmlSession", side_effect=IdentityMismatch()):
            with self.assertRaises(IdentityMismatch):
                gemm.run(args)
        torch.randn.assert_not_called()
        torch.empty.assert_not_called()
        torch.mm.assert_not_called()
