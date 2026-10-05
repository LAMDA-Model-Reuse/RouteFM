from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from routefm import checkpoints


class CheckpointResolutionTest(unittest.TestCase):
    def test_local_legacy_override_never_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"checkpoint")
            with mock.patch("huggingface_hub.hf_hub_download") as download:
                resolved = checkpoints.resolve_checkpoint("qwen", path)
            self.assertEqual(resolved.weights, path)
            self.assertIsNone(resolved.config)
            download.assert_not_called()

    def test_local_safetensors_requires_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"weights")
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                checkpoints.resolve_checkpoint("bge", path)

    def test_hub_download_is_pinned_and_checksum_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.safetensors"
            config = root / "variant-config.json"
            repository_config = root / "repository-config.json"
            weights.write_bytes(b"weights")
            config.write_bytes(b"config")
            repository_config.write_bytes(b"repository config")
            artifact = {
                "weights": "qwen/model.safetensors",
                "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
                "config": "qwen/config.json",
                "config_sha256": hashlib.sha256(b"config").hexdigest(),
            }

            def fake_download(*, filename: str, **kwargs) -> str:
                self.assertEqual(kwargs["repo_id"], "test/RouteFM")
                self.assertEqual(kwargs["revision"], "a" * 40)
                return str({
                    "config.json": repository_config,
                    "qwen/model.safetensors": weights,
                    "qwen/config.json": config,
                }[filename])

            with mock.patch.object(checkpoints, "HF_REPO_ID", "test/RouteFM"), mock.patch.object(
                checkpoints, "HF_REVISION", "a" * 40
            ), mock.patch.dict(
                checkpoints.REPOSITORY_CONFIG,
                {
                    "path": "config.json",
                    "sha256": hashlib.sha256(b"repository config").hexdigest(),
                },
                clear=True,
            ), mock.patch.dict(checkpoints.ARTIFACTS, {"qwen": artifact}, clear=True), mock.patch(
                "huggingface_hub.hf_hub_download", side_effect=fake_download
            ) as download:
                resolved = checkpoints.resolve_checkpoint("qwen")
            self.assertEqual(
                [call.kwargs["filename"] for call in download.call_args_list],
                ["config.json", "qwen/model.safetensors", "qwen/config.json"],
            )
            self.assertEqual(resolved.weights, weights)
            self.assertEqual(resolved.config, config)

    def test_hub_checksum_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"wrong")
            artifact = {
                "weights": "bge/model.safetensors",
                "weights_sha256": "0" * 64,
                "config": "bge/config.json",
                "config_sha256": "0" * 64,
            }
            with mock.patch.object(checkpoints, "HF_REVISION", "b" * 40), mock.patch.dict(
                checkpoints.REPOSITORY_CONFIG,
                {
                    "path": "config.json",
                    "sha256": hashlib.sha256(b"wrong").hexdigest(),
                },
                clear=True,
            ), mock.patch.dict(
                checkpoints.ARTIFACTS, {"bge": artifact}, clear=True
            ), mock.patch("huggingface_hub.hf_hub_download", return_value=str(path)):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    checkpoints.resolve_checkpoint("bge")


if __name__ == "__main__":
    unittest.main()
