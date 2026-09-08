import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from haven.cove import (
    COVE_ASSET_PATTERNS,
    COVE_REPO_ID,
    COVE_REPO_REVISION,
    MAX_TEXT_LENGTH,
    CoveConfig,
    build_inference_params,
    download_cove_assets,
    resolve_cove_assets,
    validate_text,
)


def _create_fake_assets(config: CoveConfig) -> None:
    assets = resolve_cove_assets(config)
    for directory in (assets.hubert, assets.bert):
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in (assets.gpt, assets.sovits, assets.reference_audio):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"test")


class CovePreparationTests(unittest.TestCase):
    def test_download_is_pinned_and_limited_to_cove_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = CoveConfig(model_dir=Path(temporary_dir))
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                _create_fake_assets(config)

            assets = download_cove_assets(config, downloader=fake_download)

            self.assertFalse(assets.missing())
            self.assertEqual(calls[0]["repo_id"], COVE_REPO_ID)
            self.assertEqual(calls[0]["revision"], COVE_REPO_REVISION)
            self.assertEqual(calls[0]["allow_patterns"], list(COVE_ASSET_PATTERNS))

    def test_existing_baked_assets_skip_network_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = CoveConfig(model_dir=Path(temporary_dir))
            _create_fake_assets(config)

            def unexpected_download(**_kwargs):
                self.fail("download should not run for a complete baked model")

            download_cove_assets(config, downloader=unexpected_download)

    def test_text_and_inference_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config = CoveConfig(model_dir=Path(temporary_dir))
            _create_fake_assets(config)
            assets = resolve_cove_assets(config)

            params = build_inference_params(
                "  Hello from Cove.  ", assets, config, speed=1.25, seed=7
            )
            self.assertEqual(params["text"], "Hello from Cove.")
            self.assertEqual(params["speed_factor"], 1.25)
            self.assertEqual(params["seed"], 7)
            self.assertEqual(params["text_split_method"], "cut4")

            with self.assertRaises(ValueError):
                validate_text("x" * (MAX_TEXT_LENGTH + 1))
            with self.assertRaises(ValueError):
                build_inference_params("hello", assets, config, speed=3.0)

    def test_environment_can_override_model_location(self) -> None:
        with patch.dict(os.environ, {"HAVEN_COVE_MODEL_DIR": "custom-models"}):
            self.assertEqual(
                CoveConfig.from_environment().model_dir,
                Path("custom-models"),
            )


if __name__ == "__main__":
    unittest.main()

