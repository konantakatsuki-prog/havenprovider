import unittest

from haven.engine import SharedTTSEngine
from providers.local import LocalProvider
from providers.modal import ModalProvider
from providers.runpod import RunPodProvider
from providers.self_hosted import SelfHostedProvider


class ProviderContractTests(unittest.TestCase):
    def test_preserves_original_response_shape(self) -> None:
        expected = {
            "ok": True,
            "text": "hello",
            "voice": "haven",
            "message": "Haven provider is alive",
        }
        self.assertEqual(expected, SharedTTSEngine().synthesize({"text": "hello"}))

    def test_all_provider_wrappers_share_the_engine(self) -> None:
        engine = SharedTTSEngine()
        payload = {"text": "hello", "voice": "test"}
        providers = (
            LocalProvider(engine),
            ModalProvider(engine),
            SelfHostedProvider(engine),
        )
        for provider in providers:
            with self.subTest(provider=type(provider).__name__):
                self.assertEqual(provider.handle(payload), providers[0].handle(payload))

    def test_runpod_job_shape_remains_compatible(self) -> None:
        result = RunPodProvider().handle_job({"input": {"text": "hello"}})
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["voice"], "haven")


if __name__ == "__main__":
    unittest.main()
