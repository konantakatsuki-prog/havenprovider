"""Replicate Cog entrypoint for the Cove GPT-SoVITS voice."""

from cog import BaseRunner, Input, Path

from haven.cove import MAX_TEXT_LENGTH, CoveSynthesizer


class Runner(BaseRunner):
    def setup(self) -> None:
        self.synthesizer = CoveSynthesizer()
        self.synthesizer.load()

    def run(
        self,
        text: str = Input(
            description="English text to speak with the Cove voice.",
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
        ),
        speed: float = Input(
            description="Speech speed multiplier.",
            default=1.0,
            ge=0.5,
            le=2.0,
        ),
        seed: int = Input(
            description="Sampling seed; use -1 for a random seed.",
            default=233333,
            ge=-1,
            le=2_147_483_647,
        ),
    ) -> Path:
        output = self.synthesizer.synthesize(text, speed=speed, seed=seed)
        return Path(output)

