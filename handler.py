from typing import Any

from providers.runpod import RunPodProvider


_provider = RunPodProvider()


def handler(job: dict[str, Any]) -> dict[str, Any]:
    return _provider.handle_job(job)


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
