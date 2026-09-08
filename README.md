# Haven / Cove TTS provider

This repository now contains a Replicate-ready Cog wrapper for the tested Cove
GPT-SoVITS voice. The older RunPod, Modal, and local provider scaffolds remain
unchanged and still use the compatibility echo engine by default.

The Replicate runner accepts English text, a speed multiplier, and a sampling
seed, then returns a WAV file. It reproduces the settings from the working
`MoCHi Cove TTS Test` Hugging Face Space.

## Replicate preparation

The deployment is deterministic:

- `cog.yaml` uses a GPU image, Python 3.10.13, and Cog SDK 0.22.0.
- `requirements-replicate.txt` pins the compatible inference stack.
- The two patched GPT-SoVITS wheels are installed from immutable Space commit
  `79d754c2f2a211471fbb3f365214020d21d8cd8b`.
- Model assets are downloaded from `MoYoYoTech/tone-models` at immutable commit
  `198ceeac1c4d5b75072b6995af2e185e4559ec1d`.
- Only nine required files are baked into the image, totaling about 1.01 GiB.

The upstream `tone-models` model card currently declares the assets as MIT
licensed. The patched GPT-SoVITS wheel declares MIT and LangSegment declares
BSD; preserve those notices if the project is redistributed.

No Replicate account, model, deployment, token, or paid resource is created by
these local preparation files.

Once Docker/WSL is available, validate locally with:

```text
cog doctor
cog build --separate-weights
cog run -i text="Hello from Cove"
```

After explicitly deciding to publish, create a **public model** on Replicate and
select an Nvidia T4 as the initial hardware, then push it with the model owner's
chosen slug:

```text
cog login
cog push r8.im/<replicate-username>/<public-model-name>
```

Use a public model, not a dedicated deployment, for the intended caller-pays
flow. Each caller should supply their own Replicate API token. Never commit,
persist, or log caller tokens in MoCHi.

## Providers

- `handler.py` / `providers.runpod`: existing RunPod serverless contract.
- `local_server.py` / `providers.local`: local HTTP server on `127.0.0.1:8000`.
- `modal_app.py` / `providers.modal`: Modal POST JSON endpoint.
- `providers.self_hosted`: reserved wrapper for the planned third provider.

All legacy providers call `haven.engine.SharedTTSEngine`, so transport-specific
code does not need to know where the model runs. The Replicate entrypoint is
`run.py:Runner` and uses `haven.cove.CoveSynthesizer` directly.

## Modal build requirements

`modal_app.py` uses `modal.Image.run_function` to download configured pretrained
model files and preload the English NLTK tagger plus `cmudict` while the image is
being built. Set `HAVEN_PRETRAINED_MODEL_URLS` to a JSON object mapping paths
inside `/opt/haven/models` to URLs before running `modal deploy`:

```text
{"pretrained/example.bin":"https://your-model-host/example.bin"}
```

The image installs `torchcodec` through `requirements-modal.txt` defaults. Modal
still uses the provider scaffold and is not part of the Replicate deployment.

The HTTP endpoint is a Modal `POST` Web Function whose JSON body is typed as a
`dict`, so clients should send:

```json
{"text":"Hello", "voice":"haven"}
```

Use `modal deploy modal_app.py` after installing the Modal CLI and authenticating.

## Local checks

```text
python -m unittest discover -s tests -v
python local_server.py
```

The unit tests do not download model files or require a GPU.
