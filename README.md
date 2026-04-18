# funasr-onnx-automodel

[![PyPI](https://img.shields.io/pypi/v/funasr-onnx-automodel.svg)](https://pypi.org/project/funasr-onnx-automodel/)
[![Python](https://img.shields.io/pypi/pyversions/funasr-onnx-automodel.svg)](https://pypi.org/project/funasr-onnx-automodel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified AutoModel interface for [funasr-onnx](https://github.com/modelscope/FunASR) that automatically orchestrates the **VAD → ASR → Punc** speech recognition pipeline with a single `generate()` call.

## Why this project?

funasr-onnx provides excellent ONNX-based speech recognition models, but using them requires manually loading and chaining multiple models (VAD, ASR, Punc) with different APIs and data formats. This package provides a **single unified interface** — just like HuggingFace's `AutoModel` — so you can run the full pipeline in 3 lines of code.

## Features

- **One interface for all models** — AutoModel automatically detects model type (Paraformer / SenseVoice)
- **Full pipeline orchestration** — VAD segmentation → ASR recognition → Punctuation restoration
- **Flexible usage** — Use full VAD+ASR+Punc pipeline, or ASR-only mode
- **Quantized by default** — Uses INT8 quantized models for fast CPU inference
- **Bypasses funasr_onnx `__init__.py`** — Avoids unnecessary heavy imports, faster startup

## Installation

```bash
pip install funasr-onnx-automodel
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install funasr-onnx-automodel
```

## Quick Start

### Full Pipeline (VAD + ASR + Punc)

```python
from funasr_onnx_automodel import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
)

results = model.generate("audio.wav")
print(results[0]["text"])
```

### ASR Only (no VAD, no Punc)

```python
from funasr_onnx_automodel import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
)

results = model.generate("short_audio.wav")
print(results[0]["text"])
```

## API Reference

### `AutoModel(model, vad_model=None, punc_model=None, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | *required* | Model ID or local path for ASR model. Model type (Paraformer/SenseVoice) is auto-detected from name. |
| `vad_model` | `str \| None` | `None` | Model ID or local path for VAD model. If `None`, ASR runs without VAD segmentation. |
| `punc_model` | `str \| None` | `None` | Model ID or local path for punctuation model. If `None`, no punctuation is added. |
| `quantize` | `bool` | `True` | Use INT8 quantized ONNX model (`model_quant.onnx`). |
| `device_id` | `str \| int` | `"-1"` | Device ID. `-1` for CPU inference. |
| `intra_op_num_threads` | `int` | `4` | Number of ONNX Runtime intra-op threads. |

### `generate(input, **kwargs)`

| Parameter | Type | Description |
|---|---|---|
| `input` | `str \| list \| np.ndarray` | Audio file path, list of paths, or 16kHz float32 numpy array. |
| `**kwargs` | | Passed through to the ASR model. |

**Returns:** `list[dict]` — Each dict contains `"text"` (recognized text) and `"timestamp"` (word-level timestamps).

## Model Download

Models are automatically downloaded from [ModelScope](https://modelscope.cn/) on first use. Use the `iic/` prefix for ModelScope model IDs. Pre-exported ONNX models have the `-onnx` suffix.

### Recommended Models

| Role | Model ID |
|---|---|
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-onnx` |
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx` |
| Punc | `iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx` |

### Audio Format

Input audio should be **16kHz mono WAV**. Use ffmpeg to convert:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## License

[MIT](LICENSE)

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) by Alibaba DAMO Academy / ModelScope
- [funasr-onnx](https://github.com/modelscope/FunASR) for ONNX model runtime
