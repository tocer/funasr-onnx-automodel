# funasr-onnx-automodel

[![PyPI](https://img.shields.io/pypi/v/funasr-onnx-automodel.svg)](https://pypi.org/project/funasr-onnx-automodel/)
[![Python](https://img.shields.io/pypi/pyversions/funasr-onnx-automodel.svg)](https://pypi.org/project/funasr-onnx-automodel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[funasr-onnx](https://github.com/modelscope/FunASR) 的统一 AutoModel 接口，只需一次 `generate()` 调用即可自动编排 **VAD → ASR → 标点** 语音识别流水线。

[English](README.md)

## 为什么需要这个项目？

funasr-onnx 提供了优秀的基于 ONNX 的语音识别模型，但使用时需要手动加载和串联多个模型（VAD、ASR、标点），且各模型 API 和数据格式不同。本包提供了**统一的 AutoModel 接口**——类似 HuggingFace 的 `AutoModel`——只需 3 行代码即可运行完整流水线。

## 特性

- **统一接口** — AutoModel 自动检测模型类型（Paraformer / SenseVoice）
- **完整流水线编排** — VAD 分段 → ASR 识别 → 标点恢复
- **灵活使用** — 支持完整 VAD+ASR+标点流水线，或仅 ASR 模式
- **默认量化** — 使用 INT8 量化模型，CPU 推理更快
- **绕过 funasr_onnx `__init__.py`** — 避免不必要的重型导入，启动更快

## 安装

```bash
pip install funasr-onnx-automodel
```

或使用 [uv](https://github.com/astral-sh/uv)：

```bash
uv pip install funasr-onnx-automodel
```

## 快速开始

### 完整流水线（VAD + ASR + 标点）

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

### 仅 ASR（无 VAD、无标点）

```python
from funasr_onnx_automodel import AutoModel

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
)

results = model.generate("short_audio.wav")
print(results[0]["text"])
```

## API 参考

### `AutoModel(model, vad_model=None, punc_model=None, quantize=True, device_id="-1", intra_op_num_threads=4, **kwargs)`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `str` | *必填* | ASR 模型的模型 ID 或本地路径。模型类型（Paraformer/SenseVoice）根据名称自动检测。 |
| `vad_model` | `str \| None` | `None` | VAD 模型的模型 ID 或本地路径。若为 `None`，则不进行 VAD 分段。 |
| `punc_model` | `str \| None` | `None` | 标点模型的模型 ID 或本地路径。若为 `None`，则不添加标点。 |
| `quantize` | `bool` | `True` | 使用 INT8 量化 ONNX 模型（`model_quant.onnx`）。 |
| `device_id` | `str \| int` | `"-1"` | 设备 ID。`-1` 表示 CPU 推理。 |
| `intra_op_num_threads` | `int` | `4` | ONNX Runtime 内部操作线程数。 |

### `generate(input, **kwargs)`

| 参数 | 类型 | 说明 |
|---|---|---|
| `input` | `str \| list \| np.ndarray` | 音频文件路径、路径列表，或 16kHz float32 numpy 数组。 |
| `**kwargs` | | 传递给 ASR 模型的额外参数。 |

**返回值：** `list[dict]` — 每个字典包含 `"text"`（识别文本）和 `"timestamp"`（字级时间戳）。

## 模型下载

首次使用时，模型会自动从 [ModelScope](https://modelscope.cn/) 下载。ModelScope 模型 ID 使用 `iic/` 前缀，预导出的 ONNX 模型带 `-onnx` 后缀。

### 推荐模型

| 用途 | 模型 ID |
|---|---|
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-onnx` |
| ASR | `iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx` |
| 标点 | `iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx` |

### 音频格式

输入音频应为 **16kHz 单声道 WAV**。使用 ffmpeg 转换：

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 许可证

[MIT](LICENSE)

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) — 阿里巴巴达摩院 / ModelScope
- [funasr-onnx](https://github.com/modelscope/FunASR) — ONNX 模型运行时
