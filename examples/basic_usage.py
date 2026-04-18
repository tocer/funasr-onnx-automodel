"""
Basic usage examples for funasr-onnx-automodel.

Before running:
  1. Convert your audio to 16kHz mono WAV:
     ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

  2. Models are auto-downloaded from ModelScope on first use.
"""

from funasr_onnx_automodel import AutoModel

# --- Full pipeline: VAD + ASR + Punc ---
print("=== Full Pipeline (VAD + ASR + Punc) ===\n")

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx",
)

results = model.generate("test.wav")
for r in results:
    print(f"Text: {r['text']}")
    print(f"Timestamps: {r['timestamp']}")

# --- ASR only (no VAD, no Punc) ---
print("\n=== ASR Only ===\n")

model = AutoModel(
    model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx",
)

results = model.generate("short_audio.wav")
for r in results:
    print(f"Text: {r['text']}")
