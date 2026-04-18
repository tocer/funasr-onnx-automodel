import importlib.util
import os
import sys

import librosa
import numpy as np


def _ensure_pkg_namespace(pkg_name):
    """Ensure a package exists in sys.modules without executing its __init__.py."""
    pkg = sys.modules.get(pkg_name)
    if pkg is not None and hasattr(pkg, "__path__"):
        return pkg
    pkg_spec = importlib.util.find_spec(pkg_name)
    pkg = importlib.util.module_from_spec(pkg_spec)
    pkg.__path__ = pkg_spec.submodule_search_locations
    sys.modules[pkg_name] = pkg
    return pkg


def _lazy_import(module_path, class_name):
    """Import a class from a submodule bypassing the package __init__.py."""
    parts = module_path.split(".")
    pkg = _ensure_pkg_namespace(parts[0])

    if module_path in sys.modules:
        return getattr(sys.modules[module_path], class_name)

    submod_file = os.path.join(pkg.__path__[0], *parts[1:-1], parts[-1] + ".py")
    spec = importlib.util.spec_from_file_location(module_path, submod_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _load_model(module_path, class_name, model_dir, **kwargs):
    """Lazy-import and instantiate a funasr_onnx model."""
    cls = _lazy_import(module_path, class_name)
    return cls(model_dir=model_dir, **kwargs)


class AutoModel:
    """Unified AutoModel interface for funasr-onnx.

    Orchestrates VAD → ASR → Punc pipeline with a single generate() call.
    """

    def __init__(
        self,
        model: str,
        vad_model: str | None = None,
        punc_model: str | None = None,
        quantize: bool = True,
        device_id: str | int = "-1",
        intra_op_num_threads: int = 4,
        **kwargs,
    ):
        common = dict(
            quantize=quantize,
            device_id=device_id,
            intra_op_num_threads=intra_op_num_threads,
            **kwargs,
        )

        # Infer ASR model type from model name
        if "sensevoice" in model.lower():
            self.asr_model = _load_model(
                "funasr_onnx.sensevoice_bin", "SenseVoiceSmall", model, **common
            )
        else:
            self.asr_model = _load_model(
                "funasr_onnx.paraformer_bin", "Paraformer", model, **common
            )

        self.vad_model = None
        if vad_model is not None:
            self.vad_model = _load_model(
                "funasr_onnx.vad_bin", "Fsmn_vad", vad_model, **common
            )

        self.punc_model = None
        if punc_model is not None:
            self.punc_model = _load_model(
                "funasr_onnx.punc_bin", "CT_Transformer", punc_model, **common
            )

    def generate(self, input, **kwargs):
        """Run VAD → ASR → Punc pipeline.

        Args:
            input: Audio file path (str), list of paths, or np.ndarray (16kHz float32).
            **kwargs: Passed through to ASR model.

        Returns:
            List of dicts: [{"text": "...", "timestamp": [...]}]
        """
        if self.vad_model is not None:
            results = self._generate_with_vad(input, **kwargs)
        else:
            results = self._generate_direct(input, **kwargs)

        if self.punc_model is not None:
            for item in results:
                text = item["text"]
                if text:
                    punc_text, _ = self.punc_model(text)
                    item["text"] = punc_text

        return results

    def _generate_with_vad(self, input, **kwargs):
        """Load audio, run VAD segmentation, ASR each segment, merge."""
        audio, _ = librosa.load(input, sr=16000)

        segments = self.vad_model(audio)
        if not segments or not segments[0]:
            return [{"text": "", "timestamp": []}]

        all_text = []
        all_timestamps = []
        for seg in segments[0]:
            start_ms, end_ms = seg
            start_sample = int(start_ms * 16)
            end_sample = int(end_ms * 16)
            chunk = audio[start_sample:end_sample]

            asr_res = self.asr_model(chunk, **kwargs)
            if asr_res:
                all_text.append(asr_res[0].get("preds", ""))
                ts = asr_res[0].get("timestamp")
                if ts:
                    all_timestamps.extend(
                        [t[0] + start_ms, t[1] + start_ms] for t in ts
                    )

        return [{"text": "".join(all_text), "timestamp": all_timestamps}]

    def _generate_direct(self, input, **kwargs):
        """Run ASR directly without VAD."""
        asr_res = self.asr_model(input, **kwargs)
        results = []
        for item in asr_res:
            if isinstance(item, dict):
                results.append({
                    "text": item.get("preds", ""),
                    "timestamp": item.get("timestamp", []),
                })
            else:
                # SenseVoiceSmall returns plain string
                results.append({"text": str(item), "timestamp": []})
        return results
