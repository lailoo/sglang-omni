"""Check AudioDiT speech serving through the pipeline contract."""

import base64
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang_omni.config import manager
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.pipeline.helpers import run_scheduler


def test_checkpoint_resolves_to_single_stage_pipeline(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "audiodit"}))
    from sglang_omni.models.longcat_audiodit.config import AudioDiTPipelineConfig

    assert (
        manager.resolve_config_cls_for_model_path(str(tmp_path))
        is AudioDiTPipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("AudioDiTForConditionalGeneration")
        is AudioDiTPipelineConfig
    )
    config = AudioDiTPipelineConfig(model_path=str(tmp_path))
    assert len(config.stages) == 1
    assert config.stages[0].factory.steps == 16
    assert config.stages[0].factory.guidance_strength == 4.0
    assert config.stages[0].factory.seed == 1024
    assert config.stages[0].terminal


def test_speech_request_returns_original_waveform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.longcat_audiodit import stages

    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    calls = []
    audio = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    class Model:
        config = types.SimpleNamespace(
            sampling_rate=24000,
            text_encoder_model="umt5",
            latent_hop=2048,
            max_wav_duration=30,
        )
        vae = types.SimpleNamespace(to_half=lambda: None)

        @classmethod
        def from_pretrained(cls, path):
            calls.append(path)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(waveform=torch.tensor(audio))

    def infer_one(*args):
        raise AssertionError("Plain speech must use the unconditioned model call")

    monkeypatch.setitem(
        sys.modules, "audiodit", types.SimpleNamespace(AudioDiTModel=Model)
    )
    monkeypatch.setitem(
        sys.modules, "batch_inference", types.SimpleNamespace(infer_one=infer_one)
    )
    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(
            normalize_text=lambda text: text.lower(),
            approx_duration_from_text=lambda text, max_duration: 1.0,
        ),
    )
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        types.SimpleNamespace(
            from_pretrained=lambda *a, **kw: (
                lambda *args, **kwargs: types.SimpleNamespace(
                    input_ids=torch.ones((1, 2), dtype=torch.long),
                    attention_mask=torch.ones((1, 2), dtype=torch.long),
                )
            )
        ),
    )
    scheduler = stages.create_inference_executor(
        "checkpoint",
        steps=16,
        guidance_strength=4.0,
        seed=1024,
        tokenizer_path="umt5",
        device="cpu",
    )
    payload = StagePayload("request", OmniRequest(inputs="Hello world"), {})
    outputs = run_scheduler(
        scheduler,
        [IncomingMessage(payload.request_id, "new_request", payload)],
        output_count=1,
    )
    assert outputs[0].type == "result"
    result = outputs[0].data
    assert calls[0] == "checkpoint"
    assert calls[1]["prompt_audio"] is None
    assert calls[1]["duration"] == 11
    assert calls[1]["steps"] == 16
    assert calls[1]["guidance_method"] == "cfg"
    assert result.data["sample_rate"] == 24000
    assert result.data["modality"] == "audio"
    np.testing.assert_array_equal(
        np.frombuffer(result.data["audio_waveform"], dtype=np.float32), audio
    )


def test_reference_data_and_transcript_reach_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.longcat_audiodit import stages

    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    captured = []
    sample = b"RIFFexample"

    class Model:
        config = types.SimpleNamespace(
            sampling_rate=24000,
            text_encoder_model="umt5",
            latent_hop=2048,
            max_wav_duration=30,
        )
        vae = types.SimpleNamespace(to_half=lambda: None)

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    def infer_one(*args):
        captured.append((args[0], args[1], Path(args[2]).read_bytes(), args[-1]))
        return np.array([0.5], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "audiodit", types.SimpleNamespace(AudioDiTModel=Model)
    )
    monkeypatch.setitem(
        sys.modules, "batch_inference", types.SimpleNamespace(infer_one=infer_one)
    )
    monkeypatch.setitem(
        sys.modules,
        "utils",
        types.SimpleNamespace(
            normalize_text=lambda text: text.lower(),
            approx_duration_from_text=lambda text, max_duration: 1.0,
        ),
    )
    monkeypatch.setattr(
        stages,
        "PreTrainedTokenizerFast",
        types.SimpleNamespace(from_pretrained=lambda *a, **kw: "tokenizer"),
    )
    scheduler = stages.create_inference_executor(
        "checkpoint",
        steps=16,
        guidance_strength=4.0,
        seed=1024,
        tokenizer_path="umt5",
        device="cpu",
    )
    reference = {
        "data": base64.b64encode(sample).decode(),
        "media_type": "audio/wav",
        "text": "Speaker",
    }
    payload = StagePayload(
        "request", OmniRequest(inputs={"text": "Speech", "references": [reference]}), {}
    )
    outputs = run_scheduler(
        scheduler,
        [IncomingMessage(payload.request_id, "new_request", payload)],
        output_count=1,
    )
    assert outputs[0].type == "result"
    assert captured == [("Speech", "Speaker", sample, "apg")]


def test_reference_requires_transcript() -> None:
    from sglang_omni.models.longcat_audiodit.stages import parse_speech_request

    payload = StagePayload(
        "request",
        OmniRequest(
            inputs={"text": "Speech", "references": [{"audio_path": "/somewhere.wav"}]}
        ),
        {},
    )
    with pytest.raises(ValueError, match="reference transcript"):
        parse_speech_request(payload)
