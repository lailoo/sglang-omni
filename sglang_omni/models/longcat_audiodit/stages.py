# SPDX-License-Identifier: Apache-2.0
"""Serialized AudioDiT synthesis for text and reference-conditioned speech."""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device


def parse_speech_request(payload: StagePayload) -> tuple[str, str, str | bytes | None]:
    """Extract a text prompt and at most one reference with its transcript."""
    inputs = payload.request.inputs
    if isinstance(inputs, str):
        text, references = inputs, []
    elif isinstance(inputs, dict):
        text = inputs.get("text", inputs.get("input", ""))
        references = inputs.get("references") or []
    else:
        raise ValueError("AudioDiT requires text input")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("AudioDiT requires non-empty text")
    else:
        pass
    if not isinstance(references, list) or len(references) > 1:
        raise ValueError("AudioDiT accepts at most one reference")
    else:
        pass
    if not references:
        return text, "", None
    else:
        pass

    reference = references[0]
    if not isinstance(reference, dict):
        raise ValueError("AudioDiT reference must be an object")
    else:
        pass
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params") or {}
    reference_text = reference.get("text") or tts_params.get("ref_text")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("AudioDiT requires a reference transcript")
    else:
        pass
    audio_path = reference.get("audio_path")
    if audio_path is not None:
        if not isinstance(audio_path, str):
            raise ValueError("AudioDiT reference audio_path must be a string")
        else:
            pass
        reference_audio = audio_path
    elif reference.get("data") is not None:
        try:
            reference_audio = base64.b64decode(reference["data"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("AudioDiT reference data must be base64") from error
    else:
        raise ValueError("AudioDiT reference requires audio data or audio_path")
    return text, reference_text, reference_audio


def create_inference_executor(
    model_path: str,
    *,
    steps: int,
    guidance_strength: float,
    seed: int,
    tokenizer_path: str | None = None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    """Load a checkpoint once and preserve sampling order across requests."""
    if steps < 2 or not np.isfinite(guidance_strength) or guidance_strength <= 0:
        raise ValueError(
            "AudioDiT steps must be at least 2 and guidance must be positive"
        )
    else:
        pass
    from audiodit import AudioDiTModel
    from batch_inference import infer_one
    from utils import approx_duration_from_text, normalize_text

    concrete_device = resolve_concrete_device(device, gpu_id)
    model = AudioDiTModel.from_pretrained(model_path).to(concrete_device)
    model.vae.to_half()
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path or model.config.text_encoder_model, local_files_only=True
    )
    torch.manual_seed(seed)
    if concrete_device.type == "cuda":
        torch.cuda.manual_seed(seed)
    else:
        pass

    def infer(payload: StagePayload) -> StagePayload:
        text, reference_text, reference_audio = parse_speech_request(payload)
        tts_params = (payload.request.metadata or {}).get("tts_params") or {}
        request_seed = tts_params.get("seed")
        if request_seed is not None:
            torch.manual_seed(int(request_seed))
            if concrete_device.type == "cuda":
                torch.cuda.manual_seed(int(request_seed))
            else:
                pass
        else:
            pass
        with torch.inference_mode():
            if reference_audio is None:
                normalized_text = normalize_text(text)
                inputs = tokenizer(
                    [normalized_text], padding="longest", return_tensors="pt"
                )
                duration_seconds = approx_duration_from_text(
                    normalized_text, model.config.max_wav_duration
                )
                duration_frames = max(
                    1,
                    int(
                        duration_seconds
                        * model.config.sampling_rate
                        // model.config.latent_hop
                    ),
                )
                output = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    prompt_audio=None,
                    duration=duration_frames,
                    steps=steps,
                    cfg_strength=guidance_strength,
                    guidance_method="cfg",
                )
                waveform = output.waveform.squeeze().cpu().numpy()
            elif isinstance(reference_audio, bytes):
                with tempfile.TemporaryDirectory() as directory:
                    audio_path = Path(directory) / "reference.wav"
                    audio_path.write_bytes(reference_audio)
                    waveform = infer_one(
                        text,
                        reference_text,
                        str(audio_path),
                        model,
                        tokenizer,
                        concrete_device,
                        steps,
                        guidance_strength,
                        "apg",
                    )
            else:
                waveform = infer_one(
                    text,
                    reference_text,
                    reference_audio,
                    model,
                    tokenizer,
                    concrete_device,
                    steps,
                    guidance_strength,
                    "apg",
                )
        waveform_array = np.asarray(waveform, dtype=np.float32)
        if waveform_array.size == 0 or not np.isfinite(waveform_array).all():
            raise RuntimeError("AudioDiT returned an empty or non-finite waveform")
        else:
            pass
        payload.data = audio_waveform_payload(
            waveform_array, sample_rate=model.config.sampling_rate, modality="audio"
        )
        return payload

    return SimpleScheduler(infer)
