# SPDX-License-Identifier: Apache-2.0
"""Single-GPU AudioDiT speech pipeline."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, PipelineConfig, StageConfig


class AudioDiTPipelineConfig(PipelineConfig):
    """Serve both AudioDiT checkpoint sizes with one serialized worker."""

    architecture: ClassVar[str] = "AudioDiTForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    speech_reference_text_required: ClassVar[bool] = True
    model_path: str
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            StageConfig(
                name="inference",
                process="pipeline",
                factory_path=(
                    "sglang_omni.models.longcat_audiodit.stages.create_inference_executor"
                ),
                factory=FactoryArgs(steps=16, guidance_strength=4.0, seed=1024),
                gpu=0,
                terminal=True,
            )
        ]
    )


EntryClass = AudioDiTPipelineConfig
