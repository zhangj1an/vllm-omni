"""Shared audio output post-processing utilities for diffusion pipelines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def build_audio_post_process_func() -> Callable[[torch.Tensor, str], Any]:
    """Create a common post-processor for audio pipeline outputs.

    - ``output_type in {"latent", "pt"}`` returns the tensor unchanged.
    - Any other output type returns CPU float numpy output.
    """

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in ("latent", "pt"):
            return audio
        return audio.detach().cpu().float().numpy()

    return post_process_func
