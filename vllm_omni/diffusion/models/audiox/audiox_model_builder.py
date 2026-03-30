from __future__ import annotations

import typing as tp

import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.audiox.audiox_conditioner import (
    MultiConditioner,
    create_audiox_fixed_conditioner_from_conditioning_config,
)
from vllm_omni.diffusion.models.audiox.audiox_pretransform import (
    AudioXVAE,
    create_pretransform_from_config,
)
from vllm_omni.diffusion.models.audiox.audiox_transformer import MMDiffusionTransformer


def create_model_from_config(model_config, od_config: OmniDiffusionConfig | None = None):
    return create_diffusion_cond_from_config(model_config, od_config=od_config)


class ConditionedDiffusionModel(nn.Module):
    def __init__(
        self,
        *args,
        supports_cross_attention: bool = False,
        supports_input_concat: bool = False,
        supports_global_cond: bool = False,
        supports_prepend_cond: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
        input_concat_cond: torch.Tensor = None,
        global_embed: torch.Tensor = None,
        prepend_cond: torch.Tensor = None,
        prepend_cond_mask: torch.Tensor = None,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = False,
        rescale_cfg: bool = False,
        **kwargs,
    ):
        raise NotImplementedError()


class ConditionedDiffusionModelWrapper(nn.Module):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        conditioner: MultiConditioner,
        io_channels,
        sample_rate,
        min_input_length: int,
        diffusion_objective: tp.Literal["v"] = "v",
        pretransform: AudioXVAE | None = None,
        cross_attn_cond_ids: list[str] | None = None,
        global_cond_ids: list[str] | None = None,
        od_config: OmniDiffusionConfig | None = None,
    ):
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config if od_config is not None else None
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids or []
        self.global_cond_ids = global_cond_ids or []
        self.min_input_length = min_input_length
        from vllm_omni.diffusion.models.audiox.audiox_maf import MAF_Block

        self.maf_block = MAF_Block()

    def get_conditioning_inputs(self, conditioning_tensors: dict[torch.Tensor, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)
                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            video_feature, text_feature, audio_feature = cross_attention_input
            refined_branches = self.maf_block(video_feature, text_feature, audio_feature)
            cross_attention_input = torch.cat(list(refined_branches.values()), dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]
                global_conds.append(global_cond_input)
            global_cond = torch.cat(global_conds, dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_embed": global_cond,
            }
        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_cond_mask": cross_attention_masks,
            "global_embed": global_cond,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


def create_diffusion_cond_from_config(config: dict[str, tp.Any], od_config: OmniDiffusionConfig | None = None):
    model_config = config["model"]
    diffusion_config = model_config["diffusion"]
    diffusion_model_config = diffusion_config["config"]
    diffusion_model_config = dict(diffusion_model_config)
    if diffusion_model_config.get("video_fps", None) is not None:
        diffusion_model_config.pop("video_fps")

    diffusion_build_kwargs = dict(diffusion_model_config)
    if od_config is not None:
        diffusion_build_kwargs["od_config"] = od_config

    diffusion_model = MMDiffusionTransformer(**diffusion_build_kwargs)
    with torch.no_grad():
        for param in diffusion_model.parameters():
            param *= 0.5

    io_channels = model_config["io_channels"]
    sample_rate = config["sample_rate"]

    diffusion_objective: tp.Literal["v"] = "v"

    conditioning_config = model_config["conditioning"]
    conditioner = create_audiox_fixed_conditioner_from_conditioning_config(conditioning_config)

    cross_attention_ids = ["video_prompt", "text_prompt", "audio_prompt"]
    global_cond_ids: list[str] = []

    pretransform_cfg = model_config["pretransform"]
    pretransform = create_pretransform_from_config(pretransform_cfg, sample_rate)
    min_input_length = pretransform.downsampling_ratio

    min_input_length *= diffusion_model.patch_size

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        diffusion_objective=diffusion_objective,
        od_config=od_config,
    )


__all__ = [
    "ConditionedDiffusionModel",
    "ConditionedDiffusionModelWrapper",
    "create_diffusion_cond_from_config",
    "create_model_from_config",
]

