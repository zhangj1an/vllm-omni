# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from vllm_omni.diffusion.registry import DiffusionModelRegistry

AUDIOX_TASKS = ("t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m")


def _write_minimal_audiox_sharded_stub(root: Path) -> None:
    """Create strict AudioX sharded layout skeleton (pipeline init does not read shard bytes)."""
    (root / "transformer").mkdir(parents=True, exist_ok=True)
    (root / "conditioners").mkdir(parents=True, exist_ok=True)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    (root / "conditioners" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    idx = {
        "_class_name": "AudioXPipeline",
        "config": "config.json",
        "weight_layout": "vllm_omni_component_sharded",
        "transformer_weights": "transformer/diffusion_pytorch_model.safetensors",
        "conditioners_weights": "conditioners/diffusion_pytorch_model.safetensors",
    }
    (root / "model_index.json").write_text(json.dumps(idx), encoding="utf-8")


def test_diffusion_registry_loads_audiox_pipeline_class():
    cls = DiffusionModelRegistry._try_load_model_cls("AudioXPipeline")
    assert cls is not None
    assert cls.__name__ == "AudioXPipeline"


def test_audiox_pipeline_audio_source_resolution_order():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    raw = AudioXPipeline.__new__(AudioXPipeline)
    raw.od_config = OmniDiffusionConfig(model="x", audiox_reference_audio_path="/default.wav")
    sp = OmniDiffusionSamplingParams()
    assert raw._resolve_audio_source({}, {}, sp, 0) == "/default.wav"

    sp2 = OmniDiffusionSamplingParams(audiox_audio_path="/req.wav")
    assert raw._resolve_audio_source({}, {}, sp2, 0) == "/req.wav"
    assert raw._resolve_audio_source({}, {"audio_path": "/extra.wav"}, sp2, 0) == "/extra.wav"

    mm_prompt = {"prompt": "p", "multi_modal_data": {"audio": "/mm.wav"}}
    assert raw._resolve_audio_source(mm_prompt, {"audio_path": "/extra.wav"}, sp2, 0) == "/mm.wav"


def test_audiox_pipeline_audio_prompt_from_tensor():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    raw = AudioXPipeline.__new__(AudioXPipeline)
    raw.od_config = OmniDiffusionConfig(model="x")
    raw._audio_conditioning_samples = 80
    sp = OmniDiffusionSamplingParams()
    wav = torch.zeros(2, 80)
    tensors = raw._audio_prompt_tensors(
        raw_prompts=[{"prompt": "x", "multi_modal_data": {"audio": wav}}],
        extra={},
        sp=sp,
        seconds_start=0.0,
        sample_rate=80,
        device=torch.device("cpu"),
    )
    assert tensors[0].shape == (2, 80)


def test_audiox_pipeline_requires_model_files(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    cfg = OmniDiffusionConfig(model=str(tmp_path), model_class_name="AudioXPipeline")
    with pytest.raises(FileNotFoundError, match="model_index.json|component-sharded|vLLM-Omni"):
        from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

        AudioXPipeline(od_config=cfg)


def test_audiox_pipeline_normalize_task_and_v2_requires_video():
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    assert AudioXPipeline._normalize_task(" V2A ") == "v2a"
    assert AudioXPipeline._text_for_task("v2a", "hello") == ""
    assert AudioXPipeline._text_for_task("tv2a", "hello") == "hello"
    assert AudioXPipeline._text_for_task("tv2m", "") == ""

    with pytest.raises(ValueError, match="non-empty text prompt"):
        AudioXPipeline._ensure_text_video_prompts("tv2a", ["  ", "ok"])

    raw = AudioXPipeline.__new__(AudioXPipeline)
    with pytest.raises(ValueError, match="requires video"):
        AudioXPipeline._video_feature_tensors(
            raw,
            task_norm="v2a",
            raw_prompts=[{"prompt": ""}],
            extra={},
            seconds_start=0.0,
            target_fps=2,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("task_norm", ["v2m", "tv2a"])
def test_audiox_pipeline_video_tasks_load_mm_tensor(task_norm: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    raw = AudioXPipeline.__new__(AudioXPipeline)
    vt = torch.zeros(4, 3, 224, 224)
    prompt = "ocean waves" if task_norm.startswith("tv2") else ""
    tensors = AudioXPipeline._video_feature_tensors(
        raw,
        task_norm=task_norm,
        raw_prompts=[{"prompt": prompt, "multi_modal_data": {"video": vt}}],
        extra={},
        seconds_start=0.0,
        target_fps=4,
        device=torch.device("cpu"),
    )
    assert tensors[0].shape == (10 * 4, 3, 224, 224)


def test_audiox_pipeline_tv2_requires_video():
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    raw = AudioXPipeline.__new__(AudioXPipeline)
    with pytest.raises(ValueError, match="requires video"):
        AudioXPipeline._video_feature_tensors(
            raw,
            task_norm="tv2m",
            raw_prompts=[{"prompt": "music"}],
            extra={},
            seconds_start=0.0,
            target_fps=2,
            device=torch.device("cpu"),
        )


def test_audiox_inference_module_importable_without_external_audiox():
    """Regression: AudioX runtime entry point is vendored in this package."""
    from vllm_omni.diffusion.models.audiox import audiox_runtime

    assert hasattr(audiox_runtime, "generate_diffusion_cond")


@pytest.mark.parametrize("task", AUDIOX_TASKS)
def test_audiox_normalize_task_all_six(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    assert AudioXPipeline._normalize_task(f" {task.upper()} ") == task


@pytest.mark.parametrize("task", ("v2a", "v2m"))
def test_audiox_text_for_task_video_only_clears_text(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    assert AudioXPipeline._text_for_task(task, "should be ignored") == ""


@pytest.mark.parametrize("task", ("t2a", "t2m", "tv2a", "tv2m"))
def test_audiox_text_for_task_preserves_prompt(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    assert AudioXPipeline._text_for_task(task, "keep me") == "keep me"


@pytest.mark.parametrize("task", ("t2a", "t2m"))
def test_audiox_text_only_tasks_zero_video_shape(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    raw = AudioXPipeline.__new__(AudioXPipeline)
    fps, seconds = 5, 2.0
    tensors = AudioXPipeline._video_feature_tensors(
        raw,
        task_norm=task,
        raw_prompts=["irrelevant"],
        extra={},
        seconds_start=0.0,
        target_fps=fps,
        device=torch.device("cpu"),
    )
    n_frames = int(10.0 * fps)
    assert tensors[0].shape == (n_frames, 3, 224, 224)
    assert tensors[0].abs().max().item() == 0.0


@pytest.mark.parametrize("task", ("tv2a", "tv2m"))
def test_audiox_tv_tasks_require_nonempty_prompt(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    with pytest.raises(ValueError, match="non-empty text prompt"):
        AudioXPipeline._ensure_text_video_prompts(task, ["valid", "  "])


@pytest.mark.parametrize("task", ("v2a", "v2m"))
def test_audiox_v2_tasks_require_video_path_or_mm(task: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    raw = AudioXPipeline.__new__(AudioXPipeline)
    with pytest.raises(ValueError, match="requires video"):
        AudioXPipeline._video_feature_tensors(
            raw,
            task_norm=task,
            raw_prompts=[""],
            extra={},
            seconds_start=0.0,
            target_fps=2,
            device=torch.device("cpu"),
        )


def test_resolve_audiox_bundle_paths_requires_default_config_name(tmp_path):
    from vllm_omni.diffusion.models.audiox.audiox_weights import resolve_audiox_bundle_paths

    (tmp_path / "custom.json").write_text("{}", encoding="utf-8")
    _write_minimal_audiox_sharded_stub(tmp_path)
    idx = {
        "_class_name": "AudioXPipeline",
        "config": "custom.json",
        "weight_layout": "vllm_omni_component_sharded",
        "transformer_weights": "transformer/diffusion_pytorch_model.safetensors",
        "conditioners_weights": "conditioners/diffusion_pytorch_model.safetensors",
    }
    (tmp_path / "model_index.json").write_text(json.dumps(idx), encoding="utf-8")

    with pytest.raises(ValueError, match="config"):
        resolve_audiox_bundle_paths(str(tmp_path))


def _audiox_pipeline_stub_for_forward():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    p = AudioXPipeline.__new__(AudioXPipeline)
    p.od_config = OmniDiffusionConfig(model="x")
    p.device = torch.device("cpu")
    # ``forward`` calls ``_conditioning_dtype()`` which reads ``next(_model.parameters())``.
    object.__setattr__(p, "_model", torch.nn.Linear(1, 1))
    p._sample_rate = 100
    p._sample_size = 200
    p._audio_conditioning_samples = 200
    p._target_fps = 5
    p._model_type = "diffusion_cond"
    # Forward now encodes conditioning tensors before diffuse().
    object.__setattr__(p, "_encode_conditioning_tensors", mock.Mock(return_value={"dummy": torch.zeros(1)}))
    p._generate_diffusion_cond = mock.Mock(return_value=torch.zeros(1, 2, 32))
    return p


@pytest.mark.parametrize(
    "task,expected_text",
    [
        ("t2a", "city ambience"),
        ("t2m", "waltz"),
        ("v2a", ""),
        ("v2m", ""),
        ("tv2a", "wind in trees"),
        ("tv2m", "symphony"),
    ],
)
def test_audiox_forward_all_six_tasks_conditioning_text(task: str, expected_text: str):
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    vt = torch.zeros(10, 3, 224, 224)
    if task in ("t2a", "t2m"):
        prompts: list = [expected_text]
        extra: dict = {}
    elif task in ("v2a", "v2m"):
        prompts = [{"prompt": "", "multi_modal_data": {"video": vt}}]
        extra = {}
    else:
        prompts = [{"prompt": expected_text, "multi_modal_data": {"video": vt}}]
        extra = {}

    sp = OmniDiffusionSamplingParams(
        audiox_task=task,
        guidance_scale=1.0,
        num_inference_steps=4,
        seed=0,
        extra_args=extra,
    )
    req = OmniDiffusionRequest(prompts=prompts, sampling_params=sp, request_ids=["0"])

    pipe = _audiox_pipeline_stub_for_forward()
    out = AudioXPipeline.forward(pipe, req)

    assert out.custom_output.get("audiox_task") == task
    pipe._generate_diffusion_cond.assert_called_once()
    kwargs = pipe._generate_diffusion_cond.call_args.kwargs
    assert "dummy" in kwargs["conditioning_tensors"]
    assert torch.equal(kwargs["conditioning_tensors"]["dummy"], torch.zeros(1))
    assert kwargs["negative_conditioning_tensors"] is None


def test_audiox_forward_negative_conditioning_when_cfg_and_neg_prompt():
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    vt = torch.zeros(8, 3, 224, 224)
    prompts = [
        {
            "prompt": "p",
            "negative_prompt": "bad",
            "multi_modal_data": {"video": vt},
        }
    ]
    sp = OmniDiffusionSamplingParams(
        audiox_task="tv2a",
        guidance_scale=4.0,
        num_inference_steps=2,
        extra_args={},
    )
    req = OmniDiffusionRequest(prompts=prompts, sampling_params=sp, request_ids=["0"])

    pipe = _audiox_pipeline_stub_for_forward()
    AudioXPipeline.forward(pipe, req)

    kwargs = pipe._generate_diffusion_cond.call_args.kwargs
    assert "dummy" in kwargs["negative_conditioning_tensors"]
    assert torch.equal(kwargs["negative_conditioning_tensors"]["dummy"], torch.zeros(1))


def test_omni_diffusion_sampling_params_has_audiox_fields():
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sp = OmniDiffusionSamplingParams(audiox_task="t2a", audiox_audio_path="/x.wav")
    assert sp.audiox_task == "t2a"
    assert sp.audiox_audio_path == "/x.wav"


def test_audiox_pre_process_func_normalizes_str_prompts(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import get_audiox_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cfg = {"sample_rate": 80, "sample_size": 800, "video_fps": 5, "model": {}}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    _write_minimal_audiox_sharded_stub(tmp_path)
    od = OmniDiffusionConfig(model=str(tmp_path), model_class_name="AudioXPipeline")
    pre = get_audiox_pre_process_func(od)
    sp = OmniDiffusionSamplingParams(audiox_task="t2a")
    req = OmniDiffusionRequest(prompts=["  hi  "], sampling_params=sp, request_ids=["0"])
    out = pre(req)
    assert out.prompts[0]["prompt"] == "hi"
    assert out.prompts[0]["multi_modal_data"] == {}
    assert "audiox_preprocess" in out.prompts[0]["additional_information"]


def test_audiox_pipeline_init_promotes_inner_model_to_float32(tmp_path, monkeypatch):
    """Regression: full ``model.to(torch.float32)`` so MAF / T5 paths do not mix bf16 vs fp32."""
    import vllm_omni.diffusion.models.audiox.pipeline_audiox as audiox_factory

    monkeypatch.setattr(
        audiox_factory,
        "create_model_from_config",
        lambda _cfg, od_config=None: torch.nn.Linear(2, 2, dtype=torch.bfloat16),
    )

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    _write_minimal_audiox_sharded_stub(tmp_path)
    ref = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    cfg = OmniDiffusionConfig(model=str(tmp_path), model_class_name="AudioXPipeline")
    pipe = AudioXPipeline(od_config=cfg)
    sd = ref.state_dict()
    pipe.load_weights((f"_model.{k}", v) for k, v in sd.items())

    assert pipe._conditioning_dtype() == torch.float32
    assert all(p.dtype == torch.float32 for p in pipe._model.parameters())


def test_audiox_pipeline_conditioning_dtype_follows_first_float_parameter():
    from vllm_omni.diffusion.models.audiox.pipeline_audiox import AudioXPipeline

    raw = AudioXPipeline.__new__(AudioXPipeline)
    object.__setattr__(raw, "_model", torch.nn.Linear(1, 1, dtype=torch.bfloat16))
    assert raw._conditioning_dtype() == torch.bfloat16

    object.__setattr__(raw, "_model", torch.nn.Linear(1, 1, dtype=torch.float32))
    assert raw._conditioning_dtype() == torch.float32
