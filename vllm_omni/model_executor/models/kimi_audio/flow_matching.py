"""Kimi-Audio flow-matching DiT (prefix-streaming)."""

import copy
import logging
import time
from functools import lru_cache

import torch
import torch.nn as nn
import yaml
from diffusers.models.embeddings import (
    Timesteps,
    get_1d_rotary_pos_embed,
    get_timestep_embedding,
)

from .dit_block import DiTBlock, FinalLayer

try:
    from torchdyn.core import NeuralODE

    NEURALODE_INSTALLED = True
except ImportError:
    NEURALODE_INSTALLED = False

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cached_zeros(numel, device="cpu", dtype=torch.float32):
    return torch.zeros(numel, device=device, dtype=dtype)


class TimestepEmbedder(nn.Module):
    """MLP is kept as ``nn.Sequential`` so state-dict keys ``mlp.0.*`` /
    ``mlp.2.*`` match Moonshot checkpoints."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = self.time_proj(t)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class SinusoidalPositionalEmbedding(nn.Module):
    """Frozen sincos table with fairseq-style cumsum positions: each
    non-padding token gets its 1-based rank within the chunk, not its
    absolute id."""

    def __init__(self, embedding_dim: int, padding_idx: int, init_size: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        table = get_timestep_embedding(
            torch.arange(init_size, dtype=torch.float32),
            embedding_dim=embedding_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=1,
            scale=1,
        )
        if padding_idx is not None:
            table[padding_idx] = 0
        self.register_buffer("weight", table, persistent=False)

    def forward(self, position_ids):
        mask = position_ids.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return self.weight[positions]


class DiTPrefix(nn.Module):
    """Flow-matching DiT with streaming KV cache and reference-audio prefill."""

    def __init__(
        self,
        input_size,
        output_size,
        semantic_vocab_size,
        hidden_size=1024,
        depth=12,
        num_heads=4,
        # mlp related
        mlp_ratio=4.0,
        # rope
        use_rope=False,
        rope_params={
            "max_position_embeddings": 4096,
            "rope_base": 10000.0,
            "rope_interpolation_factor": 1.0,
        },
        position_embedding_type="sincos",
        max_seq_len=4096,
        prompt_cfg_dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.prompt_cfg_dropout = prompt_cfg_dropout

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.semantic_token_embedding = nn.Embedding(semantic_vocab_size, hidden_size)

        self.input_linear = nn.Linear(input_size, hidden_size)

        # position embedding
        if position_embedding_type == "learnable":
            self.position_embedding = nn.Embedding(max_seq_len + 1, hidden_size)
        elif position_embedding_type == "sincos":
            self.position_embedding = SinusoidalPositionalEmbedding(
                hidden_size, 0, max_seq_len + 1
            )
        elif position_embedding_type == "skip":
            self.position_embedding = None
        else:
            raise NotImplementedError(
                "Position embedding type: {} not implemented.".format(
                    position_embedding_type
                )
            )

        self.use_rope = use_rope

        if self.use_rope:
            assert (
                hidden_size % num_heads == 0
            ), "Hidden size must be divisible by num_heads for rope position embedding."
            rope_dim = hidden_size // num_heads

            self.rotary_pos_emb = get_1d_rotary_pos_embed(
                dim=rope_dim,
                pos=rope_params["max_position_embeddings"],
                theta=rope_params["rope_base"],
                use_real=False,
                linear_factor=float(rope_params["rope_interpolation_factor"]),
            )
            if max_seq_len < rope_params["max_position_embeddings"]:
                self.rotary_pos_emb = self.rotary_pos_emb[:max_seq_len].clone()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, output_size)
        self.initialize_weights()

        self.sigma_min = 1e-4
        self.t_min = 0.0
        self.t_max = 1.0 - self.sigma_min
        self._neural_ode = None

        # CFG must be False for streaming (enforced in _ode_step); fields
        # kept so from_pretrained can accept the original signature.
        self.use_cfg = False
        self.use_cfg_rescale = True
        self.cfg_init = 1.0
        self.cfg_scale = 4.0
        self.cfg_schedule = "linear"
        self.cfg_token_id = 0

        self.incremental_state: dict = {}
        self.kv_cache_tokens = 0
        self.position_ids = None
        self.seq_len = None
        self.x_mask = None
        self.x_cond = None
        self.cu_seqlens = None
        self.cu_maxlen = None
        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None

        self.max_kv_cache_tokens = 900
        self.max_prompt_chunk = 2
        self.reserve_kv_cache_tokens = 0
        self.start_position_id = 0
        self.condition_cache: dict = {"previous_seqlen": 0}

        self.normalize_mel = False
        self.mel_mean = None
        self.mel_std = None

        self.dtype = torch.bfloat16

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _predict_velocity(
        self,
        x,
        position_ids,
        t,
        condition,
        seq_len,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        mask,
        incremental_state=None,
        nopadding=True,
    ):
        condition = self.semantic_token_embedding(condition)

        x = self.input_linear(x)

        if self.position_embedding is not None:
            position_emb = self.position_embedding(position_ids)
            x = x + position_emb

        if self.use_rope:
            bsz, seqlen = position_ids.shape
            if self.rotary_pos_emb.device != position_ids.device:
                self.rotary_pos_emb = self.rotary_pos_emb.to(position_ids.device)
            rotary_pos_emb = torch.zeros(
                (bsz, seqlen, self.rotary_pos_emb.shape[1]),
                dtype=self.rotary_pos_emb.dtype,
                device=self.rotary_pos_emb.device,
            )
            for b in range(bsz):
                cur_rope = rotary_pos_emb[b]
                cur_position_ids = position_ids[b]
                cur_rope[:] = self.rotary_pos_emb[cur_position_ids]
        else:
            rotary_pos_emb = None

        t = self.t_embedder(t)
        c = t.unsqueeze(1) + condition

        for block_idx, block in enumerate(self.blocks):
            if incremental_state is not None:
                if block_idx not in incremental_state:
                    incremental_state[block_idx] = {}
                incr = incremental_state[block_idx]
            else:
                incr = None

            x = block(
                x=x,
                c=c,
                seq_len=seq_len,
                cu_seqlens=cu_seqlens,
                cu_maxlen=cu_maxlen,
                cu_seqlens_k=cu_seqlens_k,
                cu_maxlen_k=cu_maxlen_k,
                mask=mask,
                rotary_pos_emb=rotary_pos_emb,
                incremental_state=incr,
                nopadding=nopadding,
            )

        return self.final_layer(x, c)

    forward = _predict_velocity

    def set_conditions(self, x_mask, x_cond, start_position_id, cache=None):
        if cache is None:
            cache = {}
        if self.use_cfg:
            raise NotImplementedError("CFG is not supported in streaming detokenizer.")

        self.x_mask = x_mask
        self.x_cond = x_cond

        position_ids_cur = list(
            range(start_position_id, self.x_cond.shape[1] + start_position_id)
        )
        position_ids = torch.tensor([position_ids_cur])

        self.position_ids = position_ids.to(self.x_cond.device).long()
        self.seq_len = (
            torch.Tensor([position_ids.shape[1]]).to(self.x_cond.device).long()
        )

        cu_seqlens = torch.cumsum(self.seq_len, dim=0)
        self.cu_seqlens = torch.cat(
            [torch.Tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0
        ).int()
        self.cu_maxlen = self.seq_len.cpu().max()

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens
            self.cu_maxlen_k = self.cu_maxlen
            previous_seqlen = self.seq_len
        else:
            previous_seqlen_old = cache["previous_seqlen"]
            previous_seqlen = previous_seqlen_old + self.seq_len
            cu_seqlens_k = torch.cumsum(previous_seqlen, dim=0)
            self.cu_seqlens_k = torch.cat(
                [torch.Tensor([0]).to(cu_seqlens_k.device), cu_seqlens_k], dim=0
            ).int()
            self.cu_maxlen_k = previous_seqlen.cpu().max()
        self.previous_seqlen = previous_seqlen
        return {"previous_seqlen": previous_seqlen}

    def update_incremental_state(self, condition_cache=None):
        if condition_cache is None:
            condition_cache = self.condition_cache
        assert (
            self.reserve_kv_cache_tokens <= self.max_kv_cache_tokens
        ), "reserve_kv_cache_tokens must be <= max_kv_cache_tokens"

        for _, layer_cache in self.incremental_state.items():
            layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"]["cur_k"]
            layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"]["cur_v"]

            self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            if self.kv_cache_tokens > self.max_kv_cache_tokens:
                reserve_tokens_excludeprompt = (
                    self.max_kv_cache_tokens - self.reserve_kv_cache_tokens
                )

                if self.reserve_kv_cache_tokens == 0:
                    layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"][
                        "prev_k"
                    ][:, -reserve_tokens_excludeprompt:]
                    layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"][
                        "prev_v"
                    ][:, -reserve_tokens_excludeprompt:]
                elif reserve_tokens_excludeprompt == 0:
                    layer_cache["attn_kvcache"]["prev_k"] = layer_cache["attn_kvcache"][
                        "prev_k"
                    ][:, : self.reserve_kv_cache_tokens]
                    layer_cache["attn_kvcache"]["prev_v"] = layer_cache["attn_kvcache"][
                        "prev_v"
                    ][:, : self.reserve_kv_cache_tokens]
                else:
                    layer_cache["attn_kvcache"]["prev_k"] = torch.cat(
                        [
                            layer_cache["attn_kvcache"]["prev_k"][
                                :, : self.reserve_kv_cache_tokens
                            ],
                            layer_cache["attn_kvcache"]["prev_k"][
                                :, -reserve_tokens_excludeprompt:
                            ],
                        ],
                        dim=1,
                    )
                    layer_cache["attn_kvcache"]["prev_v"] = torch.cat(
                        [
                            layer_cache["attn_kvcache"]["prev_v"][
                                :, : self.reserve_kv_cache_tokens
                            ],
                            layer_cache["attn_kvcache"]["prev_v"][
                                :, -reserve_tokens_excludeprompt:
                            ],
                        ],
                        dim=1,
                    )

                bsz = layer_cache["attn_kvcache"]["prev_k"].shape[0]
                self.previous_seqlen = (
                    torch.Tensor(
                        [
                            layer_cache["attn_kvcache"]["prev_k"].shape[1]
                            for _ in range(bsz)
                        ]
                    )
                    .to(layer_cache["attn_kvcache"]["prev_k"].device)
                    .long()
                )
                condition_cache["previous_seqlen"] = self.previous_seqlen
                self.kv_cache_tokens = layer_cache["attn_kvcache"]["prev_k"].shape[1]

            layer_cache["attn_kvcache"].pop("cur_k")
            layer_cache["attn_kvcache"].pop("cur_v")

    def clear_all_states(self):
        self.incremental_state = {}
        self.kv_cache_tokens = 0
        self.cu_seqlens = None
        self.cu_maxlen = None
        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None
        self.start_position_id = 0
        self.condition_cache = {"previous_seqlen": 0}

    def snapshot_streaming_state(self):
        """Save streaming state (NOT model weights) so the detokenizer can
        restore post-prefill state between generations from the same
        reference audio."""
        return {
            "start_position_id": self.start_position_id,
            "condition_cache": copy.deepcopy(self.condition_cache),
            "ode_wrapper": {
                "incremental_state": copy.deepcopy(self.incremental_state),
                "kv_cache_tokens": copy.deepcopy(self.kv_cache_tokens),
                "cu_seqlens": copy.deepcopy(self.cu_seqlens),
                "cu_maxlen": copy.deepcopy(self.cu_maxlen),
                "cu_seqlens_k": copy.deepcopy(self.cu_seqlens_k),
                "cu_maxlen_k": copy.deepcopy(self.cu_maxlen_k),
                "previous_seqlen": copy.deepcopy(self.previous_seqlen),
            },
        }

    def restore_streaming_state(self, state):
        if state is None:
            return
        self.start_position_id = state["start_position_id"]
        self.condition_cache = state["condition_cache"]
        ow = state["ode_wrapper"]
        self.incremental_state = ow["incremental_state"]
        self.kv_cache_tokens = ow["kv_cache_tokens"]
        self.cu_seqlens = ow["cu_seqlens"]
        self.cu_maxlen = ow["cu_maxlen"]
        self.cu_seqlens_k = ow["cu_seqlens_k"]
        self.cu_maxlen_k = ow["cu_maxlen_k"]
        self.previous_seqlen = ow["previous_seqlen"]

    def _ode_step(self, t, x):
        """Predict velocity at ``t``. ``t * 1000`` matches upstream Moonshot:
        the DiT was trained on integer timestep IDs, not [0, 1]."""
        t_long = (
            _cached_zeros(x.shape[0], device=x.device, dtype=torch.long)
            + (t * 1000).long()
        )
        if self.use_cfg:
            raise NotImplementedError("CFG is not supported in streaming detokenizer.")
        return self._predict_velocity(
            x=x,
            condition=self.x_cond,
            t=t_long,
            position_ids=self.position_ids,
            cu_seqlens=self.cu_seqlens,
            cu_maxlen=self.cu_maxlen,
            cu_seqlens_k=self.cu_seqlens_k,
            cu_maxlen_k=self.cu_maxlen_k,
            incremental_state=self.incremental_state,
            nopadding=True,
            mask=None,
            seq_len=None,
        )

    def _solve_euler(self, time_steps, xt):
        h = (self.t_max - self.t_min) / len(time_steps)
        h = h * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
        for t in time_steps:
            xt = xt + h * self._ode_step(t, xt)
        return xt

    def _solve_neural_ode(self, time_steps, xt):
        if not NEURALODE_INSTALLED:
            raise ImportError("NeuralODE is not installed; install torchdyn first.")
        if self._neural_ode is None:
            self._neural_ode = NeuralODE(
                lambda t, x, args=None: self._ode_step(t, x),
                solver="euler",
                sensitivity="adjoint",
                atol=self.sigma_min,
                rtol=self.sigma_min,
            )
        _, traj = self._neural_ode(xt, time_steps)
        return traj[-1]

    @torch.inference_mode()
    def infer_chunk(
        self,
        xt_chunk,
        semantic_tokens_chunk,
        start_position_id,
        cache=None,
        look_ahead_tokens=0,
        ode_steps=15,
        verbose=False,
        ode_solver="neural_ode_euler",
    ):
        bs = 1
        device = next(self.parameters()).device

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(device)
        xt_chunk = xt_chunk.unsqueeze(0).to(device).to(self.dtype)

        t_span = torch.linspace(0, 1, ode_steps)
        x_mask = torch.zeros(bs, xt_chunk.shape[1], device=device).bool()

        cache_ret = self.set_conditions(
            x_mask=x_mask,
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        if verbose:
            t_start = time.time()
        if ode_solver == "neural_ode_euler":
            x_t = self._solve_neural_ode(t_span, xt_chunk)
        elif ode_solver == "naive_euler":
            x_t = self._solve_euler(t_span, xt_chunk)
        else:
            raise NotImplementedError(
                "ode_solver should be in ('neural_ode_euler', 'naive_euler')"
            )

        if look_ahead_tokens > 0:
            semantic_tokens_left = semantic_tokens_chunk.view(-1)[-look_ahead_tokens:]
            cache["semantic_token"] = semantic_tokens_left
            x_t_ret = x_t[:, :-look_ahead_tokens, :]
        else:
            x_t_ret = x_t

        if look_ahead_tokens > 0:
            x_mask = torch.zeros(
                bs, xt_chunk.shape[1] - look_ahead_tokens, device=device
            ).bool()
            self.condition_cache = self.set_conditions(
                x_mask=x_mask,
                x_cond=semantic_tokens_chunk[:, :-look_ahead_tokens],
                start_position_id=start_position_id,
                cache=self.condition_cache,
            )
            # Prime the truncated KV cache with a fake near-final timestep.
            self._ode_step(torch.Tensor([0.999]).to(x_t_ret.device), x_t_ret)
        else:
            self.condition_cache = cache_ret

        if verbose:
            logger.info(f"[ODE Chunk] Time cost: {time.time() - t_start}")

        if self.normalize_mel:
            x_t_ret = x_t_ret * self.mel_std + self.mel_mean
        return x_t_ret.squeeze(0)

    @torch.inference_mode()
    def prefill(self, mel, semantic_token, chunk_size=150, verbose=False):
        """Fill the KV cache from a reference audio prompt."""
        assert mel.dim() == 2
        assert semantic_token.dim() == 1
        assert (
            semantic_token.shape[0] == mel.shape[0]
        ), "Semantic token and mel shape mismatch"
        seq_len = mel.shape[0]
        num_chunks = min(seq_len // chunk_size, self.max_prompt_chunk)
        start_pos = seq_len - num_chunks * chunk_size

        res_mel = mel[:start_pos, :]
        res_semantic_token = semantic_token[:start_pos]
        self.prefill_chunk(
            res_mel, res_semantic_token, start_position_id=self.start_position_id
        )
        self.start_position_id += start_pos
        self.update_incremental_state()
        self.reserve_kv_cache_tokens += self.kv_cache_tokens

        if verbose:
            logger.info("Prefilling prompt with {} chunks".format(num_chunks))
            t_start = time.time()

        for chunk_id in range(num_chunks):
            start = start_pos + chunk_id * chunk_size
            end = start + chunk_size
            mel_chunk = mel[start:end, :]
            semantic_token_chunk = semantic_token[start:end]

            self.prefill_chunk(
                mel_chunk,
                semantic_token_chunk,
                start_position_id=self.start_position_id,
            )
            self.start_position_id += end - start

            self.update_incremental_state()
            self.reserve_kv_cache_tokens += self.kv_cache_tokens

        if verbose:
            logger.info("Prefilling done in {:.2f} seconds".format(time.time() - t_start))

    def prefill_chunk(self, mel_chunk, semantic_tokens_chunk, start_position_id=0):
        bs = 1
        device = next(self.parameters()).device

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(device)
        mel_chunk = mel_chunk.unsqueeze(0).to(device).to(self.dtype)

        if self.normalize_mel:
            mel_chunk = (mel_chunk - self.mel_mean) / self.mel_std

        x_mask = torch.zeros(bs, mel_chunk.shape[1], device=device).bool()

        self.condition_cache = self.set_conditions(
            x_mask=x_mask,
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        x_t = torch.Tensor([0.999]).to(device)
        self._ode_step(x_t, mel_chunk)

    @classmethod
    def from_pretrained(
        cls,
        model_config,
        ckpt_path,
        device,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        use_cfg=True,
        use_cfg_rescale=True,
        cfg_init=1.5,
        cfg_scale=7.5,
        cfg_schedule="linear",
    ):
        with open(model_config, "r") as f:
            config = yaml.safe_load(f)
        dit_cfg = config["model"]["dit"]
        dit = cls(
            input_size=dit_cfg["input_size"],
            semantic_vocab_size=dit_cfg["semantic_vocab_size"] + 1,
            hidden_size=dit_cfg["hidden_size"],
            depth=dit_cfg["depth"],
            num_heads=dit_cfg["num_heads"],
            mlp_ratio=dit_cfg["mlp_ratio"],
            use_rope=dit_cfg.get("use_rope", False),
            rope_params=dit_cfg.get(
                "rope_params",
                {
                    "max_position_embeddings": 4096,
                    "rope_base": 10000,
                    "rope_interpolation_factor": 1,
                },
            ),
            position_embedding_type=dit_cfg["position_embedding_type"],
            max_seq_len=dit_cfg["max_seq_len"],
            output_size=dit_cfg["input_size"],
            prompt_cfg_dropout=0,
        )
        cfg_semantic_token_id = dit_cfg["semantic_vocab_size"]

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        speech_model_params = {
            k.replace("speech_model.", ""): v
            for k, v in state_dict.items()
            if "speech_model" in k
        }
        speech_model_params.pop("position_embedding._float_tensor", None)
        dit.load_state_dict(speech_model_params, strict=True)
        logger.info(">>> Loaded checkpoint from %s", ckpt_path)

        dit.max_prompt_chunk = max_prompt_chunk
        dit.max_kv_cache_tokens = max_kv_cache_tokens
        dit.use_cfg = use_cfg
        dit.use_cfg_rescale = use_cfg_rescale
        dit.cfg_init = cfg_init
        dit.cfg_scale = cfg_scale
        dit.cfg_schedule = cfg_schedule
        dit.cfg_token_id = cfg_semantic_token_id
        dit.normalize_mel = config.get("normalize_mel", False)
        dit.mel_mean = config.get("mel_mean")
        dit.mel_std = config.get("mel_std")

        dit.to(device).to(dit.dtype).eval()
        logger.info(
            ">>> DiTPrefix on %s: use_cfg=%s, cfg_init=%s, cfg_scale=%s, cfg_schedule=%s",
            device, use_cfg, cfg_init, cfg_scale, cfg_schedule,
        )
        return dit
