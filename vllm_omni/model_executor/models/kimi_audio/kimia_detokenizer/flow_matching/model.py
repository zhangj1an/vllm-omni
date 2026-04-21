import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    Timesteps,
    get_1d_rotary_pos_embed,
    get_timestep_embedding,
)

from .dit_block import DiTBlock, FinalLayer


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    interpolation_factor: int = 1,
    max_seq_length: int = 4096,
):
    """Complex-polar RoPE frequency table. Thin wrapper over diffusers'
    ``get_1d_rotary_pos_embed(..., use_real=False)``; semantically identical
    to the original Moonshot implementation (verified bit-exact).
    """
    freqs_cis = get_1d_rotary_pos_embed(
        dim=dim,
        pos=end,
        theta=theta,
        use_real=False,
        linear_factor=float(interpolation_factor),
    )
    if max_seq_length < end:
        freqs_cis = freqs_cis[:max_seq_length].clone()
    return freqs_cis


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps via a sinusoidal projection + 2-layer MLP.

    The sinusoidal step is delegated to ``diffusers.Timesteps`` (bit-exact to
    Kimi's original formulation with ``flip_sin_to_cos=True`` and
    ``downscale_freq_shift=0``). The MLP is kept as ``nn.Sequential`` so
    state-dict keys (``mlp.0.*``, ``mlp.2.*``) stay compatible with Moonshot
    checkpoints.
    """

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
    """Frozen sincos positional table, equivalent to Kimi's fairseq-style
    embedding at inference.

    The upstream class carried a fairseq ``incremental_state`` + ``timestep``
    interface that ``DiTPrefix`` never exercises — only the batch lookup path
    runs. This implementation builds the table once via diffusers'
    ``get_timestep_embedding(..., downscale_freq_shift=1, flip_sin_to_cos=False)``
    (matches Kimi's ``(half_dim - 1)`` normalization), zeros the
    ``padding_idx`` slot, and indexes it at forward time.

    Stored as a non-persistent buffer so it isn't serialized (the original
    class's ``self.weights`` was also a plain attribute, not a state-dict
    entry).
    """

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
        # Preserve upstream Kimi's ``make_positions`` behavior: for each
        # non-padding entry, use its 1-based rank within the chunk (not the
        # absolute id). This makes the sincos signal chunk-relative; RoPE
        # carries absolute position when ``use_rope=True``.
        mask = position_ids.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return self.weight[positions]


class DiTPrefix(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

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
        ffn_type="conv1d_conv1d",
        ffn_gated_glu=True,
        ffn_act_layer="gelu",
        ffn_conv_kernel_size=5,
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

            self.rotary_pos_emb = precompute_freqs_cis(
                rope_dim,
                rope_params["max_position_embeddings"],
                theta=rope_params["rope_base"],
                interpolation_factor=rope_params["rope_interpolation_factor"],
                max_seq_length=max_seq_len,
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    ffn_type=ffn_type,
                    ffn_conv_kernel_size=ffn_conv_kernel_size,
                    ffn_gated_glu=ffn_gated_glu,
                    ffn_act_layer=ffn_act_layer,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
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

    def forward(
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
        """
        Forward pass of DiT.
        x: (N, T, C) tensor of inputs (latent representations of speech)
        position_ids: (N, T) tensor of positional indices
        t: (N,) tensor of diffusion timesteps
        condition: (N, T) tensor of semantic tokens
        seq_len: (N,) tensor of sequence lengths
        """

        condition = self.semantic_token_embedding(condition)  # (N, T, D)

        x = self.input_linear(x)

        if self.position_embedding is not None:
            position_emb = self.position_embedding(position_ids)
            x = x + position_emb

        # ROPE
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

        t = self.t_embedder(t)  # (N, D)
        c = t.unsqueeze(1) + condition  # (N, T, D)

        for block_idx, block in enumerate(self.blocks):
            # x = block(x, c, attn_mask)  # (N, T, D)
            # XXX mask could be None because we always use full mask

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

        x = self.final_layer(x, c)  # (N, T, C)
        return x
