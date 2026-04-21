import torch
import os
from .bigvgan_wrapper import KimiBigVGAN
from .flow_matching.model import DiTPrefix


class PrefixStreamingFlowMatchingDetokenizer:
    def __init__(
        self,
        vocoder: KimiBigVGAN,
        fm: DiTPrefix,
        look_ahead_tokens: int = 0,
    ) -> None:
        self.dtype = torch.bfloat16

        self.vocoder = vocoder.to(self.dtype)

        # Attribute name kept for continuity with upstream Moonshot.
        self.semantic_fm = fm

        self.max_pos_size = 4096
        self.is_timbre_semantic_token = False
        self.pre_mel = None
        self.frame_size = 480
        self.pre_wav = None
        self.state_dict_backup = None
        self.hamming_window_cache = {}
        self.previous_chunk_left = None
        self.look_ahead_tokens = look_ahead_tokens

        self.clear_states()

    @classmethod
    def from_pretrained(
        cls,
        vocoder_config,
        vocoder_ckpt,
        fm_config,
        fm_ckpt,
        device,
        look_ahead_tokens=0,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        use_cfg=False,
        use_cfg_rescale=True,
        cfg_init=1.5,
        cfg_scale=7.5,
        cfg_schedule="linear",
    ):
        bigvgan = KimiBigVGAN.load_kimi_checkpoint(vocoder_config, vocoder_ckpt, device)
        semantic_fm = DiTPrefix.from_pretrained(
            fm_config,
            fm_ckpt,
            device,
            max_prompt_chunk=max_prompt_chunk,
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            use_cfg_rescale=use_cfg_rescale,
            cfg_init=cfg_init,
            cfg_schedule=cfg_schedule,
        )
        return cls(bigvgan, semantic_fm, look_ahead_tokens=look_ahead_tokens)

    @torch.inference_mode()
    def detokenize_streaming(
        self,
        semantic_token,
        ode_step=30,
        verbose=False,
        ode_solver="neural_ode_euler",
        is_final=False,
        upsample_factor=1,
    ):
        assert len(semantic_token.shape) == 2 and ode_step > 0
        assert semantic_token.shape[0] == 1

        semantic_token = semantic_token.repeat_interleave(upsample_factor, dim=1)

        semantic_token = semantic_token.squeeze(0)

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is not None:
            semantic_token_previous = self.previous_chunk_left["semantic_token"]
            semantic_token = torch.cat(
                [semantic_token_previous, semantic_token], dim=-1
            )

        x_t_chunk = (
            torch.randn(semantic_token.shape[0], 80)
            .to(semantic_token.device)
            .to(self.dtype)
        )

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is None:
            self.previous_chunk_left = {"semantic_token": None}

        speech_mel = self.semantic_fm.infer_chunk(
            xt_chunk=x_t_chunk,
            semantic_tokens_chunk=semantic_token,
            start_position_id=self.semantic_fm.start_position_id,
            ode_steps=ode_step,
            verbose=verbose,
            look_ahead_tokens=(
                self.look_ahead_tokens * upsample_factor if not is_final else 0
            ),
            cache=self.previous_chunk_left,
            ode_solver=ode_solver,
        )

        chunk_size = speech_mel.shape[0]
        length = speech_mel.shape[0]
        self.semantic_fm.start_position_id += length
        self.semantic_fm.update_incremental_state()
        self.semantic_fm.reserve_kv_cache_tokens += self.semantic_fm.kv_cache_tokens

        # Cross-faded streaming: first chunk returns a half-chunk of audio;
        # later chunks concat with the saved tail and Hamming-window the seam.
        if self.pre_mel is None:
            concat_mel = speech_mel
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)
            if is_final:
                self.clear_states()
                self.state_dict_backup = None
                ret_wav = concat_reconstructed_wav.float()
            else:
                reconstructed_wav = concat_reconstructed_wav[
                    :, : int(self.frame_size * chunk_size // 2)
                ]
                self.pre_wav = concat_reconstructed_wav[
                    :, -int(self.frame_size * chunk_size // 2) :
                ]
                self.pre_mel = speech_mel[-chunk_size // 2 :, :]

                ret_wav = reconstructed_wav.float()
        else:
            concat_mel = torch.cat([self.pre_mel, speech_mel], dim=0)
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)

            if is_final:
                self.clear_states()
                self.state_dict_backup = None
                ret_wav = concat_reconstructed_wav.float()
            else:
                prev_speech_len = self.pre_wav.shape[1]

                if concat_reconstructed_wav.shape[1] > prev_speech_len * 2:
                    gen_speech_len = prev_speech_len * 2
                else:
                    gen_speech_len = concat_reconstructed_wav.shape[1] // 2

                reconstructed_wav = concat_reconstructed_wav[:, :gen_speech_len]

                if gen_speech_len not in self.hamming_window_cache:
                    self.hamming_window_cache[gen_speech_len] = (
                        torch.hamming_window(gen_speech_len)
                        .to(self.dtype)
                        .to(semantic_token.device)
                        .unsqueeze(0)
                    )

                hamming_window = self.hamming_window_cache[gen_speech_len]

                # Apply smoothing to the first half chunk.
                reconstructed_wav[:, : int(gen_speech_len // 2)] = (
                    self.pre_wav[:, : int(gen_speech_len // 2)]
                    * hamming_window[:, -int(gen_speech_len // 2) :]
                    + reconstructed_wav[:, : int(gen_speech_len // 2)]
                    * hamming_window[:, : int(gen_speech_len // 2)]
                )

                res_speech_len = concat_reconstructed_wav.shape[1] - gen_speech_len
                res_mel_len = res_speech_len // self.frame_size

                self.pre_wav = concat_reconstructed_wav[:, -res_speech_len:]
                self.pre_mel = speech_mel[-res_mel_len:, :]
                ret_wav = reconstructed_wav.float()

        if (
            not is_final
            and self.semantic_fm.start_position_id + 2 * chunk_size > self.max_pos_size
        ):
            # Out of position id; reset and restore post-prefill state.
            self.semantic_fm.clear_all_states()
            self.semantic_fm.restore_streaming_state(self.state_dict_backup)

        return ret_wav

    def clear_states(self):
        self.semantic_fm.clear_all_states()
        self.previous_chunk_left = None
        self.pre_mel = None
        self.pre_wav = None


def get_audio_detokenizer(model_path):
    fm_model_config = os.path.join(model_path, "audio_detokenizer", "config.yaml")
    fm_ckpt_path = os.path.join(model_path, "audio_detokenizer", "model.pt")

    bigvgan_config_file = os.path.join(model_path, "vocoder", "config.json")
    bigvgan_ckpt_path = os.path.join(model_path, "vocoder", "model.pt")

    device = torch.cuda.current_device()
    detokenizer = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
        vocoder_config=bigvgan_config_file,
        vocoder_ckpt=bigvgan_ckpt_path,
        max_prompt_chunk=10,  # 10 * 3 = 30s
        fm_config=fm_model_config,
        fm_ckpt=fm_ckpt_path,
        device=device,
        use_cfg=False,
        look_ahead_tokens=12,
    )

    return detokenizer


def detokenize_noref(detokenizer, tokens):
    with torch.no_grad():
        detokenizer.clear_states()
        cache_speech_collection = []
        chunk_size = 150
        first_chunk_size = 100
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens, is_final=tokens.size(1) <= first_chunk_size
        )
        cache_speech_collection.append(gen_speech)
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens, is_final=(i + chunk_size >= res_tokens.size(1))
            )
            cache_speech_collection.append(gen_speech)

        gen_speech_all = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech_all
