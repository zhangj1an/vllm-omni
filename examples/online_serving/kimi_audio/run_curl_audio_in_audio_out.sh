#!/usr/bin/env bash
# Audio-in / audio-out via vLLM-Omni's /v1/chat/completions.
# Server should be launched with kimi_audio_audio_out.yaml (or
# kimi_audio_async_chunk.yaml for streaming).
#
# Output: choices[0].message.audio.data is a base64-encoded WAV at 24 kHz
# mono. We decode and save it to ./response.wav.
set -euo pipefail

PORT="${PORT:-8091}"
HOST="${HOST:-localhost}"
MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
OUT_FILE="${OUT_FILE:-response.wav}"

thinker_sampling_params='{
  "temperature": 0.6, "top_p": 0.95, "top_k": 50,
  "max_tokens": 1024, "seed": 42, "repetition_penalty": 1.0
}'
code2wav_sampling_params='{
  "temperature": 0.0, "top_p": 1.0, "top_k": -1,
  "max_tokens": 8192, "seed": 42, "detokenize": false
}'

request_body=$(cat <<EOF
{
  "model": "moonshotai/Kimi-Audio-7B-Instruct",
  "sampling_params_list": [$thinker_sampling_params, $code2wav_sampling_params],
  "modalities": ["text", "audio"],
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {"url": "$MARY_HAD_LAMB_AUDIO_URL"}
        },
        {
          "type": "text",
          "text": "Answer in audio. Briefly summarize what was said."
        }
      ]
    }
  ]
}
EOF
)

curl -sS --retry 3 --retry-delay 3 --retry-connrefused \
    -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$request_body" \
  | jq -r '.choices[0].message.audio.data' \
  | base64 -d > "$OUT_FILE"

echo "Wrote $OUT_FILE"
