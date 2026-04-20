#!/usr/bin/env bash
# Audio-in / text-out (ASR / audio QA) via vLLM-Omni's /v1/chat/completions.
# Server should be launched with kimi_audio.yaml.
set -euo pipefail

PORT="${PORT:-8091}"
HOST="${HOST:-localhost}"
MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"

thinker_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 512,
  "seed": 42,
  "repetition_penalty": 1.0
}'

request_body=$(cat <<EOF
{
  "model": "moonshotai/Kimi-Audio-7B-Instruct",
  "sampling_params_list": [$thinker_sampling_params],
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
          "text": "Please transcribe the audio."
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
    -d "$request_body" | jq '.choices[0].message.content'
