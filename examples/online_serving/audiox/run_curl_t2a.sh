#!/bin/bash
# AudioX text-to-audio request via OpenAI chat endpoint.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8099}"
PROMPT="${PROMPT:-Fireworks burst twice, followed by a period of silence before a clock begins ticking}"
OUTPUT_WAV="${OUTPUT_WAV:-audiox_t2a.wav}"

response="$(curl -sS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"${PROMPT}\"}
        ]
      }
    ],
    \"extra_body\": {
      \"audiox_task\": \"t2a\",
      \"num_inference_steps\": 250,
      \"guidance_scale\": 7.0,
      \"seed\": 42,
      \"seconds_total\": 10.0
    }
  }")"

echo "${response}" | jq .
echo "${response}" | jq -r '.choices[0].message.audio.data' | base64 -d > "${OUTPUT_WAV}"
echo "Saved audio to ${OUTPUT_WAV}"
