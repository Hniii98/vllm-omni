#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

# Step-Audio2 curl client for online serving
# Usage: bash run_curl.sh [audio_to_text|tts|text_to_audio|audio_to_audio|think]

QUERY_TYPE="${1:-audio_to_text}"
API_BASE="${API_BASE:-http://localhost:8092}"
MODALITIES=null

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(audio_to_text|tts|text_to_audio|audio_to_audio|think)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [audio_to_text|tts|text_to_audio|audio_to_audio|think]"
    echo "  audio_to_text:  Speech recognition (ASR, Step-Audio-2-mini)"
    echo "  tts:            Verbatim TTS via /v1/audio/speech (recommended, Step-Audio-2-mini-Base)"
    echo "  text_to_audio:  Text-to-speech via /v1/chat/completions"
    echo "  audio_to_audio: Speech-to-speech (S2ST, Step-Audio-2-mini)"
    echo "  think:          Two-pass reasoning + speech (Step-Audio-2-mini-Think)"
    exit 1
fi

# Default test audio URL
MARY_HAD_LAMB_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"

# Recommended TTS path: /v1/audio/speech with Step-Audio-2-mini-Base.
if [[ "$QUERY_TYPE" == "tts" ]]; then
    MODEL="${MODEL:-stepfun-ai/Step-Audio-2-mini-Base}"
    OUTPUT_WAV="${OUTPUT_WAV:-output_tts.wav}"
    echo "Query type: tts (/v1/audio/speech)"
    echo "Model: $MODEL"
    echo "API base: $API_BASE"
    curl -sS -X POST "${API_BASE}/v1/audio/speech" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$MODEL" '{
          model: $model,
          input: "你好呀，我是你的AI助手，很高兴认识你！",
          voice: "default"
        }')" \
        --output "$OUTPUT_WAV"
    echo "Audio saved to: $OUTPUT_WAV"
    echo "Done!"
    exit 0
fi

# Two-pass reasoning + speech dialogue for Step-Audio-2-mini-Think.
if [[ "$QUERY_TYPE" == "think" ]]; then
    MODEL="${MODEL:-stepfun-ai/Step-Audio-2-mini-Think}"
    SYSTEM_PROMPT="你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，你能听见用户的声音特征并在思维过程中描述出来，请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"
    echo "Query type: think (2-pass /v1/chat/completions)"
    echo "Model: $MODEL"
    echo "API base: $API_BASE"

    echo "Sending Pass 1 (reasoning)..."
    pass1_payload=$(jq -n \
        --arg model "$MODEL" \
        --arg sys "$SYSTEM_PROMPT" \
        --arg url "$MARY_HAD_LAMB_URL" \
        '{
          model: $model,
          modalities: ["text"],
          temperature: 0.7,
          max_tokens: 2048,
          seed: 42,
          stop: ["</think>"],
          continue_final_message: true,
          add_generation_prompt: false,
          messages: [
            {role: "system", content: $sys},
            {role: "user", content: [{type: "audio_url", audio_url: {url: $url}}]},
            {role: "assistant", content: "<think>"}
          ]
        }')
    pass1_output=$(curl -sS -X POST "${API_BASE}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$pass1_payload")
    pass1_error=$(echo "$pass1_output" | jq -r '.error // empty')
    if [[ -n "$pass1_error" ]]; then
        echo "Pass 1 Error: $pass1_error"
        exit 1
    fi
    think_text=$(echo "$pass1_output" | jq -r '.choices[0].message.content // empty')
    echo "Reasoning: <think>${think_text}</think>"

    echo ""
    echo "Sending Pass 2 (spoken response)..."
    assistant_pass2="<think>${think_text}</think>"$'\n\n'"<tts_start>"
    pass2_payload=$(jq -n \
        --arg model "$MODEL" \
        --arg sys "$SYSTEM_PROMPT" \
        --arg url "$MARY_HAD_LAMB_URL" \
        --arg assistant "$assistant_pass2" \
        '{
          model: $model,
          temperature: 0.7,
          max_tokens: 2048,
          seed: 42,
          continue_final_message: true,
          add_generation_prompt: false,
          messages: [
            {role: "system", content: $sys},
            {role: "user", content: [{type: "audio_url", audio_url: {url: $url}}]},
            {role: "assistant", content: $assistant}
          ]
        }')
    output=$(curl -sS -X POST "${API_BASE}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$pass2_payload")

    text_content=$(echo "$output" | jq -r '.choices[0].message.content // empty')
    if [[ -n "$text_content" ]]; then
        echo "Text output: $text_content"
    fi
    audio_data=$(echo "$output" | jq -r '.choices[0].message.audio.data // empty')
    if [[ -n "$audio_data" ]]; then
        echo "Audio output received (base64 encoded)"
    fi
    error=$(echo "$output" | jq -r '.error // empty')
    if [[ -n "$error" ]]; then
        echo "Pass 2 Error: $error"
    fi
    echo ""
    echo "Done!"
    exit 0
fi

MODEL="${MODEL:-stepfun-ai/Step-Audio-2-mini}"

# Sampling parameters for Thinker stage
thinker_sampling_params='{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": -1,
  "max_tokens": 1024,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.05
}'

# Sampling parameters for Token2Wav stage
token2wav_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 1,
  "seed": 42,
  "detokenize": false
}'

# Build request based on query type
case "$QUERY_TYPE" in
  audio_to_text)
    MODALITIES='["text"]'
    system_content='[{"type": "text", "text": "请记录下你所听到的语音内容。"}]'
    user_content='[
      {"type": "audio_url", "audio_url": {"url": "'"$MARY_HAD_LAMB_URL"'"}}
    ]'
    # Use the model EOT EOS; no additional legacy stop token.
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.05
    }'
    ;;
  text_to_audio)
    system_content='[{"type": "text", "text": "以自然的语速读出下面的文字。"}]'
    user_content='[
      {"type": "text", "text": "Hello, this is a test of Step Audio 2 text to speech synthesis."}
    ]'
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.1
    }'
    ;;
  audio_to_audio)
    system_content='[{"type": "text", "text": "You are an audio processing assistant. Listen and repeat the audio content."}]'
    user_content='[
      {"type": "audio_url", "audio_url": {"url": "'"$MARY_HAD_LAMB_URL"'"}}
    ]'
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.1
    }'
    ;;
esac

# Continue the assistant's speech prefix, matching the Python client.
assistant_message=''
generation_options=''
if [[ "$QUERY_TYPE" != "audio_to_text" ]]; then
  assistant_message=', {"role": "assistant", "content": "<tts_start>"}'
  generation_options='"continue_final_message": true, "add_generation_prompt": false,'
fi

sampling_params_list='[
  '"$thinker_sampling_params"',
  '"$token2wav_sampling_params"'
]'

echo "Query type: $QUERY_TYPE"
echo "Model: $MODEL"
echo "API base: $API_BASE"
echo "Sending request..."
echo ""

output=$(curl -sS -X POST "${API_BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "$MODEL",
  "modalities": $MODALITIES,
  "sampling_params_list": $sampling_params_list,
  $generation_options
  "messages": [
    {
      "role": "system",
      "content": $system_content
    },
    {
      "role": "user",
      "content": $user_content
    }$assistant_message
  ]
}
EOF
)

# Extract and display text content
text_content=$(echo "$output" | jq -r '.choices[0].message.content // empty')
if [[ -n "$text_content" ]]; then
    echo "Text output: $text_content"
fi

# Check for audio content
audio_data=$(echo "$output" | jq -r '.choices[0].message.audio.data // empty')
if [[ -n "$audio_data" ]]; then
    echo "Audio output received (base64 encoded)"
    echo "To save audio, use the Python client or decode the base64 data"
fi

# Check for errors
error=$(echo "$output" | jq -r '.error // empty')
if [[ -n "$error" ]]; then
    echo "Error: $error"
fi

echo ""
echo "Done!"
