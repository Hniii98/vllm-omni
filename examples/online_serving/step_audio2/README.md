# Step-Audio2: Online serving

This directory contains examples for running Step-Audio2 variants with vLLM-Omni's online serving API.

## Recommended Checkpoint and Endpoint by Task

Following upstream Step-Audio2 (`examples.py`, `examples-base.py`, `examples-think.py`):

| Task | Recommended Checkpoint | Deploy Config | Chat Template | Endpoint |
| --- | --- | --- | --- | --- |
| **Verbatim TTS** | `stepfun-ai/Step-Audio-2-mini-Base` | `step_audio_2_base.yaml` | `step_audio2_base.jinja` | `/v1/audio/speech` (`openai_speech_client.py`) |
| **ASR / S2TT / S2ST / Conversation** | `stepfun-ai/Step-Audio-2-mini` | `step_audio_2_async_chunk.yaml` or `step_audio_2.yaml` | `step_audio2.jinja` | `/v1/chat/completions` (`openai_chat_completion_client.py`) |
| **Deep-Thinking Speech Dialogue (2-pass)** | `stepfun-ai/Step-Audio-2-mini-Think` | `step_audio_2_think.yaml` | `step_audio2.jinja` | `/v1/chat/completions` (`--query-type think`) |

## Installation

Please refer to [README.md](../../../README.md)

## Launch the Server

Run these commands from the repository root. `--chat-template` is required for all
variants because the checkpoints bundle a generic Qwen `<|im_start|>` template.

### 1. Step-Audio-2-mini (ASR / S2ST / Conversation)

```bash
# Async chunk mode (recommended — lower first-packet latency)
vllm serve stepfun-ai/Step-Audio-2-mini --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2_async_chunk.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2.jinja \
    --trust-remote-code \
    --stage-overrides '{"0":{"enforce_eager":true},"1":{"enforce_eager":true}}'
```

Sequential mode:

```bash
vllm serve stepfun-ai/Step-Audio-2-mini --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2.jinja \
    --trust-remote-code \
    --stage-overrides '{"0":{"enforce_eager":true},"1":{"enforce_eager":true}}'
```

### 2. Step-Audio-2-mini-Base (Recommended for Verbatim TTS)

```bash
vllm serve stepfun-ai/Step-Audio-2-mini-Base --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2_base.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2_base.jinja \
    --trust-remote-code \
    --stage-overrides '{"0":{"enforce_eager":true},"1":{"enforce_eager":true}}'
```

### 3. Step-Audio-2-mini-Think (Two-Pass Reasoning + Speech)

```bash
vllm serve stepfun-ai/Step-Audio-2-mini-Think --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2_think.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2.jinja \
    --trust-remote-code \
    --stage-overrides '{"0":{"enforce_eager":true},"1":{"enforce_eager":true}}'
```

## Send Requests

### Verbatim TTS via `/v1/audio/speech` (Recommended: `Step-Audio-2-mini-Base`)

```bash
cd examples/online_serving/step_audio2

# Python client (defaults to stepfun-ai/Step-Audio-2-mini-Base)
python openai_speech_client.py --text "你好呀，我是你的AI助手，很高兴认识你！"

# With custom instruction
python openai_speech_client.py --text "Hello, welcome to Step-Audio2." \
    --instructions "Read this paragraph at a natural pace."

# Save to specific file
python openai_speech_client.py --text "你好世界" -o output.wav

# Or against a Step-Audio-2-mini server
python openai_speech_client.py --model stepfun-ai/Step-Audio-2-mini --text "你好世界"
```

Or via curl:

```bash
bash run_curl.sh tts

# Or direct curl:
curl -X POST http://localhost:8092/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"stepfun-ai/Step-Audio-2-mini-Base","input":"你好呀，我是你的AI助手，很高兴认识你！","voice":"default"}' \
    --output output.wav
```

This endpoint bypasses the chat template and directly triggers TTS mode (using upstream two-segment `prompt_token_ids` encoding for Base).

**Note**: Speaker voice is controlled by `STEP_AUDIO2_DEFAULT_PROMPT_WAV` env var on the server side.

### Chat Completions (ASR / S2ST / Think)

```bash
# Audio to Text (ASR, Step-Audio-2-mini)
python openai_chat_completion_client.py --query-type audio_to_text

# Audio to Audio (S2ST, Step-Audio-2-mini)
python openai_chat_completion_client.py --query-type audio_to_audio --audio-path /path/to/input.wav

# Deep-thinking speech dialogue (2-pass, Step-Audio-2-mini-Think)
python openai_chat_completion_client.py --query-type think --audio-path /path/to/round1.wav

# Multi-turn deep-thinking dialogue (strips <think> between rounds)
python openai_chat_completion_client.py --query-type think \
    --audio-path /path/to/round1.wav \
    --extra-audio-paths /path/to/round2.wav
```

| Argument | Description |
| --- | --- |
| `--query-type`, `-q` | `audio_to_text`, `text_to_audio`, `audio_to_audio`, `think` |
| `--audio-path`, `-a` | Path to input audio file (local or URL) |
| `--extra-audio-paths` | Additional audio inputs for multi-turn `think` mode |
| `--text`, `-t` | Text to synthesize (for `text_to_audio` mode) |
| `--prompt`, `-p` | Custom system prompt / instruction |
| `--model`, `-m` | Model name (defaults to upstream-recommended checkpoint per `--query-type`) |
| `--max-tokens` | Max Thinker tokens (default: `2048` for `think`, `1024` otherwise) |
| `--output-dir`, `-o` | Output directory for audio files (default: `output_online`) |
| `--api-base` | API base URL (default: `http://localhost:8092/v1`) |

### Curl

```bash
# ASR (Step-Audio-2-mini)
bash run_curl.sh audio_to_text

# Verbatim TTS via /v1/audio/speech (Step-Audio-2-mini-Base)
bash run_curl.sh tts

# S2ST (Step-Audio-2-mini)
bash run_curl.sh audio_to_audio

# Two-pass reasoning + speech (Step-Audio-2-mini-Think)
bash run_curl.sh think
```

## Output

- **Text output**: Printed to console (in `think` mode, both Pass 1 `<think>...</think>` reasoning and Pass 2 spoken transcript are logged)
- **Audio output**: Saved to `output_online/audio_0.wav` (or `output_online/round_0_audio.wav` in `think` mode, 24kHz WAV)

## Performance

### Async Chunk vs Sequential

Benchmark via `/v1/audio/speech` (4x RTX 3090, 10 prompts, concurrency=1):

| Mode | Mean TTFP | Mean E2E | Mean RTF |
| --- | --- | --- | --- |
| Sequential | 4316ms | 4316ms | 0.938 |
| **Async Chunk** | **1437ms** | 4362ms | 0.949 |

Async chunk reduces TTFP by **67%** by streaming audio token chunks from Thinker to Token2Wav as they are generated. RTF < 1 in both modes (real-time capable).

## Troubleshooting

### Server not responding

- Check if the server is running: `curl http://localhost:8092/v1/models`
- Verify the port number matches

### FileNotFoundError: prompt_wav file not found

- Ensure `default_female.wav` exists at `{model_dir}/assets/default_female.wav`
- Or set `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable when launching the server

### Audio not generated

- For verbatim TTS, use `/v1/audio/speech` (`openai_speech_client.py`) with `Step-Audio-2-mini-Base`
- For chat completions speech output, ensure the final assistant turn ends with `<tts_start>` and `continue_final_message=true`, `add_generation_prompt=false`
- For `Step-Audio-2-mini-Think`, use `--query-type think` (two-pass `/v1/chat/completions`); `/v1/audio/speech` is rejected for the `think` variant

### Out of memory

- Reduce `gpu_memory_utilization` in the deploy config
- Use a smaller batch size
