# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
OpenAI-compatible client for Step-Audio2 online serving.

Step-Audio2 is a two-stage model:
  - Stage 0 (Thinker): Audio understanding -> Text + Audio tokens
  - Stage 1 (Token2Wav): Audio tokens -> Waveform (24kHz)

Recommended checkpoint per task (matching upstream Step-Audio2):
  - ASR / S2TT / S2ST: stepfun-ai/Step-Audio-2-mini (step_audio_2.yaml + step_audio2.jinja)
  - Verbatim TTS: stepfun-ai/Step-Audio-2-mini-Base via /v1/audio/speech (openai_speech_client.py)
  - Deep-thinking speech dialogue: stepfun-ai/Step-Audio-2-mini-Think (step_audio_2_think.yaml + step_audio2.jinja)

Usage examples:
  # Audio to text (ASR, Step-Audio-2-mini)
  python openai_chat_completion_client.py --query-type audio_to_text

  # Audio to audio (S2ST, Step-Audio-2-mini)
  python openai_chat_completion_client.py --query-type audio_to_audio --audio-path input.wav

  # Two-pass deep-thinking speech dialogue (Step-Audio-2-mini-Think)
  python openai_chat_completion_client.py --query-type think --audio-path input.wav

  # Text to audio via Chat Completions (for verbatim TTS, prefer openai_speech_client.py with Base)
  python openai_chat_completion_client.py --query-type text_to_audio --text "Hello world"
"""

import base64
import logging
import os

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

DEFAULT_MODELS = {
    "audio_to_text": "stepfun-ai/Step-Audio-2-mini",
    "text_to_audio": "stepfun-ai/Step-Audio-2-mini-Base",
    "audio_to_audio": "stepfun-ai/Step-Audio-2-mini",
    "think": "stepfun-ai/Step-Audio-2-mini-Think",
}

THINK_SYSTEM_PROMPT = (
    "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，"
    "你能听见用户的声音特征并在思维过程中描述出来，"
    "请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"
)


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_base64_content_from_url(url: str) -> str:
    """Encode content from URL to base64 format."""
    with requests.get(url) as response:
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


def get_audio_url(audio_path: str | None = None) -> str:
    """Convert audio path to URL format for the API.

    Args:
        audio_path: Local file path or URL. If None, uses default test audio.

    Returns:
        Audio URL (either original URL or base64 data URL)
    """
    if not audio_path:
        return AudioAsset("mary_had_lamb").url

    if audio_path.startswith(("http://", "https://")):
        return audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect MIME type
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    else:
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_audio_to_text_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
) -> dict:
    """Build query for Audio-to-Text (ASR) mode."""
    audio_url = get_audio_url(audio_path)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ],
    }


def get_text_to_audio_query(
    text: str | None = None,
    custom_prompt: str | None = None,
) -> list[dict]:
    """Build query for Text-to-Audio (TTS) mode.

    Returns a list of messages: user message + assistant message with <tts_start>.
    The assistant message triggers TTS mode and must be used with
    continue_final_message=True so the chat template does not append <|EOT|>.
    """
    text_to_speak = text or "Hello, this is a test of Step Audio 2 text to speech."

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": text_to_speak}],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
        },
    ]


def get_audio_to_audio_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
) -> list[dict]:
    """Build query for Audio-to-Audio (S2ST) mode.

    Returns a list of messages: user message (with audio) + assistant message
    with <tts_start>. Must be used with continue_final_message=True.
    """
    audio_url = get_audio_url(audio_path)

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ],
        },
        {
            "role": "assistant",
            "content": "<tts_start>",
        },
    ]


def get_think_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
) -> dict:
    """Build one user turn for Step-Audio-2-mini-Think two-pass reasoning."""
    audio_url = get_audio_url(audio_path)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
        ],
    }


def get_system_prompt(query_type: str, custom_prompt: str | None = None, *, is_base: bool = False) -> dict:
    """Get system prompt based on query type."""
    if query_type == "audio_to_text":
        system_text = "请记录下你所听到的语音内容。"
    elif query_type == "text_to_audio":
        # Upstream Base TTS separates the instruction from the text with a newline.
        system_text = "以自然的语速读出下面的文字。\n" if is_base else "以自然的语速读出下面的文字。"
    elif query_type == "think":
        system_text = THINK_SYSTEM_PROMPT
    else:  # audio_to_audio
        system_text = "You are an audio processing assistant. Listen and repeat the audio content."

    return {
        "role": "system",
        "content": [{"type": "text", "text": custom_prompt or system_text}],
    }


QUERY_MAP = {
    "audio_to_text": get_audio_to_text_query,
    "text_to_audio": get_text_to_audio_query,
    "audio_to_audio": get_audio_to_audio_query,
    "think": get_think_query,
}


def run_think_step_audio2(client: OpenAI, args, model_name: str, max_tokens: int) -> None:
    """Run upstream two-pass reasoning + speech dialogue for Step-Audio-2-mini-Think.

    Follows upstream ``think_test_vllm`` in ``examples-think.py``:
      1. Pass 1: open ``<think>`` assistant turn, ``modalities=["text"]``, ``stop=["</think>"]``.
      2. Pass 2: append ``think_text + "</think>\\n\\n<tts_start>"`` and synthesize speech
         (terminating at ``<|EOT|>`` via ``step_audio_2_think.yaml`` defaults).
      3. Strip ``<think>...</think>`` from history before the next round.
    """
    audio_paths = [args.audio_path, *(args.extra_audio_paths or [])]
    history: list[dict] = [get_system_prompt("think", args.prompt)]

    os.makedirs(args.output_dir, exist_ok=True)

    for round_idx, audio_path in enumerate(audio_paths):
        logger.info("Think round %d: sending Pass 1 (reasoning)...", round_idx)
        history.append(get_think_query(audio_path=audio_path))
        history.append({"role": "assistant", "content": "<think>"})

        # Do not pass sampling_params_list so step_audio_2_think.yaml keeps
        # stage-0 default stop_token_ids=[151665] (<|EOT|>) and repetition_penalty=1.05.
        pass1 = client.chat.completions.create(
            messages=history,
            model=model_name,
            temperature=0.7,
            max_tokens=max_tokens,
            seed=SEED,
            stop=["</think>"],
            extra_body={
                "modalities": ["text"],
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )
        think_text = pass1.choices[0].message.content or ""
        logger.info("Round %d reasoning (<think>...): %s", round_idx, think_text)

        logger.info("Think round %d: sending Pass 2 (spoken response)...", round_idx)
        history[-1]["content"] += think_text + "</think>\n\n<tts_start>"
        pass2 = client.chat.completions.create(
            messages=history,
            model=model_name,
            temperature=0.7,
            max_tokens=max_tokens,
            seed=SEED,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )

        msg = pass2.choices[0].message
        if msg.content:
            logger.info("Round %d text output: %s", round_idx, msg.content)
        if msg.audio and msg.audio.data:
            audio_file = os.path.join(args.output_dir, f"round_{round_idx}_audio.wav")
            with open(audio_file, "wb") as f:
                f.write(base64.b64decode(msg.audio.data))
            logger.info("Round %d audio saved to: %s", round_idx, audio_file)
        else:
            logger.warning("Round %d received no audio output. Check server logs for details.", round_idx)

        # Strip reasoning trace before the next conversation turn.
        history[-1] = {"role": "assistant", "content": msg.content or ""}

    logger.info("Done!")


def run_step_audio2(args) -> None:
    """Run Step-Audio2 inference via OpenAI-compatible API."""

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
    )

    model_name = args.model or DEFAULT_MODELS[args.query_type]
    max_tokens = args.max_tokens if args.max_tokens is not None else (2048 if args.query_type == "think" else 1024)

    logger.info("Query type: %s", args.query_type)
    logger.info("Model: %s", model_name)

    if args.query_type == "think":
        run_think_step_audio2(client, args, model_name, max_tokens)
        return

    if args.query_type == "text_to_audio":
        logger.info(
            "Tip: for verbatim TTS matching upstream examples-base.py, prefer /v1/audio/speech "
            "(openai_speech_client.py) with stepfun-ai/Step-Audio-2-mini-Base."
        )

    is_base = "base" in model_name.lower()

    # Sampling parameters for each stage
    if is_base and args.query_type == "text_to_audio":
        repetition_penalty = 1.0
    elif args.query_type != "audio_to_text":
        repetition_penalty = 1.1
    else:
        repetition_penalty = 1.05

    thinker_sampling_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_tokens": max_tokens,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": repetition_penalty,
    }

    token2wav_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 1,
        "seed": SEED,
        "detokenize": False,
    }

    sampling_params_list = [
        thinker_sampling_params,
        token2wav_sampling_params,
    ]

    # Build query based on type
    query_func = QUERY_MAP[args.query_type]
    if args.query_type == "audio_to_text":
        user_message = query_func(audio_path=args.audio_path, custom_prompt=args.prompt)
    elif args.query_type == "text_to_audio":
        user_message = query_func(text=args.text, custom_prompt=args.prompt)
    else:  # audio_to_audio
        user_message = query_func(audio_path=args.audio_path, custom_prompt=args.prompt)

    system_message = get_system_prompt(args.query_type, args.prompt, is_base=is_base)

    logger.info("Sending request to server...")

    # Build messages list.
    # TTS and S2ST return a list of messages (user + assistant with <tts_start>);
    # ASR returns a single user message dict.
    if isinstance(user_message, list):
        messages = [system_message] + user_message
    else:
        messages = [system_message, user_message]

    # For TTS/S2ST, use continue_final_message so <tts_start> in the
    # assistant turn is not followed by <|EOT|>.
    extra_body: dict = {"sampling_params_list": sampling_params_list}
    if args.query_type == "audio_to_text":
        extra_body["modalities"] = ["text"]
    if args.query_type in ("text_to_audio", "audio_to_audio"):
        extra_body["continue_final_message"] = True
        extra_body["add_generation_prompt"] = False

    # Send request
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        extra_body=extra_body,
    )

    # Process response
    os.makedirs(args.output_dir, exist_ok=True)

    audio_count = 0
    for choice in chat_completion.choices:
        if choice.message.audio:
            # Audio response
            audio_data = base64.b64decode(choice.message.audio.data)
            audio_file = os.path.join(args.output_dir, f"audio_{audio_count}.wav")
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            logger.info("Audio saved to: %s", audio_file)
            audio_count += 1

        if choice.message.content:
            # Text response
            logger.info("Text output: %s", choice.message.content)

    if audio_count == 0 and args.query_type != "audio_to_text":
        logger.warning("No audio output received. Check server logs for details.")

    logger.info("Done!")


def parse_args():
    parser = FlexibleArgumentParser(description="OpenAI-compatible client for Step-Audio2 online serving")

    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="audio_to_text",
        choices=QUERY_MAP.keys(),
        help="Query type: audio_to_text (ASR), text_to_audio (TTS), audio_to_audio (S2ST), think (2-pass reasoning)",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file (local path or URL)",
    )
    parser.add_argument(
        "--extra-audio-paths",
        type=str,
        nargs="+",
        default=None,
        help="Additional audio inputs for multi-turn think mode",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (for text_to_audio mode)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom prompt/question",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=(
            "Model name. Defaults to the upstream-recommended checkpoint per query type: "
            "Step-Audio-2-mini-Base (text_to_audio), Step-Audio-2-mini-Think (think), "
            "or Step-Audio-2-mini (audio_to_text / audio_to_audio)."
        ),
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8092/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for Thinker stage (default: 2048 for think, 1024 otherwise)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output_online",
        help="Output directory for audio files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_step_audio2(args)
