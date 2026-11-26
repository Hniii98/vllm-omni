# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
This example shows how to use vLLM-Omni for running offline inference
with Step-Audio2 models for audio understanding, speech synthesis, and reasoning.

Step-Audio2 is a two-stage model:
  - Stage 0 (Thinker): Audio understanding -> Text + Audio tokens
  - Stage 1 (Token2Wav): Audio tokens -> Waveform (24kHz)

Recommended checkpoint per task (matching upstream Step-Audio2):
  - ASR / S2ST: stepfun-ai/Step-Audio-2-mini
  - Verbatim TTS: stepfun-ai/Step-Audio-2-mini-Base (two-segment prompt_token_ids)
  - Deep-thinking speech dialogue: stepfun-ai/Step-Audio-2-mini-Think (2-pass)

Usage examples:
  # Audio to text (ASR, Step-Audio-2-mini)
  python end2end.py --query-type audio_to_text --audio-path input.wav --model stepfun-ai/Step-Audio-2-mini

  # Text to audio (TTS, Step-Audio-2-mini-Base recommended)
  python end2end.py --query-type text_to_audio --text "你好呀，我是你的AI助手，很高兴认识你！" \
      --model stepfun-ai/Step-Audio-2-mini-Base

  # Audio to audio (S2ST, Step-Audio-2-mini)
  python end2end.py --query-type audio_to_audio --audio-path input.wav --model stepfun-ai/Step-Audio-2-mini

  # Two-pass deep-thinking speech dialogue (Step-Audio-2-mini-Think)
  python end2end.py --query-type think --audio-path input.wav --model stepfun-ai/Step-Audio-2-mini-Think
"""

import os
import sys
from pathlib import Path
from typing import NamedTuple

import librosa  # noqa: TID251
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from vllm.assets.audio import AudioAsset
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42
EOT_TOKEN_ID = 151665

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


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def resolve_variant(model_name: str, variant: str = "auto") -> str:
    """Resolve Step-Audio2 checkpoint variant from explicit flag or model name."""
    if variant != "auto":
        return variant
    lowered = model_name.lower()
    if "think" in lowered:
        return "think"
    if "base" in lowered:
        return "base"
    return "mini"


def _load_audio_16k(audio_path: str | None, sampling_rate: int = 16000) -> tuple[np.ndarray, int]:
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        return (audio_signal.astype(np.float32), sr)

    audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
    if audio_data[1] != sampling_rate:
        audio_signal = librosa.resample(audio_data[0], orig_sr=audio_data[1], target_sr=sampling_rate)
        return (audio_signal.astype(np.float32), sampling_rate)
    return audio_data


def get_audio_to_text_query(
    audio_path: str | None = None,
    question: str | None = None,
    sampling_rate: int = 16000,
    variant: str = "mini",
) -> QueryResult:
    """
    Audio Speech Recognition (ASR) - Convert audio to text.

    Args:
        audio_path: Path to audio file
        question: Question about the audio (optional)
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)
        variant: Checkpoint variant (``mini``, ``base``, or ``think``)

    Returns:
        QueryResult with prompt and audio data
    """
    if question is None:
        question = "请记录下你所听到的语音内容。"

    if variant == "base":
        prompt = f"{question}<audio_patch>"
    else:
        prompt = f"<|BOT|>system\n{question}<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n"

    audio_data = _load_audio_16k(audio_path, sampling_rate=sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_text_to_audio_query(
    text: str | None = None,
    sampling_rate: int = 16000,
    variant: str = "mini",
    tokenizer=None,
    question: str | None = None,
) -> QueryResult:
    """
    Text-to-Speech (TTS) - Convert text to audio.

    Args:
        text: Text to synthesize
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)
        variant: Checkpoint variant (``mini`` or ``base``)
        tokenizer: Tokenizer used for Base's upstream two-segment encoding
        question: Optional system instruction overriding the default read-aloud prompt

    Returns:
        QueryResult with text prompt or Base two-segment prompt_token_ids
    """
    if text is None:
        text = (
            "你好呀，我是你的AI助手，很高兴认识你！"
            if variant == "base"
            else ("Hello, this is a test of Step Audio 2 text to speech synthesis.")
        )
    system_prompt = question or "以自然的语速读出下面的文字。"

    if variant == "base":
        if tokenizer is None:
            raise ValueError("Base TTS requires a tokenizer for upstream two-segment encoding.")
        ids = tokenizer.encode(system_prompt + "\n", add_special_tokens=False)
        ids += tokenizer.encode(text + "<tts_start>", add_special_tokens=False)
        inputs: dict = {"prompt_token_ids": ids}
    else:
        prompt = f"<|BOT|>system\n{system_prompt}<|EOT|><|BOT|>human\n{text}<|EOT|><|BOT|>assistant\n<tts_start>"
        inputs = {"prompt": prompt}

    return QueryResult(
        inputs=inputs,
        limit_mm_per_prompt={},
    )


def get_audio_to_audio_query(
    audio_path: str | None = None,
    question: str | None = None,
    sampling_rate: int = 16000,
    variant: str = "mini",
) -> QueryResult:
    """
    Audio-to-Audio - Speech-to-speech (S2ST).

    Args:
        audio_path: Path to source audio file
        question: Question/instruction about the audio
        sampling_rate: Target sampling rate (16kHz for Step-Audio2)
        variant: Checkpoint variant (``mini``, ``base``, or ``think``)

    Returns:
        QueryResult with audio input
    """
    if question is None:
        question = "请仔细聆听这段语音，然后复述其内容。"

    if variant == "base":
        prompt = f"{question}<audio_patch><tts_start>"
    else:
        prompt = f"<|BOT|>system\n{question}<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n<tts_start>"

    audio_data = _load_audio_16k(audio_path, sampling_rate=sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_think_query(
    audio_path: str | None = None,
    question: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """Build Pass 1 reasoning prompt for Step-Audio-2-mini-Think.

    Matches upstream ``think_test`` in ``examples-think.py``: opens the
    assistant turn with ``\\n<think>\\n`` and omits ``<|EOT|>``.
    """
    system_prompt = question or THINK_SYSTEM_PROMPT
    prompt = f"<|BOT|>system\n{system_prompt}<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n\n<think>\n"
    audio_data = _load_audio_16k(audio_path, sampling_rate=sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


query_map = {
    "audio_to_text": get_audio_to_text_query,
    "text_to_audio": get_text_to_audio_query,
    "audio_to_audio": get_audio_to_audio_query,
    "think": get_think_query,
}


def _extract_first_text(omni_outputs) -> str:
    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type != "text":
            continue
        outputs_list = stage_outputs if isinstance(stage_outputs, list) else [stage_outputs]
        for output in outputs_list:
            if output.outputs:
                return output.outputs[0].text
    return ""


def _save_outputs(omni_outputs, prompts: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def _get_request_index(req_id: str, fallback: int) -> int:
        try:
            return int(req_id)
        except (TypeError, ValueError):
            if isinstance(req_id, str):
                head = req_id.split("_", 1)[0]
                if head.isdigit():
                    return int(head)
        return fallback

    for stage_outputs in omni_outputs:
        outputs_list = stage_outputs if isinstance(stage_outputs, list) else [stage_outputs]

        if stage_outputs.final_output_type == "text":
            for i, output in enumerate(outputs_list):
                request_id = _get_request_index(output.request_id, i)
                text_output = output.outputs[0].text
                prompt_display = prompts[request_id].get("prompt", "<prompt_token_ids>")

                out_txt = os.path.join(output_dir, f"{request_id:05d}_text.txt")
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"Prompt:\n{prompt_display}\n\n")
                    f.write(f"Output:\n{text_output}\n")
                print(f"Text output saved to: {out_txt}")
                print(f"Text: {text_output[:200]}")

        elif stage_outputs.final_output_type == "audio":
            for i, output in enumerate(outputs_list):
                request_id = _get_request_index(output.request_id, i)

                mm_out = {}
                if hasattr(output, "multimodal_output") and output.multimodal_output:
                    mm_out = output.multimodal_output
                elif output.outputs and hasattr(output.outputs[0], "multimodal_output"):
                    mm_out = output.outputs[0].multimodal_output or {}
                audio_tensor = None
                for key in ("audio", "wav", "waveform", "audio_pcm", "pcm", "model_outputs"):
                    if key in mm_out and mm_out[key] is not None:
                        audio_tensor = mm_out[key]
                        break

                if audio_tensor is not None:
                    output_wav = os.path.join(output_dir, f"{request_id:05d}_output.wav")
                    if isinstance(audio_tensor, list):
                        chunks = []
                        for chunk in audio_tensor:
                            if chunk is None:
                                continue
                            chunk_np = (
                                chunk.detach().cpu().numpy() if isinstance(chunk, torch.Tensor) else np.asarray(chunk)
                            )
                            if chunk_np.size > 0:
                                chunks.append(chunk_np)
                        if not chunks:
                            continue
                        audio_numpy = np.concatenate(chunks, axis=-1)
                    elif isinstance(audio_tensor, torch.Tensor):
                        audio_numpy = audio_tensor.detach().cpu().numpy()
                    else:
                        audio_numpy = np.asarray(audio_tensor)
                    sf.write(output_wav, audio_numpy, samplerate=24000)
                    print(f"Audio output saved to: {output_wav}")


def main(args):
    model_name = args.model or DEFAULT_MODELS[args.query_type]
    variant = resolve_variant(model_name, args.variant)
    max_tokens = args.max_tokens if args.max_tokens is not None else (2048 if args.query_type == "think" else 1024)

    # Check if it's a local path
    if os.path.isdir(model_name):
        config_path = os.path.join(model_name, "config.json")
        if not os.path.exists(config_path):
            sys.exit(1)

    if args.query_type == "text_to_audio" and variant == "think":
        raise ValueError(
            "Step-Audio-2-mini-Think does not define a standalone TTS prompt; "
            "use --query-type think or Step-Audio-2-mini-Base for TTS."
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if args.query_type == "text_to_audio" and variant == "base"
        else None
    )

    # Build query based on type
    if args.query_type == "audio_to_text":
        query_result = get_audio_to_text_query(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
            variant=variant,
        )
    elif args.query_type == "text_to_audio":
        query_result = get_text_to_audio_query(
            text=args.text,
            sampling_rate=args.sampling_rate,
            variant=variant,
            tokenizer=tokenizer,
            question=args.question,
        )
    elif args.query_type == "audio_to_audio":
        query_result = get_audio_to_audio_query(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
            variant=variant,
        )
    elif args.query_type == "think":
        query_result = get_think_query(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
        )
    else:
        raise ValueError(f"Unknown query type: {args.query_type}")

    # Resolve an explicit deploy config or use the bundled task/variant default.
    if args.deploy_config:
        deploy_config_path = args.deploy_config
    else:
        configs_dir = Path(__file__).parent.parent.parent.parent / "vllm_omni/deploy"
        if args.query_type == "think" or variant == "think":
            deploy_config_path = str(configs_dir / "step_audio_2_think.yaml")
        elif args.query_type == "audio_to_text":
            deploy_config_path = str(configs_dir / "step_audio_2_asr.yaml")
        elif variant == "base":
            deploy_config_path = str(configs_dir / "step_audio_2_base.yaml")
        else:
            deploy_config_path = str(configs_dir / "step_audio_2.yaml")

    omni_kwargs = {
        "model": model_name,
        "log_stats": args.enable_stats,
        "log_file": ("step_audio2_pipeline.log" if args.enable_stats else None),
        "init_sleep_seconds": args.init_sleep_seconds,
        "batch_timeout": args.batch_timeout,
        "init_timeout": args.init_timeout,
        "worker_backend": args.worker_backend,
        "ray_address": args.ray_address,
        "trust_remote_code": True,
        "deploy_config": deploy_config_path,
    }

    omni_llm = Omni(**omni_kwargs)

    token2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=SEED,
        detokenize=False,
    )

    if args.query_type == "think":
        # Two-pass reasoning + speech generation matching upstream examples-think.py.
        pass1_params = [
            SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=-1,
                max_tokens=max_tokens,
                seed=SEED,
                detokenize=True,
                stop=["</think>"],
                stop_token_ids=[EOT_TOKEN_ID],
            ),
            token2wav_sampling_params,
        ]
        pass1_outputs = omni_llm.generate([query_result.inputs], pass1_params)
        think_text = _extract_first_text(pass1_outputs)
        os.makedirs(args.output_dir, exist_ok=True)
        think_txt = os.path.join(args.output_dir, "00000_think.txt")
        with open(think_txt, "w", encoding="utf-8") as f:
            f.write(f"<think>\n{think_text}</think>\n")
        print(f"Reasoning saved to: {think_txt}")
        print(f"<think>\n{think_text}</think>")

        pass2_inputs = {
            **query_result.inputs,
            "prompt": query_result.inputs["prompt"] + think_text + "</think>\n\n<tts_start>",
        }
        pass2_params = [
            SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=-1,
                max_tokens=max_tokens,
                seed=SEED,
                detokenize=True,
                repetition_penalty=1.05,
                stop_token_ids=[EOT_TOKEN_ID],
            ),
            token2wav_sampling_params,
        ]
        prompts = [pass2_inputs]
        omni_outputs = omni_llm.generate(prompts, pass2_params)
        _save_outputs(omni_outputs, prompts, args.output_dir)
        return

    if variant == "base" and args.query_type == "text_to_audio":
        # Upstream examples-base.py uses no repetition penalty for verbatim TTS.
        repetition_penalty = 1.0
    elif args.query_type in ["text_to_audio", "audio_to_audio"]:
        repetition_penalty = 1.1
    else:
        repetition_penalty = 1.05

    thinker_kwargs: dict = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "max_tokens": max_tokens,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": repetition_penalty,
    }
    if variant == "think":
        thinker_kwargs["stop_token_ids"] = [EOT_TOKEN_ID]

    thinker_sampling_params = SamplingParams(**thinker_kwargs)

    if args.query_type == "audio_to_text":
        sampling_params_list = [thinker_sampling_params]
    else:
        sampling_params_list = [thinker_sampling_params, token2wav_sampling_params]

    if args.num_prompts > 1:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        prompts = [query_result.inputs]

    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    _save_outputs(omni_outputs, prompts, args.output_dir)


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM-Omni for offline inference with Step-Audio2")

    # Model and config
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help=(
            "Model path or HuggingFace ID. Defaults to the upstream-recommended checkpoint per query type:\n"
            "  - audio_to_text / audio_to_audio: stepfun-ai/Step-Audio-2-mini\n"
            "  - text_to_audio: stepfun-ai/Step-Audio-2-mini-Base\n"
            "  - think: stepfun-ai/Step-Audio-2-mini-Think"
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="auto",
        choices=["auto", "mini", "base", "think"],
        help="Checkpoint variant (default: auto-detect from --model name)",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Path to a deploy config YAML file (default: use bundled Step-Audio2 deploy config)",
    )
    # Query configuration
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="audio_to_text",
        choices=query_map.keys(),
        help="Query type: audio_to_text (ASR), text_to_audio (TTS), audio_to_audio (S2ST), think (2-pass reasoning)",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file (for audio_to_text, audio_to_audio, and think)",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text to synthesize (for text_to_audio mode)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom system prompt / question",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Audio sampling rate for input (default: 16000 Hz)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate in Thinker stage (default: 2048 for think, 1024 otherwise)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of times to repeat the prompt (for testing)",
    )

    # System configuration
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Batch timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Initialization timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--worker-backend",
        type=str,
        default="multi_process",
        choices=["multi_process", "ray"],
        help="Worker backend (default: multi_process)",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (for ray backend)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output_step_audio2",
        help="Output directory for generated files (default: output_step_audio2)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
