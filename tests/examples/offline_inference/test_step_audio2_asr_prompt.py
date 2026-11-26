# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
from types import ModuleType
from unittest.mock import Mock, call

import numpy as np
import pytest

from tests.examples.helpers import EXAMPLES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "question, expected_question",
    [(None, "请记录下你所听到的语音内容。"), ("Transcribe the speech.", "Transcribe the speech.")],
)
def test_asr_uses_native_step_audio2_chat_format(tmp_path, monkeypatch, question, expected_question):
    audio = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    # Prompt construction should not depend on the optional audio decoder.
    decoder = ModuleType("librosa")
    decoder.__spec__ = importlib.util.spec_from_loader("librosa", loader=None)
    load_audio = Mock(return_value=(audio, 16000))
    monkeypatch.setattr(decoder, "load", load_audio, raising=False)
    monkeypatch.setitem(sys.modules, "librosa", decoder)
    spec = importlib.util.spec_from_file_location(
        "step_audio2_example", EXAMPLES / "offline_inference/step_audio2/end2end.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    audio_path = tmp_path / "input.wav"
    audio_path.touch()
    query = example.get_audio_to_text_query(str(audio_path), question=question)
    load_audio.assert_called_once_with(str(audio_path), sr=16000)

    # Match the official StepAudio2.apply_chat_template protocol. The processor
    # expands the single audio placeholder and owns the per-chunk boundaries.
    assert query.inputs["prompt"] == (
        f"<|BOT|>system\n{expected_question}<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n"
    )
    actual_audio, sample_rate = query.inputs["multi_modal_data"]["audio"]
    np.testing.assert_array_equal(actual_audio, audio)
    assert sample_rate == 16000
    assert query.limit_mm_per_prompt == {"audio": 1}


def _load_end2end(monkeypatch, audio):
    decoder = ModuleType("librosa")
    decoder.__spec__ = importlib.util.spec_from_loader("librosa", loader=None)
    monkeypatch.setattr(decoder, "load", Mock(return_value=(audio, 16000)), raising=False)
    monkeypatch.setitem(sys.modules, "librosa", decoder)
    spec = importlib.util.spec_from_file_location(
        "step_audio2_example_variants", EXAMPLES / "offline_inference/step_audio2/end2end.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def test_base_tts_and_think_offline_prompt_builders(tmp_path, monkeypatch):
    audio = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    example = _load_end2end(monkeypatch, audio)

    assert example.resolve_variant("stepfun-ai/Step-Audio-2-mini-Base") == "base"
    assert example.resolve_variant("stepfun-ai/Step-Audio-2-mini-Think") == "think"
    assert example.resolve_variant("stepfun-ai/Step-Audio-2-mini") == "mini"

    tokenizer = Mock()
    tokenizer.encode.side_effect = [[10, 11], [20, 21]]
    tts_query = example.get_text_to_audio_query(text="hello", variant="base", tokenizer=tokenizer)
    assert tts_query.inputs == {"prompt_token_ids": [10, 11, 20, 21]}
    assert tokenizer.encode.call_args_list == [
        call("以自然的语速读出下面的文字。\n", add_special_tokens=False),
        call("hello<tts_start>", add_special_tokens=False),
    ]

    audio_path = tmp_path / "question.wav"
    audio_path.touch()
    think_query = example.get_think_query(str(audio_path))
    assert think_query.inputs["prompt"].startswith("<|BOT|>system\n")
    assert think_query.inputs["prompt"].endswith(
        "<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n\n<think>\n"
    )
    assert think_query.limit_mm_per_prompt == {"audio": 1}
