# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the Qwen3-Omni default talker voice.

The released checkpoint lists ``talker_config.speaker_id`` as
``{"chelsie": 2301, "ethan": 2302, "aiden": 2303}``. The default voice used to
be the first key, so a request that named no speaker was spoken by Chelsie,
while the reference ``generate(speaker="Ethan")`` (and other servers) use Ethan.
Benchmarks that omit ``speaker`` then compared two different voices.
"""

import logging

import pytest

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    DEFAULT_SPEAKER,
    Qwen3OmniMoeForConditionalGeneration,
    resolve_default_speaker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Order as shipped in Qwen/Qwen3-Omni-30B-A3B-Instruct config.json.
CHECKPOINT_SPEAKER_IDS = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}


def test_default_speaker_is_ethan_even_when_listed_second() -> None:
    assert DEFAULT_SPEAKER == "ethan"
    assert resolve_default_speaker(CHECKPOINT_SPEAKER_IDS) == "ethan"


def test_default_speaker_falls_back_to_first_key_without_ethan() -> None:
    assert resolve_default_speaker({"chelsie": 1, "aiden": 2}) == "chelsie"


def test_unknown_or_missing_speaker_uses_ethan_token() -> None:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.tts_text_spk_token_ids = dict(CHECKPOINT_SPEAKER_IDS)
    model.default_tts_text_spk_type = resolve_default_speaker(model.tts_text_spk_token_ids)

    assert model._get_text_spk_token_id("not-a-speaker") == CHECKPOINT_SPEAKER_IDS["ethan"]
    assert model._get_text_spk_token_id(model.default_tts_text_spk_type) == CHECKPOINT_SPEAKER_IDS["ethan"]
    assert model._get_text_spk_token_id("chelsie") == CHECKPOINT_SPEAKER_IDS["chelsie"]


@pytest.mark.parametrize("speaker", ["chelsie", "ethan", "aiden"])
def test_speaker_selection_logs_effective_token_without_mutating_default(speaker, caplog, monkeypatch):
    logger = logging.getLogger(f"{__name__}.speaker")
    monkeypatch.setattr("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni.logger", logger)
    caplog.set_level(logging.DEBUG, logger=logger.name)
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.tts_text_spk_token_ids = dict(CHECKPOINT_SPEAKER_IDS)
    model.default_tts_text_spk_type = "ethan"
    assert model._get_text_spk_token_id(speaker) == CHECKPOINT_SPEAKER_IDS[speaker]
    assert (
        caplog.records[-1].getMessage() == f"[talker] speaker={speaker} speaker_token={CHECKPOINT_SPEAKER_IDS[speaker]}"
    )
    assert model.default_tts_text_spk_type == "ethan"
    assert model._get_text_spk_token_id("unknown") == CHECKPOINT_SPEAKER_IDS["ethan"]
    assert caplog.records[-1].getMessage() == "[talker] speaker=ethan speaker_token=2302"
